"""Headless runs: the same benchmark without a window.

Rented cloud GPUs are reached over SSH and have no display, and a benchmark
worth comparing across cards has to be scriptable. This drives the same
TrainerWorker and TelemetryWorker the window does, on a QCoreApplication
event loop, prints progress and the summary table to the terminal, and always
exports CSV, PNG and the JSON report — a headless run exists to produce them.
"""
import argparse
import os
import re
import signal
import sys
from collections.abc import Callable, Iterable
from typing import Any, TextIO, TypeVar

from PyQt6.QtCore import QCoreApplication, QObject, QTimer, pyqtSignal, pyqtSlot

from gpu_bench import __version__
from gpu_bench.batch import EXIT_INTERRUPTED, Cooldown, Series, Sweep, grid
from gpu_bench.core.telemetry import TelemetryWorker, read_vram_total_mib
from gpu_bench.core.trainer import (
    DEFAULT_PRESET,
    MODELS,
    PRECISIONS,
    PRESETS,
    SEQ_LENGTHS,
    ModelSpec,
    PrecisionSpec,
    TrainerWorker,
    WorkloadPreset,
    estimate_weights_gib,
    fits_in_vram,
)
from gpu_bench.utils.display import RateTracker, format_progress, summary_rows
from gpu_bench.utils.history import History, record_run
from gpu_bench.utils.logger import BenchmarkLogger, ExportPaths
from gpu_bench.utils.report import build_report

# Workload flags. Each one overrides the preset; left out, the preset decides.
_WORKLOAD_FLAGS = ("model", "precision", "steps", "batch_size", "seq_len",
                   "gradient_checkpointing")


_Spec = TypeVar("_Spec", ModelSpec, PrecisionSpec, WorkloadPreset)


def _normalise(name: str) -> str:
    """'Qwen2.5 7B', 'qwen2.5-7b' and 'QWEN2.5_7B' all become 'qwen2.57b'."""
    return re.sub(r"[\s\-_()]", "", name.lower())


def resolve(registry: Iterable[_Spec], name: str, what: str) -> _Spec:
    """Finds a model, precision or preset by label, HF id or unique prefix.

    Labels such as "INT4 (NF4)" or "Phi-3.5 mini 3.8B" are awkward to type,
    so "int4" and "phi-3.5-mini" are accepted too — as long as they match
    exactly one entry. "qwen2.5" matches three and is refused with all three
    named, rather than silently picking one.
    """
    registry = list(registry)
    key = _normalise(name)
    exact = [spec for spec in registry
             if key in (_normalise(spec.label),
                        _normalise(getattr(spec, "model_id", "")))]
    if exact:
        return exact[0]

    matches = [spec for spec in registry if _normalise(spec.label).startswith(key)]
    if len(matches) == 1:
        return matches[0]
    labels = ", ".join(spec.label for spec in (matches or registry))
    reason = "is ambiguous between" if matches else "is not one of"
    raise argparse.ArgumentTypeError(f"{what} '{name}' {reason}: {labels}")


def _at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-bench",
        description="LoRA fine-tuning GPU benchmark. Opens the window unless "
                    "--headless is given.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window, print the results and exit")
    parser.add_argument("--list", action="store_true",
                        help="list presets, models and precisions, then exit")
    parser.add_argument("--history", nargs="?", const=20, type=_at_least(1), metavar="N",
                        help="list the N most recent runs in the history (default 20), "
                             "then exit")
    parser.add_argument("--history-import", metavar="DIR",
                        help="add every run report under DIR to the history, then exit")

    workload = parser.add_argument_group(
        "workload (headless only; each flag overrides the preset)")
    workload.add_argument("--preset", type=lambda s: resolve(PRESETS, s, "preset"),
                          default=DEFAULT_PRESET,
                          help=f"one of {', '.join(p.label for p in PRESETS)} "
                               f"(default: {DEFAULT_PRESET.label})")
    workload.add_argument("--model", type=lambda s: resolve(MODELS, s, "model"),
                          help="label, Hugging Face id or unique prefix, e.g. qwen2.5-7b")
    workload.add_argument("--precision",
                          type=lambda s: resolve(PRECISIONS, s, "precision"),
                          help="BF16, FP16, FP32, INT8 or INT4")
    workload.add_argument("--steps", type=_at_least(1), help="timed steps")
    workload.add_argument("--batch-size", type=_at_least(1), help="per-device batch size")
    workload.add_argument("--seq-len", type=int, choices=SEQ_LENGTHS,
                          help="tokens per sequence")
    workload.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                          default=None, help="recompute activations to save VRAM")
    workload.add_argument("--warmup", type=_at_least(0), default=10,
                          help="untimed steps before measuring (default: 10)")
    workload.add_argument("--output-dir", default="logs",
                          help="where CSV, PNG and report go (default: logs)")

    batch = parser.add_argument_group("several runs (headless only)")
    batch.add_argument("--repeat", type=_at_least(1), default=1, metavar="N",
                       help="run the workload N times and report the mean and spread, "
                            "outliers dropped (default: 1)")
    batch.add_argument("--sweep", nargs="+", type=parse_axis, action=_SweepAxes,
                       default={}, metavar="KEY=V1,V2",
                       help="run every combination, e.g. --sweep batch-size=1,2,4 "
                            "seq-len=256,512 precision=bf16,int4; with --repeat, each "
                            "combination is a series")
    batch.add_argument("--cooldown", type=_at_least(0), default=0, metavar="SECONDS",
                       help="between runs, wait up to this long for the GPU to cool back "
                            "to its starting temperature (default: 0, no wait)")
    return parser


# What --sweep can vary, and the spellings it accepts for each.
_SWEEP_KEYS = {
    "model": "model", "precision": "precision",
    "batch_size": "batch_size", "batch": "batch_size",
    "seq_len": "seq_len", "seq": "seq_len",
    "gradient_checkpointing": "gradient_checkpointing", "checkpointing": "gradient_checkpointing",
}
_BOOLEANS = {"on": True, "true": True, "yes": True, "1": True,
             "off": False, "false": False, "no": False, "0": False}


def parse_axis(text: str) -> tuple[str, list[Any]]:
    """'batch-size=1,2,4' → ('batch_size', [1, 2, 4]); values checked like the flags."""
    name, sep, values = text.partition("=")
    key = _SWEEP_KEYS.get(name.strip().lower().replace("-", "_"))
    if not sep or key is None:
        raise argparse.ArgumentTypeError(
            f"'{text}' is not KEY=V1,V2,... with KEY one of "
            "model, precision, batch-size, seq-len, gradient-checkpointing")
    items = [v.strip() for v in values.split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"no values given for {name}")
    parsed: list[Any] = []
    for item in items:
        if key == "model":
            parsed.append(resolve(MODELS, item, "model"))
        elif key == "precision":
            parsed.append(resolve(PRECISIONS, item, "precision"))
        elif key == "batch_size":
            parsed.append(_at_least(1)(item))
        elif key == "seq_len":
            if not item.isdigit() or int(item) not in SEQ_LENGTHS:
                raise argparse.ArgumentTypeError(
                    f"sequence length {item} is not one of {', '.join(map(str, SEQ_LENGTHS))}")
            parsed.append(int(item))
        elif item.lower() in _BOOLEANS:
            parsed.append(_BOOLEANS[item.lower()])
        else:
            raise argparse.ArgumentTypeError(f"'{item}' is not on or off")
    return key, parsed


class _SweepAxes(argparse.Action):
    """Collects --sweep's KEY=values items into one {key: values} dict."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        axes: dict[str, list[Any]] = {}
        for key, items in values:
            if key in axes:
                raise argparse.ArgumentError(self, f"{key} is swept twice")
            axes[key] = items
        setattr(namespace, self.dest, axes)


def workload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """TrainerWorker keyword arguments: the preset, with any flags on top."""
    preset = args.preset
    workload = {
        "model": preset.model,
        "precision": preset.precision,
        "steps": preset.steps,
        "batch_size": preset.batch_size,
        "seq_len": preset.seq_len,
        "gradient_checkpointing": preset.gradient_checkpointing,
    }
    for flag in _WORKLOAD_FLAGS:
        value = getattr(args, flag)
        if value is not None:
            workload[flag] = value
    workload["warmup_steps"] = args.warmup
    return workload


def gui_only_misuse(args: argparse.Namespace) -> str | None:
    """Explains why workload flags were ignored, if they were given without --headless."""
    given = [f"--{flag.replace('_', '-')}" for flag in _WORKLOAD_FLAGS
             if getattr(args, flag) is not None]
    if args.repeat != 1:
        given.append("--repeat")
    if args.cooldown:
        given.append("--cooldown")
    if args.sweep:
        given.append("--sweep")
    if not args.headless and given:
        return (f"{', '.join(given)} only apply with --headless; "
                "the window has its own Workload panel.")
    return None


def print_registry(out: TextIO | None = None) -> None:
    # Looked up per call, not bound as a default: sys.stdout may have been
    # replaced (redirected, captured) since the module was imported.
    out = out or sys.stdout
    print("Presets:", file=out)
    for p in PRESETS:
        print(f"  {p.label:<10} {p.model.label}, {p.precision.label}, {p.steps} steps, "
              f"batch {p.batch_size}, {p.seq_len} tokens"
              f"{', checkpointing' if p.gradient_checkpointing else ''}", file=out)
    print("\nModels:", file=out)
    for m in MODELS:
        print(f"  {m.label:<18} {m.model_id}  ({m.params_b:.2f}B)", file=out)
    print("\nPrecisions:", file=out)
    for precision in PRECISIONS:
        print(f"  {precision.label}", file=out)


class HeadlessRun(QObject):
    """One benchmark run wired to stdout instead of widgets.

    Emits done(exit_code) once the run is over, its threads are joined and
    the results are written. The trainer and telemetry classes are
    parameters so tests can substitute workers that start no threads.
    """

    done = pyqtSignal(int)

    def __init__(self, workload: dict[str, Any], output_dir: str = "logs",
                 out: TextIO | None = None,
                 trainer_cls: type[TrainerWorker] = TrainerWorker,
                 telemetry_cls: type[TelemetryWorker] = TelemetryWorker,
                 interval_ms: int = 250, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.workload = workload
        self.output_dir = output_dir
        self.out = out or sys.stdout
        self.exit_code: int | None = None
        # The finished run's report and where it was written, for a series.
        self.report: dict[str, Any] | None = None
        self.paths: ExportPaths | None = None
        self.error: str | None = None
        self.logger = BenchmarkLogger()
        self._device: dict[str, Any] | None = None
        self._sampler_status: dict[str, Any] | None = None
        self._loss_steps: list[int] = []
        self._loss_values: list[float] = []
        self._rate = RateTracker()
        self._total = 0
        self._last_loss: float | None = None
        self._interrupted = False

        self.telemetry = telemetry_cls(interval_ms=interval_ms)
        self.telemetry.data_updated.connect(self.logger.log)
        self.telemetry.device_info_ready.connect(self._on_device_info)
        self.telemetry.sampler_status.connect(self._on_sampler_status)
        self.telemetry.power_trace.connect(self.logger.log_power_trace)
        self.telemetry.error_occurred.connect(self._on_telemetry_error)
        self._last_telemetry_error: str | None = None

        self.trainer = trainer_cls(**workload)
        self.trainer.status_updated.connect(self._print)
        self.trainer.max_steps_ready.connect(self._on_max_steps)
        self.trainer.progress_updated.connect(self._on_progress)
        self.trainer.loss_updated.connect(self._on_loss)
        self.trainer.training_finished.connect(self._on_finished)
        self.trainer.error_occurred.connect(self._on_error)

    def _print(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    def start(self) -> None:
        w = self.workload
        self._print(
            f"Benchmark: {w['model'].label} in {w['precision'].label}, "
            f"{w['steps']} steps + {w['warmup_steps']} warm-up, batch {w['batch_size']}, "
            f"{w['seq_len']} tokens, checkpointing "
            f"{'on' if w['gradient_checkpointing'] else 'off'} "
            f"(weights ≈ {estimate_weights_gib(w['model'], w['precision']):.1f} GiB)")
        self.logger.start()
        self.telemetry.start()
        self.trainer.start()

    def request_stop(self) -> None:
        """Ctrl+C: end at the next step boundary, still writing the results."""
        self._interrupted = True
        self._print("\nStop requested — finishing the current step "
                    "(the results are still written)...")
        self.trainer.stop()

    @pyqtSlot(dict)
    def _on_device_info(self, info: dict[str, Any]) -> None:
        self._device = info
        self._print(f"GPU: {info.get('name')}, driver {info.get('driver_version') or '?'}, "
                    f"CUDA {info.get('cuda_driver_version') or '?'}")

    @pyqtSlot(dict)
    def _on_sampler_status(self, status: dict[str, Any]) -> None:
        self._sampler_status = status
        if status["active"]:
            self._print(f"Power sampler: native, {status['detail']}, "
                        f"every {status['interval_us'] / 1000:g} ms")
        else:
            self._print(f"Power sampler: {status['detail']}")

    @pyqtSlot(str)
    def _on_telemetry_error(self, message: str) -> None:
        # Repeats every poll; one line is enough.
        if message != self._last_telemetry_error:
            self._last_telemetry_error = message
            self._print(f"TELEMETRY ERROR: {message}")

    @pyqtSlot(int)
    def _on_max_steps(self, total: int) -> None:
        self._total = total

    @pyqtSlot(int)
    def _on_progress(self, step: int) -> None:
        rate = self._rate.update(step)
        # A line per step would scroll a long run off any terminal; twenty
        # updates across the run, plus the last step, are plenty.
        every = max(1, self._total // 20)
        if step % every == 0 or step == self._total:
            line = format_progress(step, self._total, rate)
            if self._last_loss is not None:
                line += f" · loss {self._last_loss:.3f}"
            self._print(line)

    @pyqtSlot(int, float)
    def _on_loss(self, step: int, loss: float) -> None:
        self._loss_steps.append(step)
        self._loss_values.append(loss)
        self._last_loss = loss

    @pyqtSlot(dict)
    def _on_finished(self, summary: dict[str, Any]) -> None:
        df = self.logger.stop()
        if df is None:
            self._print("\nNo telemetry was collected, so there is nothing to export.")
            self._print(f"Steps/sec: {summary['steps_per_sec']}")
            self._finish(1)
            return

        trace = self.logger.power_trace()
        rows = summary_rows(df, summary, trace, self._device)
        width = max(len(metric) for metric, _ in rows)
        self._print("\n=== Benchmark Summary ===")
        for metric, value in rows:
            self._print(f"{metric:<{width}}  {value}")

        report = build_report(df, summary, self._device,
                              (self._loss_steps, self._loss_values),
                              trace, self._sampler_status)
        try:
            paths = BenchmarkLogger.export(df, self.output_dir, report=report, trace=trace,
                                           summary=summary)
        except OSError as e:
            self._print(f"\nExport failed: {e}")
            self._finish(1)
            return
        self.report = report if paths.report else None
        self.paths = paths
        self._print("")
        for path in paths:
            if path:
                self._print(f"Exported: {path}")
        if paths.report:
            problem = record_run(paths.report, report)
            if problem:
                self._print(f"Note: {problem}")
        self._finish(EXIT_INTERRUPTED if self._interrupted else 0)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self.error = message
        self._print(f"\nERROR: {message}")
        df = self.logger.stop()
        if df is not None:
            try:
                for path in BenchmarkLogger.export(df, self.output_dir)[:2]:
                    self._print(f"Exported telemetry: {path}")
            except OSError as e:
                self._print(f"Export failed: {e}")
        self._finish(1)

    def _finish(self, code: int) -> None:
        # The trainer emits its result before freeing VRAM, so its thread
        # may still be running; the process must not exit underneath it.
        self.telemetry.stop()
        self.trainer.wait_for_exit()
        self.exit_code = code
        self.done.emit(code)


def split_by_fit(configs: list[dict[str, Any]], vram_total_mib: float | None
                 ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Separates the configurations whose weights alone would not fit.

    The same estimate as the window's fit hint; without a VRAM reading
    nothing is skipped, and an out-of-memory failure is caught at run time
    instead.
    """
    if vram_total_mib is None:
        return configs, []
    runnable, skipped = [], []
    for config in configs:
        if fits_in_vram(config["model"], config["precision"], vram_total_mib):
            runnable.append(config)
        else:
            need = estimate_weights_gib(config["model"], config["precision"])
            skipped.append((config, f"weights need {need:.1f} GiB of "
                                    f"{vram_total_mib / 1024:.1f} GiB"))
    return runnable, skipped


def print_history(limit: int, out: TextIO | None = None) -> None:
    out = out or sys.stdout
    with History() as history:
        rows = history.runs(limit)
        print(f"History: {history.path}", file=out)
    if not rows:
        print("No runs yet. Every exported run is added; --history-import DIR adds "
              "older ones.", file=out)
        return
    table = [("When", "GPU", "Model", "Precision", "Batch", "Seq", "Tokens/s", "Tokens/J")]
    for r in rows:
        table.append((r.created_at[:16].replace("T", " "), r.gpu or "?", r.model or "?",
                      r.precision or "?", str(r.batch_size or "?"), str(r.seq_len or "?"),
                      f"{r.tokens_per_sec:,.0f}" if r.tokens_per_sec else "—",
                      f"{r.tokens_per_joule:.1f}" if r.tokens_per_joule else "—"))
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for row in table:
        print("  ".join(cell.ljust(w) if i < 4 else cell.rjust(w)
                        for i, (cell, w) in enumerate(zip(row, widths, strict=True))),
              file=out)


def import_history(directory: str, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    if not os.path.isdir(directory):
        print(f"{directory} is not a directory.", file=out)
        return 1
    with History() as history:
        added, unreadable = history.import_dir(directory)
        print(f"Added {added} run{'' if added == 1 else 's'} to {history.path}.", file=out)
    for problem in unreadable:
        print(f"Skipped {problem}", file=out)
    return 0


def run_headless(args: argparse.Namespace) -> int:
    if args.list:
        print_registry()
        return 0
    if args.history_import:
        return import_history(args.history_import)
    if args.history:
        print_history(args.history)
        return 0

    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    workload = workload_from_args(args)

    cooldown = Cooldown(args.cooldown)
    run: HeadlessRun | Series | Sweep
    if args.sweep:
        configs, skipped = split_by_fit(grid(workload, args.sweep), read_vram_total_mib())
        run = Sweep(configs, list(args.sweep), HeadlessRun, args.repeat, args.output_dir,
                    cooldown=cooldown, skipped=skipped)
    elif args.repeat > 1:
        run = Series(lambda: HeadlessRun(workload, args.output_dir), args.repeat,
                     args.output_dir, cooldown=cooldown)
    else:
        run = HeadlessRun(workload, args.output_dir)
    run.done.connect(app.exit)

    # Python only runs signal handlers between bytecodes, and Qt's event loop
    # sits in C++ — without a timer handing control back now and then, Ctrl+C
    # would not be noticed until the run ended by itself.
    signal.signal(signal.SIGINT, lambda *_: run.request_stop())
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)

    QTimer.singleShot(0, run.start)
    return app.exec()
