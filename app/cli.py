"""Headless runs: the same benchmark without a window.

Rented cloud GPUs are reached over SSH and have no display, and a benchmark
worth comparing across cards has to be scriptable. This drives the same
TrainerWorker and TelemetryWorker the window does, on a QCoreApplication
event loop, prints progress and the summary table to the terminal, and always
exports CSV, PNG and the JSON report — a headless run exists to produce them.
"""
import argparse
import re
import signal
import sys

from PyQt6.QtCore import QCoreApplication, QObject, QTimer, pyqtSignal, pyqtSlot

from app.core.telemetry import TelemetryWorker
from app.core.trainer import (
    DEFAULT_PRESET,
    MODELS,
    PRECISIONS,
    PRESETS,
    SEQ_LENGTHS,
    TrainerWorker,
    estimate_weights_gib,
)
from app.utils.display import RateTracker, format_progress, summary_rows
from app.utils.logger import BenchmarkLogger
from app.utils.report import build_report

# Exit status after Ctrl+C, by the shell's 128 + SIGINT convention: the
# results are written, but a script can still tell the run was cut short.
EXIT_INTERRUPTED = 130

# Workload flags. Each one overrides the preset; left out, the preset decides.
_WORKLOAD_FLAGS = ("model", "precision", "steps", "batch_size", "seq_len",
                   "gradient_checkpointing")


def _normalise(name: str) -> str:
    """'Qwen2.5 7B', 'qwen2.5-7b' and 'QWEN2.5_7B' all become 'qwen2.57b'."""
    return re.sub(r"[\s\-_()]", "", name.lower())


def resolve(registry, name: str, what: str):
    """Finds a model, precision or preset by label, HF id or unique prefix.

    Labels such as "INT4 (NF4)" or "Phi-3.5 mini 3.8B" are awkward to type,
    so "int4" and "phi-3.5-mini" are accepted too — as long as they match
    exactly one entry. "qwen2.5" matches three and is refused with all three
    named, rather than silently picking one.
    """
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


def _at_least(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LoRA fine-tuning GPU benchmark. Opens the window unless "
                    "--headless is given.")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window, print the results and exit")
    parser.add_argument("--list", action="store_true",
                        help="list presets, models and precisions, then exit")

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
    return parser


def workload_from_args(args: argparse.Namespace) -> dict:
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
    if not args.headless and given:
        return (f"{', '.join(given)} only apply with --headless; "
                "the window has its own Workload panel.")
    return None


def print_registry(out=None) -> None:
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

    def __init__(self, workload: dict, output_dir: str = "logs", out=None,
                 trainer_cls=TrainerWorker, telemetry_cls=TelemetryWorker,
                 interval_ms: int = 250, parent=None):
        super().__init__(parent)
        self.workload = workload
        self.output_dir = output_dir
        self.out = out or sys.stdout
        self.exit_code: int | None = None
        self.logger = BenchmarkLogger()
        self._device: dict | None = None
        self._sampler_status: dict | None = None
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
    def _on_device_info(self, info: dict) -> None:
        self._device = info
        self._print(f"GPU: {info.get('name')}, driver {info.get('driver_version') or '?'}, "
                    f"CUDA {info.get('cuda_driver_version') or '?'}")

    @pyqtSlot(dict)
    def _on_sampler_status(self, status: dict) -> None:
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
    def _on_finished(self, summary: dict) -> None:
        df = self.logger.stop()
        if df is None:
            self._print("\nNo telemetry was collected, so there is nothing to export.")
            self._print(f"Steps/sec: {summary['steps_per_sec']}")
            self._finish(1)
            return

        rows = summary_rows(df, summary)
        width = max(len(metric) for metric, _ in rows)
        self._print("\n=== Benchmark Summary ===")
        for metric, value in rows:
            self._print(f"{metric:<{width}}  {value}")

        report = build_report(df, summary, self._device,
                              (self._loss_steps, self._loss_values))
        try:
            paths = BenchmarkLogger.export(df, self.output_dir, report=report)
        except OSError as e:
            self._print(f"\nExport failed: {e}")
            self._finish(1)
            return
        self._print("")
        for path in paths:
            if path:
                self._print(f"Exported: {path}")
        self._finish(EXIT_INTERRUPTED if self._interrupted else 0)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
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


def run_headless(args: argparse.Namespace) -> int:
    if args.list:
        print_registry()
        return 0

    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    run = HeadlessRun(workload_from_args(args), args.output_dir)
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
