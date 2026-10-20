"""Several headless runs in a row, reduced to one result.

A single HeadlessRun (gpu_bench.cli) measures once. Here they are queued
back to back on the same event loop — the next one starts only once the
last has joined its threads and freed its VRAM — and their reports are
combined: --repeat runs one workload N times for a mean with an error bar,
and --sweep runs a grid of workloads, each one a series of its own.

Back to back, every run after the first would start on a card the last one
left hot, and lose to it on clocks before measuring anything. --cooldown
waits between runs until the GPU is back near the temperature it started
at, for at most that many seconds.

The runs are made by a factory rather than constructed here, so tests can
hand in runs with threadless workers, and so this module does not import
the CLI that drives it.
"""
import itertools
import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TextIO

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from gpu_bench.core import series
from gpu_bench.core.telemetry import read_gpu_temperature
from gpu_bench.utils.logger import unique_stamp, write_json

if TYPE_CHECKING:
    from gpu_bench.cli import HeadlessRun

# Back within this much of the starting temperature counts as cooled down:
# closer, and the last degree or two of an exponential decay would take
# longer than the run itself.
COOLDOWN_MARGIN_C = 2.0
COOLDOWN_POLL_MS = 1000

# Exit status after Ctrl+C, by the shell's 128 + SIGINT convention: the
# results are written, but a script can still tell the run was cut short.
EXIT_INTERRUPTED = 130


def _say(out: TextIO, text: str) -> None:
    print(text, file=out, flush=True)


class Cooldown(QObject):
    """Waits for the GPU to get back near a baseline temperature.

    One instance is shared by everything in a batch, so the baseline is the
    temperature before the very first run, not before the latest.
    """

    def __init__(self, limit_s: float, out: TextIO | None = None,
                 read_temperature: Callable[[], float | None] = read_gpu_temperature,
                 clock: Callable[[], float] = time.monotonic,
                 poll_ms: int = COOLDOWN_POLL_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.limit_s = limit_s
        self.out = out or sys.stdout
        self.read_temperature = read_temperature
        self.clock = clock
        self.baseline_c: float | None = None
        self._then: Callable[[], None] | None = None
        self._since = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._check)

    @property
    def active(self) -> bool:
        return self._timer.isActive()

    def measure_baseline(self) -> None:
        """Reads the starting temperature; once, before the first run."""
        if self.limit_s <= 0 or self.baseline_c is not None:
            return
        self.baseline_c = self.read_temperature()
        if self.baseline_c is None:
            _say(self.out, "No GPU temperature reading, so no cool-down between runs.")
        else:
            _say(self.out, f"Starting temperature: {self.baseline_c:.0f} °C — later runs "
                           f"wait (up to {self.limit_s:g} s) to get back near it.")

    def wait(self, then: Callable[[], None]) -> None:
        """Calls `then` once cooled down, or straight away when there is nothing to wait for."""
        if self.limit_s <= 0 or self.baseline_c is None:
            then()
            return
        temp = self.read_temperature()
        if temp is not None and temp <= self.baseline_c + COOLDOWN_MARGIN_C:
            then()
            return
        _say(self.out, f"Cooling down from {temp:.0f} °C towards {self.baseline_c:.0f} °C..."
                       if temp is not None else "Cooling down...")
        self._then, self._since = then, self.clock()
        self._timer.start()

    def cancel(self) -> None:
        self._timer.stop()
        self._then = None

    def _check(self) -> None:
        assert self.baseline_c is not None
        waited = self.clock() - self._since
        temp = self.read_temperature()
        if temp is not None and temp <= self.baseline_c + COOLDOWN_MARGIN_C:
            _say(self.out, f"Cooled to {temp:.0f} °C in {waited:.0f} s.")
        elif waited >= self.limit_s:
            reading = f"{temp:.0f} °C" if temp is not None else "an unknown temperature"
            _say(self.out, f"Still at {reading} after {waited:.0f} s; starting anyway.")
        else:
            return
        then, self._then = self._then, None
        self._timer.stop()
        if then is not None:
            then()


class Series(QObject):
    """The same workload `repeat` times, then the series summary and record.

    Emits done(exit_code): 0 when every run finished, the first non-zero
    code otherwise — 130 after Ctrl+C, which ends the series after the run
    in progress. Whatever runs did finish are still summarised, and `error`
    keeps the message of a run that failed.
    """

    done = pyqtSignal(int)

    def __init__(self, make_run: Callable[[], "HeadlessRun"], repeat: int,
                 output_dir: str = "logs", out: TextIO | None = None,
                 cooldown: Cooldown | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.make_run = make_run
        self.repeat = repeat
        self.output_dir = output_dir
        self.out = out or sys.stdout
        self.cooldown = cooldown or Cooldown(0, self.out, parent=self)
        self.reports: list[dict[str, Any]] = []
        self.files: list[str] = []
        self.record: dict[str, Any] | None = None
        self.path: str | None = None
        self.error: str | None = None
        self._current: HeadlessRun | None = None
        self._finished_runs = 0
        self._interrupted = False

    def _print(self, text: str) -> None:
        _say(self.out, text)

    def start(self) -> None:
        self.cooldown.measure_baseline()
        self._start_next()

    def request_stop(self) -> None:
        self._interrupted = True
        if self.cooldown.active:
            self.cooldown.cancel()
            self._conclude(0)
        elif self._current is not None:
            self._current.request_stop()

    def _start_next(self) -> None:
        if self.repeat > 1:
            self._print(f"\n=== Run {self._finished_runs + 1} of {self.repeat} ===")
        run = self.make_run()
        run.done.connect(self._on_run_done)
        self._current = run
        run.start()

    @pyqtSlot(int)
    def _on_run_done(self, code: int) -> None:
        run = self._current
        self._finished_runs += 1
        if run is not None and run.report is not None and run.paths is not None:
            self.reports.append(run.report)
            self.files.append(os.path.basename(run.paths.report or ""))
        if run is not None and run.error:
            self.error = run.error
        if code == 0 and not self._interrupted and self._finished_runs < self.repeat:
            # Queued rather than called: the finished run's slot is still on
            # the stack, and its threads must be gone before the next starts.
            QTimer.singleShot(0, lambda: self.cooldown.wait(self._start_next))
            return
        self._conclude(code)

    def _conclude(self, code: int) -> None:
        if self.reports:
            self.record = series.build_record(self.reports, self.files)
        if len(self.reports) >= 2 and self.record is not None:
            self._print_summary(self.record)
            try:
                self.path = write_json(self.output_dir, "series", self.record)
                self._print(f"Exported: {self.path}")
            except OSError as e:
                self._print(f"Export failed: {e}")
                code = code or 1
        elif self.repeat > 1:
            self._print("\nFewer than two runs finished, so there is no series to summarise.")
        if code == 0 and self._interrupted:
            code = EXIT_INTERRUPTED
        self.done.emit(code)

    def _print_summary(self, record: dict[str, Any]) -> None:
        dropped = record["dropped"]
        title = f"\n=== Series: {record['runs']} runs"
        if dropped:
            numbers = ", ".join(str(i + 1) for i in dropped)
            one = len(dropped) == 1
            title += (f", {len(dropped)} dropped as {'an outlier' if one else 'outliers'} "
                      f"(run{'' if one else 's'} {numbers})")
        self._print(title + " ===")
        rows = [(label, series.format_aggregate(record["metrics"][key], fmt))
                for key, label, fmt, _ in series.METRICS if record["metrics"][key]]
        width = max((len(label) for label, _ in rows), default=0)
        for label, value in rows:
            self._print(f"{label:<{width}}  {value}")


# --- sweeps ---------------------------------------------------------------------

def grid(base: dict[str, Any], axes: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of the axes' values, laid over the base workload.

    In the order the axes were given, the last one varying fastest — the
    order a person reading the table expects.
    """
    keys = list(axes)
    return [{**base, **dict(zip(keys, combo, strict=True))}
            for combo in itertools.product(*(axes[k] for k in keys))]


def describe(workload: dict[str, Any], keys: Sequence[str]) -> str:
    """The swept settings of one configuration, as a short label."""
    parts = []
    for key in keys:
        value = workload[key]
        if key == "batch_size":
            parts.append(f"batch {value}")
        elif key == "seq_len":
            parts.append(f"{value} tokens")
        elif key == "gradient_checkpointing":
            parts.append("checkpointing" if value else "no checkpointing")
        else:
            parts.append(getattr(value, "label", str(value)))
    return ", ".join(parts)


def failure_reason(error: str | None) -> str:
    """What to put in the table for a configuration that did not finish."""
    if error and "out of memory" in error.lower():
        return "out of memory"
    return "failed"


def _cell(metrics: dict[str, Any], key: str, fmt: str, spread: bool = True) -> str:
    """A metric's mean, with its spread if asked, for a table cell."""
    if not metrics.get(key):
        return "—"
    if not spread:
        return fmt.format(metrics[key]["mean"])
    return series.format_aggregate(metrics[key], fmt).split("  (")[0]


class Sweep(QObject):
    """A grid of workloads, each run as a Series, into a directory of its own.

    A configuration that fails — out of memory is the usual one at the top
    of a batch-size axis — is marked as such and the sweep goes on with the
    next; Ctrl+C ends it after the run in progress. Emits done(exit_code):
    0 when at least one configuration finished, 1 when none did, 130 after
    Ctrl+C.
    """

    done = pyqtSignal(int)

    def __init__(self, configs: Sequence[dict[str, Any]], keys: Sequence[str],
                 make_run: Callable[[dict[str, Any], str], "HeadlessRun"],
                 repeat: int = 1, output_dir: str = "logs", out: TextIO | None = None,
                 cooldown: Cooldown | None = None,
                 skipped: Sequence[tuple[dict[str, Any], str]] = (),
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.configs = list(configs)
        self.keys = list(keys)
        self.make_run = make_run
        self.repeat = repeat
        self.out = out or sys.stdout
        self.cooldown = cooldown or Cooldown(0, self.out, parent=self)
        self.skipped = list(skipped)
        self.output_dir = os.path.join(output_dir,
                                       f"sweep_{unique_stamp(output_dir, 'sweep_{}')}")
        # One entry per configuration: its workload, how it ended, its series.
        self.results: list[dict[str, Any]] = []
        self._current: Series | None = None
        self._interrupted = False

    def _print(self, text: str) -> None:
        _say(self.out, text)

    def start(self) -> None:
        runs = f" × {self.repeat} runs" if self.repeat > 1 else ""
        plan = f"Sweep: {len(self.configs)} configurations{runs}, into {self.output_dir}"
        if self.skipped:
            plan += f"; {len(self.skipped)} skipped"
        self._print(plan)
        for workload, reason in self.skipped:
            self._print(f"  skipped: {describe(workload, self.keys)} — {reason}")
        self.cooldown.measure_baseline()
        if not self.configs:
            self._conclude()
            return
        self._start_next()

    def request_stop(self) -> None:
        self._interrupted = True
        if self.cooldown.active:
            self.cooldown.cancel()
            self._conclude()
        elif self._current is not None:
            self._current.request_stop()

    def _start_next(self) -> None:
        index = len(self.results)
        workload = self.configs[index]
        self._print(f"\n##### Configuration {index + 1} of {len(self.configs)}: "
                    f"{describe(workload, self.keys)} #####")
        run_series = Series(lambda: self.make_run(workload, self.output_dir), self.repeat,
                            self.output_dir, self.out, self.cooldown, parent=self)
        run_series.done.connect(self._on_config_done)
        self._current = run_series
        run_series.start()

    @pyqtSlot(int)
    def _on_config_done(self, code: int) -> None:
        run_series = self._current
        assert run_series is not None
        workload = self.configs[len(self.results)]
        if code == 0 or (code == EXIT_INTERRUPTED and run_series.record):
            status = "ok"
        elif code == EXIT_INTERRUPTED:
            status = "interrupted"
        else:
            status = failure_reason(run_series.error)
        self.results.append({"workload": workload, "status": status,
                             "record": run_series.record, "files": run_series.files})
        if code == EXIT_INTERRUPTED or self._interrupted:
            self._interrupted = True
            self._conclude()
        elif len(self.results) < len(self.configs):
            QTimer.singleShot(0, lambda: self.cooldown.wait(self._start_next))
        else:
            self._conclude()

    def _conclude(self) -> None:
        self._print_table()
        if self._interrupted:
            code = EXIT_INTERRUPTED
        else:
            code = 0 if any(r["status"] == "ok" for r in self.results) else 1
        self.done.emit(code)

    def _print_table(self) -> None:
        self._print(f"\n=== Sweep: {len(self.results)} of {len(self.configs)} "
                    "configurations run ===")
        rows = [("Configuration", "Tokens/sec", "Tokens/J", "Max °C")]
        for entry in self.results:
            label = describe(entry["workload"], self.keys)
            metrics = (entry["record"] or {}).get("metrics") or {}
            if entry["status"] != "ok" or not metrics:
                rows.append((label, entry["status"], "", ""))
                continue
            rows.append((label, _cell(metrics, "tokens_per_sec", "{:,.0f}"),
                         _cell(metrics, "tokens_per_joule", "{:.1f}"),
                         _cell(metrics, "max_temp_c", "{:.0f}", spread=False)))
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        for row in rows:
            self._print("  ".join(cell.ljust(width) if i == 0 else cell.rjust(width)
                                  for i, (cell, width) in enumerate(zip(row, widths,
                                                                        strict=True))).rstrip())
