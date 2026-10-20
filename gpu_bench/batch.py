"""Several headless runs in a row, reduced to one result.

A single HeadlessRun (gpu_bench.cli) measures once. Here they are queued
back to back on the same event loop — the next one starts only once the
last has joined its threads and freed its VRAM — and their reports are
combined: --repeat runs one workload N times for a mean with an error bar.

Back to back, every run after the first would start on a card the last one
left hot, and lose to it on clocks before measuring anything. --cooldown
waits between runs until the GPU is back near the temperature it started
the series at, for at most that many seconds.

The runs are made by a factory rather than constructed here, so tests can
hand in runs with threadless workers, and so this module does not import
the CLI that drives it.
"""
import os
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TextIO

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from gpu_bench.core import series
from gpu_bench.core.telemetry import read_gpu_temperature
from gpu_bench.utils.logger import write_json

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


class Series(QObject):
    """The same workload `repeat` times, then the series summary and record.

    Emits done(exit_code): 0 when every run finished, the first non-zero
    code otherwise — 130 after Ctrl+C, which ends the series after the run
    in progress. Whatever runs did finish are still summarised.
    """

    done = pyqtSignal(int)

    def __init__(self, make_run: Callable[[], "HeadlessRun"], repeat: int,
                 output_dir: str = "logs", out: TextIO | None = None,
                 cooldown_s: float = 0.0,
                 read_temperature: Callable[[], float | None] = read_gpu_temperature,
                 clock: Callable[[], float] = time.monotonic,
                 poll_ms: int = COOLDOWN_POLL_MS,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.make_run = make_run
        self.repeat = repeat
        self.cooldown_s = cooldown_s
        self.read_temperature = read_temperature
        self.clock = clock
        self.baseline_c: float | None = None
        self._cooling_since: float | None = None
        self._cool_timer = QTimer(self)
        self._cool_timer.setInterval(poll_ms)
        self._cool_timer.timeout.connect(self._check_cooled)
        self.output_dir = output_dir
        self.out = out or sys.stdout
        self.reports: list[dict[str, Any]] = []
        self.files: list[str] = []
        self.record: dict[str, Any] | None = None
        self.path: str | None = None
        self._current: HeadlessRun | None = None
        self._finished_runs = 0
        self._interrupted = False

    def _print(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    def start(self) -> None:
        if self.cooldown_s > 0:
            self.baseline_c = self.read_temperature()
            if self.baseline_c is None:
                self._print("No GPU temperature reading, so no cool-down between runs.")
            else:
                self._print(f"Starting temperature: {self.baseline_c:.0f} °C — later runs "
                            f"wait (up to {self.cooldown_s:g} s) to get back near it.")
        self._start_next()

    def request_stop(self) -> None:
        self._interrupted = True
        if self._cool_timer.isActive():
            self._cool_timer.stop()
            self._conclude(0)
        elif self._current is not None:
            self._current.request_stop()

    def _start_next(self) -> None:
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
        if code == 0 and not self._interrupted and self._finished_runs < self.repeat:
            # Queued rather than called: the finished run's slot is still on
            # the stack, and its threads must be gone before the next starts.
            QTimer.singleShot(0, self._between_runs)
            return
        self._conclude(code)

    def _between_runs(self) -> None:
        if self.cooldown_s <= 0 or self.baseline_c is None:
            self._start_next()
            return
        self._cooling_since = self.clock()
        temp = self.read_temperature()
        if temp is not None and temp <= self.baseline_c + COOLDOWN_MARGIN_C:
            self._start_next()
            return
        self._print(f"Cooling down from {temp:.0f} °C towards {self.baseline_c:.0f} °C..."
                    if temp is not None else "Cooling down...")
        self._cool_timer.start()

    def _check_cooled(self) -> None:
        assert self.baseline_c is not None and self._cooling_since is not None
        waited = self.clock() - self._cooling_since
        temp = self.read_temperature()
        if temp is not None and temp <= self.baseline_c + COOLDOWN_MARGIN_C:
            self._print(f"Cooled to {temp:.0f} °C in {waited:.0f} s.")
        elif waited >= self.cooldown_s:
            reading = f"{temp:.0f} °C" if temp is not None else "an unknown temperature"
            self._print(f"Still at {reading} after {waited:.0f} s; starting anyway.")
        else:
            return
        self._cool_timer.stop()
        self._start_next()

    def _conclude(self, code: int) -> None:
        if len(self.reports) >= 2:
            self.record = series.build_record(self.reports, self.files)
            self._print_summary(self.record)
            try:
                self.path = write_json(self.output_dir, "series", self.record)
                self._print(f"Exported: {self.path}")
            except OSError as e:
                self._print(f"Export failed: {e}")
                code = code or 1
        else:
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
