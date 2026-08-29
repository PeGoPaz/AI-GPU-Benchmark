"""MainWindow wiring. Skipped automatically when Qt cannot start headless."""
import os

import pytest

pytestmark = pytest.mark.ui

# A Qt platform plugin is required even offscreen; skip rather than fail when
# the runner lacks the system libraries (libEGL, xkbcommon, ...).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")

from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


class FakeTrainer(QObject):
    """Stands in for TrainerWorker: same signals, but starts no thread."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, float)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    max_steps_ready = pyqtSignal(int)

    def __init__(self, steps=150, batch_size=4, parent=None):
        super().__init__(parent)
        self._started = False

    def start(self):
        self._started = True

    def isRunning(self):
        return self._started

    def stop(self):
        self._started = False

    # closeEvent() calls this; without it PyQt6 turns the AttributeError
    # raised inside a virtual override into a hard abort.
    def wait_for_exit(self, timeout_ms=30000):
        self._started = False
        return True


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:                      # noqa: BLE001
            pytest.skip(f"Qt cannot start headless here: {exc}")
    return app


@pytest.fixture
def window(qapp, monkeypatch):
    """A MainWindow whose telemetry thread never finds a GPU."""
    import pynvml
    monkeypatch.setattr(pynvml, "nvmlInit",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("no driver")))

    from app.ui.main_window import MainWindow
    win = MainWindow()
    yield win
    win.telemetry.stop()
    win.close()


def test_telemetry_labels_show_new_metrics(window, telemetry_sample):
    window.update_telemetry_ui(telemetry_sample(9))
    assert "89" in window.lbl_mem_temp.text()
    assert "16" in window.lbl_headroom.text()


def test_unsupported_fields_render_na(window, telemetry_sample):
    """GPUs without the memory-temp field must show N/A, not crash."""
    window.update_telemetry_ui(
        telemetry_sample(0, temp_memory=None, temp_headroom_c=None))
    assert "N/A" in window.lbl_mem_temp.text()
    assert "N/A" in window.lbl_headroom.text()


def _summary_rows(window):
    return {window.summary_table.item(r, 0).text(): window.summary_table.item(r, 1).text()
            for r in range(window.summary_table.rowCount())}


def test_aborted_run_is_flagged_in_summary(window, telemetry_sample):
    window.logger.start()
    for i in range(10):
        window.logger.log(telemetry_sample(i))

    window.on_benchmark_finished({
        "total_steps": 43, "requested_steps": 150, "aborted": True,
        "elapsed_time_sec": 20.0, "steps_per_sec": 2.15,
    })

    rows = _summary_rows(window)
    assert "stopped early" in rows["Steps Completed"]
    assert "Max Mem Temp (°C)" in rows
    assert window.btn_export.isEnabled()


def test_completed_run_is_not_mislabelled(window, telemetry_sample):
    window.logger.start()
    for i in range(10):
        window.logger.log(telemetry_sample(i))

    window.on_benchmark_finished({
        "total_steps": 150, "requested_steps": 150, "aborted": False,
        "elapsed_time_sec": 70.0, "steps_per_sec": 2.14,
    })

    assert "stopped early" not in _summary_rows(window)["Steps Completed"]


def test_export_is_read_only(window, telemetry_sample, tmp_path, monkeypatch):
    """Export must not consume the run or stop collection (it used to)."""
    monkeypatch.chdir(tmp_path)
    window.logger.start()
    for i in range(10):
        window.logger.log(telemetry_sample(i))

    window.export_results()
    assert window.logger.is_logging is True

    window.logger.log(telemetry_sample(10))
    window.export_results()

    assert len(window.logger.to_dataframe()) == 11
    written = os.listdir(tmp_path / "logs")
    assert any(f.endswith(".csv") for f in written)
    assert any(f.endswith(".png") for f in written)


@pytest.fixture
def started_run(window, monkeypatch):
    """A window mid-benchmark, with the trainer thread stubbed out."""
    import app.ui.main_window as mw
    monkeypatch.setattr(mw, "TrainerWorker", FakeTrainer)
    window.start_benchmark()
    assert not window.btn_start.isEnabled()
    return window


def test_training_error_restores_the_controls(started_run):
    """A failed run used to grey out Start for good: training_finished never
    fires when run() raises, and nothing else re-enabled the button."""
    started_run.trainer.error_occurred.emit("HF Training Error: CUDA out of memory")

    assert started_run.btn_start.isEnabled()
    assert not started_run.btn_stop.isEnabled()
    assert "CUDA out of memory" in started_run.console.toPlainText()


def test_failed_run_stops_collecting_telemetry(started_run, telemetry_sample):
    """Otherwise the dead run's rows would prepend themselves to the next one."""
    started_run.logger.log(telemetry_sample(0))
    started_run.trainer.error_occurred.emit("HF Training Error: boom")

    assert started_run.logger.is_logging is False
    # Partial telemetry was collected, so it stays exportable.
    assert started_run.btn_export.isEnabled()


def test_failed_run_without_telemetry_leaves_export_disabled(started_run):
    started_run.trainer.error_occurred.emit("HF Training Error: no CUDA device")

    assert not started_run.btn_export.isEnabled()


def test_telemetry_error_signal_is_connected(window):
    """It used to be emitted into the void, leaving the dashboard on '--'."""
    assert window.telemetry.receivers(window.telemetry.error_occurred) > 0


def test_repeated_telemetry_errors_are_logged_once(window):
    """Read errors repeat every 250 ms; the console must not drown in them."""
    window._on_telemetry_error("Telemetry Read Error: GPU fell off the bus")
    window._on_telemetry_error("Telemetry Read Error: GPU fell off the bus")
    window._on_telemetry_error("Telemetry Read Error: and now something else")

    text = window.console.toPlainText()
    assert text.count("fell off the bus") == 1
    assert "something else" in text


def test_fan_and_pcie_labels_render(window, telemetry_sample):
    window.update_telemetry_ui(telemetry_sample(0))

    assert window.lbl_fan.text() == "Fan: 60% (1800 RPM)"
    assert window.lbl_pcie.text() == "PCIe: Gen4 x16"


def test_downshifted_pcie_link_shows_the_maximum(window, telemetry_sample):
    """Gen1 at idle is normal — without the max it reads as a broken slot."""
    window.update_telemetry_ui(telemetry_sample(0, pcie_gen=1))

    assert window.lbl_pcie.text() == "PCIe: Gen1 x16 (max Gen4 x16)"


def test_card_without_fans_or_pcie_data_renders_na(window, telemetry_sample):
    window.update_telemetry_ui(telemetry_sample(0, fan_speed_pct=None,
                                                fan_rpm=None, pcie_gen=None,
                                                pcie_width=None))

    assert window.lbl_fan.text() == "Fan: N/A"
    assert window.lbl_pcie.text() == "PCIe: N/A"


def test_summary_reports_peak_fan_and_link(window, telemetry_sample):
    window.logger.start()
    for i in range(5):
        window.logger.log(telemetry_sample(i))

    window.on_benchmark_finished({
        "total_steps": 150, "requested_steps": 150, "aborted": False,
        "elapsed_time_sec": 70.0, "steps_per_sec": 2.14,
    })

    rows = _summary_rows(window)
    assert rows["Peak Fan (%)"] == "64"
    assert rows["PCIe Link (peak)"] == "Gen4 x16"
