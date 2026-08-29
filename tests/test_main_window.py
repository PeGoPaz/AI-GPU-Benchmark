"""MainWindow wiring. Skipped automatically when Qt cannot start headless."""
import os

import pytest

pytestmark = pytest.mark.ui

# A Qt platform plugin is required even offscreen; skip rather than fail when
# the runner lacks the system libraries (libEGL, xkbcommon, ...).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")

from PyQt6.QtWidgets import QApplication  # noqa: E402


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
