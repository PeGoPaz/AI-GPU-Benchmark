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

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.kwargs = kwargs          # what MainWindow asked for
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


def test_disabled_buttons_drain_their_background(window):
    """Qt on its own only dims the label text. Against a saturated fill that
    was far too weak: a disabled Stop button still read as a live one."""
    for button in (window.btn_start, window.btn_stop, window.btn_export):
        button.setEnabled(True)
        live = button.grab().toImage().pixelColor(button.width() // 2, 4)
        button.setEnabled(False)
        dead = button.grab().toImage().pixelColor(button.width() // 2, 4)

        assert live != dead, button.objectName()


def test_button_styles_are_not_swapped_at_runtime(started_run):
    """Busy state is a :disabled selector now, not code rewriting stylesheets
    on every transition — which the finish and error paths both had to undo."""
    assert started_run.btn_start.styleSheet() == ""

    started_run.trainer.error_occurred.emit("HF Training Error: boom")

    assert started_run.btn_start.styleSheet() == ""


def _util_plot_labels(window):
    """Series names across both axes — the clock sits on the twin."""
    return [line.get_label()
            for axes in window.util_canvas.figure.axes
            for line in axes.get_lines()]


def _finish_run(window, telemetry_sample, n=5, **overrides):
    window.logger.start()
    for i in range(n):
        window.logger.log(telemetry_sample(i, **overrides))
    window.on_benchmark_finished({
        "total_steps": 150, "requested_steps": 150, "aborted": False,
        "elapsed_time_sec": 70.0, "steps_per_sec": 2.14,
    })


def test_fan_curve_is_plotted(window, telemetry_sample):
    _finish_run(window, telemetry_sample)

    assert "Fan (%)" in _util_plot_labels(window)


def test_passive_card_plots_no_fan_curve(window, telemetry_sample):
    """A column of None must leave the plot alone, not draw a flat empty line."""
    _finish_run(window, telemetry_sample, fan_speed_pct=None)

    labels = _util_plot_labels(window)
    assert "Fan (%)" not in labels
    assert "GPU Utilization (%)" in labels


def test_percentages_and_clock_use_separate_axes(window, telemetry_sample):
    """Sharing one axis with a ~2500 MHz clock flattened every percentage
    curve onto the zero line, which made the fan and utilization unreadable."""
    _finish_run(window, telemetry_sample)

    percent_axes, clock_axes = window.util_canvas.figure.axes
    assert percent_axes.get_ylim() == (0, 105)
    assert [line.get_label() for line in clock_axes.get_lines()] == ["SM Clock (MHz)"]


def test_workload_settings_reach_the_trainer(window, monkeypatch):
    """The run used to be hardcoded to 150 steps of TinyLlama at batch 4."""
    import app.ui.main_window as mw
    monkeypatch.setattr(mw, "TrainerWorker", FakeTrainer)

    window.spin_steps.setValue(300)
    window.spin_batch.setValue(8)
    window.spin_warmup.setValue(25)
    window.cmb_model.setCurrentIndex(1)
    window.cmb_precision.setCurrentIndex(1)          # FP16
    window.cmb_seq_len.setCurrentIndex(window.cmb_seq_len.findData(1024))
    window.chk_checkpointing.setChecked(True)
    chosen = window.cmb_model.currentData()
    window.start_benchmark()

    assert window.trainer.kwargs["steps"] == 300
    assert window.trainer.kwargs["batch_size"] == 8
    assert window.trainer.kwargs["warmup_steps"] == 25
    assert window.trainer.kwargs["model"] is chosen
    assert window.trainer.kwargs["precision"].label == "FP16"
    assert window.trainer.kwargs["seq_len"] == 1024
    assert window.trainer.kwargs["gradient_checkpointing"] is True


def test_settings_are_locked_during_a_run(started_run):
    """They describe the run in the summary, so they must not move under it."""
    assert not started_run.workload_group.isEnabled()

    started_run.trainer.error_occurred.emit("HF Training Error: boom")

    assert started_run.workload_group.isEnabled()


def test_summary_records_what_was_benchmarked(window, telemetry_sample):
    window.logger.start()
    for i in range(5):
        window.logger.log(telemetry_sample(i))

    window.on_benchmark_finished({
        "model": "Qwen2.5 1.5B", "batch_size": 8, "precision": "FP16", "seq_len": 512,
        "gradient_checkpointing": True,
        "total_steps": 300, "requested_steps": 300, "aborted": False,
        "elapsed_time_sec": 70.0, "steps_per_sec": 4.29,
    })

    rows = _summary_rows(window)
    assert rows["Model"] == "Qwen2.5 1.5B"
    assert rows["Batch Size"] == "8"
    assert rows["Precision"] == "FP16"
    assert rows["Sequence Length"] == "512"
    assert rows["Gradient Checkpointing"] == "On"


def test_tab_labels_keep_their_ampersand(window):
    """A single & is a mnemonic marker to Qt and gets swallowed, which showed
    the tabs as "Thermal _Power" and "Utilization, Clock  Fan"."""
    labels = [window.plot_tabs.tabText(i) for i in range(window.plot_tabs.count())]

    assert labels == ["Thermal && Power", "Utilization, Clock && Fan"]


def test_window_and_export_draw_the_same_series(window, telemetry_sample):
    """The tabs and the PNG used to be two hand-maintained copies of the same
    charts, and every change to one had to be repeated in the other."""
    from app.utils.plots import build_report_figure

    _finish_run(window, telemetry_sample)
    on_screen = sorted(line.get_label()
                       for canvas in (window.thermal_canvas, window.util_canvas)
                       for axes in canvas.figure.axes
                       for line in axes.get_lines())
    exported = sorted(line.get_label()
                      for axes in build_report_figure(window.logger.to_dataframe()).axes
                      for line in axes.get_lines())

    assert on_screen == exported


def test_model_tooltips_name_the_checkpoint(window):
    from PyQt6.QtCore import Qt

    for i in range(window.cmb_model.count()):
        spec = window.cmb_model.itemData(i)
        tip = window.cmb_model.itemData(i, Qt.ItemDataRole.ToolTipRole)
        assert spec.model_id in tip


def _select(combo, text_start):
    index = next(i for i in range(combo.count())
                 if combo.itemText(i).startswith(text_start))
    combo.setCurrentIndex(index)


def test_fit_hint_shows_weights_before_the_gpu_is_known(window):
    assert window.lbl_fit.text().startswith("Weights ≈ ")
    assert window.lbl_fit.property("warning") is False


def test_fit_hint_warns_when_the_model_will_not_fit(window, telemetry_sample):
    window.update_telemetry_ui(telemetry_sample(0, vram_total_mb=8192))
    _select(window.cmb_model, "Qwen2.5 7B")
    _select(window.cmb_precision, "BF16")

    assert window.lbl_fit.property("warning") is True
    assert "of 8.0 GiB" in window.lbl_fit.text()
    assert "INT4" in window.lbl_fit.text()

    _select(window.cmb_precision, "INT4")

    assert window.lbl_fit.property("warning") is False


def test_misfit_is_a_warning_not_a_lock(window, telemetry_sample, monkeypatch):
    """The estimate covers weights only, so Start stays usable — but the
    console says up front why the run is likely to die."""
    import app.ui.main_window as mw
    monkeypatch.setattr(mw, "TrainerWorker", FakeTrainer)
    window.update_telemetry_ui(telemetry_sample(0, vram_total_mb=8192))
    _select(window.cmb_model, "Qwen2.5 7B")
    _select(window.cmb_precision, "BF16")

    window.start_benchmark()

    assert window.trainer._started
    assert "WARNING: Weights" in window.console.toPlainText()



def test_window_opens_on_the_standard_preset(window):
    assert window.cmb_preset.currentText() == "Standard"
    assert window.cmb_model.currentText() == "Qwen2.5 1.5B"
    assert window.spin_steps.value() == 300
    assert window.cmb_seq_len.currentData() == 512
    assert window.chk_checkpointing.isChecked()


def test_choosing_a_preset_fills_every_field(window):
    _select(window.cmb_preset, "Stress")

    assert window.cmb_model.currentText() == "Qwen2.5 7B"
    assert window.cmb_precision.currentText() == "INT4 (NF4)"
    assert window.spin_batch.value() == 1
    assert window.cmb_preset.currentText() == "Stress"


def test_editing_a_field_switches_to_custom_and_back(window):
    window.spin_steps.setValue(310)
    assert window.cmb_preset.currentText() == "Custom"

    window.spin_steps.setValue(300)
    assert window.cmb_preset.currentText() == "Standard"


def test_picking_custom_keeps_the_current_values(window):
    _select(window.cmb_preset, "Quick")
    _select(window.cmb_preset, "Custom")

    assert window.cmb_model.currentText() == "TinyLlama 1.1B"
    assert window.spin_steps.value() == 150


def test_preset_reaches_the_trainer(window, monkeypatch, telemetry_sample):
    import app.ui.main_window as mw
    monkeypatch.setattr(mw, "TrainerWorker", FakeTrainer)
    window.update_telemetry_ui(telemetry_sample(0, vram_total_mb=8192))
    _select(window.cmb_preset, "Stress")

    window.start_benchmark()

    kwargs = window.trainer.kwargs
    assert kwargs["model"].label == "Qwen2.5 7B"
    assert kwargs["precision"].label == "INT4 (NF4)"
    assert (kwargs["steps"], kwargs["batch_size"], kwargs["seq_len"]) == (250, 1, 512)
    assert kwargs["gradient_checkpointing"] is True
    assert "WARNING" not in window.console.toPlainText()
