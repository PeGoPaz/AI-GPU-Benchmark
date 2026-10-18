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
    progress_updated = pyqtSignal(int)
    loss_updated = pyqtSignal(int, float)
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


def _scratch_settings(tmp_path):
    from PyQt6.QtCore import QSettings
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    """A MainWindow whose telemetry thread never finds a GPU.

    Settings go to a scratch file: the real ones would leak state between
    tests and overwrite the preferences of whoever runs the suite.
    """
    import pynvml
    monkeypatch.setattr(pynvml, "nvmlInit",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("no driver")))

    from app.ui.main_window import MainWindow
    win = MainWindow(settings=_scratch_settings(tmp_path))
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

    assert labels == ["Thermal && Power", "Utilization, Clock && Fan", "Loss"]


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



def test_summary_reports_tokens_compute_and_energy(window, telemetry_sample):
    window.logger.start()
    for i in range(9):
        window.logger.log(telemetry_sample(i))

    window.on_benchmark_finished({
        "total_steps": 100, "requested_steps": 100, "aborted": False,
        "elapsed_time_sec": 20.0, "steps_per_sec": 5.0,
        "tokens_per_step": 4096, "params_b": 1.5, "gradient_checkpointing": True,
        "measure_start": 102.0, "measure_end": 106.0,
    })

    rows = _summary_rows(window)
    assert rows["Tokens/sec"] == "20,480"
    assert rows["Compute (est. TFLOPS)"] == "184.3"
    assert rows["Energy, timed (kJ, counter)"] == "1.00"     # 4 s × 250 J/s
    assert rows["Efficiency (tokens/J)"] == "409.6"
    assert "20,480 tokens/sec" in window.console.toPlainText()


def test_throttle_label_names_the_reason(window, telemetry_sample):
    from app.core.telemetry import THROTTLE_SW_POWER_CAP

    window.update_telemetry_ui(telemetry_sample(0, throttle_reasons=THROTTLE_SW_POWER_CAP))
    assert window.lbl_throttle.text() == "Throttle: Power cap"
    assert window.lbl_throttle.property("warning") is True

    window.update_telemetry_ui(telemetry_sample(1, throttle_reasons=0))
    assert window.lbl_throttle.text() == "Throttle: none"
    assert window.lbl_throttle.property("warning") is False


def test_throttle_label_without_driver_support(window, telemetry_sample):
    window.update_telemetry_ui(telemetry_sample(0, throttle_reasons=None))

    assert window.lbl_throttle.text() == "Throttle: N/A"


def test_summary_reports_time_spent_throttled(window, telemetry_sample):
    from app.core.telemetry import THROTTLE_SW_POWER_CAP

    window.logger.start()
    for i in range(4):
        window.logger.log(telemetry_sample(
            i, throttle_reasons=THROTTLE_SW_POWER_CAP if i % 2 else 0))
    _finish = {"total_steps": 150, "requested_steps": 150, "aborted": False,
               "elapsed_time_sec": 70.0, "steps_per_sec": 2.14}
    window.on_benchmark_finished(_finish)

    rows = _summary_rows(window)
    assert rows["Time Power-Limited (%)"] == "50"
    assert rows["Time Thermal-Limited (%)"] == "0"



def test_progress_line_shows_rate_and_time_left():
    from app.utils.display import format_progress

    assert format_progress(10, 310, None) == "Step 10 / 310"
    assert format_progress(10, 310, 2.0) == "Step 10 / 310 · 2.00 steps/s · ETA 2:30"
    assert format_progress(0, 9000, 2.0).endswith("ETA 1:15:00")
    assert format_progress(310, 310, 2.0).endswith("ETA 0:00")


def test_charts_are_visible_and_live_during_the_run(started_run, telemetry_sample):
    """The heavier presets run for tens of minutes; a bare progress bar told
    nothing about what the card was doing until it was all over."""
    assert not started_run.plot_tabs.isHidden()
    assert started_run._live_timer.isActive()

    for i in range(5):
        started_run.update_telemetry_ui(telemetry_sample(i))
    started_run.trainer.loss_updated.emit(1, 3.2)
    started_run.trainer.loss_updated.emit(2, 3.0)
    started_run._refresh_live_charts()

    assert _util_plot_labels(started_run)
    loss_lines = started_run.loss_canvas.figure.axes[0].get_lines()
    assert list(loss_lines[0].get_ydata()) == [3.2, 3.0]


def test_live_refresh_survives_an_empty_run(started_run):
    started_run._refresh_live_charts()

    assert not started_run.util_canvas.figure.axes


def test_live_refresh_stops_when_the_run_ends(started_run, telemetry_sample):
    started_run.update_telemetry_ui(telemetry_sample(0))
    started_run.trainer.training_finished.emit({
        "total_steps": 150, "requested_steps": 150, "aborted": False,
        "elapsed_time_sec": 70.0, "steps_per_sec": 2.14,
    })

    assert not started_run._live_timer.isActive()


def test_live_refresh_stops_when_the_run_fails(started_run):
    started_run.trainer.error_occurred.emit("HF Training Error: boom")

    assert not started_run._live_timer.isActive()


def test_progress_updates_the_eta_line(started_run):
    started_run.trainer.max_steps_ready.emit(310)
    started_run.trainer.progress_updated.emit(1)

    assert started_run.progress_bar.maximum() == 310
    assert started_run.lbl_progress.text() == "Step 1 / 310"



# --- theme & persisted settings ------------------------------------------------

def _reopen(window, tmp_path):
    """Closes the window and opens a fresh one on the same settings file."""
    from app.ui.main_window import MainWindow

    window.close()
    reopened = MainWindow(settings=_scratch_settings(tmp_path))
    reopened.telemetry.stop()
    return reopened


def test_dark_theme_recolours_window_and_charts(window, telemetry_sample):
    from PyQt6.QtGui import QPalette

    from app.ui.style import DARK

    _finish_run(window, telemetry_sample)
    window.act_dark.setChecked(True)

    window_color = QApplication.instance().palette().color(QPalette.ColorRole.Window)
    assert window_color.name().upper() == DARK.window
    assert window.thermal_canvas.figure.get_facecolor()[:3] == pytest.approx(
        (0x23 / 255, 0x23 / 255, 0x23 / 255))

    window.act_dark.setChecked(False)
    assert window.thermal_canvas.figure.get_facecolor()[:3] == pytest.approx((1, 1, 1))


def test_themed_charts_keep_their_series_colours(window, telemetry_sample):
    """Temperature, power and clock axes are coloured to match their curves;
    the theme must only change the labels left at matplotlib's default."""
    from app.utils.plots import CLOCK_COLOR

    _finish_run(window, telemetry_sample)
    window.act_dark.setChecked(True)

    percent_axes, clock_axes = window.util_canvas.figure.axes
    assert clock_axes.yaxis.label.get_color() == CLOCK_COLOR
    assert percent_axes.yaxis.label.get_color() == "#DDDDDD"


def test_theme_is_remembered(window, tmp_path):
    window.act_dark.setChecked(True)

    reopened = _reopen(window, tmp_path)

    assert reopened.act_dark.isChecked()
    assert reopened._theme.name == "dark"
    reopened.act_dark.setChecked(False)              # leave the app palette light


def test_workload_is_remembered(window, tmp_path, monkeypatch):
    import app.ui.main_window as mw
    monkeypatch.setattr(mw, "TrainerWorker", FakeTrainer)
    _select(window.cmb_preset, "Extended")
    window.spin_warmup.setValue(25)
    window.start_benchmark()

    reopened = _reopen(window, tmp_path)

    assert reopened.cmb_preset.currentText() == "Extended"
    assert reopened.cmb_model.currentText() == "Qwen2.5 3B"
    assert reopened.spin_warmup.value() == 25


def test_custom_workload_is_remembered_field_by_field(window, tmp_path):
    _select(window.cmb_preset, "Quick")
    window.spin_steps.setValue(420)
    window.cmb_seq_len.setCurrentIndex(window.cmb_seq_len.findData(256))

    reopened = _reopen(window, tmp_path)

    assert reopened.cmb_preset.currentText() == "Custom"
    assert reopened.cmb_model.currentText() == "TinyLlama 1.1B"
    assert reopened.spin_steps.value() == 420
    assert reopened.cmb_seq_len.currentData() == 256


def test_stale_settings_fall_back_to_the_default(tmp_path, qapp, monkeypatch):
    """A model removed from the registry, or a hand-edited file, must not
    stop the app from starting."""
    import pynvml

    from app.ui.main_window import MainWindow
    monkeypatch.setattr(pynvml, "nvmlInit",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("no driver")))
    settings = _scratch_settings(tmp_path)
    settings.setValue("workload/preset", "Custom")
    settings.setValue("workload/model", "GPT-5 Turbo")
    settings.setValue("workload/steps", "not a number")
    settings.sync()

    win = MainWindow(settings=_scratch_settings(tmp_path))
    win.telemetry.stop()

    assert win.cmb_model.currentText() == "Qwen2.5 1.5B"
    assert win.spin_steps.value() == 300
    win.close()



# --- run report -----------------------------------------------------------------

def _exported_report(tmp_path):
    import json
    [path] = (tmp_path / "logs").glob("report_*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_a_report_for_a_finished_run(window, telemetry_sample,
                                                   tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    window._on_device_info({"name": "NVIDIA GeForce RTX 4060",
                            "driver_version": "580.82", "cuda_driver_version": "13.0"})
    window.logger.start()
    for i in range(5):
        window.logger.log(telemetry_sample(i))
    window._on_loss(1, 3.0)
    window.on_benchmark_finished({
        "model": "Qwen2.5 1.5B", "precision": "BF16", "total_steps": 150,
        "requested_steps": 150, "aborted": False, "elapsed_time_sec": 70.0,
        "steps_per_sec": 2.14,
    })

    window.export_results()

    report = _exported_report(tmp_path)
    assert report["gpu"]["driver_version"] == "580.82"
    assert report["workload"]["model"] == "Qwen2.5 1.5B"
    assert report["loss"]["values"] == [3.0]
    assert "report_" in window.console.toPlainText()


def test_failed_run_exports_telemetry_without_a_report(started_run, telemetry_sample,
                                                       tmp_path, monkeypatch):
    """No summary means nothing truthful to report — but the telemetry
    leading up to the failure is still worth keeping."""
    monkeypatch.chdir(tmp_path)
    started_run.logger.log(telemetry_sample(0))
    started_run.trainer.error_occurred.emit("HF Training Error: CUDA out of memory")

    started_run.export_results()

    written = sorted(p.suffix for p in (tmp_path / "logs").iterdir())
    assert written == [".csv", ".png"]



# --- native power sampler -----------------------------------------------------------

def test_power_label_shows_the_native_peak(window, telemetry_sample):
    window.update_telemetry_ui(telemetry_sample(0, power_w=180.0, power_peak_w=262.4))
    assert window.lbl_power.text() == "Power: 180.0 W (peak 262 W)"

    window.update_telemetry_ui(telemetry_sample(1, power_w=180.0, power_peak_w=None))
    assert window.lbl_power.text() == "Power: 180.0 W"


def test_sampler_status_is_logged(window):
    window._on_sampler_status({"active": True, "detail": "instant power", "interval_us": 2000})
    window._on_sampler_status({"active": False, "detail": "native module not built"})

    text = window.console.toPlainText()
    assert "Power sampler: native, instant power, every 2 ms" in text
    assert "Power sampler: native module not built" in text


def test_power_trace_is_collected_during_a_run(started_run, telemetry_sample):
    import numpy as np

    started_run.update_telemetry_ui(telemetry_sample(0))
    started_run._on_power_trace((np.array([100.1, 100.102]),
                                 np.array([200.0, 300.0], dtype=np.float32)))

    assert len(started_run.logger.power_trace()) == 2



def test_summary_shows_transients_when_the_trace_exists(window, telemetry_sample):
    import numpy as np

    window.logger.start()
    for i in range(5):
        window.logger.log(telemetry_sample(i))
    t = 100.0 + np.arange(1500) * 0.002
    w = np.where(np.arange(1500) % 100 == 0, 330.0, 240.0).astype(np.float32)
    window._on_power_trace((t, w))
    _finish = {"total_steps": 150, "requested_steps": 150, "aborted": False,
               "elapsed_time_sec": 70.0, "steps_per_sec": 2.14}
    window.on_benchmark_finished(_finish)

    rows = _summary_rows(window)
    assert rows["Peak Power, instant (W)"] == "330.0"
    assert rows["Native Sampling (Hz)"].startswith("500, ")
    thermal_labels = [line.get_label() for ax in window.thermal_canvas.figure.axes
                      for line in ax.get_lines()]
    assert "Power, instant (W)" in thermal_labels


def test_summary_without_a_trace_has_no_transient_rows(window, telemetry_sample):
    _finish_run(window, telemetry_sample)

    assert "Peak Power, instant (W)" not in _summary_rows(window)
