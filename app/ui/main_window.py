from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.telemetry import TelemetryWorker
from app.core.trainer import (
    DEFAULT_PRESET,
    MODELS,
    PRECISIONS,
    PRESETS,
    SEQ_LENGTHS,
    VRAM_RESERVE_GIB,
    TrainerWorker,
    estimate_weights_gib,
    fits_in_vram,
)
from app.ui import readings
from app.ui.style import LIGHT, THEMES, build_qpalette, build_stylesheet
from app.ui.widgets import header
from app.ui.widgets.header import HeaderBar
from app.ui.widgets.phase_progress import PhaseProgressBar
from app.ui.widgets.tile import MetricTile
from app.utils.display import RateTracker, format_progress, summary_rows
from app.utils.logger import BenchmarkLogger
from app.utils.plots import (
    apply_chart_theme,
    draw_loss,
    draw_thermal_power,
    draw_utilization,
)
from app.utils.report import build_report

SETTINGS_ORG = SETTINGS_APP = "AI-GPU-Benchmark"

# How often the charts are redrawn while a run is in progress. Telemetry
# arrives four times a second; redrawing that often would cost the UI thread
# more than it shows.
LIVE_REFRESH_MS = 1000


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        # Injectable so tests write to a scratch file instead of the user's
        # real preferences.
        self.settings = settings if settings is not None else QSettings(
            SETTINGS_ORG, SETTINGS_APP)
        self.setWindowTitle("AI GPU Benchmark — detecting GPU…")
        self.resize(800, 650)

        # Main Layout: the header across the top, the two panels below it.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.header = HeaderBar()
        outer_layout.addWidget(self.header)
        body = QWidget()
        outer_layout.addWidget(body, 1)
        main_layout = QHBoxLayout(body)

        # --- LEFT PANEL: Hardware Stats & Controls ---
        left_panel = QVBoxLayout()

        # 1. Hardware Dashboard: one tile per reading, in readings.TILES order.
        hw_group = QGroupBox("Live Hardware Telemetry")
        hw_layout = QGridLayout()
        hw_layout.setSpacing(8)
        self.tiles = {}
        for i, (key, title, _) in enumerate(readings.TILES):
            tile = MetricTile(title)
            self.tiles[key] = tile
            if key == "throttle":
                # Reasons can run to "Power cap, Thermal": give it the width.
                hw_layout.addWidget(tile, i // 2, 0, 1, 2)
            else:
                hw_layout.addWidget(tile, i // 2, i % 2)
        self.tiles["throttle"].setToolTip(
            "Why the clock is below its maximum, straight from the driver: "
            "power cap, power brake, or thermal slowdown."
        )
        hw_group.setLayout(hw_layout)
        left_panel.addWidget(hw_group)

        # 2. Workload settings
        self.workload_group = QGroupBox("Workload")
        workload_layout = QFormLayout()

        # Fills in every field below; editing any of them by hand turns the
        # selection into Custom, and matching a preset again turns it back.
        self.cmb_preset = QComboBox()
        for i, preset in enumerate(PRESETS):
            self.cmb_preset.addItem(preset.label, preset)
            self.cmb_preset.setItemData(i, preset.description,
                                        Qt.ItemDataRole.ToolTipRole)
        self.cmb_preset.addItem("Custom", None)

        self.cmb_model = QComboBox()
        for i, spec in enumerate(MODELS):
            self.cmb_model.addItem(spec.label, spec)
            self.cmb_model.setItemData(
                i, f"{spec.model_id}\n{spec.params_b:.2f}B parameters",
                Qt.ItemDataRole.ToolTipRole)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(10, 5000)
        self.spin_steps.setSingleStep(10)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)

        self.cmb_precision = QComboBox()
        for spec in PRECISIONS:
            self.cmb_precision.addItem(spec.label, spec)
        self.cmb_precision.setToolTip(
            "BF16 needs Ampere or newer. FP32 runs anywhere but roughly "
            "doubles VRAM and time. INT8/INT4 store the frozen base model "
            "quantized (bitsandbytes) — what fits 3B–7B models on an 8 GB card."
        )

        self.cmb_seq_len = QComboBox()
        for length in SEQ_LENGTHS:
            self.cmb_seq_len.addItem(f"{length} tokens", length)
        self.cmb_seq_len.setToolTip(
            "Tokens per training sequence. Compute and activation memory grow "
            "with it, attention quadratically — the main lever on run length."
        )

        self.chk_checkpointing = QCheckBox("Gradient checkpointing")
        self.chk_checkpointing.setToolTip(
            "Recomputes activations during the backward pass instead of storing "
            "them: far less VRAM, roughly a third more compute per step. Needed "
            "for long sequences on small cards."
        )

        self.spin_warmup = QSpinBox()
        self.spin_warmup.setRange(0, 100)
        self.spin_warmup.setValue(10)
        self.spin_warmup.setToolTip(
            "Steps run before the timer starts, to get model loading and CUDA "
            "kernel compilation out of the measured window."
        )

        workload_layout.addRow("Preset:", self.cmb_preset)
        workload_layout.addRow("Model:", self.cmb_model)
        workload_layout.addRow("Steps:", self.spin_steps)
        workload_layout.addRow("Batch size:", self.spin_batch)
        workload_layout.addRow("Precision:", self.cmb_precision)
        workload_layout.addRow("Sequence length:", self.cmb_seq_len)
        workload_layout.addRow("Warm-up steps:", self.spin_warmup)
        workload_layout.addRow(self.chk_checkpointing)

        # Updated whenever model or precision changes, and once the card's
        # VRAM size is known from the first telemetry snapshot.
        self.lbl_fit = QLabel()
        self.lbl_fit.setObjectName("fitHint")
        self.lbl_fit.setWordWrap(True)
        workload_layout.addRow(self.lbl_fit)
        self.workload_group.setLayout(workload_layout)
        left_panel.addWidget(self.workload_group)

        # 3. Controls
        ctrl_group = QGroupBox("Benchmark Controls")
        ctrl_layout = QVBoxLayout()

        self.btn_start = QPushButton("Start Stress Test")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_benchmark)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setMinimumHeight(35)
        self.btn_stop.clicked.connect(self.stop_benchmark)
        self.btn_stop.setEnabled(False)

        self.btn_export = QPushButton("Export Results")
        self.btn_export.setObjectName("btnExport")
        self.btn_export.setMinimumHeight(35)
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_export)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)
        left_panel.addStretch()

        # --- RIGHT PANEL: Console, Progress, Plots, Summary ---
        right_panel = QVBoxLayout()

        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(180)
        self.log_to_console("System initialized. Ready for benchmark.")

        self.phase_bar = PhaseProgressBar()

        self.lbl_progress = QLabel("")
        self.lbl_progress.setObjectName("progressInfo")

        # Tab widget for multiple plots
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setMinimumHeight(280)
        self.plot_tabs.setVisible(False)

        # Tab 1: Thermal & Power
        thermal_figure = Figure(figsize=(8, 3))
        self.thermal_canvas = FigureCanvas(thermal_figure)
        self.plot_tabs.addTab(self.thermal_canvas, "Thermal && Power")

        # Tab 2: Utilization & Clock
        util_figure = Figure(figsize=(8, 3))
        self.util_canvas = FigureCanvas(util_figure)
        self.plot_tabs.addTab(self.util_canvas, "Utilization, Clock && Fan")

        # Tab 3: Loss
        loss_figure = Figure(figsize=(8, 3))
        self.loss_canvas = FigureCanvas(loss_figure)
        self.plot_tabs.addTab(self.loss_canvas, "Loss")

        # Summary stats table
        self.summary_group = QGroupBox("Benchmark Summary")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_table.setMaximumHeight(200)
        summary_layout.addWidget(self.summary_table)
        self.summary_group.setLayout(summary_layout)
        self.summary_group.setVisible(False)

        right_panel.addWidget(QLabel("Execution Log:"))
        right_panel.addWidget(self.console)
        right_panel.addWidget(self.phase_bar)
        right_panel.addWidget(self.lbl_progress)
        right_panel.addWidget(self.plot_tabs, 2)
        right_panel.addWidget(self.summary_group)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        # Menu: the theme is a preference, not a benchmark setting.
        view_menu = self.menuBar().addMenu("&View")
        self.act_dark = QAction("&Dark theme", self)
        self.act_dark.setCheckable(True)
        self.act_dark.toggled.connect(self._on_dark_toggled)
        view_menu.addAction(self.act_dark)

        # Applied last, so every widget above is polished in one pass.
        self._theme = LIGHT
        self._apply_theme(LIGHT)

        self._vram_total_mib = None          # MiB, from the first snapshot
        self.cmb_model.currentIndexChanged.connect(self._update_fit_hint)
        self.cmb_precision.currentIndexChanged.connect(self._update_fit_hint)

        self._applying_preset = False
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_selected)
        for signal in (self.cmb_model.currentIndexChanged,
                       self.spin_steps.valueChanged,
                       self.spin_batch.valueChanged,
                       self.cmb_precision.currentIndexChanged,
                       self.cmb_seq_len.currentIndexChanged,
                       self.chk_checkpointing.toggled):
            signal.connect(self._sync_preset_combo)
        self._restore_settings()
        self._update_fit_hint()

        # --- INITIALIZE THREADS ---
        self.logger = BenchmarkLogger()
        self._last_telemetry_error = None
        self.telemetry = TelemetryWorker(interval_ms=250)
        self.telemetry.data_updated.connect(self.update_telemetry_ui)
        self.telemetry.gpu_name_ready.connect(self._on_gpu_name_ready)
        self.telemetry.device_info_ready.connect(self._on_device_info)
        self.telemetry.sampler_status.connect(self._on_sampler_status)
        self.telemetry.power_trace.connect(self._on_power_trace)
        self.telemetry.error_occurred.connect(self._on_telemetry_error)
        self.telemetry.start()

        self.trainer = None
        self._device_info = None
        self._sampler_status = None
        self._last_summary = None

        # Live charts: redrawn from what the logger has collected so far.
        self._loss_steps, self._loss_values = [], []
        self._run_warmup = 0
        self._total_steps = 0
        self._rate = RateTracker()
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(LIVE_REFRESH_MS)
        self._live_timer.timeout.connect(self._refresh_live_charts)

    def log_to_console(self, text: str):
        """Appends text to the console window."""
        self.console.append(text)

    @pyqtSlot(str)
    def _on_gpu_name_ready(self, gpu_name: str):
        """Updates the window title once the GPU name is detected."""
        self.setWindowTitle(f"AI GPU Benchmark — {gpu_name}")
        self.header.set_gpu_name(gpu_name)
        self.log_to_console(f"Detected GPU: {gpu_name}")

    @pyqtSlot(dict)
    def _on_device_info(self, info: dict):
        self._device_info = info
        self.header.set_device_info(info)
        if info.get("driver_version"):
            self.log_to_console(
                f"Driver {info['driver_version']}, CUDA {info.get('cuda_driver_version') or '?'}")

    @pyqtSlot(dict)
    def _on_sampler_status(self, status: dict):
        self._sampler_status = status
        self.header.set_sampler(status)
        if status["active"]:
            self.log_to_console(f"Power sampler: native, {status['detail']}, "
                                f"every {status['interval_us'] / 1000:g} ms")
        else:
            self.log_to_console(f"Power sampler: {status['detail']}")

    @pyqtSlot(object)
    def _on_power_trace(self, chunk):
        self.logger.log_power_trace(chunk)

    @pyqtSlot(str)
    def _on_telemetry_error(self, message: str):
        """Surfaces NVML failures instead of leaving the dashboard on '--'.

        Read errors repeat on every poll, so an identical consecutive message
        is swallowed rather than filling the console with the same line four
        times a second.
        """
        if message == self._last_telemetry_error:
            return
        self._last_telemetry_error = message
        if message.startswith("NVML Initialization Failed"):
            self.header.set_gpu_unavailable(message)
        self.log_to_console(f"TELEMETRY ERROR: {message}")

    @pyqtSlot(dict)
    def update_telemetry_ui(self, data: dict):
        """Updates the dashboard with live PyNVML data."""
        self.logger.log(data)
        if self._vram_total_mib is None and data.get('vram_total_mb'):
            self._vram_total_mib = data['vram_total_mb']
            self._update_fit_hint()
        for key, _, read in readings.TILES:
            self.tiles[key].set_reading(read(data))

    # --- theme & persisted settings --------------------------------------------

    def _apply_theme(self, theme):
        self._theme = theme
        app = QApplication.instance()
        if isinstance(app, QApplication):
            # App-wide rather than on the window: combo popups and tooltips
            # are top-level windows and would not inherit it.
            app.setPalette(build_qpalette(theme, self.style().standardPalette()))
        self.setStyleSheet(build_stylesheet(theme))
        self.phase_bar.set_theme(theme)
        self.header.set_theme(theme)
        for canvas in (self.thermal_canvas, self.util_canvas, self.loss_canvas):
            self._theme_canvas(canvas)

    def _theme_canvas(self, canvas):
        apply_chart_theme(canvas.figure, self._theme.chart_bg,
                          self._theme.chart_fg, self._theme.chart_grid)
        canvas.draw()

    @pyqtSlot(bool)
    def _on_dark_toggled(self, dark: bool):
        self._apply_theme(THEMES["dark" if dark else "light"])
        self.settings.setValue("ui/theme", self._theme.name)

    def _save_settings(self):
        """Remembers the workload, so the next launch opens on the last run."""
        s = self.settings
        s.setValue("ui/theme", self._theme.name)
        s.setValue("workload/preset", self.cmb_preset.currentText())
        s.setValue("workload/model", self.cmb_model.currentText())
        s.setValue("workload/precision", self.cmb_precision.currentText())
        s.setValue("workload/steps", self.spin_steps.value())
        s.setValue("workload/batch_size", self.spin_batch.value())
        s.setValue("workload/seq_len", self.cmb_seq_len.currentData())
        s.setValue("workload/gradient_checkpointing", self.chk_checkpointing.isChecked())
        s.setValue("workload/warmup_steps", self.spin_warmup.value())
        s.sync()

    def _setting_int(self, key: str, default: int) -> int:
        """An int from the settings file, or the default if it is not one."""
        try:
            return self.settings.value(key, default, type=int)
        except (TypeError, ValueError):
            return default

    def _restore_settings(self):
        """Reapplies saved settings, skipping anything that no longer exists.

        Values are matched by label, so a model dropped from the registry or a
        hand-edited file falls back to the default instead of raising.
        """
        s = self.settings
        preset_label = s.value("workload/preset", DEFAULT_PRESET.label, type=str)
        preset = next((p for p in PRESETS if p.label == preset_label), None)
        self._apply_preset(preset or DEFAULT_PRESET)

        if preset_label == "Custom":
            self._applying_preset = True
            try:
                for combo, key in ((self.cmb_model, "workload/model"),
                                   (self.cmb_precision, "workload/precision")):
                    index = combo.findText(s.value(key, "", type=str))
                    if index >= 0:
                        combo.setCurrentIndex(index)
                seq_index = self.cmb_seq_len.findData(
                    self._setting_int("workload/seq_len", 0))
                if seq_index >= 0:
                    self.cmb_seq_len.setCurrentIndex(seq_index)
                self.spin_steps.setValue(
                    self._setting_int("workload/steps", self.spin_steps.value()))
                self.spin_batch.setValue(
                    self._setting_int("workload/batch_size", self.spin_batch.value()))
                self.chk_checkpointing.setChecked(
                    s.value("workload/gradient_checkpointing",
                            self.chk_checkpointing.isChecked(), type=bool))
            finally:
                self._applying_preset = False
            self._sync_preset_combo()

        self.spin_warmup.setValue(
            self._setting_int("workload/warmup_steps", self.spin_warmup.value()))
        # setChecked fires toggled, which applies the theme.
        self.act_dark.setChecked(s.value("ui/theme", "light", type=str) == "dark")

    def _apply_preset(self, preset):
        """Writes a preset into the fields without bouncing back to Custom."""
        self._applying_preset = True
        try:
            self.cmb_model.setCurrentIndex(MODELS.index(preset.model))
            self.cmb_precision.setCurrentIndex(PRECISIONS.index(preset.precision))
            self.spin_steps.setValue(preset.steps)
            self.spin_batch.setValue(preset.batch_size)
            self.cmb_seq_len.setCurrentIndex(SEQ_LENGTHS.index(preset.seq_len))
            self.chk_checkpointing.setChecked(preset.gradient_checkpointing)
        finally:
            self._applying_preset = False
        self._sync_preset_combo()

    def _matching_preset(self):
        """The preset the fields currently spell out, or None for Custom."""
        current = (self.cmb_model.currentData(), self.cmb_precision.currentData(),
                   self.spin_steps.value(), self.spin_batch.value(),
                   self.cmb_seq_len.currentData(), self.chk_checkpointing.isChecked())
        for preset in PRESETS:
            if current == (preset.model, preset.precision, preset.steps,
                           preset.batch_size, preset.seq_len,
                           preset.gradient_checkpointing):
                return preset
        return None

    def _on_preset_selected(self, index: int):
        preset = self.cmb_preset.itemData(index)
        # Picking Custom by hand keeps whatever the fields already hold.
        if preset is not None:
            self._apply_preset(preset)

    def _sync_preset_combo(self):
        if self._applying_preset:
            return
        preset = self._matching_preset()
        index = len(PRESETS) if preset is None else PRESETS.index(preset)
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.setCurrentIndex(index)
        self.cmb_preset.blockSignals(False)

    def _update_fit_hint(self):
        """Shows the weights' footprint against the card, flagging a misfit.

        A warning rather than a lock: the estimate covers weights only, and
        someone may well want to watch a run hit the wall on purpose.
        """
        model = self.cmb_model.currentData()
        precision = self.cmb_precision.currentData()
        weights = estimate_weights_gib(model, precision)

        if self._vram_total_mib is None:
            text, misfit = f"Weights ≈ {weights:.1f} GiB", False
        else:
            total = self._vram_total_mib / 1024
            misfit = not fits_in_vram(model, precision, self._vram_total_mib)
            text = f"Weights ≈ {weights:.1f} GiB of {total:.1f} GiB VRAM"
            if misfit:
                text += (f" — too big: a run needs ~{VRAM_RESERVE_GIB:.0f} GiB "
                         "free on top of the weights. Try INT4 or a smaller model.")

        self.lbl_fit.setText(text)
        # A dynamic property drives the stylesheet; Qt only re-reads it on a
        # re-polish, so the colour would otherwise stick to the first state.
        self.lbl_fit.setProperty("warning", misfit)
        self.lbl_fit.style().unpolish(self.lbl_fit)
        self.lbl_fit.style().polish(self.lbl_fit)

    def start_benchmark(self):
        """Disables UI and fires up the PyTorch training thread."""
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.phase_bar.start(self.spin_warmup.value(), self.spin_steps.value())
        self.header.set_status(header.LOADING)
        self.summary_group.setVisible(False)
        self._loss_steps, self._loss_values = [], []
        self._rate = RateTracker()
        self._run_warmup = self.spin_warmup.value()
        self.lbl_progress.setText("")
        # Visible from the start and redrawn as data arrives: a run on the
        # heavier presets is far too long to watch a bare progress bar.
        for canvas in (self.thermal_canvas, self.util_canvas, self.loss_canvas):
            canvas.figure.clear()
            self._theme_canvas(canvas)
        self.plot_tabs.setVisible(True)
        self._live_timer.start()
        self.log_to_console("\n--- Starting Synthetic Workload ---")
        if self.lbl_fit.property("warning"):
            self.log_to_console(f"WARNING: {self.lbl_fit.text()}")
        self.logger.start()

        # Locked for the duration: the summary reports these values, so they
        # must describe the run that actually happened.
        self.workload_group.setEnabled(False)
        self._save_settings()

        self.trainer = TrainerWorker(
            steps=self.spin_steps.value(),
            batch_size=self.spin_batch.value(),
            model=self.cmb_model.currentData(),
            warmup_steps=self.spin_warmup.value(),
            precision=self.cmb_precision.currentData(),
            seq_len=self.cmb_seq_len.currentData(),
            gradient_checkpointing=self.chk_checkpointing.isChecked(),
        )
        self.trainer.status_updated.connect(self.log_to_console)
        self.trainer.progress_updated.connect(self._on_progress)
        self.trainer.loss_updated.connect(self._on_loss)
        self.trainer.max_steps_ready.connect(self._on_max_steps)
        self.trainer.training_finished.connect(self.on_benchmark_finished)
        self.trainer.error_occurred.connect(self.on_benchmark_error)
        self.trainer.start()

    def stop_benchmark(self):
        """Requests an early stop; the run ends at the next step boundary."""
        if self.trainer and self.trainer.isRunning():
            self.log_to_console("\nStop requested — finishing current step...")
            self.trainer.stop()
            self.header.set_status(header.STOPPING)
        self.btn_stop.setEnabled(False)

    @pyqtSlot(int)
    def _on_max_steps(self, total: int):
        self._total_steps = total
        self.phase_bar.set_total(total)

    @pyqtSlot(int)
    def _on_progress(self, step: int):
        """Moves the bar and recomputes rate and ETA."""
        self.phase_bar.set_step(step)
        if self.header.status() != header.STOPPING:
            self.header.set_status(header.WARMUP if self.phase_bar.phase() == "warm-up"
                                   else header.MEASURING)
        rate = self._rate.update(step)
        self.lbl_progress.setText(format_progress(step, self._total_steps, rate))

    @pyqtSlot(int, float)
    def _on_loss(self, step: int, loss: float):
        self._loss_steps.append(step)
        self._loss_values.append(loss)

    def _refresh_live_charts(self):
        df = self.logger.to_dataframe()
        if df is not None and len(df) >= 2:
            self._draw_telemetry(df)
        self._draw_loss()

    @pyqtSlot(dict)
    def on_benchmark_finished(self, summary: dict):
        self._live_timer.stop()
        self.phase_bar.finish()
        self.header.set_status(header.STOPPED if summary.get('aborted') else header.DONE)
        self.btn_stop.setEnabled(False)
        self._last_summary = summary
        df = self.logger.stop()
        if df is not None:
            self._render_plots(df, summary)
        if summary.get('aborted'):
            self.log_to_console(
                f"\nBenchmark stopped early: {summary['total_steps']}"
                f"/{summary['requested_steps']} steps in {summary['elapsed_time_sec']}s"
            )
        else:
            self.log_to_console(f"\nBenchmark Complete! Time: {summary['elapsed_time_sec']}s")
        self.log_to_console(f"Throughput: {summary['steps_per_sec']} steps/sec")
        if summary.get('tokens_per_step') and summary['steps_per_sec']:
            self.log_to_console(
                f"           {summary['steps_per_sec'] * summary['tokens_per_step']:,.0f} tokens/sec")
        self._set_idle_controls()
        self.btn_export.setEnabled(True)

    @pyqtSlot(str)
    def on_benchmark_error(self, message: str):
        """Recovers the UI from a failed run.

        TrainerWorker.run() emits error_occurred and returns without ever
        emitting training_finished, so nothing else re-enables the controls:
        before this, any training failure left Start greyed out for good and
        the app had to be restarted.
        """
        self._live_timer.stop()
        self._last_summary = None
        self.log_to_console(f"ERROR: {message}")
        # Close the run, or its partial telemetry leaks into the next one.
        df = self.logger.stop()
        self.phase_bar.reset()
        self.header.set_status(header.FAILED)
        self._set_idle_controls()
        self.btn_export.setEnabled(df is not None)

    def _set_idle_controls(self):
        """Returns the control panel to its ready-for-a-new-run state."""
        self.workload_group.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)

    def _draw_telemetry(self, df):
        trace = self.logger.power_trace()
        for canvas, draw in ((self.thermal_canvas, lambda ax: draw_thermal_power(ax, df, trace)),
                             (self.util_canvas, lambda ax: draw_utilization(ax, df))):
            figure = canvas.figure
            figure.clear()
            draw(figure.add_subplot(111))
            figure.tight_layout()
            self._theme_canvas(canvas)

    def _draw_loss(self):
        figure = self.loss_canvas.figure
        figure.clear()
        draw_loss(figure.add_subplot(111), self._loss_steps, self._loss_values,
                  self._run_warmup)
        figure.tight_layout()
        self._theme_canvas(self.loss_canvas)

    def _render_plots(self, df, summary: dict):
        """Final draw of all three tabs once the run is over."""
        self._draw_telemetry(df)
        self._draw_loss()

        self.plot_tabs.setVisible(True)
        self.plot_tabs.setCurrentIndex(0)

        # --- Summary Stats Table ---
        self._render_summary(df, summary)
        self.log_to_console("Summary and plots rendered.")

    def _render_summary(self, df, summary: dict):
        """Populates the summary table with computed benchmark metrics."""
        metrics = summary_rows(df, summary, self.logger.power_trace())

        self.summary_table.setRowCount(len(metrics))
        for row, (metric, value) in enumerate(metrics):
            item_m = QTableWidgetItem(metric)
            item_m.setFlags(item_m.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item_v = QTableWidgetItem(value)
            item_v.setFlags(item_v.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.summary_table.setItem(row, 0, item_m)
            self.summary_table.setItem(row, 1, item_v)

        self.summary_group.setVisible(True)

    def export_results(self):
        """Exports the last benchmark's telemetry to CSV and renders plot PNG."""
        df = self.logger.to_dataframe()
        if df is None:
            self.log_to_console("No data to export.")
            return
        trace = self.logger.power_trace()
        report = None
        if self._last_summary is not None:
            report = build_report(df, self._last_summary, self._device_info,
                                  (self._loss_steps, self._loss_values),
                                  trace, self._sampler_status)
        try:
            paths = BenchmarkLogger.export(df, report=report, trace=trace)
        except OSError as e:
            self.log_to_console(f"Export failed: {e}")
            return
        for path in paths:
            if path:
                self.log_to_console(f"Exported: {path}")

    def closeEvent(self, event):
        """Clean up threads on exit."""
        self._live_timer.stop()
        self._save_settings()
        self.telemetry.stop()
        if self.trainer and self.trainer.isRunning():
            # Shutdown is the one place we genuinely must block, so the
            # training thread cannot outlive the window it reports into.
            self.trainer.wait_for_exit()
        event.accept()
