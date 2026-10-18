from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.metrics import compute_stats
from app.core.telemetry import TelemetryWorker, decode_throttle_reasons
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
from app.ui.style import STYLESHEET
from app.utils.logger import BenchmarkLogger
from app.utils.plots import draw_thermal_power, draw_utilization


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI GPU Benchmark — detecting GPU…")
        self.resize(800, 650)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL: Hardware Stats & Controls ---
        left_panel = QVBoxLayout()

        # 1. Hardware Dashboard
        hw_group = QGroupBox("Live Hardware Telemetry")
        hw_layout = QVBoxLayout()
        self.lbl_temp = QLabel("Temp: -- °C")
        self.lbl_mem_temp = QLabel("Mem Temp: -- °C")
        self.lbl_headroom = QLabel("Thermal Headroom: -- °C")
        self.lbl_power = QLabel("Power: -- W")
        self.lbl_clock = QLabel("Clock: -- MHz")
        self.lbl_vram = QLabel("VRAM: -- / -- MiB")
        self.lbl_fan = QLabel("Fan: -- %")
        self.lbl_pcie = QLabel("PCIe: --")
        self.lbl_throttle = QLabel("Throttle: --")
        self.lbl_throttle.setToolTip(
            "Why the clock is below its maximum, straight from the driver: "
            "power cap, power brake, or thermal slowdown."
        )

        for lbl in [self.lbl_temp, self.lbl_mem_temp, self.lbl_headroom,
                    self.lbl_power, self.lbl_clock, self.lbl_vram,
                    self.lbl_fan, self.lbl_pcie, self.lbl_throttle]:
            lbl.setProperty("role", "metric")
            hw_layout.addWidget(lbl)

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

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

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
        right_panel.addWidget(self.progress_bar)
        right_panel.addWidget(self.plot_tabs, 2)
        right_panel.addWidget(self.summary_group)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        # Applied last, so every widget above is polished in one pass.
        self.setStyleSheet(STYLESHEET)

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
        self._apply_preset(DEFAULT_PRESET)
        self._update_fit_hint()

        # --- INITIALIZE THREADS ---
        self.logger = BenchmarkLogger()
        self._last_telemetry_error = None
        self.telemetry = TelemetryWorker(interval_ms=250)
        self.telemetry.data_updated.connect(self.update_telemetry_ui)
        self.telemetry.gpu_name_ready.connect(self._on_gpu_name_ready)
        self.telemetry.error_occurred.connect(self._on_telemetry_error)
        self.telemetry.start()

        self.trainer = None

    def log_to_console(self, text: str):
        """Appends text to the console window."""
        self.console.append(text)

    @pyqtSlot(str)
    def _on_gpu_name_ready(self, gpu_name: str):
        """Updates the window title once the GPU name is detected."""
        self.setWindowTitle(f"AI GPU Benchmark — {gpu_name}")
        self.log_to_console(f"Detected GPU: {gpu_name}")

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
        self.log_to_console(f"TELEMETRY ERROR: {message}")

    @pyqtSlot(dict)
    def update_telemetry_ui(self, data: dict):
        """Updates the dashboard with live PyNVML data."""
        self.logger.log(data)
        if self._vram_total_mib is None and data.get('vram_total_mb'):
            self._vram_total_mib = data['vram_total_mb']
            self._update_fit_hint()
        self.lbl_temp.setText(f"Temp: {data['temp_gpu']} °C")

        if data.get('temp_memory') is not None:
            self.lbl_mem_temp.setText(f"Mem Temp: {data['temp_memory']:.0f} °C")
        else:
            self.lbl_mem_temp.setText("Mem Temp: N/A")

        if data.get('temp_headroom_c') is not None:
            self.lbl_headroom.setText(
                f"Thermal Headroom: {data['temp_headroom_c']:.0f} °C"
            )
        else:
            self.lbl_headroom.setText("Thermal Headroom: N/A")

        self.lbl_power.setText(f"Power: {data['power_w']} W")
        self.lbl_clock.setText(f"Clock: {data['sm_clock_mhz']} MHz")
        self.lbl_vram.setText(f"VRAM: {data['vram_used_mb']} / {data['vram_total_mb']} MiB")
        self.lbl_fan.setText(self._format_fan(data))
        self.lbl_pcie.setText(self._format_pcie(data))
        self._show_throttle(data.get('throttle_reasons'))

    def _show_throttle(self, mask):
        reasons = decode_throttle_reasons(mask)
        if mask is None:
            self.lbl_throttle.setText("Throttle: N/A")
        else:
            self.lbl_throttle.setText(f"Throttle: {', '.join(reasons) or 'none'}")

        # Re-polished only on a change of state: this runs four times a second.
        throttled = bool(reasons)
        if self.lbl_throttle.property("warning") != throttled:
            self.lbl_throttle.setProperty("warning", throttled)
            self.lbl_throttle.style().unpolish(self.lbl_throttle)
            self.lbl_throttle.style().polish(self.lbl_throttle)

    @staticmethod
    def _format_fan(data: dict) -> str:
        pct, rpm = data.get('fan_speed_pct'), data.get('fan_rpm')
        if pct is None:
            return "Fan: N/A"
        if rpm:
            return f"Fan: {pct:.0f}% ({rpm} RPM)"
        return f"Fan: {pct:.0f}%"

    @staticmethod
    def _format_pcie(data: dict) -> str:
        gen, width = data.get('pcie_gen'), data.get('pcie_width')
        if gen is None or width is None:
            return "PCIe: N/A"

        text = f"PCIe: Gen{gen} x{width}"
        gen_max, width_max = data.get('pcie_gen_max'), data.get('pcie_width_max')
        # Only worth showing when the link is downshifted — otherwise it is
        # noise repeating what the current values already say.
        if gen_max and width_max and (gen, width) != (gen_max, width_max):
            text += f" (max Gen{gen_max} x{width_max})"
        return text

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
        self.progress_bar.setValue(0)
        self.plot_tabs.setVisible(False)
        self.summary_group.setVisible(False)
        self.log_to_console("\n--- Starting Synthetic Workload ---")
        if self.lbl_fit.property("warning"):
            self.log_to_console(f"WARNING: {self.lbl_fit.text()}")
        self.logger.start()

        # Locked for the duration: the summary reports these values, so they
        # must describe the run that actually happened.
        self.workload_group.setEnabled(False)

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
        self.trainer.progress_updated.connect(lambda step, loss: self.progress_bar.setValue(step))
        self.trainer.max_steps_ready.connect(lambda steps: self.progress_bar.setRange(0, steps))
        self.trainer.training_finished.connect(self.on_benchmark_finished)
        self.trainer.error_occurred.connect(self.on_benchmark_error)
        self.trainer.start()

    def stop_benchmark(self):
        """Requests an early stop; the run ends at the next step boundary."""
        if self.trainer and self.trainer.isRunning():
            self.log_to_console("\nStop requested — finishing current step...")
            self.trainer.stop()
        self.btn_stop.setEnabled(False)

    @pyqtSlot(dict)
    def on_benchmark_finished(self, summary: dict):
        self.btn_stop.setEnabled(False)
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
        self.log_to_console(f"ERROR: {message}")
        # Close the run, or its partial telemetry leaks into the next one.
        df = self.logger.stop()
        self.progress_bar.setValue(0)
        self._set_idle_controls()
        self.btn_export.setEnabled(df is not None)

    def _set_idle_controls(self):
        """Returns the control panel to its ready-for-a-new-run state."""
        self.workload_group.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)

    def _render_plots(self, df, summary: dict):
        """Draws thermal/power and utilization/clock plots into tabs."""
        for canvas, draw in ((self.thermal_canvas, draw_thermal_power),
                             (self.util_canvas, draw_utilization)):
            figure = canvas.figure
            figure.clear()
            draw(figure.add_subplot(111), df)
            figure.tight_layout()
            canvas.draw()

        self.plot_tabs.setVisible(True)
        self.plot_tabs.setCurrentIndex(0)

        # --- Summary Stats Table ---
        self._render_summary(df, summary)
        self.log_to_console("Summary and plots rendered.")

    def _render_summary(self, df, summary: dict):
        """Populates the summary table with computed benchmark metrics."""
        stats = compute_stats(df, summary)

        steps_label = f"{summary['total_steps']}"
        if summary.get('aborted'):
            steps_label += f" of {summary['requested_steps']} (stopped early)"

        checkpointing = "—"
        if "gradient_checkpointing" in summary:
            checkpointing = "On" if summary["gradient_checkpointing"] else "Off"

        metrics = [
            ("Model", summary.get("model", "—")),
            ("Batch Size", str(summary.get("batch_size", "—"))),
            ("Precision", summary.get("precision", "—")),
            ("Sequence Length", str(summary.get("seq_len", "—"))),
            ("Gradient Checkpointing", checkpointing),
            ("Warm-up Steps", str(summary.get("warmup_steps", "—"))),
            ("Steps Completed", steps_label),
            ("Duration (s)", f"{summary['elapsed_time_sec']:.2f}"),
            ("Steps/sec", f"{summary['steps_per_sec']:.2f}"),
        ]

        # Throughput and energy need the timed window, which a run stopped
        # during warm-up never opened.
        if stats['tokens_per_sec'] is not None:
            metrics.append(("Tokens/sec", f"{stats['tokens_per_sec']:,.0f}"))
        if stats['est_tflops'] is not None:
            metrics.append(("Compute (est. TFLOPS)", f"{stats['est_tflops']:.1f}"))
        if stats['energy_j'] is not None:
            metrics.append((f"Energy, timed (kJ, {stats['energy_source']})",
                            f"{stats['energy_j'] / 1000:.2f}"))
            metrics.append(("Efficiency (tokens/J)", f"{stats['tokens_per_joule']:.1f}"))

        metrics += [
            ("Avg Temp (°C)", f"{stats['avg_temp_c']:.1f}"),
            ("Max Temp (°C)", f"{stats['max_temp_c']:.0f}"),
            ("Avg Power (W)", f"{stats['avg_power_w']:.1f}"),
            ("Peak Power (W)", f"{stats['peak_power_w']:.1f}"),
            ("Avg VRAM (MiB)", f"{stats['avg_vram_mb']:.1f}"),
            ("Peak VRAM (MiB)", f"{stats['peak_vram_mb']:.1f}"),
        ]

        # Optional hardware: memory temperature needs a GPU exposing the
        # field, and fan and PCIe readings are absent on passive cards.
        if stats['max_mem_temp_c'] is not None:
            metrics.append(("Max Mem Temp (°C)", f"{stats['max_mem_temp_c']:.0f}"))
        if stats['min_headroom_c'] is not None:
            metrics.append(("Min Thermal Headroom (°C)", f"{stats['min_headroom_c']:.0f}"))
        if stats['peak_fan_pct'] is not None:
            metrics.append(("Peak Fan (%)", f"{stats['peak_fan_pct']:.0f}"))
        if stats['power_limited_pct'] is not None:
            metrics.append(("Time Power-Limited (%)", f"{stats['power_limited_pct']:.0f}"))
            metrics.append(("Time Thermal-Limited (%)", f"{stats['thermal_limited_pct']:.0f}"))
        if stats['pcie_gen_peak'] is not None and stats['pcie_width_peak'] is not None:
            metrics.append(("PCIe Link (peak)",
                            f"Gen{stats['pcie_gen_peak']:.0f} x{stats['pcie_width_peak']:.0f}"))

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
        try:
            csv_path, plot_path = BenchmarkLogger.export(df)
        except OSError as e:
            self.log_to_console(f"Export failed: {e}")
            return
        self.log_to_console(f"Exported: {csv_path}")
        self.log_to_console(f"Exported: {plot_path}")

    def closeEvent(self, event):
        """Clean up threads on exit."""
        self.telemetry.stop()
        if self.trainer and self.trainer.isRunning():
            # Shutdown is the one place we genuinely must block, so the
            # training thread cannot outlive the window it reports into.
            self.trainer.wait_for_exit()
        event.accept()
