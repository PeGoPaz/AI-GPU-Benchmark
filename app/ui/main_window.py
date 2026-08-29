from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.telemetry import TelemetryWorker
from app.core.trainer import TrainerWorker
from app.utils.logger import BenchmarkLogger


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

        for lbl in [self.lbl_temp, self.lbl_mem_temp, self.lbl_headroom,
                    self.lbl_power, self.lbl_clock, self.lbl_vram,
                    self.lbl_fan, self.lbl_pcie]:
            lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
            hw_layout.addWidget(lbl)

        hw_group.setLayout(hw_layout)
        left_panel.addWidget(hw_group)

        # 2. Controls
        ctrl_group = QGroupBox("Benchmark Controls")
        ctrl_layout = QVBoxLayout()

        self.btn_start = QPushButton("Start Stress Test")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #2E8B57; color: white;")
        self.btn_start.clicked.connect(self.start_benchmark)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(35)
        self.btn_stop.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_benchmark)
        self.btn_stop.setEnabled(False)

        self.btn_export = QPushButton("Export Results")
        self.btn_export.setMinimumHeight(35)
        self.btn_export.setStyleSheet("background-color: #4682B4; color: white;")
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
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(180)
        self.console.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: monospace;")
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
        self.plot_tabs.addTab(self.thermal_canvas, "Thermal & Power")

        # Tab 2: Utilization & Clock
        util_figure = Figure(figsize=(8, 3))
        self.util_canvas = FigureCanvas(util_figure)
        self.plot_tabs.addTab(self.util_canvas, "Utilization & Clock")

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

    def start_benchmark(self):
        """Disables UI and fires up the PyTorch training thread."""
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("background-color: #555555; color: white;")
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.plot_tabs.setVisible(False)
        self.summary_group.setVisible(False)
        self.log_to_console("\n--- Starting Synthetic Workload ---")
        self.logger.start()

        self.trainer = TrainerWorker(steps=150, batch_size=4)
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
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #2E8B57; color: white;")

    def _render_plots(self, df, summary: dict):
        """Draws thermal/power and utilization/clock plots into tabs."""
        # --- Tab 1: Thermal & Power ---
        tf = self.thermal_canvas.figure
        tf.clear()
        ax1 = tf.add_subplot(111)
        ax1.plot(df['time_sec'], df['temp_gpu'], color='#ff4c4c',
                 label='GPU Temp (°C)', linewidth=2)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Temperature (°C)', color='#ff4c4c')
        ax1.tick_params(axis='y', labelcolor='#ff4c4c')
        ax1.set_title('Thermal & Power Curve')

        ax2 = ax1.twinx()
        ax2.plot(df['time_sec'], df['power_w'], color='#4c72ff',
                 label='Power Draw (W)', linewidth=2, linestyle='--')
        ax2.set_ylabel('Power (W)', color='#4c72ff')
        ax2.tick_params(axis='y', labelcolor='#4c72ff')

        ax1.grid(True, alpha=0.2)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        tf.tight_layout()
        self.thermal_canvas.draw()

        # --- Tab 2: Utilization & Clock ---
        uf = self.util_canvas.figure
        uf.clear()
        ax3 = uf.add_subplot(111)
        ax3.plot(df['time_sec'], df['gpu_util_pct'], color='#ffa500',
                 label='GPU Utilization (%)', linewidth=2)
        ax3.plot(df['time_sec'], df['sm_clock_mhz'], color='#00bfff',
                 label='SM Clock (MHz)', linewidth=2, linestyle='--')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Utilization (%)')
        ax3.set_title('Utilization & Clock Speed')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.2)
        uf.tight_layout()
        self.util_canvas.draw()

        self.plot_tabs.setVisible(True)
        self.plot_tabs.setCurrentIndex(0)

        # --- Summary Stats Table ---
        self._render_summary(df, summary)
        self.log_to_console("Summary and plots rendered.")

    def _render_summary(self, df, summary: dict):
        """Populates the summary table with computed benchmark metrics."""
        avg_temp = df['temp_gpu'].mean()
        max_temp = df['temp_gpu'].max()
        avg_power = df['power_w'].mean()
        peak_power = df['power_w'].max()
        avg_vram = df['vram_used_mb'].mean()
        peak_vram = df['vram_used_mb'].max()

        steps_per_sec = summary['steps_per_sec']
        efficiency = steps_per_sec / avg_power if avg_power > 0 else 0

        steps_label = f"{summary['total_steps']}"
        if summary.get('aborted'):
            steps_label += f" of {summary['requested_steps']} (stopped early)"

        metrics = [
            ("Steps Completed", steps_label),
            ("Duration (s)", f"{summary['elapsed_time_sec']:.2f}"),
            ("Steps/sec", f"{steps_per_sec:.2f}"),
            ("Efficiency (steps/sec/W)", f"{efficiency:.4f}"),
            ("Avg Temp (°C)", f"{avg_temp:.1f}"),
            ("Max Temp (°C)", f"{max_temp}"),
            ("Avg Power (W)", f"{avg_power:.1f}"),
            ("Peak Power (W)", f"{peak_power:.1f}"),
            ("Avg VRAM (MiB)", f"{avg_vram:.1f}"),
            ("Peak VRAM (MiB)", f"{peak_vram:.1f}"),
        ]

        # Memory temperature is only present on GPUs that expose the field.
        if 'temp_memory' in df.columns and df['temp_memory'].notna().any():
            metrics.append(("Max Mem Temp (°C)", f"{df['temp_memory'].max():.0f}"))
        if 'temp_headroom_c' in df.columns and df['temp_headroom_c'].notna().any():
            metrics.append(
                ("Min Thermal Headroom (°C)", f"{df['temp_headroom_c'].min():.0f}")
            )

        # Cooling and interconnect: absent on passively cooled cards.
        if 'fan_speed_pct' in df.columns and df['fan_speed_pct'].notna().any():
            metrics.append(("Peak Fan (%)", f"{df['fan_speed_pct'].max():.0f}"))
        if ('pcie_gen' in df.columns and df['pcie_gen'].notna().any()
                and df['pcie_width'].notna().any()):
            # The link under load is the interesting one; at idle it downshifts.
            metrics.append(
                ("PCIe Link (peak)",
                 f"Gen{df['pcie_gen'].max():.0f} x{df['pcie_width'].max():.0f}")
            )

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
