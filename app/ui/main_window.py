from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QProgressBar, QTextEdit
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from app.core.telemetry import TelemetryWorker
from app.core.trainer import TrainerWorker
from app.utils.logger import BenchmarkLogger

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI GPU Benchmark — detecting GPU…")
        self.resize(800, 600)

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
        self.lbl_hotspot = QLabel("Hotspot: -- °C")
        self.lbl_power = QLabel("Power: -- W")
        self.lbl_clock = QLabel("Clock: -- MHz")
        self.lbl_vram = QLabel("VRAM: -- / -- MiB")
        
        # Make the fonts slightly larger for the dashboard
        for lbl in [self.lbl_temp, self.lbl_hotspot, self.lbl_power, self.lbl_clock, self.lbl_vram]:
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
        
        ctrl_layout.addWidget(self.btn_start)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)
        left_panel.addStretch() # Pushes everything up

        # --- RIGHT PANEL: Console & Progress ---
        right_panel = QVBoxLayout()
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(220)
        self.console.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: monospace;")
        self.log_to_console("System initialized. Ready for benchmark.")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Plot area (hidden until a benchmark finishes)
        self.plot_figure = Figure(figsize=(8, 3))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.setMinimumHeight(260)
        self.plot_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(QLabel("Thermal & Power Curve"))
        plot_layout.addWidget(self.plot_canvas)
        self.plot_container.setVisible(False)

        right_panel.addWidget(QLabel("Execution Log:"))
        right_panel.addWidget(self.console)
        right_panel.addWidget(self.progress_bar)
        right_panel.addWidget(self.plot_container, 1)

        # Add panels to main layout (Left takes 1 part, Right takes 2 parts of width)
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        # --- INITIALIZE THREADS ---
        self.logger = BenchmarkLogger()
        self.telemetry = TelemetryWorker(interval_ms=250)
        self.telemetry.data_updated.connect(self.update_telemetry_ui)
        self.telemetry.gpu_name_ready.connect(self._on_gpu_name_ready)
        self.telemetry.start()
        
        self.trainer = None

    def log_to_console(self, text: str):
        """Appends text to the mock terminal window."""
        self.console.append(text)

    @pyqtSlot(str)
    def _on_gpu_name_ready(self, gpu_name: str):
        """Updates the window title once the GPU name is detected."""
        self.setWindowTitle(f"AI GPU Benchmark — {gpu_name}")
        self.log_to_console(f"Detected GPU: {gpu_name}")

    @pyqtSlot(dict)
    def update_telemetry_ui(self, data: dict):
        """Updates the dashboard with live PyNVML data."""
        self.logger.log(data)
        self.lbl_temp.setText(f"Temp: {data['temp_gpu']} °C")
        
        # Hotspot may be None if GPU doesn't expose it
        if data.get('temp_hotspot') is not None:
            self.lbl_hotspot.setText(f"Hotspot: {data['temp_hotspot']} °C")
        else:
            self.lbl_hotspot.setText("Hotspot: N/A")
        
        self.lbl_power.setText(f"Power: {data['power_w']} W")
        self.lbl_clock.setText(f"Clock: {data['sm_clock_mhz']} MHz")
        self.lbl_vram.setText(f"VRAM: {data['vram_used_mb']} / {data['vram_total_mb']} MiB")

    def start_benchmark(self):
        """Disables UI and fires up the PyTorch training thread."""
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("background-color: #555555; color: white;")
        self.progress_bar.setValue(0)
        self.plot_container.setVisible(False)
        self.log_to_console("\n--- Starting Synthetic Workload ---")
        self.logger.start()

        self.trainer = TrainerWorker(steps=150, batch_size=4)
        self.trainer.status_updated.connect(self.log_to_console)
        self.trainer.progress_updated.connect(lambda step, loss: self.progress_bar.setValue(step))
        self.trainer.max_steps_ready.connect(lambda steps: self.progress_bar.setRange(0, steps))
        self.trainer.training_finished.connect(self.on_benchmark_finished)
        self.trainer.error_occurred.connect(lambda err: self.log_to_console(f"ERROR: {err}"))
        self.trainer.start()

    @pyqtSlot(dict)
    def on_benchmark_finished(self, summary: dict):
        df = self.logger.stop()
        if df is not None:
            self._render_plot(df)
        self.log_to_console(f"\nBenchmark Complete! Time: {summary['elapsed_time_sec']}s")
        self.log_to_console(f"Throughput: {summary['steps_per_sec']} steps/sec")
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #2E8B57; color: white;")

    def _render_plot(self, df):
        """Draws the thermal & power curves onto the embedded matplotlib canvas."""
        self.plot_figure.clear()

        ax1 = self.plot_figure.add_subplot(111)
        ax1.plot(df['time_sec'], df['temp_gpu'], color='#ff4c4c',
                 label='GPU Temp (°C)', linewidth=2)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Temperature (°C)', color='#ff4c4c')
        ax1.tick_params(axis='y', labelcolor='#ff4c4c')

        ax2 = ax1.twinx()
        ax2.plot(df['time_sec'], df['power_w'], color='#4c72ff',
                 label='Power Draw (W)', linewidth=2, linestyle='--')
        ax2.set_ylabel('Power (W)', color='#4c72ff')
        ax2.tick_params(axis='y', labelcolor='#4c72ff')

        ax1.set_title('GPU Thermal & Power Curve During Stress Test')
        ax1.grid(True, alpha=0.2)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        self.plot_figure.tight_layout()
        self.plot_canvas.draw()
        self.plot_container.setVisible(True)
        self.log_to_console("Plot rendered.")

    def closeEvent(self, event):
        """Clean up threads on exit."""
        self.telemetry.stop()
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop()
        event.accept()