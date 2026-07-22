from app.utils.logger import BenchmarkLogger
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGroupBox, QLabel, QPushButton, QProgressBar, QTextEdit
)

from app.core.telemetry import TelemetryWorker
from app.core.trainer import TrainerWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI GPU Benchmark - RTX 4070 Ti Super")
        self.resize(800, 500)

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
        self.lbl_power = QLabel("Power: -- W")
        self.lbl_clock = QLabel("Clock: -- MHz")
        self.lbl_vram = QLabel("VRAM: -- / -- MiB")
        
        # Make the fonts slightly larger for the dashboard
        for lbl in [self.lbl_temp, self.lbl_power, self.lbl_clock, self.lbl_vram]:
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
        self.console.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: monospace;")
        self.log_to_console("System initialized. Ready for benchmark.")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)

        right_panel.addWidget(QLabel("Execution Log:"))
        right_panel.addWidget(self.console)
        right_panel.addWidget(self.progress_bar)

        # Add panels to main layout (Left takes 1 part, Right takes 2 parts of width)
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        # --- INITIALIZE THREADS ---
        self.logger = BenchmarkLogger()
        self.telemetry = TelemetryWorker(interval_ms=250)
        self.telemetry.data_updated.connect(self.update_telemetry_ui)
        self.telemetry.start()
        
        self.trainer = None

    def log_to_console(self, text: str):
        """Appends text to the mock terminal window."""
        self.console.append(text)

    @pyqtSlot(dict)
    def update_telemetry_ui(self, data: dict):
        """Updates the dashboard with live PyNVML data."""
        self.logger.log(data)
        self.lbl_temp.setText(f"Temp: {data['temp_gpu']} °C")
        self.lbl_power.setText(f"Power: {data['power_w']} W")
        self.lbl_clock.setText(f"Clock: {data['sm_clock_mhz']} MHz")
        self.lbl_vram.setText(f"VRAM: {data['vram_used_mb']} / {data['vram_total_mb']} MiB")

    def start_benchmark(self):
        """Disables UI and fires up the PyTorch training thread."""
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("background-color: #555555; color: white;")
        self.progress_bar.setValue(0)
        self.log_to_console("\n--- Starting Synthetic Workload ---")
        self.logger.start()

        self.trainer = TrainerWorker(steps=150, batch_size=4)
        self.trainer.status_updated.connect(self.log_to_console)
        self.trainer.progress_updated.connect(lambda step, loss: self.progress_bar.setValue(step))
        self.trainer.training_finished.connect(self.on_benchmark_finished)
        self.trainer.error_occurred.connect(lambda err: self.log_to_console(f"ERROR: {err}"))
        self.trainer.start()

    @pyqtSlot(dict)
    def on_benchmark_finished(self, summary: dict):
        plot_file = self.logger.stop_and_save()
        if plot_file:
            self.log_to_console(f"Data saved! Graph generated at: {plot_file}")
        self.log_to_console(f"\nBenchmark Complete! Time: {summary['elapsed_time_sec']}s")
        self.log_to_console(f"Throughput: {summary['steps_per_sec']} steps/sec")
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #2E8B57; color: white;")

    def closeEvent(self, event):
        """Clean up threads on exit."""
        self.telemetry.stop()
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop()
        event.accept()