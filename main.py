import sys
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QProgressBar
)

from app.core.telemetry import TelemetryWorker
from app.core.trainer import TrainerWorker


class StressTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step 3 Test: Dual-Thread Concurrent Benchmark")
        self.resize(500, 320)

        # UI Components
        self.layout = QVBoxLayout()
        
        self.telemetry_label = QLabel("Telemetry: Initializing...")
        self.telemetry_label.setStyleSheet("font-size: 13px; font-family: monospace;")
        
        self.status_label = QLabel("Status: Ready to stress test.")
        self.status_label.setStyleSheet("font-weight: bold;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)

        self.start_btn = QPushButton("Start 1000-Step Benchmark Load")
        self.start_btn.clicked.connect(self.start_training)

        self.layout.addWidget(self.telemetry_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.start_btn)
        self.setLayout(self.layout)

        # 1. Start Telemetry Worker Thread
        self.telemetry = TelemetryWorker(interval_ms=250)
        self.telemetry.data_updated.connect(self.update_telemetry)
        self.telemetry.start()

        self.trainer = None

    @pyqtSlot(dict)
    def update_telemetry(self, data: dict):
        self.telemetry_label.setText(
            f"<b>GPU Temp:</b> {data['temp_gpu']} °C | "
            f"<b>Power:</b> {data['power_w']} W | "
            f"<b>Usage:</b> {data['gpu_util_pct']} %<br>"
            f"<b>Clock:</b> {data['sm_clock_mhz']} MHz | "
            f"<b>VRAM:</b> {data['vram_used_mb']} / {data['vram_total_mb']} MiB"
        )

    def start_training(self):
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # 2. Start Trainer Worker Thread
        self.trainer = TrainerWorker(steps=1000, batch_size=256)
        self.trainer.status_updated.connect(lambda msg: self.status_label.setText(f"Status: {msg}"))
        self.trainer.progress_updated.connect(self.update_progress)
        self.trainer.training_finished.connect(self.on_training_finished)
        self.trainer.error_occurred.connect(lambda err: self.status_label.setText(f"Error: {err}"))
        self.trainer.start()

    @pyqtSlot(int, float)
    def update_progress(self, step: int, loss: float):
        self.progress_bar.setValue(step)

    @pyqtSlot(dict)
    def on_training_finished(self, summary: dict):
        self.status_label.setText(
            f"Done in {summary['elapsed_time_sec']}s ({summary['steps_per_sec']} steps/sec)"
        )
        self.start_btn.setEnabled(True)

    def closeEvent(self, event):
        self.telemetry.stop()
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StressTestWindow()
    window.show()
    sys.exit(app.exec())