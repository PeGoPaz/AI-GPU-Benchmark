import sys
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from app.core.telemetry import TelemetryWorker


class TestWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step 2 Test: Real-Time Telemetry Feed")
        self.resize(450, 200)

        # UI Setup
        self.layout = QVBoxLayout()
        self.label = QLabel("Initializing Telemetry Thread...")
        self.label.setStyleSheet("font-size: 14px; font-family: monospace;")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Initialize & Start Telemetry Thread
        self.telemetry_thread = TelemetryWorker(interval_ms=250)
        self.telemetry_thread.data_updated.connect(self.update_telemetry_ui)
        self.telemetry_thread.error_occurred.connect(self.show_error)
        self.telemetry_thread.start()

    @pyqtSlot(dict)
    def update_telemetry_ui(self, data: dict):
        """Receives live dictionary snapshots from background thread."""
        text = (
            f"<b>GPU Temp:</b> {data['temp_gpu']} °C<br>"
            f"<b>Power Draw:</b> {data['power_w']} W<br>"
            f"<b>Core Usage:</b> {data['gpu_util_pct']} %<br>"
            f"<b>Clock Speed:</b> {data['sm_clock_mhz']} MHz<br>"
            f"<b>VRAM Used:</b> {data['vram_used_mb']} / {data['vram_total_mb']} MiB"
        )
        self.label.setText(text)

    @pyqtSlot(str)
    def show_error(self, err_msg: str):
        self.label.setText(f"<font color='red'>{err_msg}</font>")

    def closeEvent(self, event):
        """Ensure thread shuts down safely when closing window."""
        self.telemetry_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())