import time
import pynvml
from PyQt6.QtCore import QThread, pyqtSignal


class TelemetryWorker(QThread):
    # Signals to pass hardware telemetry data back to the UI thread
    data_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, interval_ms: int = 250, parent=None):
        super().__init__(parent)
        self.interval_ms = interval_ms
        self._is_running = True

    def run(self):
        """Main thread loop polling NVML at the given interval."""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            self.error_occurred.emit(f"NVML Initialization Failed: {str(e)}")
            return

        while self._is_running:
            try:
                # 1. Temperature (°C)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # 2. VRAM Usage (Convert bytes to MiB)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used = mem_info.used / (1024 * 1024)
                vram_total = mem_info.total / (1024 * 1024)

                # 3. Power Draw (Convert mW to Watts)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

                # 4. Core Utilization (%)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # 5. Core Clock Speed (MHz)
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )

                # Package stats into a telemetry snapshot
                telemetry = {
                    "timestamp": time.time(),
                    "temp_gpu": temp,
                    "vram_used_mb": round(vram_used, 2),
                    "vram_total_mb": round(vram_total, 2),
                    "power_w": round(power, 2),
                    "gpu_util_pct": utilization.gpu,
                    "sm_clock_mhz": sm_clock,
                }

                # Broadcast data to UI
                self.data_updated.emit(telemetry)

            except Exception as e:
                self.error_occurred.emit(f"Telemetry Read Error: {str(e)}")

            self.msleep(self.interval_ms)

        # Cleanup NVML when thread stops
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def stop(self):
        """Safely signals the thread loop to terminate."""
        self._is_running = False
        self.wait()