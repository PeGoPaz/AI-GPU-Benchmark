import time
import pynvml
from PyQt6.QtCore import QThread, pyqtSignal


def _read_hotspot_temperature(handle):
    """Attempt to read GPU hotspot (junction) temperature.
    Returns int or None if unavailable on this GPU/driver combo.
    """
    # Method 1: Try NVML extended temperature sensor
    # Sensor types: 0=GPU, 1=Memory, 2=Power Supply (varies by GPU)
    # Some drivers support additional sensor types for hotspot
    try:
        # Try sensor type 2 (sometimes hotspot on newer GPUs)
        return pynvml.nvmlDeviceGetTemperature(handle, 2)
    except Exception:
        pass

    # Method 2: Try alternative sensor index
    try:
        return pynvml.nvmlDeviceGetTemperature(handle, 3)
    except Exception:
        pass

    return None


class TelemetryWorker(QThread):
    # Signals to pass hardware telemetry data back to the UI thread
    data_updated = pyqtSignal(dict)
    gpu_name_ready = pyqtSignal(str)
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

            # Emit GPU name once at startup
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8", errors="replace")
            # Some nvidia-ml-py versions return str already
            self.gpu_name_ready.emit(gpu_name)
        except Exception as e:
            self.error_occurred.emit(f"NVML Initialization Failed: {str(e)}")
            return

        while self._is_running:
            try:
                # 1. Temperature (°C)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # 2. Hotspot/Junction Temperature (°C)
                temp_hotspot = _read_hotspot_temperature(handle)

                # 3. VRAM Usage (Convert bytes to MiB)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used = mem_info.used / (1024 * 1024)
                vram_total = mem_info.total / (1024 * 1024)

                # 4. Power Draw (Convert mW to Watts)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

                # 5. Core Utilization (%)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # 6. Core Clock Speed (MHz)
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )

                # Package stats into a telemetry snapshot
                telemetry = {
                    "timestamp": time.time(),
                    "temp_gpu": temp,
                    "temp_hotspot": temp_hotspot,  # May be None if unsupported
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