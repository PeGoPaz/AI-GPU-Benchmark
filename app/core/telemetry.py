import time

import pynvml
from PyQt6.QtCore import QThread, pyqtSignal

# NVML exposes exactly one temperature sensor via nvmlDeviceGetTemperature
# (NVML_TEMPERATURE_COUNT == 1, sensor 0 == GPU core). The junction/hotspot
# sensor is NOT reachable through the public NVML API on any driver — tools
# that display it read it over NvAPI or raw BAR0 MMIO. Passing sensor index
# 2 or 3 just raises NVML_ERROR_INVALID_ARGUMENT on every single call.
#
# What IS available on Ampere and newer are these field values, which we use
# as the honest substitute: memory temperature, and the thermal headroom
# (T.Limit) remaining before the GPU throttles below base clock.
_FIELD_MEMORY_TEMP = pynvml.NVML_FI_DEV_MEMORY_TEMP
_FIELD_GPU_TLIMIT = pynvml.NVML_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT

_VALUE_READERS = {
    pynvml.NVML_VALUE_TYPE_DOUBLE: lambda v: v.dVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_INT: lambda v: v.uiVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG: lambda v: v.ulVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: lambda v: v.ullVal,
    pynvml.NVML_VALUE_TYPE_SIGNED_INT: lambda v: v.siVal,
    pynvml.NVML_VALUE_TYPE_SIGNED_LONG_LONG: lambda v: v.sllVal,
    pynvml.NVML_VALUE_TYPE_UNSIGNED_SHORT: lambda v: v.usVal,
}


def read_field_values(handle, field_ids: list[int]) -> dict[int, float]:
    """Reads NVML field values, returning {field_id: value} for successful reads.

    Fields the GPU/driver does not support are omitted rather than raising, so
    a missing key means "unsupported on this hardware".
    """
    result: dict[int, float] = {}
    try:
        values = pynvml.nvmlDeviceGetFieldValues(handle, list(field_ids))
    except Exception:
        return result

    for value in values:
        if value.nvmlReturn != pynvml.NVML_SUCCESS:
            continue
        reader = _VALUE_READERS.get(value.valueType)
        if reader is not None:
            result[value.fieldId] = reader(value.value)

    return result


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

                # 2. Memory temperature & thermal headroom (Ampere+ field values).
                #    These replace the old bogus "hotspot" probe: NVML has no
                #    public junction sensor. Both keys stay None when unsupported.
                fields = read_field_values(
                    handle, [_FIELD_MEMORY_TEMP, _FIELD_GPU_TLIMIT]
                )
                temp_memory = fields.get(_FIELD_MEMORY_TEMP)
                temp_headroom = fields.get(_FIELD_GPU_TLIMIT)

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
                    "temp_memory": temp_memory,      # None if unsupported
                    "temp_headroom_c": temp_headroom,  # None if unsupported
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
