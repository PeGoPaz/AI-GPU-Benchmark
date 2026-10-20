import time

import pynvml
from PyQt6.QtCore import QThread, pyqtSignal

from gpu_bench.core.power_sampler import describe_status, open_sampler

# nvmlDeviceGetTemperature takes sensor 0 (GPU core) and, since NVML 13,
# sensor 1 (NVML_TEMPERATURE_GPU_MAX). The junction/hotspot sensor is NOT
# reachable through the public NVML API on any driver — tools that display it
# read it over NvAPI or raw BAR0 MMIO. Passing sensor index 2 or 3 just raises
# NVML_ERROR_INVALID_ARGUMENT on every single call.
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


# Why the clock is below its maximum, as a bitmask. The values are nvml.h's
# and are written out here rather than taken from pynvml, because NVML is
# renaming the constants (ClocksThrottleReason* → ClocksEventReason*, the old
# names due to go with CUDA 14) while the bits themselves stay the same.
THROTTLE_SW_POWER_CAP = 0x04
THROTTLE_HW_SLOWDOWN = 0x08
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80

# Only the reasons that hold a working card back. NVML also reports GpuIdle,
# SyncBoost and the application/display clock settings, but those are the
# card resting or obeying a setting, not running out of power or cooling.
THROTTLE_LABELS = (
    (THROTTLE_SW_POWER_CAP, "Power cap"),
    (THROTTLE_HW_POWER_BRAKE, "Power brake"),
    (THROTTLE_SW_THERMAL, "Thermal"),
    (THROTTLE_HW_THERMAL, "Thermal (HW)"),
    (THROTTLE_HW_SLOWDOWN, "HW slowdown"),
)

POWER_LIMIT_BITS = THROTTLE_SW_POWER_CAP | THROTTLE_HW_POWER_BRAKE
THERMAL_LIMIT_BITS = THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL


def decode_throttle_reasons(mask: int | None) -> list[str]:
    """Human-readable reasons in a throttle bitmask; empty when unthrottled."""
    if not mask:
        return []
    return [label for bit, label in THROTTLE_LABELS if int(mask) & bit]


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


def _try_read(fn, *args):
    """Calls an NVML getter, returning None when this GPU or driver lacks it.

    Unlike the core temperature and power readings, fan and PCIe queries are
    genuinely optional: passively cooled datacenter cards raise
    NVML_ERROR_NOT_SUPPORTED for anything fan-related, and drivers older than
    the RPM entry point raise NVML_ERROR_FUNCTION_NOT_FOUND. Neither should
    cost us the rest of the snapshot.
    """
    try:
        return fn(*args)
    except Exception:
        return None


def read_fan_speed_pct(handle, num_fans: int | None) -> float | None:
    """Highest fan duty cycle on the card, or None if it exposes no fans.

    The maximum across fans is the number worth watching: fans pegged at 100%
    mean the card has run out of thermal room. Cards that predate the indexed
    getter still answer the unindexed one.
    """
    if not num_fans:
        return _try_read(pynvml.nvmlDeviceGetFanSpeed, handle)

    speeds = [speed for fan in range(num_fans)
              if (speed := _try_read(pynvml.nvmlDeviceGetFanSpeed_v2, handle, fan))
              is not None]
    return max(speeds) if speeds else None


def read_pcie_link(handle) -> tuple[int | None, int | None]:
    """Current PCIe generation and width.

    Polled rather than read once: the link drops to a lower generation at idle
    to save power and comes back up under load, which is exactly the transition
    a benchmark wants to show.
    """
    return (_try_read(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle),
            _try_read(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle))


def read_throttle_reasons(handle) -> int | None:
    """The current throttle bitmask, None if the driver cannot say.

    The getter was renamed along with the constants. Newer drivers answer
    the new name; drivers older than the rename only know the old one.
    """
    for name in ("nvmlDeviceGetCurrentClocksEventReasons",
                 "nvmlDeviceGetCurrentClocksThrottleReasons"):
        getter = getattr(pynvml, name, None)
        if getter is None:
            continue
        mask = _try_read(getter, handle)
        if mask is not None:
            return mask
    return None


def format_cuda_version(version: int | None) -> str | None:
    """NVML's packed CUDA version (12040) as it is usually written (12.4)."""
    if not version:
        return None
    return f"{version // 1000}.{(version % 1000) // 10}"


def _decode(value):
    """Some nvidia-ml-py versions return bytes for strings, others str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def read_device_info(handle, gpu_name: str) -> dict:
    """What a run report needs to say which card, driver and limits it ran on.

    Read once — none of it changes while the app is open. The enforced power
    limit is the context for the throttle reasons: a card pinned at its power
    cap is drawing exactly this.
    """
    power_limit_mw = _try_read(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
    memory = _try_read(pynvml.nvmlDeviceGetMemoryInfo, handle)
    return {
        "name": gpu_name,
        "driver_version": _decode(_try_read(pynvml.nvmlSystemGetDriverVersion)),
        "cuda_driver_version": format_cuda_version(
            _try_read(pynvml.nvmlSystemGetCudaDriverVersion)),
        "vram_total_mb": round(memory.total / (1024 * 1024), 2) if memory else None,
        "power_limit_w": power_limit_mw / 1000.0 if power_limit_mw else None,
        "pcie_gen_max": _try_read(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle),
        "pcie_width_max": _try_read(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle),
    }


class TelemetryWorker(QThread):
    # Signals to pass hardware telemetry data back to the UI thread
    data_updated = pyqtSignal(dict)
    gpu_name_ready = pyqtSignal(str)
    device_info_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    # (unix seconds, watts) numpy arrays from the native sampler, once per
    # poll; never emitted when the module is not built.
    power_trace = pyqtSignal(object)
    # Once, after NVML is up: whether native sampling is active, and why not.
    sampler_status = pyqtSignal(dict)

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
            gpu_name = _decode(pynvml.nvmlDeviceGetName(handle))
            self.gpu_name_ready.emit(gpu_name)
            self.device_info_ready.emit(read_device_info(handle, gpu_name))
        except Exception as e:
            self.error_occurred.emit(f"NVML Initialization Failed: {str(e)}")
            return

        # Fixed for the lifetime of the card, so read them once rather than
        # every 250 ms. The link maxima give the current values context: a
        # Gen4 x16 card sitting at Gen1 x16 is idling, not misconfigured.
        num_fans = _try_read(pynvml.nvmlDeviceGetNumFans, handle)
        pcie_gen_max = _try_read(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle)
        pcie_width_max = _try_read(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle)

        # The native sampler opens its own NVML reference (init is counted),
        # so it can come and go without disturbing this loop.
        sampler, detail = open_sampler()
        if sampler is not None:
            try:
                sampler.start()
            except Exception as e:
                sampler, detail = None, f"native sampler failed to start: {e}"
        self.sampler_status.emit(describe_status(sampler, detail))

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

                # 7. Cooling and interconnect. Every one of these is optional
                #    hardware, so a card without them reports None rather than
                #    losing the whole snapshot.
                fan_pct = read_fan_speed_pct(handle, num_fans)
                fan_rpm = _try_read(pynvml.nvmlDeviceGetFanSpeedRPM, handle)
                pcie_gen, pcie_width = read_pcie_link(handle)

                # 8. Cumulative energy (mJ since the driver loaded), Volta and
                #    newer. Integrated by the card itself, so unlike sampled
                #    power it misses no spike between two polls.
                energy_mj = _try_read(pynvml.nvmlDeviceGetTotalEnergyConsumption, handle)

                # 9. Why the clock is where it is. A falling SM clock on its
                #    own does not say whether the card hit its power limit or
                #    its temperature limit; this does.
                throttle = read_throttle_reasons(handle)

                # 10. Everything the native sampler read since the last poll.
                #     The peak goes into the row; the trace itself travels
                #     separately, at its own rate.
                trace = sampler.drain() if sampler is not None else None
                power_peak = (float(trace[1].max())
                              if trace is not None and len(trace[1]) else None)

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
                    "fan_speed_pct": fan_pct,        # None if passively cooled
                    "fan_rpm": fan_rpm,              # None on older drivers
                    "pcie_gen": pcie_gen,
                    "pcie_width": pcie_width,
                    "pcie_gen_max": pcie_gen_max,
                    "pcie_width_max": pcie_width_max,
                    "energy_mj": energy_mj,          # None before Volta
                    "throttle_reasons": throttle,    # bitmask, None if unknown
                    "power_peak_w": power_peak,      # None without the native sampler
                }

                # Broadcast data to UI
                self.data_updated.emit(telemetry)
                if trace is not None and len(trace[0]):
                    self.power_trace.emit(trace)

            except Exception as e:
                self.error_occurred.emit(f"Telemetry Read Error: {str(e)}")

            self.msleep(self.interval_ms)

        # Cleanup NVML when thread stops
        if sampler is not None:
            sampler.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def stop(self):
        """Safely signals the thread loop to terminate."""
        self._is_running = False
        self.wait()
