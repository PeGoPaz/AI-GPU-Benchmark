"""TelemetryWorker / NVML field-value decoding.

Context for the odd-looking assertions: nvmlDeviceGetTemperature never had the
sensors 2 and 3 the old hotspot probe asked for (index 0 is the GPU core, and
NVML 13 added index 1, NVML_TEMPERATURE_GPU_MAX). There is no public
junction/hotspot sensor, so the dashboard reports memory temperature and
thermal headroom via field values instead.
"""
import types
from unittest import mock

import pynvml

from gpu_bench.core import telemetry

MEM = telemetry._FIELD_MEMORY_TEMP
TLIM = telemetry._FIELD_GPU_TLIMIT


def test_nvml_has_no_sensor_2_or_3():
    """Guards the premise of the fix: sensors 2/3 never existed.

    This used to assert NVML_TEMPERATURE_COUNT == 1, which nvidia-ml-py 13
    broke by adding NVML_TEMPERATURE_GPU_MAX at index 1. The count was never
    the point — the indices the hotspot probe asked for were.
    """
    assert pynvml.NVML_TEMPERATURE_COUNT <= 2
    assert pynvml.NVML_TEMPERATURE_GPU == 0


def test_dead_hotspot_probe_is_gone():
    assert not hasattr(telemetry, "_read_hotspot_temperature")


def test_field_values_decoded_by_type(nvml_field_value):
    values = [
        nvml_field_value(MEM, 78, pynvml.NVML_VALUE_TYPE_UNSIGNED_INT),
        nvml_field_value(TLIM, 22, pynvml.NVML_VALUE_TYPE_SIGNED_INT),
    ]
    with mock.patch.object(pynvml, "nvmlDeviceGetFieldValues", return_value=values):
        assert telemetry.read_field_values(None, [MEM, TLIM]) == {MEM: 78, TLIM: 22}


def test_unsupported_field_is_omitted_not_raised(nvml_field_value):
    """A GPU without the memory-temp field must degrade, not crash."""
    values = [
        nvml_field_value(MEM, 0, ret=3),           # NVML_ERROR_NOT_SUPPORTED
        nvml_field_value(TLIM, 15, pynvml.NVML_VALUE_TYPE_SIGNED_INT),
    ]
    with mock.patch.object(pynvml, "nvmlDeviceGetFieldValues", return_value=values):
        assert telemetry.read_field_values(None, [MEM, TLIM]) == {TLIM: 15}


def test_whole_call_failure_degrades_to_empty():
    with mock.patch.object(pynvml, "nvmlDeviceGetFieldValues",
                           side_effect=Exception("NVML_ERROR_NOT_SUPPORTED")):
        assert telemetry.read_field_values(None, [MEM, TLIM]) == {}


def test_worker_reports_nvml_init_failure_and_exits():
    """No NVIDIA driver (the CI case) must surface an error, not hang or crash."""
    worker = telemetry.TelemetryWorker(interval_ms=10)
    errors = []
    worker.error_occurred = mock.Mock(emit=errors.append)
    worker.gpu_name_ready = mock.Mock(emit=lambda *_: None)
    worker.data_updated = mock.Mock(emit=lambda *_: None)

    with mock.patch.object(pynvml, "nvmlInit", side_effect=Exception("no driver")):
        worker.run()

    assert errors and "NVML Initialization Failed" in errors[0]


# --- fan and PCIe ------------------------------------------------------------
# Both are optional hardware: datacenter cards are passively cooled, and the
# RPM entry point is missing on older drivers.

def test_fan_speed_takes_the_hottest_of_several_fans():
    with mock.patch.object(pynvml, "nvmlDeviceGetFanSpeed_v2",
                           side_effect=[55, 71, 60]):
        assert telemetry.read_fan_speed_pct(None, 3) == 71


def test_fan_speed_falls_back_to_the_unindexed_getter():
    """Cards that cannot report a fan count still answer the old call."""
    with mock.patch.object(pynvml, "nvmlDeviceGetFanSpeed", return_value=48):
        assert telemetry.read_fan_speed_pct(None, None) == 48


def test_passive_card_reports_no_fan():
    with mock.patch.object(pynvml, "nvmlDeviceGetFanSpeed_v2",
                           side_effect=Exception("NVML_ERROR_NOT_SUPPORTED")):
        assert telemetry.read_fan_speed_pct(None, 2) is None


def test_pcie_link_read_as_generation_and_width():
    with mock.patch.object(pynvml, "nvmlDeviceGetCurrPcieLinkGeneration",
                           return_value=4), \
         mock.patch.object(pynvml, "nvmlDeviceGetCurrPcieLinkWidth",
                           return_value=16):
        assert telemetry.read_pcie_link(None) == (4, 16)


def test_unsupported_pcie_query_degrades_to_none():
    with mock.patch.object(pynvml, "nvmlDeviceGetCurrPcieLinkGeneration",
                           side_effect=Exception("NVML_ERROR_NOT_SUPPORTED")), \
         mock.patch.object(pynvml, "nvmlDeviceGetCurrPcieLinkWidth",
                           return_value=16):
        assert telemetry.read_pcie_link(None) == (None, 16)


# --- the poll loop end to end ------------------------------------------------

def _patch_nvml(monkeypatch, **overrides):
    """Stands up the whole NVML surface the worker touches, on a fake 4090."""
    defaults = {
        "nvmlInit": lambda: None,
        "nvmlShutdown": lambda: None,
        "nvmlDeviceGetHandleByIndex": lambda i: "handle",
        "nvmlDeviceGetName": lambda h: "NVIDIA GeForce RTX 4090",
        "nvmlDeviceGetNumFans": lambda h: 3,
        "nvmlDeviceGetMaxPcieLinkGeneration": lambda h: 4,
        "nvmlDeviceGetMaxPcieLinkWidth": lambda h: 16,
        "nvmlDeviceGetTemperature": lambda h, s: 72,
        "nvmlDeviceGetFieldValues": lambda h, f: [],
        "nvmlDeviceGetMemoryInfo": lambda h: types.SimpleNamespace(
            used=6 * 1024 ** 3, total=24 * 1024 ** 3),
        "nvmlDeviceGetPowerUsage": lambda h: 320_000,
        "nvmlDeviceGetUtilizationRates": lambda h: types.SimpleNamespace(
            gpu=99, memory=60),
        "nvmlDeviceGetClockInfo": lambda h, c: 2610,
        "nvmlDeviceGetFanSpeed_v2": lambda h, fan: 55 + fan,
        "nvmlDeviceGetFanSpeed": lambda h: 55,
        "nvmlDeviceGetFanSpeedRPM": lambda h: 1900,
        "nvmlDeviceGetCurrPcieLinkGeneration": lambda h: 4,
        "nvmlDeviceGetCurrPcieLinkWidth": lambda h: 16,
        "nvmlDeviceGetTotalEnergyConsumption": lambda h: 123_456_789,
        "nvmlSystemGetDriverVersion": lambda: "580.82",
        "nvmlSystemGetCudaDriverVersion": lambda: 13000,
        "nvmlDeviceGetEnforcedPowerLimit": lambda h: 450_000,
        "nvmlDeviceGetCudaComputeCapability": lambda h: (8, 9),
        "nvmlDeviceGetMaxClockInfo": lambda h, c: 2790,
    }
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(pynvml, name, fn)
    # Deterministic whether or not the native module is installed here.
    monkeypatch.setattr(telemetry, "open_sampler", lambda: (None, "not built (test)"))


def _one_poll(worker, device=None, traces=None, status=None):
    """Runs the worker's loop for exactly one snapshot."""
    snapshots, errors = [], []
    worker.device_info_ready = mock.Mock(
        emit=(device.append if device is not None else lambda *_: None))
    worker.power_trace = mock.Mock(
        emit=(traces.append if traces is not None else lambda *_: None))
    worker.sampler_status = mock.Mock(
        emit=(status.append if status is not None else lambda *_: None))

    def capture(sample):
        snapshots.append(sample)
        worker._is_running = False          # one iteration is enough

    worker.data_updated = mock.Mock(emit=capture)
    worker.gpu_name_ready = mock.Mock(emit=lambda *_: None)
    worker.error_occurred = mock.Mock(emit=errors.append)
    worker.run()
    return snapshots, errors


def test_snapshot_carries_fan_and_pcie(monkeypatch):
    _patch_nvml(monkeypatch)
    snapshots, errors = _one_poll(telemetry.TelemetryWorker(interval_ms=1))

    assert not errors
    snap = snapshots[0]
    assert snap["fan_speed_pct"] == 57                       # max of fans 0..2
    assert snap["fan_rpm"] == 1900
    assert (snap["pcie_gen"], snap["pcie_width"]) == (4, 16)
    assert (snap["pcie_gen_max"], snap["pcie_width_max"]) == (4, 16)


def test_passive_card_still_reports_core_telemetry(monkeypatch):
    """Missing fans must cost the fan reading, not the whole snapshot."""
    def unsupported(*_a, **_k):
        raise Exception("NVML_ERROR_NOT_SUPPORTED")

    _patch_nvml(monkeypatch,
                nvmlDeviceGetNumFans=unsupported,
                nvmlDeviceGetFanSpeed=unsupported,
                nvmlDeviceGetFanSpeed_v2=unsupported,
                nvmlDeviceGetFanSpeedRPM=unsupported)
    snapshots, errors = _one_poll(telemetry.TelemetryWorker(interval_ms=1))

    assert not errors
    snap = snapshots[0]
    assert snap["fan_speed_pct"] is None
    assert snap["fan_rpm"] is None
    assert snap["temp_gpu"] == 72
    assert snap["power_w"] == 320.0


def test_snapshot_carries_the_energy_counter(monkeypatch):
    _patch_nvml(monkeypatch)
    snapshots, _ = _one_poll(telemetry.TelemetryWorker(interval_ms=1))

    assert snapshots[0]["energy_mj"] == 123_456_789


def test_pre_volta_card_has_no_energy_counter(monkeypatch):
    def unsupported(*_a, **_k):
        raise Exception("NVML_ERROR_NOT_SUPPORTED")

    _patch_nvml(monkeypatch, nvmlDeviceGetTotalEnergyConsumption=unsupported)
    snapshots, errors = _one_poll(telemetry.TelemetryWorker(interval_ms=1))

    assert not errors
    assert snapshots[0]["energy_mj"] is None


# --- throttle reasons ---------------------------------------------------------

def test_throttle_bits_match_nvml():
    """Written out locally because NVML is renaming the constants; the bits
    must still be the driver's."""
    assert telemetry.THROTTLE_SW_POWER_CAP == pynvml.nvmlClocksThrottleReasonSwPowerCap
    assert telemetry.THROTTLE_HW_SLOWDOWN == pynvml.nvmlClocksThrottleReasonHwSlowdown
    assert telemetry.THROTTLE_SW_THERMAL == pynvml.nvmlClocksThrottleReasonSwThermalSlowdown
    assert telemetry.THROTTLE_HW_THERMAL == pynvml.nvmlClocksThrottleReasonHwThermalSlowdown
    assert telemetry.THROTTLE_HW_POWER_BRAKE == \
        pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown


def test_decode_names_each_limiting_reason():
    mask = telemetry.THROTTLE_SW_POWER_CAP | telemetry.THROTTLE_SW_THERMAL

    assert telemetry.decode_throttle_reasons(mask) == ["Power cap", "Thermal"]


def test_idle_and_settings_bits_are_not_throttling():
    """GpuIdle (0x1) and SyncBoost (0x10) are the card resting, not held back."""
    assert telemetry.decode_throttle_reasons(0x01 | 0x10) == []
    assert telemetry.decode_throttle_reasons(None) == []


def test_throttle_read_falls_back_to_the_old_getter(monkeypatch):
    """Drivers older than the rename only know the ThrottleReasons name."""
    def missing(_h):
        raise Exception("NVML_ERROR_FUNCTION_NOT_FOUND")

    monkeypatch.setattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", missing)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons",
                        lambda h: telemetry.THROTTLE_SW_POWER_CAP)

    assert telemetry.read_throttle_reasons("handle") == telemetry.THROTTLE_SW_POWER_CAP


def test_snapshot_carries_the_throttle_mask(monkeypatch):
    _patch_nvml(monkeypatch,
                nvmlDeviceGetCurrentClocksEventReasons=lambda h: 0x04)
    snapshots, _ = _one_poll(telemetry.TelemetryWorker(interval_ms=1))

    assert snapshots[0]["throttle_reasons"] == 0x04



# --- static device info --------------------------------------------------------

def test_cuda_version_is_unpacked():
    assert telemetry.format_cuda_version(12040) == "12.4"
    assert telemetry.format_cuda_version(13000) == "13.0"
    assert telemetry.format_cuda_version(None) is None


def test_device_info_is_sent_once_at_startup(monkeypatch):
    _patch_nvml(monkeypatch)
    device = []
    _one_poll(telemetry.TelemetryWorker(interval_ms=1), device=device)

    assert device == [{
        "name": "NVIDIA GeForce RTX 4090",
        "driver_version": "580.82",
        "cuda_driver_version": "13.0",
        "vram_total_mb": 24576.0,
        "power_limit_w": 450.0,
        "pcie_gen_max": 4,
        "pcie_width_max": 16,
        "compute_capability": [8, 9],
        "max_sm_clock_mhz": 2790,
    }]


def test_device_info_tolerates_missing_queries(monkeypatch):
    def unsupported(*_a, **_k):
        raise Exception("NVML_ERROR_NOT_SUPPORTED")

    _patch_nvml(monkeypatch, nvmlSystemGetDriverVersion=unsupported,
                nvmlDeviceGetEnforcedPowerLimit=unsupported,
                nvmlDeviceGetCudaComputeCapability=unsupported)
    device = []
    _one_poll(telemetry.TelemetryWorker(interval_ms=1), device=device)

    assert device[0]["driver_version"] is None
    assert device[0]["power_limit_w"] is None
    assert device[0]["compute_capability"] is None



# --- native power sampler ----------------------------------------------------------

class FakeSampler:
    """Stands in for nvml_sampler.PowerSampler: one canned batch per drain."""

    description = "instant power (test)"
    interval_us = 2000

    def __init__(self, watts=(180.0, 260.0, 210.0)):
        import numpy as np
        self.batch = (np.array([100.000, 100.002, 100.004]),
                      np.array(watts, dtype=np.float32))
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def drain(self):
        return self.batch


def test_native_peak_and_trace_ride_along_with_the_poll(monkeypatch):
    _patch_nvml(monkeypatch)
    sampler = FakeSampler()
    monkeypatch.setattr(telemetry, "open_sampler", lambda: (sampler, sampler.description))
    traces, status = [], []

    snapshots, errors = _one_poll(telemetry.TelemetryWorker(interval_ms=1),
                                  traces=traces, status=status)

    assert not errors
    assert snapshots[0]["power_peak_w"] == 260.0
    assert len(traces) == 1 and list(traces[0][1]) == [180.0, 260.0, 210.0]
    assert status[0]["active"] is True
    assert status[0]["detail"] == "instant power (test)"
    assert status[0]["interval_us"] == 2000
    assert sampler.started and sampler.stopped


def test_without_the_module_telemetry_is_unchanged(monkeypatch):
    _patch_nvml(monkeypatch)
    traces, status = [], []

    snapshots, errors = _one_poll(telemetry.TelemetryWorker(interval_ms=1),
                                  traces=traces, status=status)

    assert not errors
    assert snapshots[0]["power_peak_w"] is None
    assert snapshots[0]["power_w"] == 320.0
    assert traces == []
    assert status[0] == {"active": False, "detail": "not built (test)"}


def test_sampler_that_fails_to_start_degrades_to_polling(monkeypatch):
    _patch_nvml(monkeypatch)
    sampler = FakeSampler()
    sampler.start = mock.Mock(side_effect=RuntimeError("thread creation failed"))
    monkeypatch.setattr(telemetry, "open_sampler", lambda: (sampler, sampler.description))
    status = []

    snapshots, errors = _one_poll(telemetry.TelemetryWorker(interval_ms=1), status=status)

    assert not errors
    assert snapshots[0]["power_peak_w"] is None
    assert status[0]["active"] is False
    assert "failed to start" in status[0]["detail"]


def test_one_off_temperature_reading_initialises_and_shuts_down(monkeypatch):
    calls = []
    _patch_nvml(monkeypatch, nvmlInit=lambda: calls.append("init"),
                nvmlShutdown=lambda: calls.append("shutdown"))

    assert telemetry.read_gpu_temperature() == 72.0
    assert calls == ["init", "shutdown"]
