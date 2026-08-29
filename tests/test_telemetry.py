"""TelemetryWorker / NVML field-value decoding.

Context for the odd-looking assertions: NVML exposes exactly one temperature
sensor (NVML_TEMPERATURE_COUNT == 1, sensor 0 = GPU core). There is no public
junction/hotspot sensor, so the dashboard reports memory temperature and
thermal headroom via field values instead.
"""
import types
from unittest import mock

import pynvml

from app.core import telemetry

MEM = telemetry._FIELD_MEMORY_TEMP
TLIM = telemetry._FIELD_GPU_TLIMIT


def test_nvml_exposes_single_temperature_sensor():
    """Guards the premise of the fix: sensors 2/3 never existed."""
    assert pynvml.NVML_TEMPERATURE_COUNT == 1


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
    }
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(pynvml, name, fn)


def _one_poll(worker):
    """Runs the worker's loop for exactly one snapshot."""
    snapshots, errors = [], []

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
