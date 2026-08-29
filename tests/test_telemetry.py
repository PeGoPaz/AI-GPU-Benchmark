"""TelemetryWorker / NVML field-value decoding.

Context for the odd-looking assertions: NVML exposes exactly one temperature
sensor (NVML_TEMPERATURE_COUNT == 1, sensor 0 = GPU core). There is no public
junction/hotspot sensor, so the dashboard reports memory temperature and
thermal headroom via field values instead.
"""
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
