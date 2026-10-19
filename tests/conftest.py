"""Shared fixtures and stubs so the suite runs without a GPU or the ML stack.

The app imports torch/transformers/peft/datasets at module level and talks to
NVML on a real NVIDIA card. CI runners have neither, so we install lightweight
stand-ins before any `app.*` module is imported.
"""
import sys
import types
from unittest import mock

import pytest

# --- Stub the heavy ML stack -------------------------------------------------
# Must happen at import time, before app.core.trainer is pulled in.
for _name in ("torch", "transformers", "peft", "datasets"):
    sys.modules.setdefault(_name, mock.MagicMock())

# TrainerCallback is used as a base class, so it has to be a real class.
sys.modules["transformers"].TrainerCallback = object


@pytest.fixture
def telemetry_sample():
    """Factory for a single NVML telemetry snapshot."""
    def _make(i=0, **overrides):
        s = {
            "timestamp": 100.0 + i,
            "temp_gpu": 65 + i,
            "temp_memory": 80 + i,
            "temp_headroom_c": 25 - i,
            "vram_used_mb": 5000 + i * 10,
            "vram_total_mb": 24564,
            "power_w": 250 + i,
            "gpu_util_pct": 98,
            "sm_clock_mhz": 2600,
            "fan_speed_pct": 60 + i,
            "fan_rpm": 1800 + i * 10,
            "pcie_gen": 4,
            "pcie_width": 16,
            "pcie_gen_max": 4,
            "pcie_width_max": 16,
            "energy_mj": 1_000_000 + i * 250_000,
            "throttle_reasons": 0,
        }
        s.update(overrides)
        return s
    return _make


@pytest.fixture
def nvml_field_value():
    """Builds a fake nvmlFieldValue_t as returned by nvmlDeviceGetFieldValues."""
    import pynvml

    def _make(field_id, value, value_type=None, ret=None):
        value_type = pynvml.NVML_VALUE_TYPE_UNSIGNED_INT if value_type is None else value_type
        ret = pynvml.NVML_SUCCESS if ret is None else ret
        union = types.SimpleNamespace(dVal=0, uiVal=0, ulVal=0, ullVal=0,
                                      sllVal=0, siVal=0, usVal=0)
        attr = {
            pynvml.NVML_VALUE_TYPE_DOUBLE: "dVal",
            pynvml.NVML_VALUE_TYPE_UNSIGNED_INT: "uiVal",
            pynvml.NVML_VALUE_TYPE_SIGNED_INT: "siVal",
        }[value_type]
        setattr(union, attr, value)
        return types.SimpleNamespace(fieldId=field_id, nvmlReturn=ret,
                                     valueType=value_type, value=union)
    return _make


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for every test that builds widgets.

    Imported here rather than at the top, so modules that need no Qt widgets
    never load the platform plugin. Skips when Qt cannot start headless.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")
    app = widgets.QApplication.instance()
    if app is None:
        try:
            app = widgets.QApplication([])
        except Exception as exc:                      # noqa: BLE001
            pytest.skip(f"Qt cannot start headless here: {exc}")
    return app
