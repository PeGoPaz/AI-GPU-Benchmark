"""open_sampler(): the native module when it works, one line of why when it does not."""
import sys
import types

import pytest

from app.core import power_sampler


def _fake_module(monkeypatch, *, raises=None):
    module = types.SimpleNamespace(__version__="0.1.0")

    class PowerSampler:
        def __init__(self, device_index=0, interval_us=2000):
            if raises is not None:
                raise raises
            self.device_index = device_index
            self.interval_us = interval_us
            self.description = "instant power (NVML_FI_DEV_POWER_INSTANT)"

    module.PowerSampler = PowerSampler
    monkeypatch.setitem(sys.modules, "nvml_sampler", module)
    return module


def test_missing_module_falls_back_to_polling(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvml_sampler", None)      # import fails

    sampler, detail = power_sampler.open_sampler()

    assert sampler is None
    assert detail == power_sampler.NOT_BUILT
    assert "pip install ./native" in detail


def test_unreachable_nvml_is_explained(monkeypatch):
    _fake_module(monkeypatch, raises=RuntimeError("could not load NVML: libnvidia-ml.so.1"))

    sampler, detail = power_sampler.open_sampler()

    assert sampler is None
    assert detail == "native sampler unavailable: could not load NVML: libnvidia-ml.so.1"


def test_working_module_hands_back_an_unstarted_sampler(monkeypatch):
    _fake_module(monkeypatch)

    sampler, detail = power_sampler.open_sampler(interval_us=1000)

    assert sampler.interval_us == 1000
    assert detail.startswith("instant power")


@pytest.mark.parametrize("active", [True, False])
def test_status_says_what_the_report_needs(monkeypatch, active):
    _fake_module(monkeypatch)
    sampler, detail = power_sampler.open_sampler() if active else (None, "not built")

    status = power_sampler.describe_status(sampler, detail)

    assert status["active"] is active
    assert status["detail"] == detail
    if active:
        assert status["interval_us"] == 2000
        assert status["module_version"] == "0.1.0"
