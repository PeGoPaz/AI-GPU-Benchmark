"""Tile readings: formatting and thresholds, without a window."""
import pytest

from app.core.telemetry import THROTTLE_SW_POWER_CAP, THROTTLE_SW_THERMAL
from app.ui import readings


@pytest.mark.parametrize("temp, state", [(65, "normal"), (80, "warn"), (86, "warn"),
                                         (87, "crit"), (95, "crit")])
def test_temperature_thresholds(telemetry_sample, temp, state):
    assert readings.temperature(telemetry_sample(0, temp_gpu=temp)).state == state


@pytest.mark.parametrize("room, state", [(25, "normal"), (10, "warn"), (5, "crit")])
def test_headroom_gets_tighter_as_it_shrinks(telemetry_sample, room, state):
    assert readings.headroom(telemetry_sample(0, temp_headroom_c=room)).state == state


def test_missing_readings_say_so_rather_than_zero(telemetry_sample):
    sample = telemetry_sample(0, temp_headroom_c=None, fan_speed_pct=None, fan_rpm=None,
                              pcie_gen=None, throttle_reasons=None, temp_memory=None)

    for read in (readings.headroom, readings.fan, readings.pcie, readings.throttle,
                 readings.memory_temperature):
        reading = read(sample)
        assert reading.value == "N/A", read.__name__
        assert reading.number is None, read.__name__


def test_vram_is_shown_in_gib_with_the_share_in_use(telemetry_sample):
    reading = readings.vram(telemetry_sample(0, vram_used_mb=7782, vram_total_mb=8188))

    assert (reading.value, reading.unit) == ("7.6", "GiB")
    assert reading.caption == "of 8.0 · 95%"
    assert reading.state == "warn"


def test_power_caption_prefers_the_native_peak(telemetry_sample):
    assert readings.power(telemetry_sample(0, power_peak_w=262.4)).caption == "peak 262 W"
    assert readings.power(telemetry_sample(0, power_peak_w=None)).caption == "250 ms average"


def test_throttle_lists_every_reason(telemetry_sample):
    reading = readings.throttle(telemetry_sample(
        0, throttle_reasons=THROTTLE_SW_POWER_CAP | THROTTLE_SW_THERMAL))

    assert reading.value == "Power cap, Thermal"
    assert reading.state == "warn"


def test_numbers_feed_sparklines_and_gauges(telemetry_sample):
    sample = telemetry_sample(3)

    assert readings.temperature(sample).number == sample["temp_gpu"]
    assert readings.clock(sample).number == sample["sm_clock_mhz"]



def test_power_at_its_limit_is_a_warning(telemetry_sample):
    assert readings.power(telemetry_sample(0, power_w=110.0), limit_w=115.0).state == "warn"
    assert readings.power(telemetry_sample(0, power_w=80.0), limit_w=115.0).state == "normal"
    assert readings.power(telemetry_sample(0, power_w=110.0)).state == "normal"   # limit unknown
