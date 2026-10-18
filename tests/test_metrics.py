"""Derived numbers: tokens/sec, estimated TFLOPS and energy per token."""
import pandas as pd
import pytest

from app.core import metrics


def _df(sample, n=9, **overrides):
    """Telemetry polled every second, timestamps 100..100+n-1."""
    df = pd.DataFrame([sample(i, **overrides) for i in range(n)])
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def _summary(**overrides):
    s = {
        "total_steps": 100, "requested_steps": 100, "aborted": False,
        "elapsed_time_sec": 20.0, "steps_per_sec": 5.0,
        "tokens_per_step": 4096, "params_b": 1.5, "gradient_checkpointing": False,
        "measure_start": 102.0, "measure_end": 106.0,
    }
    s.update(overrides)
    return s


def test_lora_costs_4n_flops_per_token_and_6n_with_checkpointing():
    """Frozen weights need no weight gradients — the 2N that full
    fine-tuning's 6N carries. Checkpointing adds back a 2N recompute."""
    assert metrics.flops_per_token(1.0, False) == 4e9
    assert metrics.flops_per_token(1.0, True) == 6e9


def test_tokens_per_sec_is_independent_of_how_the_step_is_shaped(telemetry_sample):
    """5 steps/sec at 4096 tokens a step and 20 steps/sec at 1024 are the same
    rate. Steps/sec alone made them look four times apart."""
    df = _df(telemetry_sample)
    a = metrics.compute_stats(df, _summary(steps_per_sec=5.0, total_steps=100,
                                           tokens_per_step=4096))
    b = metrics.compute_stats(df, _summary(steps_per_sec=20.0, total_steps=400,
                                           tokens_per_step=1024))

    assert a["tokens_per_sec"] == b["tokens_per_sec"] == 100 * 4096 / 20.0


def test_tflops_follow_from_tokens_and_model_size(telemetry_sample):
    stats = metrics.compute_stats(_df(telemetry_sample), _summary())

    assert stats["est_tflops"] == pytest.approx(20480 * 4 * 1.5e9 / 1e12)


def test_energy_prefers_the_hardware_counter(telemetry_sample):
    df = _df(telemetry_sample, energy_mj=None)
    df["energy_mj"] = [1_000_000 + i * 300_000 for i in range(len(df))]

    energy, source = metrics.energy_joules(df)

    assert source == "counter"
    assert energy == pytest.approx(8 * 300.0)


def test_energy_falls_back_to_integrating_power(telemetry_sample):
    """A constant 250 W for 8 seconds is 2000 J."""
    df = _df(telemetry_sample, power_w=250.0, energy_mj=None)

    energy, source = metrics.energy_joules(df)

    assert source == "power samples"
    assert energy == pytest.approx(2000.0)


def test_energy_covers_only_the_timed_window(telemetry_sample):
    """Model loading and warm-up draw power too, but produce no measured
    tokens; counting them would understate tokens per joule."""
    df = _df(telemetry_sample, power_w=200.0, energy_mj=None)
    stats = metrics.compute_stats(df, _summary(measure_start=102.0, measure_end=106.0))

    assert stats["energy_j"] == pytest.approx(4 * 200.0)
    assert stats["tokens_per_joule"] == pytest.approx(100 * 4096 / 800.0)


def test_run_stopped_in_warmup_reports_no_throughput(telemetry_sample):
    stats = metrics.compute_stats(
        _df(telemetry_sample),
        _summary(total_steps=0, elapsed_time_sec=0.0, steps_per_sec=0.0,
                 measure_start=None, measure_end=None))

    assert stats["tokens_per_sec"] is None
    assert stats["energy_j"] is None


def test_optional_readings_come_back_none(telemetry_sample):
    df = _df(telemetry_sample, temp_memory=None, fan_speed_pct=None,
             pcie_gen=None, pcie_width=None)
    stats = metrics.compute_stats(df, _summary())

    assert stats["max_mem_temp_c"] is None
    assert stats["peak_fan_pct"] is None
    assert stats["pcie_gen_peak"] is None
    assert stats["avg_temp_c"] == pytest.approx(df["temp_gpu"].mean())


def test_summary_without_token_counts_still_computes_thermals(telemetry_sample):
    """Older summaries, and hand-built ones in the UI tests, lack the fields."""
    stats = metrics.compute_stats(_df(telemetry_sample),
                                  {"total_steps": 10, "elapsed_time_sec": 5.0})

    assert stats["tokens_per_sec"] is None
    assert stats["max_temp_c"] == 73


def test_share_of_time_power_and_thermal_limited(telemetry_sample):
    from app.core import telemetry

    df = _df(telemetry_sample, n=4)
    df["throttle_reasons"] = [0, telemetry.THROTTLE_SW_POWER_CAP,
                              telemetry.THROTTLE_SW_POWER_CAP | telemetry.THROTTLE_HW_THERMAL,
                              0x01]                                  # idle only
    stats = metrics.compute_stats(df, _summary(measure_start=None, measure_end=None))

    assert stats["power_limited_pct"] == 50.0
    assert stats["thermal_limited_pct"] == 25.0


def test_driver_without_throttle_reasons_reports_none(telemetry_sample):
    stats = metrics.compute_stats(_df(telemetry_sample, throttle_reasons=None), _summary())

    assert stats["power_limited_pct"] is None
