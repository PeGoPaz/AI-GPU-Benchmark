"""Derived numbers: tokens/sec, estimated TFLOPS and energy per token."""
import numpy as np
import pandas as pd
import pytest

from gpu_bench.core import metrics


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
    from gpu_bench.core import telemetry

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



# --- native power trace ------------------------------------------------------------

def _trace(start=100.0, n=2001, step=0.002, base=200.0, spike_every=100, spike=320.0):
    """A 500 Hz trace sitting at `base` watts with a one-sample spike every 200 ms."""
    import numpy as np
    t = start + np.arange(n) * step
    w = np.full(n, base)
    w[::spike_every] = spike
    return pd.DataFrame({"timestamp": t, "power_w": w, "time_sec": t - start})


def test_transients_show_what_the_polls_average_away():
    stats = metrics.transient_stats(_trace())

    assert stats["native_peak_power_w"] == 320.0
    assert stats["native_p99_power_w"] == pytest.approx(320.0)     # 1 % of samples are spikes
    assert stats["transient_ratio"] == pytest.approx(320.0 / (200.0 + 120.0 / 100), rel=1e-3)


def test_update_rate_counts_changes_not_reads():
    """Up, then back down, at every spike: 2 changes per 200 ms is 10 Hz,
    however many times the sampler asked in between."""
    stats = metrics.transient_stats(_trace())

    assert stats["native_sample_hz"] == pytest.approx(500.0)
    assert stats["native_update_hz"] == pytest.approx(10.0, rel=0.01)


def test_trace_energy_matches_the_area_under_the_curve():
    flat = _trace(spike=200.0)                   # 200 W for 4 s

    assert metrics.transient_stats(flat)["energy_trace_j"] == pytest.approx(800.0)


def test_trace_figures_cover_the_timed_window(telemetry_sample):
    stats = metrics.compute_stats(_df(telemetry_sample), _summary(measure_start=102.0,
                                                                  measure_end=104.0),
                                  trace=_trace(start=100.0, n=2501))

    assert stats["energy_trace_j"] == pytest.approx(2.0 * (200.0 + 1.2), rel=0.01)


def test_no_trace_means_no_transient_figures(telemetry_sample):
    stats = metrics.compute_stats(_df(telemetry_sample), _summary())

    assert stats["native_peak_power_w"] is None
    assert stats["native_update_hz"] is None


def test_step_durations_start_at_the_timed_window():
    durations = metrics.step_durations({"measure_start": 10.0,
                                        "step_times": [11.0, 12.5, 13.0]})

    assert list(durations) == [1.0, 1.5, 0.5]


def test_step_durations_are_empty_without_stamps():
    assert len(metrics.step_durations({"measure_start": 10.0})) == 0
    assert len(metrics.step_durations({"measure_start": None, "step_times": [1.0]})) == 0


def test_step_time_stats_catch_the_slow_tail():
    durations = np.array([1.0] * 99 + [3.0])

    stats = metrics.step_time_stats(durations)

    assert stats["step_time_p50_s"] == 1.0
    assert stats["step_time_p99_s"] > 1.0
    assert 0.1 < stats["step_time_cv"] < 0.3


def test_step_time_stats_need_two_steps():
    assert metrics.step_time_stats(np.array([1.0])) == dict.fromkeys(
        ("step_time_p50_s", "step_time_p99_s", "step_time_cv"))


def _polls(t0, t1, every, watts):
    """250 ms-style polls with a matching energy counter."""
    t = np.arange(t0, t1 + 1e-9, every)
    return pd.DataFrame({"timestamp": t, "power_w": watts,
                         "energy_mj": (t - t0) * watts * 1000})


def _trace_500hz(t0, t1, power_at):
    t = np.arange(t0, t1 + 1e-9, 0.002)
    return pd.DataFrame({"timestamp": t, "power_w": [power_at(x) for x in t]})


def test_step_energy_comes_from_the_trace_when_there_is_one():
    summary = {"measure_start": 100.0, "step_times": [101.0, 102.0, 103.0, 104.0]}
    # 200 W through the second step, 100 W otherwise.
    trace = _trace_500hz(99.0, 105.0, lambda x: 200.0 if 101.0 <= x < 102.0 else 100.0)

    energies, source = metrics.step_energy(summary, _polls(99, 105, 0.25, 100.0), trace)

    assert source == "instant trace"
    assert energies == pytest.approx([100.0, 200.0, 100.0, 100.0], abs=0.5)


def test_step_energy_falls_back_to_the_counter():
    summary = {"measure_start": 100.0, "step_times": [102.0, 104.0, 106.0]}

    energies, source = metrics.step_energy(summary, _polls(99, 107, 0.25, 150.0))

    assert source == "counter"
    assert energies == pytest.approx([300.0, 300.0, 300.0])


def test_steps_too_short_for_the_polls_get_no_energy():
    summary = {"measure_start": 100.0, "step_times": [100.3, 100.6, 100.9]}

    energies, source = metrics.step_energy(summary, _polls(99, 102, 0.25, 150.0))

    assert len(energies) == 0 and source is None


def test_a_trace_that_misses_part_of_the_window_is_not_used():
    summary = {"measure_start": 100.0, "step_times": [102.0, 104.0, 106.0]}
    late_trace = _trace_500hz(103.0, 107.0, lambda x: 999.0)

    _, source = metrics.step_energy(summary, _polls(99, 107, 0.25, 150.0), late_trace)

    assert source == "counter"


def test_per_step_energy_reaches_the_stats(telemetry_sample):
    df = _df(telemetry_sample)             # one poll a second, 100..108
    summary = {"total_steps": 2, "elapsed_time_sec": 8.0, "measure_start": 100.0,
               "measure_end": 108.0, "step_times": [104.0, 108.0]}

    stats = metrics.compute_stats(df, summary)

    assert stats["step_energy_source"] == "counter"
    assert stats["step_energy_p50_j"] > 0


def _polled(masks, start=100.0, **columns):
    return pd.DataFrame({"timestamp": start + np.arange(len(masks), dtype=float),
                         "throttle_reasons": pd.array(masks, dtype="Int64"), **columns})


def test_throttling_onset_is_the_first_sustained_streak():
    from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP as P

    window = _polled([0, 0, P, 0, 0, P, P, P, P, 0])

    assert metrics.throttle_onset(window, 100.0, P) == 5.0


def test_a_lone_flagged_poll_is_not_an_onset():
    from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP as P

    assert metrics.throttle_onset(_polled([0, P, 0, P, 0, 0]), 100.0, P) is None


def test_no_onset_when_the_driver_reports_no_reasons():
    assert metrics.throttle_onset(_polled([None] * 6), 100.0, 0xFF) is None
    assert metrics.throttle_onset(pd.DataFrame({"timestamp": [1.0]}), 0.0, 0xFF) is None


def test_clock_and_throughput_drift_compare_the_quarters():
    clock = [2600.0] * 10 + [2500.0] * 10 + [2400.0] * 10
    window = _polled([0] * 30, sm_clock_mhz=clock)
    durations = np.array([1.0] * 10 + [1.1] * 10)

    stats = metrics.drift_stats(window, durations)

    assert stats["clock_drift"] == pytest.approx(2400 / 2600 - 1)
    assert stats["throughput_drift"] == pytest.approx(1 / 1.1 - 1)


def test_drift_needs_enough_of_a_window():
    stats = metrics.drift_stats(_polled([0] * 4, sm_clock_mhz=[2600.0] * 4), np.ones(4))

    assert stats == {"clock_drift": None, "throughput_drift": None}


def test_onset_and_drift_reach_the_summary(telemetry_sample):
    from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP as P
    from gpu_bench.utils.display import summary_rows

    df = _df(telemetry_sample, n=12)
    df["throttle_reasons"] = [0, 0, 0, 0, P, P, P, P, P, P, P, P]
    summary = {"total_steps": 8, "requested_steps": 8, "elapsed_time_sec": 10.0,
               "steps_per_sec": 0.8, "measure_start": 101.0, "measure_end": 111.0,
               "step_times": [102.0, 103.0, 104.0, 105.0, 106.2, 107.4, 108.6, 109.8]}

    rows = dict(summary_rows(df, summary))

    assert rows["Throttling Began (into timing)"] == "power limit at 3 s"
    assert rows["Throughput Drift (first → last quarter)"] == "-16.7 %"
