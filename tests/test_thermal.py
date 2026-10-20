"""The first-order heating fit: τ, the steady temperature, and when to give up."""
import numpy as np
import pandas as pd
import pytest

from gpu_bench.core import thermal
from gpu_bench.utils.display import thermal_verdict


def _curve(tau, start=40.0, steady=80.0, seconds=300.0, hz=4.0, quantise=True, noise=0.0,
           seed=0):
    t = np.arange(0, seconds, 1 / hz)
    temps = steady - (steady - start) * np.exp(-t / tau)
    if noise:
        temps = temps + np.random.default_rng(seed).normal(0, noise, len(t))
    if quantise:                    # NVML reports whole degrees
        temps = np.round(temps)
    return 1000.0 + t, temps


def test_recovers_a_clean_curve():
    fit = thermal.fit_heating(*_curve(tau=60, quantise=False))

    assert fit.tau_s == pytest.approx(60, rel=0.02)
    assert fit.steady_c == pytest.approx(80, abs=0.2)
    assert fit.start_c == pytest.approx(40, abs=0.2)
    assert fit.r_squared > 0.999


def test_whole_degree_readings_still_fit():
    fit = thermal.fit_heating(*_curve(tau=90, noise=0.4))

    assert fit.tau_s == pytest.approx(90, rel=0.1)
    assert fit.steady_c == pytest.approx(80, abs=1.5)


def test_a_long_run_has_reached_steady_state():
    fit = thermal.fit_heating(*_curve(tau=50, seconds=400))

    assert fit.saturated
    assert fit.progress > 0.95


def test_a_short_run_says_it_ended_early_and_where_it_was_heading():
    fit = thermal.fit_heating(*_curve(tau=120, seconds=150))

    assert not fit.saturated
    assert 0.6 < fit.progress < 0.85
    assert fit.steady_c == pytest.approx(80, abs=3)


@pytest.mark.parametrize("temps", [
    np.full(400, 65.0),                         # already hot: no curve to fit
    np.linspace(80, 50, 400),                   # cooling
    np.r_[np.full(200, 40.0), np.full(200, 70.0)],   # a step, not a curve
])
def test_refuses_what_is_not_a_heating_curve(temps):
    assert thermal.fit_heating(np.arange(len(temps)) * 0.25, temps) is None


def test_too_few_points_is_no_fit():
    t, temps = _curve(tau=30, seconds=4)

    assert thermal.fit_heating(t, temps) is None


def test_run_fit_starts_when_the_load_does():
    t, temps = _curve(tau=60, quantise=False)
    idle = pd.DataFrame({"timestamp": np.arange(990, 1000, 0.25), "temp_gpu": 75.0,
                         "gpu_util_pct": 0})       # hot from a previous run, idle
    loaded = pd.DataFrame({"timestamp": t, "temp_gpu": temps, "gpu_util_pct": 98})
    df = pd.concat([idle, loaded], ignore_index=True)

    fit = thermal.fit_run(df)

    assert fit.tau_s == pytest.approx(60, rel=0.02)


def test_run_fit_stops_at_the_end_of_the_timed_window():
    t, temps = _curve(tau=60, quantise=False)
    cooling = pd.DataFrame({"timestamp": np.arange(1300, 1330, 0.25), "temp_gpu": 45.0,
                            "gpu_util_pct": 0})
    df = pd.concat([pd.DataFrame({"timestamp": t, "temp_gpu": temps, "gpu_util_pct": 98}),
                    cooling], ignore_index=True)

    assert thermal.fit_run(df, end=1299.9).tau_s == pytest.approx(60, rel=0.02)


def test_no_utilisation_column_is_no_fit():
    assert thermal.fit_run(pd.DataFrame({"timestamp": [1.0], "temp_gpu": [50]})) is None


def test_verdict_reads_the_fit():
    base = {"thermal_tau_s": 60.0, "thermal_steady_c": 81.4}

    assert thermal_verdict({**base, "thermal_saturated": True,
                            "thermal_progress": 0.99}) == "reached, at 81 °C"
    assert "72% of the way, heading for 81 °C" in thermal_verdict(
        {**base, "thermal_saturated": False, "thermal_progress": 0.72})
    assert "run longer than 180 s" in thermal_verdict(
        {**base, "thermal_saturated": False, "thermal_progress": 0.3})


def test_thermal_rows_reach_the_summary(telemetry_sample):
    from gpu_bench.utils.display import summary_rows

    t, temps = _curve(tau=60)
    df = pd.DataFrame([telemetry_sample(i) for i in range(len(t))])
    df["timestamp"], df["temp_gpu"] = t, temps
    summary = {"total_steps": 10, "requested_steps": 10, "elapsed_time_sec": 300.0,
               "steps_per_sec": 0.03}

    rows = dict(summary_rows(df, summary))

    assert rows["Thermal Time Constant τ (s)"].startswith("6")
    assert rows["Thermal Steady State"].startswith("reached")
