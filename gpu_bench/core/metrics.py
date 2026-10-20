"""Derived benchmark numbers: throughput, compute rate, energy.

Kept apart from the window so the same figures can be shown on screen, written
into a report or printed by a headless run, and tested without Qt. Everything
here returns raw numbers; formatting is the caller's business.

Steps/sec on its own is not comparable between runs: a step at batch 1 and
512 tokens is an eighth of the work of one at batch 2 and 2048. Tokens/sec
normalises that, and the energy figures say what the throughput cost.
"""
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from gpu_bench.core import thermal
from gpu_bench.core.peak import peak_tflops
from gpu_bench.core.telemetry import POWER_LIMIT_BITS, THERMAL_LIMIT_BITS


def flops_per_token(params_b: float, gradient_checkpointing: bool) -> float:
    """Approximate training FLOPs per token for LoRA on a frozen base model.

    Forward is 2N and the backward pass through the activations another 2N.
    Full fine-tuning adds 2N more for weight gradients — the familiar 6N —
    but a frozen base needs none, and the adapters' share is negligible at
    rank 8. Checkpointing recomputes the forward pass: +2N.

    Attention-score FLOPs are left out, as is usual for this estimate. They
    are a few percent at 512 tokens and grow with sequence length, so
    long-sequence runs read somewhat low.
    """
    return params_b * 1e9 * (6 if gradient_checkpointing else 4)


def timed_window(df: pd.DataFrame, start: float | None,
                 end: float | None) -> pd.DataFrame:
    """The telemetry rows inside the measured window, warm-up and loading excluded."""
    if start is None or end is None:
        return df.iloc[0:0]
    return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]


def energy_joules(df: pd.DataFrame) -> tuple[float | None, str | None]:
    """Energy drawn across the rows of df, and where the number came from.

    NVML's cumulative energy counter (Volta and newer) is preferred: it is
    integrated by the card itself, so it includes the millisecond power spikes
    a 250 ms poll steps right over. Without it, the power samples are
    integrated with the trapezoid rule instead.
    """
    if 'energy_mj' in df.columns:
        counter = df['energy_mj'].dropna()
        if len(counter) >= 2:
            return (counter.iloc[-1] - counter.iloc[0]) / 1000.0, "counter"

    if len(df) >= 2:
        return trapezoid_joules(df['timestamp'], df['power_w']), "power samples"

    return None, None


def limited_share(df: pd.DataFrame, bits: int) -> float | None:
    """Percentage of polls in which any of the given throttle bits was set."""
    if 'throttle_reasons' not in df.columns:
        return None
    masks = df['throttle_reasons'].dropna()
    if masks.empty:
        return None
    return float((masks.astype('int64') & bits).astype(bool).mean() * 100)


def trapezoid_joules(timestamps: Sequence[float] | np.ndarray | pd.Series,
                     watts: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Energy under a power curve sampled at arbitrary times."""
    t = np.asarray(timestamps, dtype=float)
    p = np.asarray(watts, dtype=float)
    if len(t) < 2:
        return 0.0
    return float(((p[1:] + p[:-1]) / 2 * (t[1:] - t[:-1])).sum())


def transient_stats(trace: pd.DataFrame | None) -> dict[str, Any]:
    """What the native sampler's trace shows that the 250 ms polls cannot.

    The peak and p99 of instantaneous power, how far the peak sits above the
    mean, the rate samples arrived at, and the rate the reading actually
    changed at — NVML refreshes less often than it can be asked, so the
    second number, not the first, is the trace's real resolution. Energy
    from the trace is a cross-check on the card's own counter.
    """
    empty = dict.fromkeys(('native_peak_power_w', 'native_p99_power_w', 'transient_ratio',
                           'native_sample_hz', 'native_update_hz', 'energy_trace_j'))
    if trace is None or len(trace) < 2:
        return empty

    watts = trace['power_w'].to_numpy(dtype=float)
    seconds = trace['timestamp'].to_numpy(dtype=float)
    duration = seconds[-1] - seconds[0]
    mean = watts.mean()
    return {
        'native_peak_power_w': float(watts.max()),
        'native_p99_power_w': float(np.percentile(watts, 99)),
        'transient_ratio': float(watts.max() / mean) if mean > 0 else None,
        'native_sample_hz': (len(watts) - 1) / duration if duration > 0 else None,
        'native_update_hz': (float(np.count_nonzero(np.diff(watts))) / duration
                             if duration > 0 else None),
        'energy_trace_j': trapezoid_joules(seconds, watts),
    }


def step_durations(summary: dict[str, Any]) -> np.ndarray:
    """Seconds each timed step took, from the trainer's step-end stamps.

    The first step is measured from the opening of the timed window, so the
    durations add up to the window exactly. Empty for summaries from before
    the stamps existed, and for a run stopped during warm-up.
    """
    ends = summary.get('step_times') or []
    start = summary.get('measure_start')
    if start is None or not ends:
        return np.empty(0)
    return np.diff(np.concatenate([[start], ends]))


def step_time_stats(durations: np.ndarray) -> dict[str, float | None]:
    """Median and p99 step time, and the jitter as a coefficient of variation.

    The average hides what the distribution shows: a run whose every
    twentieth step takes twice as long — a data-loader stall, a clock dip —
    averages out to a plausible number. The p99 and the CV do not.
    """
    if len(durations) < 2:
        return dict.fromkeys(('step_time_p50_s', 'step_time_p99_s', 'step_time_cv'))
    mean = float(durations.mean())
    return {
        'step_time_p50_s': float(np.median(durations)),
        'step_time_p99_s': float(np.percentile(durations, 99)),
        'step_time_cv': float(durations.std(ddof=1)) / mean if mean > 0 else None,
    }


def _cumulative_joules(seconds: np.ndarray, watts: np.ndarray) -> np.ndarray:
    """Energy drawn from the first sample up to each sample (trapezoid rule)."""
    return np.concatenate([[0.0], np.cumsum((watts[1:] + watts[:-1]) / 2 * np.diff(seconds))])


# A source is used for per-step energy only if it samples at least this many
# times per median step; coarser, and the split between neighbouring steps
# would be interpolation rather than measurement.
_SAMPLES_PER_STEP = 4


def step_energy(summary: dict[str, Any], df: pd.DataFrame,
                trace: pd.DataFrame | None = None) -> tuple[np.ndarray, str | None]:
    """Joules each timed step drew, and which reading they were taken from.

    The energy curve is read at every step boundary and differenced. The
    native trace is preferred — at 500 Hz it resolves even sub-second steps
    — then the card's energy counter and finally the 250 ms power polls,
    each only if it is fine-grained enough for the steps at hand and spans
    the whole timed window. Empty when none is.
    """
    durations = step_durations(summary)
    if len(durations) == 0:
        return np.empty(0), None
    edges = np.concatenate([[summary['measure_start']], summary['step_times']])
    resolution = float(np.median(durations)) / _SAMPLES_PER_STEP

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    if trace is not None and len(trace) >= 2:
        t = trace['timestamp'].to_numpy(dtype=float)
        candidates.append(("instant trace", t,
                           _cumulative_joules(t, trace['power_w'].to_numpy(dtype=float))))
    if 'energy_mj' in df.columns:
        counter = df[['timestamp', 'energy_mj']].dropna()
        if len(counter) >= 2:
            candidates.append(("counter", counter['timestamp'].to_numpy(dtype=float),
                               counter['energy_mj'].to_numpy(dtype=float) / 1000.0))
    if len(df) >= 2:
        t = df['timestamp'].to_numpy(dtype=float)
        candidates.append(("power samples", t,
                           _cumulative_joules(t, df['power_w'].to_numpy(dtype=float))))

    for source, seconds, joules in candidates:
        covers = seconds[0] <= edges[0] and seconds[-1] >= edges[-1]
        if covers and float(np.median(np.diff(seconds))) <= resolution:
            return np.diff(np.interp(edges, seconds, joules)), source
    return np.empty(0), None


def step_energy_stats(energies: np.ndarray, source: str | None) -> dict[str, Any]:
    """Median and p95 energy per step, its spread, and where it came from."""
    if len(energies) < 2:
        return dict.fromkeys(('step_energy_p50_j', 'step_energy_p95_j',
                              'step_energy_cv', 'step_energy_source'))
    mean = float(energies.mean())
    return {
        'step_energy_p50_j': float(np.median(energies)),
        'step_energy_p95_j': float(np.percentile(energies, 95)),
        'step_energy_cv': float(energies.std(ddof=1)) / mean if mean > 0 else None,
        'step_energy_source': source,
    }


# Throttling has begun once its bits hold for this many polls in a row: a
# single flagged poll is the driver trimming a transient, not a limit reached.
ONSET_STREAK = 3


def throttle_onset(window: pd.DataFrame, start: float, bits: int,
                   streak: int = ONSET_STREAK) -> float | None:
    """Seconds into the timed window at which throttling for these bits set in.

    None when it never held for `streak` consecutive polls, and when the
    driver reports no throttle reasons at all.
    """
    if 'throttle_reasons' not in window.columns or len(window) < streak:
        return None
    masks = window['throttle_reasons']
    if not masks.notna().any():
        return None
    flags = (masks.fillna(0).astype('int64') & bits).astype(bool).to_numpy()
    held = np.convolve(flags, np.ones(streak, dtype=int), mode='valid') == streak
    if not held.any():
        return None
    return float(window['timestamp'].iloc[int(np.argmax(held))] - start)


def quarter_drift(first: np.ndarray, last: np.ndarray) -> float | None:
    """Relative change of the median from the first quarter to the last."""
    if len(first) < 2 or len(last) < 2:
        return None
    before = float(np.median(first))
    return float(np.median(last)) / before - 1 if before > 0 else None


def drift_stats(window: pd.DataFrame, durations: np.ndarray) -> dict[str, float | None]:
    """How much the SM clock and the step rate sagged over the timed window.

    The first quarter against the last, by median: a card that holds its
    boost clock for a minute and then settles lower shows it here even when
    the averages look fine. Throughput comes from the step durations — rate
    is their inverse — so the two figures can be read side by side.
    """
    stats: dict[str, float | None] = {'clock_drift': None, 'throughput_drift': None}
    if 'sm_clock_mhz' in window.columns and len(window) >= 8:
        t = window['timestamp'].to_numpy(dtype=float)
        clock = window['sm_clock_mhz'].to_numpy(dtype=float)
        q1, q3 = t[0] + (t[-1] - t[0]) / 4, t[-1] - (t[-1] - t[0]) / 4
        stats['clock_drift'] = quarter_drift(clock[t <= q1], clock[t >= q3])
    if len(durations) >= 8:
        quarter = len(durations) // 4
        slowdown = quarter_drift(durations[:quarter], durations[-quarter:])
        if slowdown is not None:
            stats['throughput_drift'] = 1 / (1 + slowdown) - 1
    return stats


def thermal_stats(df: pd.DataFrame, end: float | None) -> dict[str, Any]:
    """The fitted heating curve's numbers; all None when there is no curve."""
    fit = thermal.fit_run(df, end)
    if fit is None:
        return dict.fromkeys(('thermal_tau_s', 'thermal_steady_c', 'thermal_r2',
                              'thermal_progress', 'thermal_saturated'))
    return {
        'thermal_tau_s': fit.tau_s,
        'thermal_steady_c': fit.steady_c,
        'thermal_r2': fit.r_squared,
        'thermal_progress': fit.progress,
        'thermal_saturated': fit.saturated,
    }


def _column_stat(df: pd.DataFrame, column: str, how: str) -> float | None:
    """max/min/mean of an optional column, None when it was never reported."""
    if column not in df.columns or not df[column].notna().any():
        return None
    return float(getattr(df[column], how)())


def compute_stats(df: pd.DataFrame, summary: dict[str, Any],
                  trace: pd.DataFrame | None = None,
                  device: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the summary table shows, as numbers.

    Optional readings (memory temperature, fan, PCIe) come back as None on
    hardware that does not report them, and so do the throughput figures when
    the summary predates them or the timed window never opened. The
    transient figures need the native sampler's trace and are None without it,
    and MFU needs the device info telemetry reads at start-up.
    """
    stats: dict[str, Any] = {
        'avg_temp_c': float(df['temp_gpu'].mean()),
        'max_temp_c': float(df['temp_gpu'].max()),
        'avg_power_w': float(df['power_w'].mean()),
        'peak_power_w': float(df['power_w'].max()),
        'avg_vram_mb': float(df['vram_used_mb'].mean()),
        'peak_vram_mb': float(df['vram_used_mb'].max()),
        'max_mem_temp_c': _column_stat(df, 'temp_memory', 'max'),
        'min_headroom_c': _column_stat(df, 'temp_headroom_c', 'min'),
        'peak_fan_pct': _column_stat(df, 'fan_speed_pct', 'max'),
        # The link under load is the interesting one; at idle it downshifts.
        'pcie_gen_peak': _column_stat(df, 'pcie_gen', 'max'),
        'pcie_width_peak': _column_stat(df, 'pcie_width', 'max'),
        'tokens_per_sec': None,
        'est_tflops': None,
        'energy_j': None,
        'energy_source': None,
        'tokens_per_joule': None,
        'power_limited_pct': None,
        'thermal_limited_pct': None,
        'peak_tflops': None,
        'peak_basis': None,
        'mfu': None,
        'hfu': None,
    }

    # Over the measured steps when there were any, else over the whole run:
    # loading and warm-up rarely throttle and would only dilute the share.
    window = timed_window(df, summary.get('measure_start'), summary.get('measure_end'))
    throttle_rows = window if len(window) else df
    stats['power_limited_pct'] = limited_share(throttle_rows, POWER_LIMIT_BITS)
    stats['thermal_limited_pct'] = limited_share(throttle_rows, THERMAL_LIMIT_BITS)
    start = summary.get('measure_start')
    for key, bits in (('power_limit_onset_s', POWER_LIMIT_BITS),
                      ('thermal_limit_onset_s', THERMAL_LIMIT_BITS)):
        stats[key] = throttle_onset(window, start, bits) if start is not None else None

    # The same window for the trace: loading draws little and would only
    # pull the mean down and the transient ratio up.
    if trace is not None:
        trace_window = timed_window(trace, summary.get('measure_start'),
                                    summary.get('measure_end'))
        stats.update(transient_stats(trace_window if len(trace_window) else trace))
    else:
        stats.update(transient_stats(None))

    durations = step_durations(summary)
    stats.update(step_time_stats(durations))
    stats.update(drift_stats(window, durations))
    stats.update(step_energy_stats(*step_energy(summary, df, trace)))
    stats.update(thermal_stats(df, summary.get('measure_end')))

    tokens_per_step = summary.get('tokens_per_step')
    elapsed = summary.get('elapsed_time_sec') or 0.0
    if not tokens_per_step or elapsed <= 0:
        return stats

    tokens = summary['total_steps'] * tokens_per_step
    stats['tokens_per_sec'] = tokens / elapsed

    if summary.get('params_b'):
        flops = flops_per_token(summary['params_b'],
                                bool(summary.get('gradient_checkpointing')))
        stats['est_tflops'] = stats['tokens_per_sec'] * flops / 1e12
        peak = peak_tflops(device, summary.get('sm_count'), summary.get('compute_dtype'))
        if peak is not None:
            # MFU counts the model's FLOPs only — the 4N of a LoRA step,
            # never the recompute checkpointing adds — so it does not reward
            # doing work twice. HFU counts what the hardware actually ran.
            model_tflops = stats['tokens_per_sec'] * flops_per_token(
                summary['params_b'], gradient_checkpointing=False) / 1e12
            stats['peak_tflops'] = peak.tflops
            stats['peak_basis'] = f"{peak.architecture}, {peak.unit}"
            stats['mfu'] = model_tflops / peak.tflops
            stats['hfu'] = stats['est_tflops'] / peak.tflops

    energy, source = energy_joules(window)
    if energy:
        stats['energy_j'] = energy
        stats['energy_source'] = source
        stats['tokens_per_joule'] = tokens / energy

    return stats
