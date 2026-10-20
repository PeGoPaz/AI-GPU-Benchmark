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


def _column_stat(df: pd.DataFrame, column: str, how: str) -> float | None:
    """max/min/mean of an optional column, None when it was never reported."""
    if column not in df.columns or not df[column].notna().any():
        return None
    return float(getattr(df[column], how)())


def compute_stats(df: pd.DataFrame, summary: dict[str, Any],
                  trace: pd.DataFrame | None = None) -> dict[str, Any]:
    """Everything the summary table shows, as numbers.

    Optional readings (memory temperature, fan, PCIe) come back as None on
    hardware that does not report them, and so do the throughput figures when
    the summary predates them or the timed window never opened. The
    transient figures need the native sampler's trace and are None without it.
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
    }

    # Over the measured steps when there were any, else over the whole run:
    # loading and warm-up rarely throttle and would only dilute the share.
    window = timed_window(df, summary.get('measure_start'), summary.get('measure_end'))
    throttle_rows = window if len(window) else df
    stats['power_limited_pct'] = limited_share(throttle_rows, POWER_LIMIT_BITS)
    stats['thermal_limited_pct'] = limited_share(throttle_rows, THERMAL_LIMIT_BITS)

    # The same window for the trace: loading draws little and would only
    # pull the mean down and the transient ratio up.
    if trace is not None:
        trace_window = timed_window(trace, summary.get('measure_start'),
                                    summary.get('measure_end'))
        stats.update(transient_stats(trace_window if len(trace_window) else trace))
    else:
        stats.update(transient_stats(None))

    stats.update(step_time_stats(step_durations(summary)))

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

    energy, source = energy_joules(window)
    if energy:
        stats['energy_j'] = energy
        stats['energy_source'] = source
        stats['tokens_per_joule'] = tokens / energy

    return stats
