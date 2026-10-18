"""Derived benchmark numbers: throughput, compute rate, energy.

Kept apart from the window so the same figures can be shown on screen, written
into a report or printed by a headless run, and tested without Qt. Everything
here returns raw numbers; formatting is the caller's business.

Steps/sec on its own is not comparable between runs: a step at batch 1 and
512 tokens is an eighth of the work of one at batch 2 and 2048. Tokens/sec
normalises that, and the energy figures say what the throughput cost.
"""
import pandas as pd


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
        t = df['timestamp'].to_numpy(dtype=float)
        p = df['power_w'].to_numpy(dtype=float)
        return float(((p[1:] + p[:-1]) / 2 * (t[1:] - t[:-1])).sum()), "power samples"

    return None, None


def _column_stat(df: pd.DataFrame, column: str, how: str) -> float | None:
    """max/min/mean of an optional column, None when it was never reported."""
    if column not in df.columns or not df[column].notna().any():
        return None
    return float(getattr(df[column], how)())


def compute_stats(df: pd.DataFrame, summary: dict) -> dict:
    """Everything the summary table shows, as numbers.

    Optional readings (memory temperature, fan, PCIe) come back as None on
    hardware that does not report them, and so do the throughput figures when
    the summary predates them or the timed window never opened.
    """
    stats: dict = {
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
    }

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

    window = timed_window(df, summary.get('measure_start'), summary.get('measure_end'))
    energy, source = energy_joules(window)
    if energy:
        stats['energy_j'] = energy
        stats['energy_source'] = source
        stats['tokens_per_joule'] = tokens / energy

    return stats
