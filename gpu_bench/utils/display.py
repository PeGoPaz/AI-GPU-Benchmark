"""Text shown to people, shared by the window and the headless CLI.

Both present the same run — a summary table, a progress line — and used to
be on course for two copies of the formatting, the way the charts once were.
"""
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from gpu_bench.core.metrics import compute_stats


class RateTracker:
    """Steps per second, measured from the first completed step.

    Not from Start: model loading can take longer than the run itself on a
    cold cache, and counting it would make every early ETA wildly pessimistic.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._first: tuple[float, int] | None = None

    def update(self, step: int) -> float | None:
        now = self._clock()
        if self._first is None:
            self._first = (now, step)
        first_time, first_step = self._first
        return (step - first_step) / (now - first_time) if now > first_time else None


def format_progress(step: int, total: int, steps_per_sec: float | None) -> str:
    """The line under the progress bar: position, live rate and time left."""
    text = f"Step {step} / {total}"
    if steps_per_sec:
        remaining = max(0, total - step) / steps_per_sec
        minutes, seconds = divmod(int(round(remaining)), 60)
        hours, minutes = divmod(minutes, 60)
        eta = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
        text += f" · {steps_per_sec:.2f} steps/s · ETA {eta}"
    return text


# Below this share of the way to T∞, the curve has bent too little for its
# end point to be extrapolated with any confidence.
_PREDICTABLE_PROGRESS = 0.5


def thermal_verdict(stats: dict[str, Any]) -> str:
    """Whether the run reached thermal steady state, and if not, how far it got."""
    steady = f"{stats['thermal_steady_c']:.0f} °C"
    if stats['thermal_saturated']:
        return f"reached, at {steady}"
    progress = stats['thermal_progress']
    if progress < _PREDICTABLE_PROGRESS:
        return (f"not reached — still heating steeply at the end; run longer than "
                f"{3 * stats['thermal_tau_s']:.0f} s for sustained numbers")
    return (f"not reached — {progress:.0%} of the way, heading for {steady}; "
            f"sustained numbers may be lower")


def summary_rows(df: pd.DataFrame, summary: dict[str, Any],
                 trace: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """The summary table as (metric, value) pairs, ready to display."""
    stats = compute_stats(df, summary, trace)

    steps_label = f"{summary['total_steps']}"
    if summary.get('aborted'):
        steps_label += f" of {summary['requested_steps']} (stopped early)"

    checkpointing = "—"
    if "gradient_checkpointing" in summary:
        checkpointing = "On" if summary["gradient_checkpointing"] else "Off"

    metrics = [
        ("Model", summary.get("model", "—")),
        ("Batch Size", str(summary.get("batch_size", "—"))),
        ("Precision", summary.get("precision", "—")),
        ("Sequence Length", str(summary.get("seq_len", "—"))),
        ("Gradient Checkpointing", checkpointing),
        ("Warm-up Steps", str(summary.get("warmup_steps", "—"))),
        ("Steps Completed", steps_label),
        ("Duration (s)", f"{summary['elapsed_time_sec']:.2f}"),
        ("Steps/sec", f"{summary['steps_per_sec']:.2f}"),
    ]

    if stats['step_time_p50_s'] is not None:
        metrics.append(("Step Time p50 / p99 (s)",
                        f"{stats['step_time_p50_s']:.3f} / {stats['step_time_p99_s']:.3f}"))
        if stats['step_time_cv'] is not None:
            metrics.append(("Step-Time Jitter (CV)", f"{stats['step_time_cv'] * 100:.1f} %"))
    if stats['step_energy_p50_j'] is not None:
        metrics.append((f"Energy per Step p50 / p95 (J, {stats['step_energy_source']})",
                        f"{stats['step_energy_p50_j']:.1f} / {stats['step_energy_p95_j']:.1f}"))
        if stats['step_energy_cv'] is not None:
            metrics.append(("Step-Energy Spread (CV)", f"{stats['step_energy_cv'] * 100:.1f} %"))

    # Throughput and energy need the timed window, which a run stopped
    # during warm-up never opened.
    if stats['tokens_per_sec'] is not None:
        metrics.append(("Tokens/sec", f"{stats['tokens_per_sec']:,.0f}"))
    if stats['est_tflops'] is not None:
        metrics.append(("Compute (est. TFLOPS)", f"{stats['est_tflops']:.1f}"))
    if stats['energy_j'] is not None:
        metrics.append((f"Energy, timed (kJ, {stats['energy_source']})",
                        f"{stats['energy_j'] / 1000:.2f}"))
        metrics.append(("Efficiency (tokens/J)", f"{stats['tokens_per_joule']:.1f}"))

    metrics += [
        ("Avg Temp (°C)", f"{stats['avg_temp_c']:.1f}"),
        ("Max Temp (°C)", f"{stats['max_temp_c']:.0f}"),
    ]
    if stats['thermal_tau_s'] is not None:
        metrics.append(("Thermal Time Constant τ (s)",
                        f"{stats['thermal_tau_s']:.0f} (fit R² {stats['thermal_r2']:.2f})"))
        metrics.append(("Thermal Steady State", thermal_verdict(stats)))
    metrics += [
        ("Avg Power (W)", f"{stats['avg_power_w']:.1f}"),
        ("Peak Power (W)", f"{stats['peak_power_w']:.1f}"),
    ]

    # From the native sampler only: what the 250 ms polls cannot see.
    if stats['native_peak_power_w'] is not None:
        metrics += [
            ("Peak Power, instant (W)", f"{stats['native_peak_power_w']:.1f}"),
            ("Power p99, instant (W)", f"{stats['native_p99_power_w']:.1f}"),
        ]
        if stats['transient_ratio'] is not None:
            metrics.append(("Transient Ratio (peak / mean)", f"{stats['transient_ratio']:.2f}×"))
        if stats['native_sample_hz'] is not None:
            metrics.append(("Native Sampling (Hz)",
                            f"{stats['native_sample_hz']:.0f}, "
                            f"{stats['native_update_hz']:.0f} new readings/s"))
        if stats['energy_j'] is not None and stats['energy_trace_j']:
            metrics.append(("Energy, instant trace (kJ)", f"{stats['energy_trace_j'] / 1000:.2f}"))

    metrics += [
        ("Avg VRAM (MiB)", f"{stats['avg_vram_mb']:.1f}"),
        ("Peak VRAM (MiB)", f"{stats['peak_vram_mb']:.1f}"),
    ]

    # Optional hardware: memory temperature needs a GPU exposing the
    # field, and fan and PCIe readings are absent on passive cards.
    if stats['max_mem_temp_c'] is not None:
        metrics.append(("Max Mem Temp (°C)", f"{stats['max_mem_temp_c']:.0f}"))
    if stats['min_headroom_c'] is not None:
        metrics.append(("Min Thermal Headroom (°C)", f"{stats['min_headroom_c']:.0f}"))
    if stats['peak_fan_pct'] is not None:
        metrics.append(("Peak Fan (%)", f"{stats['peak_fan_pct']:.0f}"))
    if stats['power_limited_pct'] is not None:
        metrics.append(("Time Power-Limited (%)", f"{stats['power_limited_pct']:.0f}"))
        metrics.append(("Time Thermal-Limited (%)", f"{stats['thermal_limited_pct']:.0f}"))
    onsets = [f"{name} limit at {stats[key]:.0f} s" for name, key in
              (("power", 'power_limit_onset_s'), ("thermal", 'thermal_limit_onset_s'))
              if stats[key] is not None]
    if onsets:
        metrics.append(("Throttling Began (into timing)", ", ".join(onsets)))
    if stats['clock_drift'] is not None:
        metrics.append(("SM Clock Drift (first → last quarter)",
                        f"{stats['clock_drift'] * 100:+.1f} %"))
    if stats['throughput_drift'] is not None:
        metrics.append(("Throughput Drift (first → last quarter)",
                        f"{stats['throughput_drift'] * 100:+.1f} %"))
    if stats['pcie_gen_peak'] is not None and stats['pcie_width_peak'] is not None:
        metrics.append(("PCIe Link (peak)",
                        f"Gen{stats['pcie_gen_peak']:.0f} x{stats['pcie_width_peak']:.0f}"))

    return metrics


@dataclass(frozen=True)
class Headline:
    """One of the few numbers worth reading first, as a card shows it."""
    label: str
    value: str
    unit: str = ""
    caption: str = ""


def headline_cards(df: pd.DataFrame, summary: dict[str, Any],
                   trace: pd.DataFrame | None = None) -> list[Headline]:
    """The run in five numbers: rate, compute, efficiency, peak power, peak heat.

    The full table still has everything; these are what people compare runs
    by. A figure the run cannot support — no timed window, no energy reading —
    falls back to the nearest one it can, rather than showing a dash.
    """
    stats = compute_stats(df, summary, trace)
    cards = []

    if stats['tokens_per_sec'] is not None:
        cards.append(Headline("Throughput", f"{stats['tokens_per_sec']:,.0f}", "tokens/s",
                              f"{summary['steps_per_sec']:.2f} steps/s"))
    else:
        cards.append(Headline("Throughput", f"{summary['steps_per_sec']:.2f}", "steps/s",
                              "no token count for this run"))

    if stats['est_tflops'] is not None:
        cards.append(Headline("Compute", f"{stats['est_tflops']:.1f}", "TFLOPS", "estimated"))

    if stats['tokens_per_joule'] is not None:
        cards.append(Headline("Efficiency", f"{stats['tokens_per_joule']:.0f}", "tokens/J",
                              f"{stats['energy_j'] / 1000:.1f} kJ over the timed window"))

    if stats['native_peak_power_w'] is not None:
        cards.append(Headline("Peak Power", f"{stats['native_peak_power_w']:.0f}", "W",
                              f"instant · {stats['peak_power_w']:.0f} W in 250 ms polls"))
    else:
        cards.append(Headline("Peak Power", f"{stats['peak_power_w']:.0f}", "W",
                              f"250 ms polls · avg {stats['avg_power_w']:.0f} W"))

    cards.append(Headline("Max Temp", f"{stats['max_temp_c']:.0f}", "°C",
                          f"avg {stats['avg_temp_c']:.0f} °C"))
    return cards
