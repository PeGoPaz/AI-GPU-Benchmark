"""Text shown to people, shared by the window and the headless CLI.

Both present the same run — a summary table, a progress line — and used to
be on course for two copies of the formatting, the way the charts once were.
"""
import time
from collections.abc import Callable

from app.core.metrics import compute_stats


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


def summary_rows(df, summary: dict, trace=None) -> list[tuple[str, str]]:
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
    if stats['pcie_gen_peak'] is not None and stats['pcie_width_peak'] is not None:
        metrics.append(("PCIe Link (peak)",
                        f"Gen{stats['pcie_gen_peak']:.0f} x{stats['pcie_width_peak']:.0f}"))

    return metrics
