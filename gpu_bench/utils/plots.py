"""matplotlib charts for everything drawn once: the exported PNG, the report.

The live charts in the window are pyqtgraph (gpu_bench/ui/live_charts.py), but they
take their series names, colours and throttle spans from here, so the two
cannot drift apart the way two hand-maintained copies of each chart once did.
Each function draws one panel onto an axes it is handed, which leaves figure
layout as the only thing the callers decide.
"""
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from gpu_bench.core.telemetry import POWER_LIMIT_BITS, THERMAL_LIMIT_BITS

TEMP_COLOR = '#ff4c4c'
POWER_COLOR = '#4c72ff'
UTIL_COLOR = '#ffa500'
FAN_COLOR = '#a06bff'
CLOCK_COLOR = '#00bfff'
POWER_LIMIT_COLOR = '#ff8c00'
THERMAL_LIMIT_COLOR = '#ff2020'

LOSS_COLOR = '#2e8b57'

THERMAL_TITLE = 'Thermal & Power Curve'
UTILIZATION_TITLE = 'Utilization, Clock & Fan'
LOSS_TITLE = 'Training Loss'


def _merge_legends(ax: Axes, twin: Axes) -> None:
    """One legend for a panel whose series are split across two y-axes."""
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = twin.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')


def throttle_spans(df: pd.DataFrame, bits: int) -> list[tuple[float, float]]:
    """(start, end) times of each stretch where any of the bits was set.

    A stretch runs to the poll after its last flagged one, so a single
    flagged sample still covers the interval it stands for.
    """
    if 'throttle_reasons' not in df.columns:
        return []
    times = df['time_sec'].tolist()
    # A mix of masks and None arrives as a float column with NaN gaps.
    flags = [not pd.isna(mask) and bool(int(mask) & bits)
             for mask in df['throttle_reasons']]

    spans, start = [], None
    for i, flagged in enumerate(flags):
        if flagged and start is None:
            start = times[i]
        elif not flagged and start is not None:
            spans.append((start, times[i]))
            start = None
    if start is not None:
        spans.append((start, times[-1]))
    return spans


def _shade_throttling(ax: Axes, df: pd.DataFrame) -> None:
    """Shades where the card was held back, so a clock dip has a stated cause."""
    for bits, color, label in ((POWER_LIMIT_BITS, POWER_LIMIT_COLOR, 'Power-limited'),
                               (THERMAL_LIMIT_BITS, THERMAL_LIMIT_COLOR, 'Thermal-limited')):
        for i, (start, end) in enumerate(throttle_spans(df, bits)):
            # Labelled once, so the legend gets one entry however many spans.
            ax.axvspan(start, end, color=color, alpha=0.12, linewidth=0,
                       label=label if i == 0 else None)


def envelope(x: ArrayLike, y: ArrayLike,
             max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Thins a long trace for drawing without losing its extremes.

    Plain decimation (every n-th point) would drop exactly the one-sample
    spikes the native trace exists to show. Each bucket keeps its minimum
    and its maximum instead, in time order, so the drawn envelope is the
    same at any zoom a window can show.
    """
    x, y = np.asarray(x), np.asarray(y)
    if len(y) <= max_points:
        return x, y
    edges = np.linspace(0, len(y), max_points // 2 + 1, dtype=int)
    keep = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        segment = y[start:end]
        keep.extend(sorted({start + int(segment.argmin()), start + int(segment.argmax())}))
    return x[keep], y[keep]


def draw_thermal_power(ax: Axes, df: pd.DataFrame,
                       trace: pd.DataFrame | None = None) -> Axes:
    """GPU temperature on the left axis, power draw on a twin. Returns the twin.

    With the native sampler's trace, instantaneous power is drawn faintly
    behind the 250 ms polls — the gap between the two is the point.
    """
    ax.plot(df['time_sec'], df['temp_gpu'], color=TEMP_COLOR,
            label='GPU Temp (°C)', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)', color=TEMP_COLOR)
    ax.tick_params(axis='y', labelcolor=TEMP_COLOR)
    ax.set_title(THERMAL_TITLE)

    twin = ax.twinx()
    if trace is not None and len(trace):
        t, w = envelope(trace['time_sec'], trace['power_w'])
        twin.plot(t, w, color=POWER_COLOR, alpha=0.4, linewidth=0.7,
                  label='Power, instant (W)')
    twin.plot(df['time_sec'], df['power_w'], color=POWER_COLOR,
              label='Power Draw (W)', linewidth=2, linestyle='--')
    twin.set_ylabel('Power (W)', color=POWER_COLOR)
    twin.tick_params(axis='y', labelcolor=POWER_COLOR)

    ax.grid(True, alpha=0.2)
    _merge_legends(ax, twin)
    return twin


def draw_utilization(ax: Axes, df: pd.DataFrame) -> Axes:
    """Utilization and fan on a 0–105 % axis, SM clock on a twin. Returns the twin."""
    ax.plot(df['time_sec'], df['gpu_util_pct'], color=UTIL_COLOR,
            label='GPU Utilization (%)', linewidth=2)

    # Fan duty cycle belongs on the percentage axis beside utilization.
    # Absent on passively cooled cards, so it is drawn only when collected.
    if 'fan_speed_pct' in df.columns and df['fan_speed_pct'].notna().any():
        ax.plot(df['time_sec'], df['fan_speed_pct'], color=FAN_COLOR,
                label='Fan (%)', linewidth=2, linestyle=':')

    _shade_throttling(ax, df)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Utilization / Fan (%)')
    ax.set_ylim(0, 105)
    ax.set_title(UTILIZATION_TITLE)

    # The clock needs its own axis: at ~2500 MHz it shares no useful scale
    # with percentages, and plotting both on one axis flattened every
    # percentage curve onto the zero line.
    twin = ax.twinx()
    twin.plot(df['time_sec'], df['sm_clock_mhz'], color=CLOCK_COLOR,
              label='SM Clock (MHz)', linewidth=2, linestyle='--')
    twin.set_ylabel('Clock (MHz)', color=CLOCK_COLOR)
    twin.tick_params(axis='y', labelcolor=CLOCK_COLOR)

    ax.grid(True, alpha=0.2)
    _merge_legends(ax, twin)
    return twin


def draw_loss(ax: Axes, steps: list[int], losses: list[float],
              warmup_steps: int = 0) -> None:
    """Training loss per optimizer step, with the start of timing marked.

    Not a convergence plot — the run is far too short for that — but a
    cheap check that the adapters are training at all: a flat line means
    gradients are not reaching them.
    """
    ax.plot(steps, losses, color=LOSS_COLOR, label='Loss', linewidth=1.5)
    if warmup_steps and steps and steps[-1] >= warmup_steps:
        ax.axvline(warmup_steps, color='grey', linestyle=':', linewidth=1,
                   label='Timing starts')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(LOSS_TITLE)
    ax.grid(True, alpha=0.2)
    if steps:
        ax.legend(loc='upper right')


def draw_step_durations(ax: Axes, durations: np.ndarray) -> None:
    """Every timed step's duration in order, with the median for reference.

    A drift upwards is the card throttling as it heats; isolated spikes are
    stalls — the data loader, the host, a checkpoint write.
    """
    steps = np.arange(1, len(durations) + 1)
    ax.plot(steps, durations, color=POWER_COLOR, linewidth=1, label='Step time')
    ax.axhline(float(np.median(durations)), color='grey', linestyle=':', linewidth=1,
               label='Median')
    ax.set_xlabel('Timed step')
    ax.set_ylabel('Seconds')
    ax.set_title('Step Duration')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')


def draw_step_energy(ax: Axes, energies: np.ndarray, source: str | None = None) -> None:
    """How the energy of a step is distributed: one narrow peak is a steady run."""
    ax.hist(energies, bins=min(40, max(5, len(energies) // 5)), color=TEMP_COLOR, alpha=0.8)
    ax.axvline(float(np.median(energies)), color='grey', linestyle=':', linewidth=1)
    ax.set_xlabel(f"Joules per step{f' ({source})' if source else ''}")
    ax.set_ylabel('Steps')
    ax.set_title('Energy per Step')
    ax.grid(True, alpha=0.2)


def build_step_figure(durations: np.ndarray, energies: np.ndarray,
                      source: str | None = None) -> Figure | None:
    """Step duration over the run beside the per-step energy histogram.

    None without step stamps. Without a fine enough energy reading the
    duration plot stands alone.
    """
    if len(durations) < 2:
        return None
    fig = Figure(figsize=(10, 3.2))
    if len(energies) >= 2:
        left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
        draw_step_durations(left, durations)
        draw_step_energy(right, energies, source)
    else:
        draw_step_durations(fig.add_subplot(111), durations)
    fig.tight_layout()
    return fig


def build_report_figure(df: pd.DataFrame, trace: pd.DataFrame | None = None) -> Figure:
    """Both panels stacked, as written to the exported PNG.

    A bare Figure rather than pyplot: pyplot keeps a global registry of open
    figures and binds them to the GUI backend, neither of which an export
    running inside a Qt app has any use for.
    """
    fig = Figure(figsize=(10, 7))
    top, bottom = fig.subplots(2, 1)
    draw_thermal_power(top, df, trace)
    draw_utilization(bottom, df)
    fig.tight_layout()
    return fig
