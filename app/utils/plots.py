"""Chart drawing shared by the window's tabs and the exported PNG.

Both used to carry their own copy of every series, colour and axis label, so
each change to a chart had to be made twice, in two modules — and the two had
already started to drift apart. Each function here draws one panel onto an
axes it is handed, which leaves figure layout (two tabs on screen, two stacked
panels on disk) as the only thing the callers still decide.
"""
from matplotlib.axes import Axes
from matplotlib.figure import Figure

TEMP_COLOR = '#ff4c4c'
POWER_COLOR = '#4c72ff'
UTIL_COLOR = '#ffa500'
FAN_COLOR = '#a06bff'
CLOCK_COLOR = '#00bfff'

THERMAL_TITLE = 'Thermal & Power Curve'
UTILIZATION_TITLE = 'Utilization, Clock & Fan'


def _merge_legends(ax: Axes, twin: Axes) -> None:
    """One legend for a panel whose series are split across two y-axes."""
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = twin.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')


def draw_thermal_power(ax: Axes, df) -> Axes:
    """GPU temperature on the left axis, power draw on a twin. Returns the twin."""
    ax.plot(df['time_sec'], df['temp_gpu'], color=TEMP_COLOR,
            label='GPU Temp (°C)', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)', color=TEMP_COLOR)
    ax.tick_params(axis='y', labelcolor=TEMP_COLOR)
    ax.set_title(THERMAL_TITLE)

    twin = ax.twinx()
    twin.plot(df['time_sec'], df['power_w'], color=POWER_COLOR,
              label='Power Draw (W)', linewidth=2, linestyle='--')
    twin.set_ylabel('Power (W)', color=POWER_COLOR)
    twin.tick_params(axis='y', labelcolor=POWER_COLOR)

    ax.grid(True, alpha=0.2)
    _merge_legends(ax, twin)
    return twin


def draw_utilization(ax: Axes, df) -> Axes:
    """Utilization and fan on a 0–105 % axis, SM clock on a twin. Returns the twin."""
    ax.plot(df['time_sec'], df['gpu_util_pct'], color=UTIL_COLOR,
            label='GPU Utilization (%)', linewidth=2)

    # Fan duty cycle belongs on the percentage axis beside utilization.
    # Absent on passively cooled cards, so it is drawn only when collected.
    if 'fan_speed_pct' in df.columns and df['fan_speed_pct'].notna().any():
        ax.plot(df['time_sec'], df['fan_speed_pct'], color=FAN_COLOR,
                label='Fan (%)', linewidth=2, linestyle=':')

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


def build_report_figure(df) -> Figure:
    """Both panels stacked, as written to the exported PNG.

    A bare Figure rather than pyplot: pyplot keeps a global registry of open
    figures and binds them to the GUI backend, neither of which an export
    running inside a Qt app has any use for.
    """
    fig = Figure(figsize=(10, 7))
    top, bottom = fig.subplots(2, 1)
    draw_thermal_power(top, df)
    draw_utilization(bottom, df)
    fig.tight_layout()
    return fig
