"""Shared chart drawing: one definition of each panel for screen and export."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from gpu_bench.utils import plots


def _df(sample, n=5, **overrides):
    df = pd.DataFrame([sample(i, **overrides) for i in range(n)])
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def _labels(*axes):
    return [line.get_label() for ax in axes for line in ax.get_lines()]


def test_thermal_panel_splits_temperature_and_power(telemetry_sample):
    ax = Figure().add_subplot(111)
    twin = plots.draw_thermal_power(ax, _df(telemetry_sample))

    assert _labels(ax) == ["GPU Temp (°C)"]
    assert _labels(twin) == ["Power Draw (W)"]
    assert ax.get_title() == plots.THERMAL_TITLE


def test_utilization_panel_keeps_clock_on_its_own_axis(telemetry_sample):
    ax = Figure().add_subplot(111)
    twin = plots.draw_utilization(ax, _df(telemetry_sample))

    assert _labels(ax) == ["GPU Utilization (%)", "Fan (%)"]
    assert _labels(twin) == ["SM Clock (MHz)"]
    assert ax.get_ylim() == (0, 105)


def test_legend_covers_both_axes(telemetry_sample):
    ax = Figure().add_subplot(111)
    plots.draw_utilization(ax, _df(telemetry_sample))

    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["GPU Utilization (%)", "Fan (%)", "SM Clock (MHz)"]


def test_passive_card_draws_no_fan_series(telemetry_sample):
    ax = Figure().add_subplot(111)
    plots.draw_utilization(ax, _df(telemetry_sample, fan_speed_pct=None))

    assert "Fan (%)" not in _labels(ax)


def test_report_figure_stacks_both_panels(telemetry_sample):
    fig = plots.build_report_figure(_df(telemetry_sample))

    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == [plots.THERMAL_TITLE, plots.UTILIZATION_TITLE]


def test_throttle_spans_cover_each_flagged_stretch(telemetry_sample):
    from gpu_bench.core.telemetry import POWER_LIMIT_BITS, THROTTLE_SW_POWER_CAP

    df = _df(telemetry_sample, n=6)
    df["throttle_reasons"] = [0, THROTTLE_SW_POWER_CAP, THROTTLE_SW_POWER_CAP, 0,
                              THROTTLE_SW_POWER_CAP, None]

    assert plots.throttle_spans(df, POWER_LIMIT_BITS) == [(1.0, 3.0), (4.0, 5.0)]


def test_power_limited_stretches_are_shaded_once_in_the_legend(telemetry_sample):
    from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP

    df = _df(telemetry_sample, n=6)
    df["throttle_reasons"] = [0, THROTTLE_SW_POWER_CAP, 0, THROTTLE_SW_POWER_CAP, 0, 0]
    ax = Figure().add_subplot(111)
    plots.draw_utilization(ax, df)

    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend.count("Power-limited") == 1
    assert "Thermal-limited" not in legend
    assert len(ax.patches) == 2


def test_unthrottled_run_draws_no_shading(telemetry_sample):
    ax = Figure().add_subplot(111)
    plots.draw_utilization(ax, _df(telemetry_sample))

    assert not ax.patches



def test_loss_plot_marks_where_timing_starts():
    ax = Figure().add_subplot(111)
    plots.draw_loss(ax, [1, 2, 3, 4], [3.0, 2.8, 2.7, 2.6], warmup_steps=2)

    assert _labels(ax) == ["Loss", "Timing starts"]
    assert ax.get_title() == plots.LOSS_TITLE


def test_loss_plot_before_the_first_step_is_empty_not_an_error():
    ax = Figure().add_subplot(111)
    plots.draw_loss(ax, [], [], warmup_steps=10)

    assert "Timing starts" not in _labels(ax)
    assert ax.get_legend() is None



# --- native power trace ------------------------------------------------------------

def test_envelope_keeps_a_single_sample_spike():
    """Every-n-th decimation would drop the one sample the trace exists for."""
    import numpy as np
    y = np.full(100_000, 200.0)
    y[54_321] = 450.0
    x = np.arange(len(y))

    ex, ey = plots.envelope(x, y, max_points=1000)

    assert len(ey) <= 1000
    assert ey.max() == 450.0
    assert list(ex) == sorted(ex)


def test_short_trace_is_drawn_as_is():
    import numpy as np
    x, y = np.arange(10), np.arange(10.0)

    ex, ey = plots.envelope(x, y)

    assert list(ey) == list(y)


def test_instant_power_is_drawn_behind_the_polls(telemetry_sample):
    import numpy as np
    df = _df(telemetry_sample)
    t = np.linspace(0, 4, 2000)
    trace = pd.DataFrame({"time_sec": t, "timestamp": 100 + t, "power_w": 250 + 50 * np.sin(t * 40)})
    ax = Figure().add_subplot(111)

    twin = plots.draw_thermal_power(ax, df, trace)

    assert _labels(twin) == ["Power, instant (W)", "Power Draw (W)"]


def test_report_figure_takes_the_trace_too(telemetry_sample):
    import numpy as np
    t = np.linspace(0, 4, 50)
    trace = pd.DataFrame({"time_sec": t, "timestamp": 100 + t, "power_w": np.full(50, 260.0)})

    fig = plots.build_report_figure(_df(telemetry_sample), trace)

    assert "Power, instant (W)" in [line.get_label() for ax in fig.axes for line in ax.get_lines()]


def test_step_figure_needs_two_steps():
    assert plots.build_step_figure(np.array([1.0]), np.empty(0)) is None


def test_step_figure_adds_the_energy_histogram_when_there_is_energy():
    durations = np.array([1.0, 1.1, 0.9, 1.0])

    alone = plots.build_step_figure(durations, np.empty(0))
    both = plots.build_step_figure(durations, np.array([100.0, 110.0, 90.0, 100.0]), "counter")

    assert len(alone.axes) == 1
    assert [ax.get_title() for ax in both.axes] == ["Step Duration", "Energy per Step"]
    assert "counter" in both.axes[1].get_xlabel()
