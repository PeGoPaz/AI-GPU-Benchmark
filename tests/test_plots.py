"""Shared chart drawing: one definition of each panel for screen and export."""
import pandas as pd
from matplotlib.figure import Figure

from app.utils import plots


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
