"""LiveCharts: pyqtgraph curves fed one telemetry snapshot at a time."""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.ui


@pytest.fixture
def charts(qapp):
    from gpu_bench.ui.live_charts import LiveCharts
    c = LiveCharts()
    c.resize(800, 500)
    return c


def _y(charts, name):
    return list(charts.curves[name].getData()[1] if charts.curves[name].getData()[1] is not None
                else [])


def test_time_starts_at_the_first_snapshot(charts, telemetry_sample):
    from gpu_bench.ui.live_charts import TEMP

    for i in range(4):
        charts.append(telemetry_sample(i))
    charts.refresh()

    x, y = charts.curves[TEMP].getData()
    assert list(x) == [0.0, 1.0, 2.0, 3.0]
    assert list(y) == [65, 66, 67, 68]


def test_native_batches_are_thinned_but_keep_their_peak(charts, telemetry_sample):
    from gpu_bench.ui.live_charts import POWER_INSTANT

    charts.append(telemetry_sample(0))
    seconds = 100.0 + np.arange(125) * 0.002
    watts = np.full(125, 200.0)
    watts[77] = 340.0
    charts.append_trace((seconds, watts))
    charts.refresh()

    y = _y(charts, POWER_INSTANT)
    assert len(y) <= 4
    assert max(y) == 340.0


def test_trace_before_any_telemetry_is_ignored(charts):
    """No time origin yet, so there is nothing to place it against."""
    from gpu_bench.ui.live_charts import POWER_INSTANT

    charts.append_trace((np.array([1.0, 2.0]), np.array([100.0, 110.0])))
    charts.refresh()

    assert _y(charts, POWER_INSTANT) == []


def test_power_limited_stretches_are_shaded_once(charts, telemetry_sample):
    from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP

    for i, mask in enumerate([0, THROTTLE_SW_POWER_CAP, THROTTLE_SW_POWER_CAP, 0]):
        charts.append(telemetry_sample(i, throttle_reasons=mask))
    charts.refresh()
    first = charts._shading[:]
    charts.refresh()

    assert [(a, b) for a, b, _ in charts.shaded_spans()] == [(1.0, 3.0)]
    assert charts._shading == first, "unchanged spans must not be rebuilt"


def test_passive_card_draws_no_fan_curve(charts, telemetry_sample):
    from gpu_bench.ui.live_charts import FAN

    for i in range(3):
        charts.append(telemetry_sample(i, fan_speed_pct=None))
    charts.refresh()

    assert not charts.curves[FAN].isVisible()


def test_warmup_line_appears_once_timing_starts(charts):
    charts.set_warmup(10)
    for step in range(1, 10):
        charts.append_loss(step, 3.0)
    charts.refresh()
    assert not charts.warmup_line.isVisible()

    charts.append_loss(10, 2.9)
    charts.refresh()
    assert charts.warmup_line.isVisible()
    assert charts.warmup_line.value() == 10


def test_reset_starts_a_clean_run(charts, telemetry_sample):
    from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP
    from gpu_bench.ui.live_charts import LOSS, TEMP

    charts.append(telemetry_sample(0, throttle_reasons=THROTTLE_SW_POWER_CAP))
    charts.append(telemetry_sample(1))
    charts.append_loss(1, 3.0)
    charts.refresh()
    charts.reset()

    assert _y(charts, TEMP) == []
    assert _y(charts, LOSS) == []
    assert charts.shaded_spans() == []


def test_final_draw_uses_the_complete_run(charts, telemetry_sample):
    from gpu_bench.ui.live_charts import POWER, POWER_INSTANT

    df = pd.DataFrame([telemetry_sample(i) for i in range(20)])
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    t = np.linspace(0, 19, 10_000)
    trace = pd.DataFrame({"time_sec": t, "timestamp": 100 + t, "power_w": np.full(10_000, 250.0)})
    trace.loc[4321, "power_w"] = 400.0

    charts.show_run(df, trace)

    assert len(_y(charts, POWER)) == 20
    assert len(_y(charts, POWER_INSTANT)) <= 4000
    assert max(_y(charts, POWER_INSTANT)) == 400.0



def test_theme_recolours_every_legend_label(charts):
    from gpu_bench.ui.style import DARK

    charts._set_instant(np.array([0.0, 1.0]), np.array([100.0, 120.0]))
    charts.set_theme(DARK)

    # pyqtgraph renders the colour into each label's HTML.
    for _, label in charts.thermal_plot.legend.items:
        assert f"color:{DARK.chart_fg.lower()}" in label.item.toHtml().replace(" ", "").lower()


# --- crosshair ----------------------------------------------------------------------

def _fed(charts, telemetry_sample, n=10, **overrides):
    for i in range(n):
        charts.append(telemetry_sample(i, **overrides))
    charts.refresh()


def test_readout_lists_every_series_at_the_nearest_sample(charts, telemetry_sample):
    _fed(charts, telemetry_sample)

    assert charts.readout_text(3.4) == ("t 3.0 s   68 °C   253 W   util 98 %   "
                                        "fan 63 %   2600 MHz")


def test_readout_skips_a_fan_the_card_does_not_have(charts, telemetry_sample):
    _fed(charts, telemetry_sample, fan_speed_pct=None)

    assert "fan" not in charts.readout_text(2.0)


def test_crosshair_runs_through_both_plots(charts, telemetry_sample):
    _fed(charts, telemetry_sample)

    charts.show_crosshair(5.0)

    assert all(line.isVisible() and line.value() == 5.0 for line in charts.crosshairs)
    assert "t 5.0 s" in charts.readout.text

    charts.hide_crosshair()
    assert not any(line.isVisible() for line in charts.crosshairs)
    assert charts.readout.text == ""


def test_crosshair_before_any_data_stays_hidden(charts):
    charts.show_crosshair(5.0)

    assert not any(line.isVisible() for line in charts.crosshairs)


def test_mouse_over_a_plot_moves_the_crosshair(charts, telemetry_sample):
    from PyQt6.QtCore import QPointF

    _fed(charts, telemetry_sample)
    charts.show()
    charts.thermal_plot.setXRange(0, 9, padding=0)
    scene_pos = charts.thermal_plot.vb.mapViewToScene(QPointF(6.0, 70.0))

    charts._on_mouse_moved((scene_pos,))

    assert charts.crosshairs[0].value() == pytest.approx(6.0, abs=0.05)


def test_a_new_run_clears_the_crosshair(charts, telemetry_sample):
    _fed(charts, telemetry_sample)
    charts.show_crosshair(4.0)

    charts.reset()

    assert charts.readout_text(4.0) == ""
    assert not any(line.isVisible() for line in charts.crosshairs)
