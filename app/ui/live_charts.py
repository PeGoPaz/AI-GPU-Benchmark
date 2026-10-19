"""Live charts, drawn with pyqtgraph.

matplotlib redrew every figure from scratch once a second: tolerable for a
150-step run, a visible stutter by the end of a long one, and never smooth.
pyqtgraph keeps its curves and only swaps their data, so the charts follow
telemetry at its own 4 Hz and can be zoomed and panned while a run goes on.

The exported PNG stays on matplotlib (app/utils/plots.py): it is drawn once,
needs no interaction, and prints better. The series, colours and the
throttle shading are the same on both, taken from the same constants.
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTabWidget, QWidget

from app.core.telemetry import POWER_LIMIT_BITS, THERMAL_LIMIT_BITS
from app.ui.style import LIGHT, Palette
from app.utils.plots import (
    CLOCK_COLOR,
    FAN_COLOR,
    LOSS_COLOR,
    POWER_COLOR,
    POWER_LIMIT_COLOR,
    TEMP_COLOR,
    THERMAL_LIMIT_COLOR,
    UTIL_COLOR,
    envelope,
    throttle_spans,
)

pg.setConfigOptions(antialias=True)

# Series names, shared with the legend and the tests.
TEMP, POWER, POWER_INSTANT = "GPU Temp (°C)", "Power Draw (W)", "Power, instant (W)"
UTIL, FAN, CLOCK = "GPU Utilization (%)", "Fan (%)", "SM Clock (MHz)"
LOSS = "Loss"


def _pen(color: str, width: float = 2.0, style=Qt.PenStyle.SolidLine, alpha: int = 255):
    c = pg.mkColor(color)
    c.setAlpha(alpha)
    return pg.mkPen(c, width=width, style=style)


def _twin_view(plot: pg.PlotItem) -> pg.ViewBox:
    """A second y-axis on the right of a plot, sharing its x-axis.

    pyqtgraph's way of doing matplotlib's twinx(): a separate ViewBox laid
    over the plot's own, kept the same size and linked to it in x.
    """
    view = pg.ViewBox()
    plot.showAxis("right")
    scene = plot.scene()
    if scene is not None:
        scene.addItem(view)
    plot.getAxis("right").linkToView(view)
    view.setXLink(plot)

    def follow():
        view.setGeometry(plot.vb.sceneBoundingRect())
        view.linkedViewChanged(plot.vb, view.XAxis)

    plot.vb.sigResized.connect(follow)
    follow()
    return view


class LiveCharts(QTabWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme: Palette = LIGHT
        self.curves: dict[str, pg.PlotDataItem] = {}
        self._shading: list[pg.LinearRegionItem] = []
        self._spans: list[tuple[float, float, str]] = []
        self.reset()

        # --- Telemetry: two plots stacked, one shared time axis ----------------------
        self.telemetry_view = pg.GraphicsLayoutWidget()
        self.thermal_plot = self.telemetry_view.addPlot(row=0, col=0)
        self.util_plot = self.telemetry_view.addPlot(row=1, col=0)
        self.util_plot.setXLink(self.thermal_plot)
        for plot in (self.thermal_plot, self.util_plot):
            plot.showGrid(x=True, y=True, alpha=0.2)
        self.thermal_plot.setLabel("left", "Temperature (°C)", color=TEMP_COLOR)
        self.util_plot.setLabel("left", "Utilization / Fan (%)")
        self.util_plot.setLabel("bottom", "Time (seconds)")
        self.util_plot.setYRange(0, 105, padding=0)
        self.util_plot.vb.setMouseEnabled(y=False)

        self.power_view = _twin_view(self.thermal_plot)
        self.thermal_plot.setLabel("right", "Power (W)", color=POWER_COLOR)
        self.clock_view = _twin_view(self.util_plot)
        self.util_plot.setLabel("right", "Clock (MHz)", color=CLOCK_COLOR)

        thermal_legend = self.thermal_plot.addLegend(offset=(10, 10))
        util_legend = self.util_plot.addLegend(offset=(10, 10))

        self.curves[TEMP] = self.thermal_plot.plot(name=TEMP, pen=_pen(TEMP_COLOR))
        # Behind the polls, faint: the 250 ms line is the summary, the instant
        # trace the detail it smooths away.
        self.curves[POWER_INSTANT] = pg.PlotDataItem(pen=_pen(POWER_COLOR, 1, alpha=100))
        self.curves[POWER] = pg.PlotDataItem(pen=_pen(POWER_COLOR, style=Qt.PenStyle.DashLine))
        for name in (POWER_INSTANT, POWER):
            self.power_view.addItem(self.curves[name])
        thermal_legend.addItem(self.curves[POWER], POWER)
        self._instant_in_legend = False
        self._thermal_legend = thermal_legend

        self.curves[UTIL] = self.util_plot.plot(name=UTIL, pen=_pen(UTIL_COLOR))
        self.curves[FAN] = self.util_plot.plot(name=FAN, pen=_pen(FAN_COLOR, style=Qt.PenStyle.DotLine))
        self.curves[CLOCK] = pg.PlotDataItem(pen=_pen(CLOCK_COLOR, style=Qt.PenStyle.DashLine))
        self.clock_view.addItem(self.curves[CLOCK])
        util_legend.addItem(self.curves[CLOCK], CLOCK)

        # --- Loss -----------------------------------------------------------------------
        self.loss_view = pg.PlotWidget()
        self.loss_plot = self.loss_view.getPlotItem()
        self.loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Step")
        self.loss_plot.addLegend(offset=(-10, 10))
        self.curves[LOSS] = self.loss_plot.plot(name=LOSS, pen=_pen(LOSS_COLOR, 1.5))
        self.warmup_line = pg.InfiniteLine(angle=90, movable=False,
                                           pen=_pen("#888888", 1, Qt.PenStyle.DotLine),
                                           label="timing starts",
                                           labelOpts={"position": 0.92, "color": "#888888"})
        self.loss_plot.addItem(self.warmup_line)
        self.warmup_line.hide()

        self.addTab(self.telemetry_view, "Telemetry")
        self.addTab(self.loss_view, "Loss")
        self.set_theme(LIGHT)

    # --- collecting -----------------------------------------------------------------------

    def reset(self) -> None:
        """A new run: the charts start empty and time starts at zero."""
        self._t0: float | None = None
        self._rows: dict[str, list] = {k: [] for k in
                                       ("t", "temp", "power", "util", "fan", "clock", "throttle")}
        self._trace_t: list[np.ndarray] = []
        self._trace_w: list[np.ndarray] = []
        self._loss_steps: list[int] = []
        self._loss_values: list[float] = []
        self._warmup = 0
        for curve in getattr(self, "curves", {}).values():
            curve.setData([], [])
        if hasattr(self, "warmup_line"):
            self.warmup_line.hide()
            self._set_spans([])

    def set_warmup(self, steps: int) -> None:
        self._warmup = steps

    def append(self, snapshot: dict) -> None:
        if self._t0 is None:
            self._t0 = snapshot["timestamp"]
        r = self._rows
        r["t"].append(snapshot["timestamp"] - self._t0)
        r["temp"].append(snapshot["temp_gpu"])
        r["power"].append(snapshot["power_w"])
        r["util"].append(snapshot["gpu_util_pct"])
        fan = snapshot.get("fan_speed_pct")
        r["fan"].append(np.nan if fan is None else fan)
        r["clock"].append(snapshot["sm_clock_mhz"])
        r["throttle"].append(snapshot.get("throttle_reasons"))

    def append_trace(self, chunk) -> None:
        """A native sampler batch, thinned on arrival to its min and max.

        Keeping every sample for live drawing would mean redrawing hundreds of
        thousands of points four times a second; a batch's extremes are what
        the eye needs, and the final draw uses the full trace.
        """
        seconds, watts = np.asarray(chunk[0]), np.asarray(chunk[1])
        if len(seconds) == 0 or self._t0 is None:
            return
        t, w = envelope(seconds - self._t0, watts, max_points=4)
        self._trace_t.append(t)
        self._trace_w.append(w)

    def append_loss(self, step: int, loss: float) -> None:
        self._loss_steps.append(step)
        self._loss_values.append(loss)

    # --- drawing --------------------------------------------------------------------------

    def refresh(self) -> None:
        """Pushes what has been collected into the curves. Cheap: no redraw."""
        r = self._rows
        t = np.asarray(r["t"], dtype=float)
        self.curves[TEMP].setData(t, np.asarray(r["temp"], dtype=float))
        self.curves[POWER].setData(t, np.asarray(r["power"], dtype=float))
        self.curves[UTIL].setData(t, np.asarray(r["util"], dtype=float))
        self.curves[CLOCK].setData(t, np.asarray(r["clock"], dtype=float))
        fan = np.asarray(r["fan"], dtype=float)
        self._set_fan(t, fan)
        if self._trace_t:
            self._set_instant(np.concatenate(self._trace_t), np.concatenate(self._trace_w))
        self._shade(pd.DataFrame({"time_sec": t, "throttle_reasons": r["throttle"]}))
        self._draw_loss()

    def show_run(self, df: pd.DataFrame, trace: pd.DataFrame | None = None) -> None:
        """The final draw from the logger's complete data, as the export has it."""
        t = df["time_sec"].to_numpy(dtype=float)
        self.curves[TEMP].setData(t, df["temp_gpu"].to_numpy(dtype=float))
        self.curves[POWER].setData(t, df["power_w"].to_numpy(dtype=float))
        self.curves[UTIL].setData(t, df["gpu_util_pct"].to_numpy(dtype=float))
        self.curves[CLOCK].setData(t, df["sm_clock_mhz"].to_numpy(dtype=float))
        fan = (df["fan_speed_pct"].to_numpy(dtype=float) if "fan_speed_pct" in df.columns
               else np.full(len(t), np.nan))
        self._set_fan(t, fan)
        if trace is not None and len(trace):
            self._set_instant(*envelope(trace["time_sec"], trace["power_w"]))
        else:
            self._set_instant(np.array([]), np.array([]))
        self._shade(df)
        self._draw_loss()

    def _set_fan(self, t: np.ndarray, fan: np.ndarray) -> None:
        # Absent on passively cooled cards: no curve at all, not a flat zero.
        if np.isfinite(fan).any():
            self.curves[FAN].setData(t, fan, connect="finite")
            self.curves[FAN].show()
        else:
            self.curves[FAN].setData([], [])
            self.curves[FAN].hide()

    def _set_instant(self, t: np.ndarray, w: np.ndarray) -> None:
        self.curves[POWER_INSTANT].setData(t, w)
        if len(t) and not self._instant_in_legend:
            self._thermal_legend.addItem(self.curves[POWER_INSTANT], POWER_INSTANT)
            self._instant_in_legend = True

    def _draw_loss(self) -> None:
        self.curves[LOSS].setData(self._loss_steps, self._loss_values)
        if self._warmup and self._loss_steps and self._loss_steps[-1] >= self._warmup:
            self.warmup_line.setValue(self._warmup)
            self.warmup_line.show()

    def _shade(self, df: pd.DataFrame) -> None:
        spans = [(start, end, color)
                 for bits, color in ((POWER_LIMIT_BITS, POWER_LIMIT_COLOR),
                                     (THERMAL_LIMIT_BITS, THERMAL_LIMIT_COLOR))
                 for start, end in throttle_spans(df, bits)]
        self._set_spans(spans)

    def _set_spans(self, spans: list[tuple[float, float, str]]) -> None:
        # Rebuilt only when they change: four times a second otherwise.
        if spans == self._spans:
            return
        for item in self._shading:
            self.util_plot.removeItem(item)
        self._shading = []
        for start, end, color in spans:
            brush = pg.mkColor(color)
            brush.setAlpha(40)
            region = pg.LinearRegionItem((start, end), movable=False, brush=brush,
                                         pen=pg.mkPen(None))
            region.setZValue(-10)
            self.util_plot.addItem(region)
            self._shading.append(region)
        self._spans = spans

    def shaded_spans(self) -> list[tuple[float, float, str]]:
        return list(self._spans)

    def loss_data(self) -> tuple[list[int], list[float]]:
        return list(self._loss_steps), list(self._loss_values)

    # --- theme ----------------------------------------------------------------------------

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        for view in (self.telemetry_view, self.loss_view):
            view.setBackground(theme.chart_bg)
        coloured = {(self.thermal_plot, "left"): TEMP_COLOR,
                    (self.thermal_plot, "right"): POWER_COLOR,
                    (self.util_plot, "right"): CLOCK_COLOR}
        for plot in (self.thermal_plot, self.util_plot, self.loss_plot):
            for side in ("left", "bottom", "right"):
                axis = plot.getAxis(side)
                # An axis that names one curve keeps that curve's colour.
                color = coloured.get((plot, side), theme.chart_fg)
                axis.setPen(pg.mkPen(theme.chart_grid))
                axis.setTextPen(pg.mkPen(color))
                if axis.label is not None:
                    axis.label.setDefaultTextColor(pg.mkColor(color))
            legend = plot.legend
            if legend is not None:
                # setLabelTextColor only stores the colour for labels added
                # later; the ones already there keep theirs until re-rendered.
                legend.setLabelTextColor(theme.chart_fg)
                for _, label in legend.items:
                    label.setText(label.text, color=theme.chart_fg)
                legend.setBrush(pg.mkBrush(pg.mkColor(theme.chart_bg)))
                legend.setPen(pg.mkPen(theme.chart_grid))

    def axis_text_color(self, plot: pg.PlotItem, side: str) -> str:
        return plot.getAxis(side).textPen().color().name().upper()
