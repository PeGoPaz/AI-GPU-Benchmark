"""File → Compare runs: several exported runs side by side.

One run's numbers mean little alone; against a baseline — the same card
yesterday, another card, another precision — they start to. This reads the
JSON reports export writes, lines their results up in one table with the
best value of each comparable row marked, and overlays their power and
temperature curves from the CSVs the reports name.
"""
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gpu_bench.ui.style import LIGHT, Palette

# One colour per run, distinct in both themes.
RUN_COLORS = ("#4c72ff", "#ff8c00", "#2e8b57", "#d64545", "#a06bff", "#00a6c8")

HIGHER, LOWER, NEUTRAL = 1, -1, 0

# (label, results key, format, which way is better)
METRICS = (
    ("Tokens/sec", "tokens_per_sec", "{:,.0f}", HIGHER),
    ("Compute (est. TFLOPS)", "est_tflops", "{:.1f}", HIGHER),
    ("MFU", "mfu", "{:.1%}", HIGHER),
    ("Efficiency (tokens/J)", "tokens_per_joule", "{:.1f}", HIGHER),
    ("Steps/sec", "steps_per_sec", "{:.2f}", HIGHER),
    ("Peak power, instant (W)", "native_peak_power_w", "{:.0f}", NEUTRAL),
    ("Peak power, polls (W)", "peak_power_w", "{:.0f}", NEUTRAL),
    ("Avg power (W)", "avg_power_w", "{:.0f}", NEUTRAL),
    ("Max temp (°C)", "max_temp_c", "{:.0f}", LOWER),
    ("Time power-limited (%)", "power_limited_pct", "{:.0f}", LOWER),
    ("Time thermal-limited (%)", "thermal_limited_pct", "{:.0f}", LOWER),
    ("Duration (s)", "elapsed_time_sec", "{:.1f}", NEUTRAL),
)


@dataclass
class Run:
    path: str
    report: dict[str, Any]
    telemetry: pd.DataFrame | None
    note: str = ""          # e.g. why the curves are missing

    @property
    def label(self) -> str:
        workload = self.report.get("workload") or {}
        gpu = (self.report.get("gpu") or {}).get("name") or "unknown GPU"
        return " · ".join(str(p) for p in (workload.get("model"), workload.get("precision"), gpu)
                          if p)


def load_run(path: str) -> Run:
    """Reads a report and the telemetry CSV it names. Raises ValueError with
    a readable reason when the report itself cannot be used."""
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except OSError as e:
        raise ValueError(f"{os.path.basename(path)}: {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{os.path.basename(path)}: not valid JSON ({e.msg})") from e
    if not isinstance(report, dict) or "results" not in report:
        raise ValueError(f"{os.path.basename(path)}: not a run report")

    csv_name = (report.get("files") or {}).get("csv")
    if not csv_name:
        return Run(path, report, None, "no telemetry CSV named")
    csv_path = os.path.join(os.path.dirname(path), csv_name)
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError):
        return Run(path, report, None, f"{csv_name} not found or unreadable")
    if "time_sec" not in df.columns:
        df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return Run(path, report, df)


def load_runs(paths: list[str]) -> tuple[list[Run], list[str]]:
    runs, errors = [], []
    for path in paths:
        try:
            runs.append(load_run(path))
        except ValueError as e:
            errors.append(str(e))
    return runs, errors


def comparison_rows(runs: list[Run]) -> list[tuple[str, list[str], int | None]]:
    """(metric, one formatted value per run, index of the best or None).

    Rows no run reports are dropped. "Best" is only marked where a direction
    makes sense and at least two runs report the value — a single number is
    not better than anything.
    """
    rows = []
    for label, key, fmt, direction in METRICS:
        raw = [(run.report.get("results") or {}).get(key) for run in runs]
        if all(v is None for v in raw):
            continue
        values = [fmt.format(v) if v is not None else "—" for v in raw]
        known = [(v, i) for i, v in enumerate(raw) if v is not None]
        best = None
        if direction != NEUTRAL and len(known) >= 2:
            best = (max(known) if direction == HIGHER else min(known))[1]
        rows.append((label, values, best))
    return rows


class CompareWindow(QWidget):
    def __init__(self, runs: list[Run], errors: list[str] | None = None,
                 theme: Palette = LIGHT, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(f"Compare {len(runs)} runs — AI GPU Benchmark")
        self.resize(1100, 760)
        self.runs = runs
        self._theme = theme

        layout = QVBoxLayout(self)
        self.lbl_errors = QLabel("\n".join(f"Skipped {e}" for e in (errors or [])))
        self.lbl_errors.setObjectName("fitHint")
        self.lbl_errors.setProperty("warning", True)
        self.lbl_errors.setVisible(bool(errors))
        self.lbl_errors.setWordWrap(True)
        layout.addWidget(self.lbl_errors)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._fill_table()

        self.plots = pg.GraphicsLayoutWidget()
        self.power_plot = self.plots.addPlot(row=0, col=0, title="Power (W)")
        self.temp_plot = self.plots.addPlot(row=1, col=0, title="Temperature (°C)")
        self.temp_plot.setXLink(self.power_plot)
        self.temp_plot.setLabel("bottom", "Time (seconds)")
        self.curves: dict[str, list[pg.PlotDataItem]] = {"power": [], "temp": []}
        self._fill_plots()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.plots)
        splitter.setSizes([300, 460])
        layout.addWidget(splitter, 1)
        self.set_theme(theme)

    def _fill_table(self) -> None:
        rows = comparison_rows(self.runs)
        self.table.setColumnCount(len(self.runs))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(
            [f"{i + 1}. {run.label}" for i, run in enumerate(self.runs)])
        self.table.setVerticalHeaderLabels([label for label, _, _ in rows])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bold = QFont(self.font())
        bold.setBold(True)
        for r, (_, values, best) in enumerate(rows):
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == best:
                    item.setFont(bold)
                    item.setForeground(QBrush(QColor(self._theme.ok)))
                    item.setToolTip("best of these runs")
                self.table.setItem(r, c, item)
        self.best = {label: best for label, _, best in rows}

    def _fill_plots(self) -> None:
        legend = self.power_plot.addLegend(offset=(10, 10))
        for i, run in enumerate(self.runs):
            color = RUN_COLORS[i % len(RUN_COLORS)]
            name = f"{i + 1}. {run.label}"
            if run.telemetry is None:
                name += f" ({run.note})"
                legend.addItem(pg.PlotDataItem(pen=pg.mkPen(color)), name)
                continue
            df = run.telemetry
            pen = pg.mkPen(color, width=2)
            self.curves["power"].append(
                self.power_plot.plot(df["time_sec"].to_numpy(), df["power_w"].to_numpy(),
                                     pen=pen, name=name))
            self.curves["temp"].append(
                self.temp_plot.plot(df["time_sec"].to_numpy(), df["temp_gpu"].to_numpy(),
                                    pen=pen))

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        self.plots.setBackground(theme.chart_bg)
        for plot in (self.power_plot, self.temp_plot):
            for side in ("left", "bottom"):
                axis = plot.getAxis(side)
                axis.setPen(pg.mkPen(theme.chart_grid))
                axis.setTextPen(pg.mkPen(theme.chart_fg))
            plot.setTitle(plot.titleLabel.text, color=theme.chart_fg)
            plot.showGrid(x=True, y=True, alpha=0.2)
