"""View → History: every exported run, sortable and filterable, one click from its report.

Qt's model/view split rather than a QTableWidget: the model wraps the rows
the history database returns and says how each cell displays and sorts —
numbers sort as numbers, missing values last — and a QSortFilterProxyModel
on top does sorting and filtering without the model knowing. Selected runs
go to the same comparison window File → Compare runs opens.

Under the table, a trend chart plots one figure over time for whatever the
filter leaves, one line per card and model — a driver update or a new
cooler shows up as a step.
"""
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
    QUrl,
)
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from gpu_bench.ui.compare_window import RUN_COLORS
from gpu_bench.ui.style import LIGHT, Palette
from gpu_bench.utils.history import History, HistoryRow

# (header, row attribute, format; None formats as text)
COLUMNS: tuple[tuple[str, str, str | None], ...] = (
    ("When", "created_at", None),
    ("GPU", "gpu", None),
    ("Model", "model", None),
    ("Precision", "precision", None),
    ("Batch", "batch_size", "{:d}"),
    ("Seq", "seq_len", "{:d}"),
    ("Tokens/s", "tokens_per_sec", "{:,.0f}"),
    ("Tokens/J", "tokens_per_joule", "{:.1f}"),
    ("MFU", "mfu", "{:.1%}"),
    ("Max °C", "max_temp_c", "{:.0f}"),
)

# What the trend chart can plot: (label, row attribute, scale for display).
TREND_METRICS: tuple[tuple[str, str, float], ...] = (
    ("Tokens/s", "tokens_per_sec", 1.0),
    ("Tokens/J", "tokens_per_joule", 1.0),
    ("MFU (%)", "mfu", 100.0),
    ("Max temperature (°C)", "max_temp_c", 1.0),
)


def _timestamp(created_at: str) -> float | None:
    try:
        return datetime.fromisoformat(created_at).timestamp()
    except ValueError:
        return None


def trend_series(rows: Sequence[HistoryRow], attribute: str,
                 scale: float = 1.0) -> dict[str, tuple[list[float], list[float]]]:
    """One (times, values) series per card and model, each in time order.

    Rows without the figure, or without a readable time, are left out: a
    run that never measured efficiency has no point on the efficiency line.
    """
    points: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        value = getattr(row, attribute)
        when = _timestamp(row.created_at)
        if value is None or when is None:
            continue
        points.setdefault(f"{row.gpu or '?'} · {row.model or '?'}", []).append(
            (when, float(value) * scale))
    return {name: ([t for t, _ in sorted(p)], [v for _, v in sorted(p)])
            for name, p in sorted(points.items())}


# Sorting reads this role, so 9,800 sorts below 12,000 rather than above it.
SORT_ROLE = Qt.ItemDataRole.UserRole

_Index = QModelIndex | QPersistentModelIndex
# The invalid index, Qt's "no parent": a module-level singleton rather than a
# new QModelIndex() built for every default argument.
_ROOT = QModelIndex()


class HistoryModel(QAbstractTableModel):
    def __init__(self, rows: Sequence[HistoryRow] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows = list(rows)

    def set_rows(self, rows: Sequence[HistoryRow]) -> None:
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent: _Index = _ROOT) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: _Index = _ROOT) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section][0]
        return None

    def data(self, index: _Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        _, attribute, fmt = COLUMNS[index.column()]
        value = getattr(row, attribute)
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None:
                return "—"
            if attribute == "created_at":
                return str(value)[:16].replace("T", " ")
            return fmt.format(value) if fmt else str(value)
        if role == SORT_ROLE:
            # Missing values sort as the lowest of their kind.
            if fmt is None:
                return "" if value is None else str(value)
            return float("-inf") if value is None else float(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and fmt is not None:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ToolTipRole:
            return row.report_path
        return None


class HistoryWindow(QWidget):
    """The history in a table; `compare` receives the reports picked for comparison."""

    def __init__(self, compare: Callable[[list[str]], Any] | None = None,
                 history_path: str | None = None, theme: Palette = LIGHT,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Run history — AI GPU Benchmark")
        self.resize(1100, 820)
        self.compare = compare
        self.history_path = history_path
        self._theme = theme

        self.model = HistoryModel(parent=self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(SORT_ROLE)
        self.proxy.setFilterKeyColumn(-1)                  # every column
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by GPU, model, precision…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._on_filter)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(lambda index: self.open_report(index.row()))
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            # The GPU name is the long one; it takes the slack.
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        vertical = self.table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
        selection = self.table.selectionModel()
        if selection is not None:
            selection.selectionChanged.connect(self._update_buttons)

        self.btn_open = QPushButton("Open report")
        self.btn_open.clicked.connect(lambda: self.open_report())
        self.btn_compare = QPushButton("Compare selected")
        self.btn_compare.setToolTip("Two or more runs side by side")
        self.btn_compare.clicked.connect(self.compare_selected)
        self.btn_remove = QPushButton("Remove from history")
        self.btn_remove.setToolTip("Forget the selected runs; their files are kept")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_import = QPushButton("Import folder…")
        self.btn_import.setToolTip("Add every run report under a folder")
        self.btn_import.clicked.connect(self.import_folder)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("fitHint")

        self.cmb_metric = QComboBox()
        for label, attribute, _ in TREND_METRICS:
            self.cmb_metric.addItem(label, attribute)
        self.cmb_metric.currentIndexChanged.connect(lambda _: self.update_trend())
        self.trend_plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.trend_plot.setMinimumHeight(220)
        self.trend_legend = self.trend_plot.addLegend(offset=(10, 10))
        self.trend_curves: dict[str, pg.PlotDataItem] = {}
        self.proxy.rowsInserted.connect(lambda *_: self.update_trend())
        self.proxy.rowsRemoved.connect(lambda *_: self.update_trend())
        self.proxy.modelReset.connect(self.update_trend)
        self.proxy.layoutChanged.connect(lambda *_: self.update_trend())

        buttons = QHBoxLayout()
        for button in (self.btn_open, self.btn_compare, self.btn_remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_import)

        trend = QWidget()
        trend_layout = QVBoxLayout(trend)
        trend_layout.setContentsMargins(0, 0, 0, 0)
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Trend of"))
        picker.addWidget(self.cmb_metric)
        picker.addStretch(1)
        trend_layout.addLayout(picker)
        trend_layout.addWidget(self.trend_plot, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(trend)
        splitter.setSizes([480, 300])

        layout = QVBoxLayout(self)
        layout.addWidget(self.filter)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.lbl_status)
        self.set_theme(theme)
        self.refresh()

    def _history(self) -> History:
        return History(self.history_path)

    def refresh(self) -> None:
        with self._history() as history:
            self.model.set_rows(history.runs())
            path = history.path
        self.lbl_status.setText(f"{len(self.model.rows)} runs · {path}")
        self._update_buttons()

    def _on_filter(self, text: str) -> None:
        self.proxy.setFilterFixedString(text)
        self._update_buttons()
        self.update_trend()

    def visible_rows(self) -> list[HistoryRow]:
        """The runs the filter leaves, whatever order they are sorted in."""
        return [self.model.rows[self.proxy.mapToSource(self.proxy.index(r, 0)).row()]
                for r in range(self.proxy.rowCount())]

    def update_trend(self) -> None:
        """Redraws the chosen figure over time for the runs on show."""
        index = max(0, self.cmb_metric.currentIndex())
        label, attribute, scale = TREND_METRICS[index]
        for curve in self.trend_curves.values():
            self.trend_plot.removeItem(curve)
        self.trend_legend.clear()
        self.trend_curves = {}
        for i, (name, (times, values)) in enumerate(
                trend_series(self.visible_rows(), attribute, scale).items()):
            color = RUN_COLORS[i % len(RUN_COLORS)]
            self.trend_curves[name] = self.trend_plot.plot(
                times, values, name=name, pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=6, symbolBrush=color, symbolPen=None)
        self.trend_plot.setLabel("left", label)

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        self.trend_plot.setBackground(theme.chart_bg)
        for side in ("left", "bottom"):
            axis = self.trend_plot.getAxis(side)
            axis.setPen(pg.mkPen(theme.chart_grid))
            axis.setTextPen(pg.mkPen(theme.chart_fg))
        self.trend_plot.showGrid(x=True, y=True, alpha=0.2)
        self.trend_legend.setLabelTextColor(theme.chart_fg)
        # Translucent, so the lines stay readable underneath it.
        backdrop = pg.mkColor(theme.chart_bg)
        backdrop.setAlpha(210)
        self.trend_legend.setBrush(pg.mkBrush(backdrop))
        self.update_trend()

    def selected_rows(self) -> list[HistoryRow]:
        """The selected runs, in the order the table shows them."""
        selection = self.table.selectionModel()
        if selection is None:
            return []
        rows = sorted(index.row() for index in selection.selectedRows())
        return [self.model.rows[self.proxy.mapToSource(self.proxy.index(r, 0)).row()]
                for r in rows]

    def _update_buttons(self) -> None:
        count = len(self.selected_rows())
        self.btn_open.setEnabled(count == 1)
        self.btn_compare.setEnabled(count >= 2 and self.compare is not None)
        self.btn_remove.setEnabled(count >= 1)

    def open_report(self, view_row: int | None = None) -> None:
        """Opens a run's HTML page in the browser, or its JSON if there is no page."""
        if view_row is None:
            rows = self.selected_rows()
            if len(rows) != 1:
                return
            row = rows[0]
        else:
            row = self.model.rows[self.proxy.mapToSource(self.proxy.index(view_row, 0)).row()]
        target = row.html_path or row.report_path
        if not os.path.exists(target):
            self.lbl_status.setText(f"{target} no longer exists — remove it from the history, "
                                    "or import the folder it moved to.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def compare_selected(self) -> None:
        rows = self.selected_rows()
        if len(rows) >= 2 and self.compare is not None:
            self.compare([r.report_path for r in rows])

    def remove_selected(self) -> None:
        rows = self.selected_rows()
        with self._history() as history:
            for row in rows:
                history.remove(row.report_path)
        self.refresh()

    def import_folder(self, directory: str | None = None) -> None:
        if directory is None:
            directory = QFileDialog.getExistingDirectory(self, "Import run reports", "logs")
        if not directory:
            return
        with self._history() as history:
            added, unreadable = history.import_dir(directory)
        self.refresh()
        skipped = f", {len(unreadable)} unreadable" if unreadable else ""
        self.lbl_status.setText(f"Imported {added} runs from {directory}{skipped} · "
                                f"{len(self.model.rows)} in the history")
