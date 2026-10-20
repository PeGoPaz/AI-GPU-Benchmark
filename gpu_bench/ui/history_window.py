"""View → History: every exported run, sortable and filterable, one click from its report.

Qt's model/view split rather than a QTableWidget: the model wraps the rows
the history database returns and says how each cell displays and sorts —
numbers sort as numbers, missing values last — and a QSortFilterProxyModel
on top does sorting and filtering without the model knowing. Selected runs
go to the same comparison window File → Compare runs opens.
"""
import os
from collections.abc import Callable, Sequence
from typing import Any

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
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

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
                 history_path: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Run history — AI GPU Benchmark")
        self.resize(1100, 640)
        self.compare = compare
        self.history_path = history_path

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

        buttons = QHBoxLayout()
        for button in (self.btn_open, self.btn_compare, self.btn_remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_import)

        layout = QVBoxLayout(self)
        layout.addWidget(self.filter)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.lbl_status)
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
