"""What an area shows before it has anything to show.

The charts and the summary used to be hidden until a run produced them,
which left the right half of a freshly opened window as an empty grey gap
with nothing to say what belonged there or how to fill it.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class EmptyState(QWidget):
    def __init__(self, title: str, hint: str, pixmap: QPixmap | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        self.lbl_icon = QLabel()
        self.lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if pixmap is not None:
            self.lbl_icon.setPixmap(pixmap)
        else:
            self.lbl_icon.hide()
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("emptyTitle")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint = QLabel(hint)
        self.lbl_hint.setObjectName("emptyHint")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint.setWordWrap(True)
        # A centred word-wrapped label shrinks to its narrowest wrap; a
        # minimum width keeps it reading as sentences, not a column.
        self.lbl_hint.setMinimumWidth(320)
        self.lbl_hint.setMaximumWidth(480)

        layout.addWidget(self.lbl_icon)
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_hint, 0, Qt.AlignmentFlag.AlignCenter)

    def set_text(self, title: str, hint: str) -> None:
        self.lbl_title.setText(title)
        self.lbl_hint.setText(hint)

    def title(self) -> str:
        return self.lbl_title.text()

    def hint(self) -> str:
        return self.lbl_hint.text()
