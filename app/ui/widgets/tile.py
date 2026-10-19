"""One telemetry reading as a card: title, a big value, its unit, a caption.

The value's colour carries the state — normal, warn, crit — through a dynamic
property the stylesheet selects on, so the thresholds live in readings.py
and the colours in style.py, and this widget knows neither.
"""
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.ui.readings import NORMAL, Reading


class MetricTile(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tile")
        self.setProperty("state", NORMAL)

        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setObjectName("tileTitle")
        self.lbl_value = QLabel("--")
        self.lbl_value.setObjectName("tileValue")
        self.lbl_unit = QLabel("")
        self.lbl_unit.setObjectName("tileUnit")
        self.lbl_caption = QLabel("")
        self.lbl_caption.setObjectName("tileCaption")

        value_row = QHBoxLayout()
        value_row.setSpacing(4)
        value_row.addWidget(self.lbl_value)
        value_row.addWidget(self.lbl_unit)
        value_row.addStretch()

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._layout.setSpacing(2)
        self._layout.addWidget(self.lbl_title)
        self._layout.addLayout(value_row)
        self._layout.addWidget(self.lbl_caption)

    def set_reading(self, reading: Reading) -> None:
        self.lbl_value.setText(reading.value)
        self.lbl_unit.setText(reading.unit)
        self.lbl_caption.setText(reading.caption)
        self.setToolTip(f"{reading.value} {reading.unit}  {reading.caption}".strip())
        # Re-polished only on a change: readings arrive four times a second.
        if self.property("state") != reading.state:
            self.setProperty("state", reading.state)
            for widget in (self, self.lbl_value):
                style = widget.style()
                if style is not None:
                    style.unpolish(widget)
                    style.polish(widget)

    def value_text(self) -> str:
        return self.lbl_value.text()

    def caption_text(self) -> str:
        return self.lbl_caption.text()

    def state(self) -> str:
        return str(self.property("state"))
