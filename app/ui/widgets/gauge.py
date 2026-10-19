"""A 270° arc gauge for the readings that have a natural full scale.

Temperature, power, VRAM and utilization each have a ceiling — 100 °C, the
card's power limit, its memory, 100 % — and how close a reading sits to it
matters more than its digits. An arc shows that at a glance; the number in
the middle keeps the precision.
"""
import math

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.ui.readings import CRIT, NORMAL, WARN
from app.ui.style import LIGHT, Palette

START_DEG = 225     # 7:30 on a clock face
SPAN_DEG = -270     # clockwise round to 4:30


def nice_ceiling(value: float) -> float:
    """Rounds a scale maximum up to 1, 2 or 5 × 10^n, so the scale reads cleanly."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


class ArcGauge(QWidget):
    def __init__(self, title: str, unit: str, maximum: float | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme: Palette = LIGHT
        self._title = title
        self._unit = unit
        self._value: float | None = None
        self._text = "--"
        self._caption = ""
        self._state = NORMAL
        # None: grow to fit what has been seen, for a scale nobody reported.
        self._auto = maximum is None
        self._maximum = maximum if maximum is not None else 1.0
        self.setMinimumSize(88, 124)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(110, 132)

    # --- inputs -----------------------------------------------------------------------

    def set_value(self, value: float | None, text: str, state: str = NORMAL,
                  caption: str = "") -> None:
        self._value = value
        self._text = text
        self._state = state
        self._caption = caption
        if self._auto and value is not None and value > self._maximum:
            self._maximum = nice_ceiling(value * 1.1)
        self.update()

    def set_maximum(self, maximum: float) -> None:
        """A known full scale — the card's power limit, its total VRAM."""
        self._auto = False
        self._maximum = max(maximum, 1e-9)
        self.update()

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        self.update()

    # --- outputs ----------------------------------------------------------------------

    def fraction(self) -> float:
        if self._value is None:
            return 0.0
        return min(1.0, max(0.0, self._value / self._maximum))

    def maximum(self) -> float:
        return self._maximum

    def text(self) -> str:
        return self._text

    def caption(self) -> str:
        return self._caption

    def state(self) -> str:
        return self._state

    def color(self) -> str:
        t = self._theme
        return {WARN: t.warning, CRIT: t.crit}.get(self._state, t.accent)

    # --- painting ---------------------------------------------------------------------

    # Below the arc: the title, then the caption ("peak 262 W"), which is
    # far too long to fit inside the arc on a gauge this size.
    LABELS_H = 30

    def arc_geometry(self) -> tuple[QRectF, float, float]:
        """(the arc's bounding square, stroke thickness, side) for this size."""
        side = min(self.width(), self.height() - self.LABELS_H) - 8
        thickness = max(6.0, side * 0.09)
        rect = QRectF((self.width() - side) / 2, 6, side, side).adjusted(
            thickness / 2, thickness / 2, -thickness / 2, -thickness / 2)
        return rect, thickness, side

    def paintEvent(self, event) -> None:
        t = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect, thickness, side = self.arc_geometry()

        pen = QPen(QColor(t.track), thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, START_DEG * 16, SPAN_DEG * 16)

        if self._value is not None and self.fraction() > 0:
            pen.setColor(QColor(self.color()))
            painter.setPen(pen)
            painter.drawArc(rect, START_DEG * 16, int(SPAN_DEG * 16 * self.fraction()))

        value_font = QFont(self.font())
        value_font.setBold(True)
        value_font.setPixelSize(max(12, int(side * 0.24)))
        painter.setFont(value_font)
        painter.setPen(QColor(t.text_strong if self._state == NORMAL else self.color()))
        value_rect = QRectF(rect.left(), rect.center().y() - side * 0.2, rect.width(), side * 0.32)
        painter.drawText(value_rect, Qt.AlignmentFlag.AlignCenter, self._text)

        small = QFont(self.font())
        small.setPixelSize(max(9, int(side * 0.1)))
        painter.setFont(small)
        painter.setPen(QColor(t.muted))
        unit_rect = QRectF(rect.left(), value_rect.bottom(), rect.width(), side * 0.15)
        painter.drawText(unit_rect, Qt.AlignmentFlag.AlignCenter, self._unit)

        labels_top = self.height() - self.LABELS_H
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPixelSize(10)
        painter.setFont(title_font)
        painter.drawText(QRectF(0, labels_top, self.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, self._title.upper())
        if self._caption:
            caption_font = QFont(self.font())
            caption_font.setPixelSize(10)
            painter.setFont(caption_font)
            caption = painter.fontMetrics().elidedText(
                self._caption, Qt.TextElideMode.ElideRight, self.width() - 4)
            painter.drawText(QRectF(0, labels_top + 14, self.width(), 14),
                             Qt.AlignmentFlag.AlignCenter, caption)
        painter.end()
