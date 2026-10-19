"""The last minute of a reading, as a line under its value.

A single number says where a reading is; the trend says where it is going —
whether the clock is sagging, the fan still ramping, the headroom shrinking.
The scale is the minute's own range, so a two-degree drift is visible, and
the range itself is printed at the end so the line cannot exaggerate it.
"""
from collections import deque

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.ui.style import LIGHT, Palette

# Telemetry arrives every 250 ms: 240 points is one minute.
HISTORY_POINTS = 240


class Sparkline(QWidget):
    LABEL_PX = 9

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme: Palette = LIGHT
        self._color: str | None = None
        self._values: deque[float] = deque(maxlen=HISTORY_POINTS)
        self.setMinimumHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(160, 24)

    def append(self, value: float | None) -> None:
        # A missing reading is skipped rather than drawn as zero: a card that
        # stops reporting its fan has not stopped its fan.
        if value is not None:
            self._values.append(float(value))
            self.update()

    def values(self) -> list[float]:
        return list(self._values)

    def set_color(self, color: str | None) -> None:
        """None follows the theme's accent; a state colour overrides it."""
        self._color = color
        self.update()

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        self.update()

    def range_label(self) -> str:
        if not self._values:
            return ""
        low, high = min(self._values), max(self._values)
        return f"{low:.0f}–{high:.0f}" if high != low else f"{high:.0f}"

    def _label_font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(self.LABEL_PX)
        return font

    def line_rect(self) -> QRectF:
        """Where the line goes: the full width, less room for the range label."""
        label_w = QFontMetrics(self._label_font()).horizontalAdvance(self.range_label())
        return QRectF(1, 3, max(1.0, self.width() - label_w - 8), self.height() - 6)

    def paintEvent(self, event) -> None:
        if len(self._values) < 2:
            return
        t = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.line_rect()

        low, high = min(self._values), max(self._values)
        span = high - low or 1.0      # a flat minute draws a flat, centred line
        mid = (high + low) / 2
        step = rect.width() / (HISTORY_POINTS - 1)
        # Right-aligned: the newest point is always at the right edge, and a
        # history shorter than a minute grows in from the left.
        x0 = rect.right() - step * (len(self._values) - 1)

        def y(value: float) -> float:
            if high == low:
                return rect.center().y() - (value - mid)
            return rect.bottom() - (value - low) / span * rect.height()

        points = [QPointF(x0 + i * step, y(v)) for i, v in enumerate(self._values)]
        line = QPainterPath(points[0])
        for point in points[1:]:
            line.lineTo(point)

        color = QColor(self._color or t.accent)
        fill_path = QPainterPath(line)
        fill_path.lineTo(points[-1].x(), rect.bottom())
        fill_path.lineTo(points[0].x(), rect.bottom())
        fill_path.closeSubpath()
        fill = QColor(color)
        fill.setAlpha(40)
        painter.fillPath(fill_path, fill)

        painter.setPen(QPen(color, 1.5))
        painter.drawPath(line)
        painter.setBrush(color)
        painter.drawEllipse(points[-1], 2.0, 2.0)

        painter.setFont(self._label_font())
        painter.setPen(QColor(t.muted))
        label_x = rect.right() + 4
        painter.drawText(QRectF(label_x, 0, self.width() - label_x, self.height()),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         self.range_label())
        painter.end()
