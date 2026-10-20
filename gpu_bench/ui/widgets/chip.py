"""A small rounded label in a state colour: run status, sampler mode.

Painted rather than styled: its colour follows a value that changes several
times a run, and a per-instance stylesheet rewritten on every change is
exactly what style.py was introduced to get rid of.
"""
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class Chip(QWidget):
    PAD_X, PAD_Y = 10, 3

    def __init__(self, text: str = "", color: str = "#888888",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._color = color
        font = self.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 0.9)
        self.setFont(font)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set(self, text: str, color: str) -> None:
        if (text, color) != (self._text, self._color):
            self._text, self._color = text, color
            self.updateGeometry()
            self.update()

    def text(self) -> str:
        return self._text

    def color(self) -> str:
        return self._color

    def sizeHint(self) -> QSize:
        metrics = QFontMetrics(self.font())
        return QSize(metrics.horizontalAdvance(self._text) + 2 * self.PAD_X,
                     metrics.height() + 2 * self.PAD_Y)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        color = QColor(self._color)
        fill = QColor(color)
        fill.setAlpha(40)
        painter.setPen(QPen(color, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()
