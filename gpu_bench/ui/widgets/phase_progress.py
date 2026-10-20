"""A progress bar that knows a run has phases.

A plain percentage hid the two things worth knowing about a run in progress:
whether the model is still loading (which can take longer than the run on a
cold cache) and whether the steps going by are warm-up — executed but not
timed — or the measured ones. The bar draws the run's plan as two segments,
warm-up then measured, fills them as steps complete, and animates while the
model loads, before there is any step to count.
"""
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from gpu_bench.ui.style import LIGHT, Palette

IDLE, LOADING, WARMUP, MEASURING, DONE = "idle", "loading", "warm-up", "measuring", "done"


class PhaseProgressBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._theme: Palette = LIGHT
        self._warmup = 0
        self._total = 0
        self._step = 0
        self._loading = False
        self._finished = False
        self._stripe_offset = 0.0
        self._animation = QTimer(self)
        self._animation.setInterval(40)
        self._animation.timeout.connect(self._advance_stripes)

    # --- driving it ---------------------------------------------------------------

    def start(self, warmup_steps: int, measured_steps: int) -> None:
        """A new run: the plan is known, the first step is not here yet."""
        self._warmup = max(0, warmup_steps)
        self._total = self._warmup + max(0, measured_steps)
        self._step = 0
        self._finished = False
        self._loading = True
        self._animation.start()
        self.update()

    def set_total(self, total_steps: int) -> None:
        """The trainer's own count, warm-up included, once it has one."""
        self._total = max(total_steps, self._warmup)
        self.update()

    def set_step(self, step: int) -> None:
        if self._loading:
            self._loading = False
            self._animation.stop()
        self._step = min(step, self._total) if self._total else step
        self.update()

    def finish(self) -> None:
        self._loading = False
        self._animation.stop()
        self._finished = True
        self.update()

    def reset(self) -> None:
        self._loading = False
        self._animation.stop()
        self._finished = False
        self._warmup = self._total = self._step = 0
        self.update()

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        self.update()

    # --- reading it ----------------------------------------------------------------

    def phase(self) -> str:
        if self._loading:
            return LOADING
        if self._finished:
            return DONE
        if self._total == 0:
            return IDLE
        if self._step < self._warmup:
            return WARMUP
        return MEASURING

    def text(self) -> str:
        phase = self.phase()
        measured_total = self._total - self._warmup
        measured_done = max(0, self._step - self._warmup)
        if phase == IDLE:
            return ""
        if phase == LOADING:
            return "Loading model…"
        if phase == WARMUP:
            return f"Warm-up {self._step} / {self._warmup}"
        pct = 100 * measured_done / measured_total if measured_total else 100
        prefix = "Done" if phase == DONE else "Measuring"
        return f"{prefix} {measured_done} / {measured_total} · {pct:.0f}%"

    # --- painting -----------------------------------------------------------------

    def _advance_stripes(self) -> None:
        self._stripe_offset = (self._stripe_offset + 1.5) % 24
        self.update()

    def paintEvent(self, event) -> None:
        t = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2

        track = QPainterPath()
        track.addRoundedRect(rect, radius, radius)
        painter.setClipPath(track)
        painter.fillRect(rect, QColor(t.track))

        if self._loading:
            self._paint_stripes(painter, rect)
        elif self._total:
            warm_w = rect.width() * self._warmup / self._total
            warm_done = rect.width() * min(self._step, self._warmup) / self._total
            measured_done = rect.width() * max(0, self._step - self._warmup) / self._total

            # The plan, faintly: where warm-up ends is visible before it does.
            plan = QColor(t.warning)
            plan.setAlpha(45)
            painter.fillRect(QRectF(rect.left(), rect.top(), warm_w, rect.height()), plan)

            painter.fillRect(QRectF(rect.left(), rect.top(), warm_done, rect.height()),
                             QColor(t.warning))
            painter.fillRect(QRectF(rect.left() + warm_w, rect.top(), measured_done,
                                    rect.height()), QColor(t.ok))

        painter.setClipping(False)
        painter.setPen(QPen(QColor(t.border), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(track)

        painter.setPen(QColor(t.text_strong))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()

    def _paint_stripes(self, painter: QPainter, rect: QRectF) -> None:
        stripe = QColor(self._theme.accent)
        stripe.setAlpha(70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(stripe))
        h = rect.height()
        x = rect.left() - h + self._stripe_offset
        while x < rect.right():
            path = QPainterPath()
            path.moveTo(x, rect.bottom())
            path.lineTo(x + 12, rect.bottom())
            path.lineTo(x + 12 + h, rect.top())
            path.lineTo(x + h, rect.top())
            path.closeSubpath()
            painter.drawPath(path)
            x += 24
