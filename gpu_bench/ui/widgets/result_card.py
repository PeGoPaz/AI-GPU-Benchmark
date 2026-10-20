"""A headline result: one of the few numbers worth reading first.

A tile with a bigger value and an accent colour, so the finished run's
throughput and efficiency stand out from the details table beneath them.
"""
from PyQt6.QtWidgets import QWidget

from gpu_bench.ui.readings import Reading
from gpu_bench.ui.widgets.tile import MetricTile
from gpu_bench.utils.display import Headline


class ResultCard(MetricTile):
    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent=parent)
        self.setObjectName("resultCard")
        self.lbl_value.setObjectName("resultValue")

    def set_headline(self, headline: Headline) -> None:
        self.lbl_title.setText(headline.label.upper())
        self.set_reading(Reading(headline.value, headline.unit, headline.caption))

    def title_text(self) -> str:
        return self.lbl_title.text()
