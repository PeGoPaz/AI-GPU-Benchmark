"""The strip across the top: which card, which driver, what the run is doing.

The GPU used to appear only in the window title, the driver in a console
line, and the run's state nowhere at all. A benchmark result is meaningless
without the first two, and the third is the first thing anyone glancing at
the window wants to know.
"""
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gpu_bench.ui.style import LIGHT, Palette
from gpu_bench.ui.widgets.chip import Chip

IDLE = "Idle"
LOADING = "Loading"
WARMUP = "Warm-up"
MEASURING = "Measuring"
STOPPING = "Stopping"
DONE = "Done"
STOPPED = "Stopped"
FAILED = "Failed"


def device_details(info: dict) -> str:
    """'8.0 GiB · driver 580.82 · CUDA 13.0 · 115 W limit · PCIe Gen4 x8', skipping gaps."""
    parts = []
    if info.get("vram_total_mb"):
        parts.append(f"{info['vram_total_mb'] / 1024:.1f} GiB")
    if info.get("driver_version"):
        parts.append(f"driver {info['driver_version']}")
    if info.get("cuda_driver_version"):
        parts.append(f"CUDA {info['cuda_driver_version']}")
    if info.get("power_limit_w"):
        parts.append(f"{info['power_limit_w']:.0f} W limit")
    if info.get("pcie_gen_max") and info.get("pcie_width_max"):
        parts.append(f"PCIe Gen{info['pcie_gen_max']} x{info['pcie_width_max']}")
    return " · ".join(parts)


class HeaderBar(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("header")
        self._theme: Palette = LIGHT
        self._status = IDLE
        self._sampler_active: bool | None = None

        self.lbl_app = QLabel("AI GPU BENCHMARK")
        self.lbl_app.setObjectName("headerApp")
        self.lbl_gpu = QLabel("Detecting GPU…")
        self.lbl_gpu.setObjectName("headerGpu")
        self.lbl_details = QLabel("")
        self.lbl_details.setObjectName("headerDetails")

        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(self.lbl_app)
        text.addWidget(self.lbl_gpu)
        text.addWidget(self.lbl_details)

        self.sampler_chip = Chip()
        self.sampler_chip.setToolTip("How power is sampled: the native C++ sampler "
                                     "(instant readings every few ms) or 250 ms polls.")
        self.sampler_chip.hide()
        self.status_chip = Chip()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.addLayout(text, 1)
        layout.addWidget(self.sampler_chip)
        layout.addSpacing(8)
        layout.addWidget(self.status_chip)
        self._refresh_chips()

    # --- inputs -----------------------------------------------------------------------

    def set_gpu_name(self, name: str) -> None:
        self.lbl_gpu.setText(name)

    def set_gpu_unavailable(self, reason: str) -> None:
        self.lbl_gpu.setText("No NVIDIA GPU detected")
        self.lbl_details.setText(reason)

    def set_device_info(self, info: dict) -> None:
        if info.get("name"):
            self.lbl_gpu.setText(info["name"])
        self.lbl_details.setText(device_details(info))

    def set_sampler(self, status: dict) -> None:
        self._sampler_active = bool(status.get("active"))
        self.sampler_chip.setToolTip(status.get("detail", ""))
        self.sampler_chip.show()
        self._refresh_chips()

    def set_status(self, status: str) -> None:
        self._status = status
        self._refresh_chips()

    def set_theme(self, theme: Palette) -> None:
        self._theme = theme
        self._refresh_chips()

    # --- outputs ----------------------------------------------------------------------

    def status(self) -> str:
        return self._status

    def _status_color(self) -> str:
        t = self._theme
        return {
            LOADING: t.accent, WARMUP: t.warning, MEASURING: t.ok, STOPPING: t.warning,
            DONE: t.ok, STOPPED: t.warning, FAILED: t.crit,
        }.get(self._status, t.muted)

    def _refresh_chips(self) -> None:
        self.status_chip.set(self._status, self._status_color())
        if self._sampler_active is not None:
            if self._sampler_active:
                self.sampler_chip.set("Native sampler", self._theme.ok)
            else:
                self.sampler_chip.set("250 ms polls", self._theme.muted)
