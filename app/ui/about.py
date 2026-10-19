"""Help → About: what this build is made of, ready to paste into a bug report.

A benchmark result depends on the versions behind it as much as on the card,
and "which torch, which driver, was the native sampler on?" is the first
question about any odd number. The dialog answers it and copies the answer.
"""
import platform
import sys
from pathlib import Path

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, Qt
from PyQt6.QtGui import QGuiApplication, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.utils.report import package_versions

ICON_PATH = Path(__file__).with_name("assets") / "icon.svg"
REPO_URL = "https://github.com/PeGoPaz/AI-GPU-Benchmark"


def app_icon() -> QIcon:
    return QIcon(str(ICON_PATH))


def native_module_status() -> str:
    try:
        import nvml_sampler
    except ImportError:
        return "not built (pip install ./native)"
    info = nvml_sampler.build_info()
    return f"{info['version']}, {info['compiler']}, {info['build_type']}"


def about_rows(device: dict | None = None, sampler: dict | None = None) -> list[tuple[str, str]]:
    """(label, value) pairs: platform, toolkit, ML stack, native module, GPU."""
    versions = package_versions()
    rows = [
        ("Python", f"{platform.python_version()} ({sys.platform})"),
        ("Qt / PyQt", f"{QT_VERSION_STR} / {PYQT_VERSION_STR}"),
    ]
    for name in ("torch", "transformers", "peft", "bitsandbytes", "nvidia-ml-py"):
        rows.append((name, versions.get(name) or "not installed"))
    rows.append(("Native module", native_module_status()))
    if sampler is not None:
        rows.append(("Power sampling", ("native, " if sampler.get("active") else "")
                     + str(sampler.get("detail", ""))))
    if device:
        rows.append(("GPU", str(device.get("name", "unknown"))))
        rows.append(("Driver / CUDA", f"{device.get('driver_version') or '?'} / "
                                      f"{device.get('cuda_driver_version') or '?'}"))
    return rows


def about_text(rows: list[tuple[str, str]]) -> str:
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


class AboutDialog(QDialog):
    def __init__(self, rows: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About AI GPU Benchmark")
        self.setWindowIcon(app_icon())
        self.rows = rows

        icon = QLabel()
        icon.setPixmap(app_icon().pixmap(56, 56))
        title = QLabel("<b style='font-size:16px'>AI GPU Benchmark</b><br>"
                       "LoRA fine-tuning throughput and GPU telemetry")
        link = QLabel(f"<a href='{REPO_URL}'>{REPO_URL}</a>")
        link.setOpenExternalLinks(True)

        form = QFormLayout()
        for label, value in rows:
            field = QLabel(value)
            field.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(f"{label}:", field)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setToolTip("Copy these details, e.g. for a bug report")
        self.btn_copy.clicked.connect(self.copy)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton(self.btn_copy, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(link)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def copy(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(about_text(self.rows))
