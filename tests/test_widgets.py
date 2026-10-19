"""Dashboard widgets on their own, outside the window."""
import pytest

pytestmark = pytest.mark.ui


@pytest.fixture
def phase_bar(qapp):
    from app.ui.widgets.phase_progress import PhaseProgressBar
    bar = PhaseProgressBar()
    bar.resize(400, 28)
    yield bar
    bar.reset()


def _pixel(widget, x_fraction):
    image = widget.grab().toImage()
    return image.pixelColor(int(image.width() * x_fraction), image.height() // 2)


# --- phase progress bar ------------------------------------------------------------

def test_phase_bar_starts_idle_and_silent(phase_bar):
    assert phase_bar.phase() == "idle"
    assert phase_bar.text() == ""


def test_phase_bar_animates_until_the_first_step(phase_bar):
    phase_bar.start(warmup_steps=10, measured_steps=90)
    assert phase_bar.phase() == "loading"
    assert phase_bar._animation.isActive()

    phase_bar.set_step(1)

    assert phase_bar.phase() == "warm-up"
    assert not phase_bar._animation.isActive()


def test_phase_bar_counts_warmup_and_measured_steps_separately(phase_bar):
    phase_bar.start(warmup_steps=10, measured_steps=90)
    phase_bar.set_step(4)
    assert phase_bar.text() == "Warm-up 4 / 10"

    phase_bar.set_step(55)
    assert phase_bar.text() == "Measuring 45 / 90 · 50%"

    phase_bar.finish()
    assert phase_bar.text() == "Done 45 / 90 · 50%"


def test_no_warmup_goes_straight_to_measuring(phase_bar):
    phase_bar.start(warmup_steps=0, measured_steps=50)
    phase_bar.set_step(1)

    assert phase_bar.phase() == "measuring"


def test_warmup_and_measured_segments_are_drawn_in_their_own_colours(phase_bar):
    from PyQt6.QtGui import QColor

    from app.ui.style import LIGHT

    phase_bar.start(warmup_steps=50, measured_steps=50)
    phase_bar.set_step(100)

    assert _pixel(phase_bar, 0.10) == QColor(LIGHT.warning)
    assert _pixel(phase_bar, 0.90) == QColor(LIGHT.ok)


def test_theme_switch_recolours_the_bar(phase_bar):
    from PyQt6.QtGui import QColor

    from app.ui.style import DARK

    phase_bar.start(warmup_steps=0, measured_steps=10)
    phase_bar.set_step(10)
    phase_bar.set_theme(DARK)

    assert _pixel(phase_bar, 0.5) == QColor(DARK.ok)


# --- header bar ---------------------------------------------------------------------

@pytest.fixture
def header_bar(qapp):
    from app.ui.widgets.header import HeaderBar
    return HeaderBar()


def test_device_details_skip_what_the_driver_did_not_say():
    from app.ui.widgets.header import device_details

    full = {"vram_total_mb": 8188.0, "driver_version": "580.82",
            "cuda_driver_version": "13.0", "power_limit_w": 115.0,
            "pcie_gen_max": 4, "pcie_width_max": 8}
    assert device_details(full) == "8.0 GiB · driver 580.82 · CUDA 13.0 · 115 W limit · PCIe Gen4 x8"
    assert device_details({"driver_version": "580.82"}) == "driver 580.82"
    assert device_details({}) == ""


def test_header_shows_the_card_it_was_told_about(header_bar):
    header_bar.set_device_info({"name": "NVIDIA GeForce RTX 4060", "driver_version": "580.82"})

    assert header_bar.lbl_gpu.text() == "NVIDIA GeForce RTX 4060"
    assert header_bar.lbl_details.text() == "driver 580.82"


def test_status_chip_colour_follows_the_state(header_bar):
    from app.ui.style import LIGHT
    from app.ui.widgets import header

    header_bar.set_status(header.MEASURING)
    assert header_bar.status_chip.text() == "Measuring"
    assert header_bar.status_chip.color() == LIGHT.ok

    header_bar.set_status(header.FAILED)
    assert header_bar.status_chip.color() == LIGHT.crit


def test_sampler_chip_appears_once_the_mode_is_known(header_bar):
    assert header_bar.sampler_chip.isHidden()

    header_bar.set_sampler({"active": False, "detail": "native module not built"})

    assert not header_bar.sampler_chip.isHidden()
    assert header_bar.sampler_chip.text() == "250 ms polls"
    assert header_bar.sampler_chip.toolTip() == "native module not built"


def test_chip_grows_with_its_text(qapp):
    from app.ui.widgets.chip import Chip

    short, long = Chip("Idle"), Chip("Measuring transients")

    assert long.sizeHint().width() > short.sizeHint().width()
