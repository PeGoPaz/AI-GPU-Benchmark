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
