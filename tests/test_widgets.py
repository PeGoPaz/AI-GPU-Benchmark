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



# --- arc gauge ----------------------------------------------------------------------

@pytest.fixture
def gauge(qapp):
    from app.ui.widgets.gauge import ArcGauge
    g = ArcGauge("Temperature", "°C", maximum=100)
    g.resize(140, 140)
    return g


@pytest.mark.parametrize("value, expected", [(0.4, 0.5), (3, 5), (7, 10), (120, 200),
                                             (230, 250), (0, 1)])
def test_scales_round_up_to_readable_numbers(value, expected):
    from app.ui.widgets.gauge import nice_ceiling

    assert nice_ceiling(value) == expected


def test_fraction_is_clamped_to_the_scale(gauge):
    gauge.set_value(150, "150")
    assert gauge.fraction() == 1.0

    gauge.set_value(-5, "-5")
    assert gauge.fraction() == 0.0

    gauge.set_value(None, "N/A")
    assert gauge.fraction() == 0.0


def test_unscaled_gauge_grows_to_fit_and_never_shrinks(qapp):
    from app.ui.widgets.gauge import ArcGauge
    g = ArcGauge("Power", "W")

    g.set_value(180, "180")
    assert g.maximum() == 200
    g.set_value(40, "40")
    assert g.maximum() == 200


def _top_of_arc(gauge):
    """The pixel at 12 o'clock on the arc: halfway along its 270° sweep."""
    rect, _, _ = gauge.arc_geometry()
    image = gauge.grab().toImage()
    return image.pixelColor(int(rect.center().x()), int(rect.top()))


def test_arc_fills_clockwise_from_the_start(gauge):
    from PyQt6.QtGui import QColor

    from app.ui.style import LIGHT

    gauge.set_value(40, "40")
    assert _top_of_arc(gauge) == QColor(LIGHT.track)

    gauge.set_value(60, "60")
    assert _top_of_arc(gauge) == QColor(LIGHT.accent)


def test_arc_takes_the_state_colour(gauge):
    from PyQt6.QtGui import QColor

    from app.ui.style import LIGHT

    gauge.set_value(95, "95", state="crit")

    assert _top_of_arc(gauge) == QColor(LIGHT.crit)



# --- sparkline ----------------------------------------------------------------------

@pytest.fixture
def sparkline(qapp):
    from app.ui.widgets.sparkline import Sparkline
    line = Sparkline()
    line.resize(200, 24)
    return line


def test_missing_readings_are_skipped_not_drawn_as_zero(sparkline):
    for value in (60, None, 61, None, 62):
        sparkline.append(value)

    assert sparkline.values() == [60, 61, 62]


def test_history_is_bounded_to_a_minute(sparkline):
    from app.ui.widgets.sparkline import HISTORY_POINTS

    for i in range(1000):
        sparkline.append(i)

    assert len(sparkline.values()) == HISTORY_POINTS
    assert sparkline.values()[0] == 1000 - HISTORY_POINTS


def test_flat_history_paints_without_dividing_by_zero(sparkline):
    for _ in range(50):
        sparkline.append(70)

    assert not sparkline.grab().toImage().isNull()


def test_newest_point_sits_at_the_right_edge(sparkline):
    from PyQt6.QtGui import QColor

    from app.ui.style import LIGHT

    for i in range(10):
        sparkline.append(i)
    rect = sparkline.line_rect()
    image = sparkline.grab().toImage()

    # The newest value is the maximum, so its end dot sits at the top-right
    # corner. Its centre is solid; the rim around it is antialiased.
    centre = (int(rect.right()), int(rect.top()))
    around = [image.pixelColor(centre[0] + dx, centre[1] + dy)
              for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    assert QColor(LIGHT.accent) in around



def test_range_label_gets_the_room_it_needs(sparkline):
    sparkline.append(2310)
    narrow = sparkline.line_rect().width()
    sparkline.append(2460)

    assert sparkline.range_label() == "2310–2460"
    assert sparkline.line_rect().width() < narrow


# --- icons ----------------------------------------------------------------------------

def test_every_glyph_renders_in_the_requested_colour(qapp):
    from PyQt6.QtGui import QColor

    from app.ui import icons

    for name in icons.GLYPHS:
        image = icons.icon(name, "#ff0000").pixmap(24, 24).toImage()
        reds = [image.pixelColor(x, y) for x in range(24) for y in range(24)
                if image.pixelColor(x, y).alpha() == 255]
        assert reds and all(c == QColor("#ff0000") for c in reds), name
