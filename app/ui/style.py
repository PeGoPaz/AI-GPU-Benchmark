"""Application styling, kept out of the layout code.

Two reasons this is not a pile of setStyleSheet() calls next to each widget.
Appearance is readable in one place without tracing the layout, and widget
state is expressed as a selector rather than as code that rewrites a
stylesheet on every transition — start_benchmark() used to swap the Start
button's sheet to grey and both the finish and error paths had to swap it
back.

A theme is a Palette. It drives three things that each need telling
separately: the QPalette Fusion paints standard widgets with, the stylesheet
for the custom-styled ones, and the matplotlib colours of the charts.
"""
from dataclasses import dataclass

from PyQt6.QtGui import QColor, QPalette

START_GREEN = "#2E8B57"
START_GREEN_HOVER = "#379E64"
START_GREEN_PRESSED = "#256F45"

STOP_RED = "#8B0000"
STOP_RED_HOVER = "#A50000"
STOP_RED_PRESSED = "#6E0000"

EXPORT_BLUE = "#4682B4"
EXPORT_BLUE_HOVER = "#5490C4"
EXPORT_BLUE_PRESSED = "#3A6E99"

CONSOLE_BG = "#1E1E1E"
CONSOLE_FG = "#00FF00"


@dataclass(frozen=True)
class Palette:
    """Every colour that differs between the light and the dark theme.

    window/base/text are None for the light theme: it keeps Fusion's own
    standard palette rather than an imitation of it.
    """

    name: str
    disabled_bg: str
    disabled_fg: str
    muted: str
    warning: str
    chart_bg: str
    chart_fg: str
    chart_grid: str
    # Dashboard tokens: cards sit a step above the window on `surface`,
    # outlined in `border`; `track` is the unfilled part of bars and gauges.
    # ok / warning / crit colour state, accent marks what is interactive.
    surface: str
    border: str
    track: str
    text_strong: str
    ok: str
    crit: str
    accent: str
    window: str | None = None
    base: str | None = None
    text: str | None = None
    highlight: str | None = None


LIGHT = Palette(
    name="light",
    disabled_bg="#3C3C3C", disabled_fg="#8A8A8A",
    muted="#6E6E6E",
    # Darker than the dark theme's amber: #D08A00 on a near-white window
    # falls short of readable contrast.
    warning="#B36B00",
    chart_bg="#FFFFFF", chart_fg="#222222", chart_grid="#B0B0B0",
    surface="#FFFFFF", border="#D5D5D5", track="#E4E4E4", text_strong="#1A1A1A",
    ok="#2E8B57", crit="#C62828", accent="#3A6EA5",
)

DARK = Palette(
    name="dark",
    disabled_bg="#454545", disabled_fg="#9A9A9A",
    muted="#A0A0A0",
    warning="#E0A030",
    chart_bg="#232323", chart_fg="#DDDDDD", chart_grid="#5A5A5A",
    surface="#343434", border="#474747", track="#3E3E3E", text_strong="#F2F2F2",
    ok="#3CB371", crit="#E05252", accent="#5B9BD5",
    window="#2B2B2B", base="#1F1F1F", text="#E6E6E6", highlight="#3A6E99",
)

THEMES = {theme.name: theme for theme in (LIGHT, DARK)}


def build_qpalette(theme: Palette, standard: QPalette) -> QPalette:
    """The QPalette for a theme; the light one is Fusion's standard palette."""
    if theme.window is None:
        return QPalette(standard)

    window, base, text = QColor(theme.window), QColor(theme.base), QColor(theme.text)
    palette = QPalette(standard)
    role = QPalette.ColorRole
    for r, color in ((role.Window, window), (role.Button, window),
                     (role.Base, base), (role.AlternateBase, window),
                     (role.ToolTipBase, base),
                     (role.WindowText, text), (role.Text, text),
                     (role.ButtonText, text), (role.ToolTipText, text),
                     (role.Highlight, QColor(theme.highlight)),
                     (role.HighlightedText, QColor("#FFFFFF"))):
        palette.setColor(r, color)

    muted = QColor(theme.disabled_fg)
    for r in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, r, muted)
    return palette


def build_stylesheet(theme: Palette) -> str:
    return f"""
/* Declaring a border switches Qt to the stylesheet box model, so the fill is
   exactly the colour named below. Left to the native Fusion primitive, the
   button keeps a gradient over it and #2E8B57 renders as #5BB783. */
QPushButton#btnStart, QPushButton#btnStop, QPushButton#btnExport {{
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    color: white;
}}

QPushButton#btnStart, QPushButton#btnStop {{
    font-weight: bold;
}}

QPushButton#btnStart {{ background-color: {START_GREEN}; }}
QPushButton#btnStart:hover {{ background-color: {START_GREEN_HOVER}; }}
QPushButton#btnStart:pressed {{ background-color: {START_GREEN_PRESSED}; }}

QPushButton#btnStop {{ background-color: {STOP_RED}; }}
QPushButton#btnStop:hover {{ background-color: {STOP_RED_HOVER}; }}
QPushButton#btnStop:pressed {{ background-color: {STOP_RED_PRESSED}; }}

QPushButton#btnExport {{ background-color: {EXPORT_BLUE}; }}
QPushButton#btnExport:hover {{ background-color: {EXPORT_BLUE_HOVER}; }}
QPushButton#btnExport:pressed {{ background-color: {EXPORT_BLUE_PRESSED}; }}

/* Spelled out per button on purpose: a bare QPushButton:disabled loses to
   the id selectors above under CSS specificity, which is exactly how the
   disabled state went unstyled before. Qt's own disabled rendering only dims
   the label text, and on a saturated background that read as a live button. */
QPushButton#btnStart:disabled,
QPushButton#btnStop:disabled,
QPushButton#btnExport:disabled {{
    background-color: {theme.disabled_bg};
    color: {theme.disabled_fg};
}}

QLabel#fitHint, QLabel#progressInfo {{
    color: {theme.muted};
}}

QLabel#fitHint[warning="true"] {{
    color: {theme.warning};
    font-weight: bold;
}}

QFrame#tile {{
    background-color: {theme.surface};
    border: 1px solid {theme.border};
    border-radius: 8px;
}}

QLabel#tileTitle {{
    color: {theme.muted};
    font-size: 10px;
    font-weight: bold;
}}

QLabel#tileValue {{
    color: {theme.text_strong};
    font-size: 22px;
    font-weight: bold;
}}

QFrame#tile[state="warn"] QLabel#tileValue {{ color: {theme.warning}; }}
QFrame#tile[state="crit"] QLabel#tileValue {{ color: {theme.crit}; }}

QLabel#tileUnit, QLabel#tileCaption {{
    color: {theme.muted};
    font-size: 11px;
}}

QFrame#header {{
    background-color: {theme.surface};
    border-bottom: 1px solid {theme.border};
}}

QLabel#headerApp {{
    color: {theme.muted};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
}}

QLabel#headerGpu {{
    color: {theme.text_strong};
    font-size: 18px;
    font-weight: bold;
}}

QLabel#headerDetails {{
    color: {theme.muted};
}}

QTextEdit#console {{
    background-color: {CONSOLE_BG};
    color: {CONSOLE_FG};
    font-family: monospace;
}}
"""


STYLESHEET = build_stylesheet(LIGHT)
