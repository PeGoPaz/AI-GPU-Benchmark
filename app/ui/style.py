"""Application styling, kept out of the layout code.

Two reasons this is not a pile of setStyleSheet() calls next to each widget.
Appearance is readable in one place without tracing the layout, and widget
state is expressed as a selector rather than as code that rewrites a
stylesheet on every transition — start_benchmark() used to swap the Start
button's sheet to grey and both the finish and error paths had to swap it
back.

A second theme means a second palette and rebuilding STYLESHEET from it;
nothing in main_window.py has to change.
"""

START_GREEN = "#2E8B57"
START_GREEN_HOVER = "#379E64"
START_GREEN_PRESSED = "#256F45"

STOP_RED = "#8B0000"
STOP_RED_HOVER = "#A50000"
STOP_RED_PRESSED = "#6E0000"

EXPORT_BLUE = "#4682B4"
EXPORT_BLUE_HOVER = "#5490C4"
EXPORT_BLUE_PRESSED = "#3A6E99"

DISABLED_BG = "#3C3C3C"
DISABLED_FG = "#8A8A8A"

CONSOLE_BG = "#1E1E1E"
CONSOLE_FG = "#00FF00"

STYLESHEET = f"""
QLabel[role="metric"] {{
    font-size: 14px;
    font-weight: bold;
}}

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
    background-color: {DISABLED_BG};
    color: {DISABLED_FG};
}}

QTextEdit#console {{
    background-color: {CONSOLE_BG};
    color: {CONSOLE_FG};
    font-family: monospace;
}}
"""
