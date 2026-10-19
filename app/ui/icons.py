"""Toolbar icons, as SVG drawn in whatever colour the theme needs.

Embedded rather than shipped as files: five small glyphs are not worth an
asset pipeline, and an SVG with the colour substituted in renders crisply at
any device pixel ratio in either theme.
"""
from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

_STROKE = 'stroke="{c}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"'

GLYPHS = {
    "start": '<path d="M8 5v14l11-7z" fill="{c}"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2" fill="{c}"/>',
    "export": f'<path d="M12 3v12M8 11l4 4 4-4M5 20h14" {_STROKE}/>',
    "theme": '<path d="M20 13.5A8 8 0 1 1 10.5 4a6.5 6.5 0 0 0 9.5 9.5z" fill="{c}"/>',
    "log": f'<path d="M5 6h14M5 12h14M5 18h9" {_STROKE}/>',
    "report": f'<path d="M7 3h7l4 4v14H7zM14 3v4h4M10 12h5M10 16h5" {_STROKE}/>',
}


def svg(name: str, color: str) -> str:
    body = GLYPHS[name].replace("{c}", color)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'


def icon(name: str, color: str, size: int = 24) -> QIcon:
    """The glyph as an icon, rendered at 1× and 2× for high-DPI screens."""
    renderer = QSvgRenderer(QByteArray(svg(name, color).encode()))
    result = QIcon()
    for scale in (1, 2):
        pixmap = QPixmap(size * scale, size * scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, size * scale, size * scale))
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    return result
