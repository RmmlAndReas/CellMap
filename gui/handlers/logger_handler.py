"""Logger console handler."""

from qtpy.QtGui import QColor, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import QApplication


def _is_dark_theme():
    """Check if the application is using dark theme."""
    app = QApplication.instance()
    if app is None:
        return False
    palette = app.palette()
    window_color = palette.color(palette.Window)
    # Check if window background is dark (brightness < 128)
    brightness = (window_color.red() + window_color.green() + window_color.blue()) / 3
    return brightness < 128


def append_text(console, text, color):
    """Append colored text to logger console."""
    textCursor = console.textCursor()
    textCursor.movePosition(QTextCursor.End)
    console.setTextCursor(textCursor)
    format = QTextCharFormat()
    format.setForeground(color)
    console.setCurrentCharFormat(format)
    console.insertPlainText(text)
    console.verticalScrollBar().setValue(console.verticalScrollBar().maximum())


def set_html_red(console, text):
    """Append red text to console."""
    append_text(console, text, QColor(255, 0, 0))


def set_html_black(console, text):
    """Append black text to console in light theme, green text in dark theme."""
    if _is_dark_theme():
        # Use green text in dark theme
        append_text(console, text, QColor(0, 255, 0))
    else:
        # Use black text in light theme
        append_text(console, text, QColor(0, 0, 0))
