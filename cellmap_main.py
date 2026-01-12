"""Entry point script for CellMap."""
import os
import sys
from pathlib import Path

# Add package root to path so gui and other modules can be imported
package_root = Path(__file__).parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from utils.qt_settings import set_UI
set_UI()
from gui.main_window import TissueAnalyzer
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon, QPixmap, QPainter
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtCore import Qt
from pathlib import Path

def create_icon_from_svg(logo_path):
    """Create QIcon from SVG with multiple sizes for Windows taskbar and title bar."""
    icon = QIcon()
    sizes = [16, 32, 48, 64, 128, 256, 512]
    renderer = QSvgRenderer(str(logo_path))
    if renderer.isValid():
        for size in sizes:
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pixmap)
    return icon

def main():
    """Main entry point for CellMap."""
    import sys
    import traceback
    
    def exception_handler(exctype, value, tb):
        """Global exception handler to catch unhandled exceptions."""
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                import json
                error_data = {"error": str(value), "type": str(exctype), "traceback": ''.join(traceback.format_exception(exctype, value, tb))}
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"cellmap_main.py:exception_handler","message":"Unhandled exception caught","data":error_data,"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        traceback.print_exception(exctype, value, tb)
        sys.__excepthook__(exctype, value, tb)
    
    sys.excepthook = exception_handler
    
    # #region agent log
    try:
        with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
            f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"cellmap_main.py:main:entry","message":"main() started","data":{},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
    except: pass
    # #endregion
    try:
        app = QApplication(sys.argv)
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"cellmap_main.py:main:after_app","message":"QApplication created","data":{},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
        except: pass
        # #endregion
        # Set application icon from logo file with multiple sizes
        logo_path = Path(__file__).parent / "logo" / "logo.svg"
        if logo_path.exists():
            app.setWindowIcon(create_icon_from_svg(logo_path))
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"cellmap_main.py:main:before_window","message":"About to create TissueAnalyzer","data":{},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
        except: pass
        # #endregion
        w = TissueAnalyzer()
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"cellmap_main.py:main:after_window","message":"TissueAnalyzer created","data":{},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
        except: pass
        # #endregion
        w.show()
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write('{"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"cellmap_main.py:main:after_show","message":"Window shown, about to exec","data":{},"timestamp":' + str(int(__import__('time').time()*1000)) + '}\n')
        except: pass
        # #endregion
        sys.exit(app.exec())
    except Exception as e:
        # #region agent log
        try:
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                import traceback
                import json
                error_data = {"error": str(e), "traceback": traceback.format_exc()}
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"cellmap_main.py:main:exception","message":"Exception in main","data":error_data,"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except: pass
        # #endregion
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
