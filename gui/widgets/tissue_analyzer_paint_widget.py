from gui.widgets.scrollable_paint_widget import scrollable_paint
from gui.widgets.paint_widget import Createpaintwidget
from utils.qt_settings import set_UI
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtCore import Qt
import sys
import json
import time

# #region agent log
def _debug_log(location, message, data, hypothesis_id=None):
    try:
        log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
        entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass
# #endregion

class tascrollablepaint(scrollable_paint):
    """
    A customized version of the scrollable_paint class for TA-like drawing area.

    Args:
        overriding_paint_widget (Createpaintwidget, optional): Custom paint widget to override the default behavior. Defaults to None.
    """

    def __init__(self, overriding_paint_widget=None):
        if overriding_paint_widget is None:
            class overriding_apply(Createpaintwidget):
                """
                Custom paint widget with TA-like functions.
                """

                def apply(self):
                    """
                    Apply the drawing with a minimal cell size of 0.
                    """
                    # #region agent log
                    _debug_log("tissue_analyzer_paint_widget.py:24", "overriding_apply.apply() called", {}, "E")
                    # #endregion
                    # Debug: verify apply is being called
                    from utils.logger import TA_logger
                    logger = TA_logger()
                    logger.info('apply() called - processing user drawing')
                    self.apply_drawing(minimal_cell_size=0)

                def shift_apply(self):
                    """
                    Apply the drawing with a minimal cell size of 10.
                    """
                    self.apply_drawing(minimal_cell_size=10)

                def ctrl_m_apply(self):
                    """
                    Manually reseed the watershed with the selected channel.
                    """
                    self.manually_reseeded_wshed()

                def m_apply(self):
                    """
                    Toggle the visibility of the mask.
                    """
                    self.maskVisible = not self.maskVisible
                    self.update()

                def save(self):
                    """
                    Save the mask.
                    """
                    self.save_mask()

            super().__init__(custom_paint_panel=overriding_apply())
        else:
            super().__init__(custom_paint_panel=overriding_paint_widget)





