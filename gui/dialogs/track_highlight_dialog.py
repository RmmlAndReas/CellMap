"""Dialog for confirming track highlight operation."""

from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QDialogButtonBox)
from qtpy.QtCore import Qt
from utils.logger import TA_logger

logger = TA_logger()


class TrackHighlightDialog(QDialog):
    """Dialog for confirming track highlight operation."""
    
    def __init__(self, parent=None, track_id=None):
        super(TrackHighlightDialog, self).__init__(parent)
        self.setWindowTitle("Highlight Track")
        self.setMinimumSize(400, 150)
        
        self.track_id = track_id
        self.highlight_confirmed = False
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Info label
        track_hex = f"0x{track_id:06x}" if track_id is not None else "N/A"
        
        info_text = f"Highlight the track?\n\n"
        info_text += f"This will highlight track {track_hex} in all frames where it is present."
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        button_box.accepted.connect(self.accept_highlight)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def accept_highlight(self):
        """Handle Yes button click."""
        self.highlight_confirmed = True
        self.accept()


def show_track_highlight_dialog(parent, track_id):
    """
    Show the track highlight dialog and return if highlight was confirmed.
    
    Args:
        parent: Parent window
        track_id: Track ID to highlight
    
    Returns:
        bool: True if highlight was confirmed, False otherwise
    """
    dialog = TrackHighlightDialog(parent=parent, track_id=track_id)
    dialog.exec()
    return dialog.highlight_confirmed
