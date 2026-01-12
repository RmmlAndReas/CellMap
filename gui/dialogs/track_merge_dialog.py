"""Dialog for confirming track merge operation."""

from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QDialogButtonBox)
from qtpy.QtCore import Qt
from utils.logger import TA_logger

logger = TA_logger()


class TrackMergeDialog(QDialog):
    """Dialog for confirming track merge operation."""
    
    def __init__(self, parent=None, track_id_x=None, track_id_y=None):
        super(TrackMergeDialog, self).__init__(parent)
        self.setWindowTitle("Merge Tracks")
        self.setMinimumSize(400, 150)
        
        self.track_id_x = track_id_x
        self.track_id_y = track_id_y
        self.merge_confirmed = False
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Info label
        track_x_hex = f"0x{track_id_x:06x}" if track_id_x is not None else "N/A"
        track_y_hex = f"0x{track_id_y:06x}" if track_id_y is not None else "N/A"
        
        info_text = f"Do you want to merge track {track_x_hex} with track {track_y_hex}?\n\n"
        info_text += f"This will merge all frames of track {track_y_hex} into track {track_x_hex}."
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        button_box.accepted.connect(self.accept_merge)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def accept_merge(self):
        """Handle Yes button click."""
        self.merge_confirmed = True
        self.accept()


def show_track_merge_dialog(parent, track_id_x, track_id_y):
    """
    Show the track merge dialog and return if merge was confirmed.
    
    Args:
        parent: Parent window
        track_id_x: First track ID (will be kept)
        track_id_y: Second track ID (will be merged into track_id_x)
    
    Returns:
        bool: True if merge was confirmed, False otherwise
    """
    dialog = TrackMergeDialog(parent=parent, track_id_x=track_id_x, track_id_y=track_id_y)
    dialog.exec()
    return dialog.merge_confirmed
