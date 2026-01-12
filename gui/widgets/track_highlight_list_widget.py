"""Widget for managing track highlights with enable/disable buttons."""

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QListWidget, QListWidgetItem)
from qtpy.QtCore import Qt
from utils.logger import TA_logger

logger = TA_logger()


class TrackHighlightListWidget(QWidget):
    """Widget showing all track IDs with enable/disable buttons."""
    
    def __init__(self, parent=None):
        super(TrackHighlightListWidget, self).__init__(parent)
        self.main_window = parent
        self.track_items = {}  # track_id -> dict with 'item', 'button', 'label'
        
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("Track Highlights:")
        label.setToolTip("Enable/disable track highlighting")
        layout.addWidget(label)
        
        # Scrollable list
        self.list_widget = QListWidget()
        self.list_widget.setToolTip("Click Enable/Disable to toggle track highlighting")
        layout.addWidget(self.list_widget)
        
        self.setLayout(layout)
    
    def populate_tracks(self, track_ids):
        """
        Populate the list with track IDs, with highlighted tracks at the top.
        
        Args:
            track_ids: Set or list of track IDs to display
        """
        self.list_widget.clear()
        self.track_items.clear()
        
        if not track_ids:
            return
        
        # Separate highlighted and non-highlighted tracks
        highlighted_tracks = []
        non_highlighted_tracks = []
        
        if (self.main_window and 
            hasattr(self.main_window, 'highlighted_track_ids')):
            for track_id in track_ids:
                if track_id in self.main_window.highlighted_track_ids:
                    highlighted_tracks.append(track_id)
                else:
                    non_highlighted_tracks.append(track_id)
        else:
            non_highlighted_tracks = list(track_ids)
        
        # Sort each group
        highlighted_tracks.sort()
        non_highlighted_tracks.sort()
        
        # Combine: highlighted first, then non-highlighted
        sorted_track_ids = highlighted_tracks + non_highlighted_tracks
        
        for track_id in sorted_track_ids:
            # Create item widget
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(5, 2, 5, 2)
            
            # Track ID label
            track_label = QLabel(f"Track 0x{track_id:06x}")
            track_label.setMinimumWidth(100)
            item_layout.addWidget(track_label)
            
            # Enable/Disable button
            toggle_button = QPushButton("Enable")
            toggle_button.setCheckable(True)
            toggle_button.setMaximumWidth(80)
            
            # Check if track is currently highlighted
            if (self.main_window and 
                hasattr(self.main_window, 'highlighted_track_ids') and
                track_id in self.main_window.highlighted_track_ids):
                toggle_button.setChecked(True)
                toggle_button.setText("Disable")
            
            # Connect button
            toggle_button.clicked.connect(
                lambda checked, tid=track_id, btn=toggle_button: 
                self._toggle_track_highlight(tid, checked, btn)
            )
            
            item_layout.addWidget(toggle_button)
            item_layout.addStretch()
            
            item_widget.setLayout(item_layout)
            
            # Create list item
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, item_widget)
            
            self.track_items[track_id] = {
                'item': list_item,
                'button': toggle_button,
                'label': track_label
            }
    
    def _toggle_track_highlight(self, track_id, enabled, button):
        """Toggle highlight for a track."""
        if not self.main_window or not hasattr(self.main_window, 'highlighted_track_ids'):
            return
        
        if enabled:
            self.main_window.highlighted_track_ids.add(track_id)
            logger.info(f'Enabled highlight for track {track_id:06x}')
        else:
            self.main_window.highlighted_track_ids.discard(track_id)
            logger.info(f'Disabled highlight for track {track_id:06x}')
        
        # Re-populate to re-sort (highlighted tracks to top)
        current_track_ids = set(self.track_items.keys())
        if current_track_ids:
            self.populate_tracks(current_track_ids)
        
        # Trigger repaint
        if hasattr(self.main_window, 'paint') and hasattr(self.main_window.paint, 'paint'):
            self.main_window.paint.paint.update()
    
    def update_button_states(self):
        """Update button states based on current highlighted_track_ids."""
        if not self.main_window or not hasattr(self.main_window, 'highlighted_track_ids'):
            return
        
        for track_id, item_data in self.track_items.items():
            button = item_data['button']
            is_highlighted = track_id in self.main_window.highlighted_track_ids
            button.setChecked(is_highlighted)
            button.setText("Disable" if is_highlighted else "Enable")
