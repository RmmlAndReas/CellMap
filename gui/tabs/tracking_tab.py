"""Tracking tab for cell tracking."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QScrollArea, QGroupBox, QLabel, QSpinBox
from qtpy.QtCore import Qt


def create_tracking_tab(parent):
    """Create and return the tracking tab widget.
    
    Args:
        parent: The main window instance (for connecting callbacks)
    
    Returns:
        QScrollArea: Scrollable container with the tracking tab
    """
    tab2b_scroll = QScrollArea()
    tab2b_scroll.setWidgetResizable(True)
    tab2b_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    tab2b_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    tab2b = QWidget()
    tab2b_scroll.setWidget(tab2b)
    
    layout = QGridLayout()
    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 1)
    
    # Left column
    left_column = QVBoxLayout()
    
    # Tracking group box
    tracking_group = QGroupBox("1. Tracking")
    tracking_layout = QVBoxLayout()
    
    # Button row with track button and defaults button
    button_layout = QHBoxLayout()
    track_cells_dynamic_button = QPushButton("Track cells")
    track_cells_dynamic_button.setToolTip(
        "Track cells (dynamic tissue).\n"
        "Use this method whenever the cells (or the field of view) move a lot between consecutive frames.")
    track_cells_dynamic_button.clicked.connect(parent.track_cells_dynamic)
    button_layout.addWidget(track_cells_dynamic_button)
    
    # Defaults button
    tracking_defaults_button = QPushButton("default parameters")
    tracking_defaults_button.clicked.connect(parent.reset_tracking_defaults)
    tracking_defaults_button.setToolTip("Reset all tracking parameters to default values")
    button_layout.addWidget(tracking_defaults_button)
    
    tracking_layout.addLayout(button_layout)
    
    # Pyramidal depth parameter
    pyramidal_layout = QHBoxLayout()
    pyramidal_label = QLabel("Pyramidal depth:")
    pyramidal_label.setToolTip(
        "Depth of pyramidal registration (1-5).\n"
        "Higher values improve alignment but increase computation time.\n"
        "For large cells, lower values may be sufficient and faster.")
    pyramidal_spinbox = QSpinBox()
    pyramidal_spinbox.setMinimum(1)
    pyramidal_spinbox.setMaximum(5)
    pyramidal_spinbox.setValue(3)
    pyramidal_spinbox.setToolTip(
        "Depth of pyramidal registration (1-5).\n"
        "Higher values improve alignment but increase computation time.\n"
        "For large cells, lower values may be sufficient and faster.")
    pyramidal_layout.addWidget(pyramidal_label)
    pyramidal_layout.addWidget(pyramidal_spinbox)
    pyramidal_layout.addStretch()
    tracking_layout.addLayout(pyramidal_layout)
    
    # Max iterations parameter
    max_iter_layout = QHBoxLayout()
    max_iter_label = QLabel("Max optimization iterations:")
    max_iter_label.setToolTip(
        "Maximum number of optimization iterations for swapping correction (5-30).\n"
        "The algorithm will stop early if no improvement is detected.\n"
        "Higher values may help with difficult tracking cases.")
    max_iter_spinbox = QSpinBox()
    max_iter_spinbox.setMinimum(5)
    max_iter_spinbox.setMaximum(30)
    max_iter_spinbox.setValue(15)
    max_iter_spinbox.setToolTip(
        "Maximum number of optimization iterations for swapping correction (5-30).\n"
        "The algorithm will stop early if no improvement is detected.\n"
        "Higher values may help with difficult tracking cases.")
    max_iter_layout.addWidget(max_iter_label)
    max_iter_layout.addWidget(max_iter_spinbox)
    max_iter_layout.addStretch()
    tracking_layout.addLayout(max_iter_layout)
    
    tracking_group.setLayout(tracking_layout)
    left_column.addWidget(tracking_group)
    
    left_column.addStretch()
    
    # Add columns to grid layout
    left_widget = QWidget()
    left_widget.setLayout(left_column)
    
    layout.addWidget(left_widget, 0, 0)
    
    # Right column
    right_column = QVBoxLayout()
    
    # Tracking correction group box
    tracking_correction_group = QGroupBox("2.Tracking correction")
    tracking_correction_layout = QVBoxLayout()
    
    # Track completeness overlay button
    completeness_overlay_button = QPushButton("Show track completness")
    completeness_overlay_button.setCheckable(True)
    completeness_overlay_button.setToolTip(
        "Enable/disable overlay showing track completeness.\n"
        "Green = complete tracks (present in all frames)\n"
        "Red = incomplete tracks (missing in some frames)")
    completeness_overlay_button.setChecked(False)
    completeness_overlay_button.clicked.connect(lambda checked: parent.toggle_completeness_overlay(checked))
    tracking_correction_layout.addWidget(completeness_overlay_button)
    
    # Track merge mode button
    track_merge_button = QPushButton("Track Merge Mode")
    track_merge_button.setCheckable(True)
    track_merge_button.setToolTip(
        "Enable track merge mode.\n"
        "When enabled:\n"
        "1. Click a cell in frame A (source track)\n"
        "2. Navigate to frame B and click another cell (target track)\n"
        "3. Confirm merge in dialog\n"
        "The target track will be merged into the source track.")
    track_merge_button.clicked.connect(parent.toggle_track_merge_mode)
    tracking_correction_layout.addWidget(track_merge_button)
    
    tracking_correction_group.setLayout(tracking_correction_layout)
    right_column.addWidget(tracking_correction_group)
    
    right_column.addStretch()
    
    # Add right column to grid layout
    right_widget = QWidget()
    right_widget.setLayout(right_column)
    
    layout.addWidget(right_widget, 0, 1)
    
    tab2b.setLayout(layout)
    
    # Store references on parent
    parent.track_cells_dynamic_button = track_cells_dynamic_button
    parent.tracking_defaults_button = tracking_defaults_button
    parent.pyramidal_depth_spinbox = pyramidal_spinbox
    parent.max_iter_spinbox = max_iter_spinbox
    parent.completeness_overlay_checkbox = completeness_overlay_button  # Keep old name for backward compatibility
    parent.completeness_overlay_button = completeness_overlay_button
    parent.track_merge_button = track_merge_button
    parent.tab2b = tab2b
    
    return tab2b_scroll
