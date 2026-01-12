"""CellPose segmentation tab."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QCheckBox, QComboBox, QGroupBox, QDoubleSpinBox, QSpinBox, QScrollArea
from qtpy.QtCore import Qt


def create_cellpose_tab(parent):
    """Create and return the CellPose segmentation tab widget.
    
    Args:
        parent: The main window instance (for connecting callbacks)
    
    Returns:
        QScrollArea: Scrollable container with the CellPose tab
    """
    tab1_scroll = QScrollArea()
    tab1_scroll.setWidgetResizable(True)
    tab1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    tab1_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    tab1 = QWidget()
    tab1_scroll.setWidget(tab1)
    
    layout = QGridLayout()
    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 1)
    
    # Left column widgets
    left_column = QVBoxLayout()
    
    # ROIs label (will be updated dynamically)
    cellpose_rois_label = QLabel("0 ROIs")
    cellpose_rois_label.setToolTip("Number of segmented regions of interest (ROIs). Each ROI corresponds to one detected cell.")
    left_column.addWidget(cellpose_rois_label)
    
    # Button row with run button
    button_layout = QHBoxLayout()
    pushButton0 = QPushButton("run CellPose")
    pushButton0.clicked.connect(parent.cellpose_seg)
    pushButton0.setToolTip("Run CellPose segmentation. Detects and segments cells in the image.")
    button_layout.addWidget(pushButton0)
    
    # Defaults button
    cellpose_defaults_button = QPushButton("default parameters")
    cellpose_defaults_button.clicked.connect(parent.reset_cellpose_defaults)
    cellpose_defaults_button.setToolTip("Reset all Cellpose parameters to default values")
    button_layout.addWidget(cellpose_defaults_button)
    
    left_column.addLayout(button_layout)
    
    # Hole filling group box
    cellpose_hole_filling_group = QGroupBox("Fill Holes")
    hole_filling_group_layout = QVBoxLayout()
    
    # Maximum size parameter
    max_size_layout = QHBoxLayout()
    max_size_layout.addWidget(QLabel("Maximum size:"))
    cellpose_max_hole_size_spin = QSpinBox()
    cellpose_max_hole_size_spin.setMinimum(0)
    cellpose_max_hole_size_spin.setMaximum(10000)
    cellpose_max_hole_size_spin.setValue(500)
    cellpose_max_hole_size_spin.setSpecialValueText("All")
    cellpose_max_hole_size_spin.setToolTip("Maximum size (in pixels) of holes/gaps to fill. Set to 0 to fill all holes regardless of size.")
    max_size_layout.addWidget(cellpose_max_hole_size_spin)
    hole_filling_group_layout.addLayout(max_size_layout)
    
    # Fill Holes button
    cellpose_fill_holes_button = QPushButton("Fill Holes")
    cellpose_fill_holes_button.clicked.connect(parent.cellpose_fill_holes)
    cellpose_fill_holes_button.setToolTip("Fill holes in CellPose segmentation results. Processes selected images and fills gaps between cells.")
    hole_filling_group_layout.addWidget(cellpose_fill_holes_button)
    
    cellpose_hole_filling_group.setLayout(hole_filling_group_layout)
    left_column.addWidget(cellpose_hole_filling_group)
    
    # Additional settings group (always active) - two columns
    cellpose_additional_settings_group = QGroupBox("parameters")
    additional_settings_layout = QHBoxLayout()  # Changed to horizontal for two columns
    
    # Left column of additional settings
    left_settings_column = QVBoxLayout()
    
    # Use GPU checkbox
    cellpose_use_gpu_check = QCheckBox("use GPU")
    cellpose_use_gpu_check.setToolTip("Run segmentation on the GPU if available. Significantly faster for large images or batches. Falls back to CPU if no compatible GPU is detected.")
    left_settings_column.addWidget(cellpose_use_gpu_check)
    
    # Model selection dropdown
    model_layout = QHBoxLayout()
    model_layout.addWidget(QLabel("model:"))
    cellpose_model_combo = QComboBox()
    cellpose_model_combo.addItems(["cyto", "cyto2", "nuclei"])
    cellpose_model_combo.setCurrentText("cyto")
    cellpose_model_combo.setToolTip("Cellpose model type: cyto (cytoplasm), cyto2 (improved cytoplasm), or nuclei")
    model_layout.addWidget(cellpose_model_combo)
    left_settings_column.addLayout(model_layout)
    
    # Diameter
    diameter_layout = QHBoxLayout()
    diameter_layout.addWidget(QLabel("diameter:"))
    cellpose_diameter_spin = QDoubleSpinBox()
    cellpose_diameter_spin.setMinimum(0)
    cellpose_diameter_spin.setMaximum(1000)
    cellpose_diameter_spin.setValue(15)
    cellpose_diameter_spin.setSpecialValueText("Auto")
    cellpose_diameter_spin.setToolTip("Expected average cell diameter in pixels. This is the most important parameter and strongly affects segmentation quality. Too small causes over-segmentation; too large causes merged cells.")
    diameter_layout.addWidget(cellpose_diameter_spin)
    left_settings_column.addLayout(diameter_layout)
    
    # Flow threshold
    flow_layout = QHBoxLayout()
    flow_layout.addWidget(QLabel("flow threshold:"))
    cellpose_flow_threshold_spin = QDoubleSpinBox()
    cellpose_flow_threshold_spin.setMinimum(0.0)
    cellpose_flow_threshold_spin.setMaximum(10.0)
    cellpose_flow_threshold_spin.setSingleStep(0.1)
    cellpose_flow_threshold_spin.setValue(1.0)
    cellpose_flow_threshold_spin.setToolTip("Threshold for consistency of predicted flow fields. Higher values are stricter and remove uncertain masks; lower values keep more detections but may include errors.")
    flow_layout.addWidget(cellpose_flow_threshold_spin)
    left_settings_column.addLayout(flow_layout)
    
    # Right column of additional settings
    right_settings_column = QVBoxLayout()
    
    # Cellprob threshold
    cellprob_layout = QHBoxLayout()
    cellprob_layout.addWidget(QLabel("cellprob threshold:"))
    cellpose_cellprob_threshold_spin = QDoubleSpinBox()
    cellpose_cellprob_threshold_spin.setMinimum(-6.0)
    cellpose_cellprob_threshold_spin.setMaximum(6.0)
    cellpose_cellprob_threshold_spin.setSingleStep(0.1)
    cellpose_cellprob_threshold_spin.setValue(0.0)
    cellpose_cellprob_threshold_spin.setToolTip("Minimum cell probability required to keep a detected region. Increase to suppress background or noise; set to 0.0 to keep all potential cells.")
    cellprob_layout.addWidget(cellpose_cellprob_threshold_spin)
    right_settings_column.addLayout(cellprob_layout)
    
    # Norm percentiles
    norm_percentiles_label = QLabel("norm percentiles:")
    norm_percentiles_label.setToolTip("Intensity normalization percentiles applied before segmentation. Pixel values below the lower percentile and above the upper percentile are clipped to improve robustness.")
    right_settings_column.addWidget(norm_percentiles_label)
    
    norm_lower_layout = QHBoxLayout()
    norm_lower_layout.addWidget(QLabel("  lower:"))
    cellpose_norm_percentile_lower_spin = QDoubleSpinBox()
    cellpose_norm_percentile_lower_spin.setMinimum(0.0)
    cellpose_norm_percentile_lower_spin.setMaximum(100.0)
    cellpose_norm_percentile_lower_spin.setSingleStep(0.1)
    cellpose_norm_percentile_lower_spin.setValue(1.0)
    cellpose_norm_percentile_lower_spin.setToolTip("Lower percentile used for intensity normalization. Typically left at 1.0.")
    norm_lower_layout.addWidget(cellpose_norm_percentile_lower_spin)
    right_settings_column.addLayout(norm_lower_layout)
    
    norm_upper_layout = QHBoxLayout()
    norm_upper_layout.addWidget(QLabel("  upper:"))
    cellpose_norm_percentile_upper_spin = QDoubleSpinBox()
    cellpose_norm_percentile_upper_spin.setMinimum(0.0)
    cellpose_norm_percentile_upper_spin.setMaximum(100.0)
    cellpose_norm_percentile_upper_spin.setSingleStep(0.1)
    cellpose_norm_percentile_upper_spin.setValue(99.0)
    cellpose_norm_percentile_upper_spin.setToolTip("Upper percentile used for intensity normalization. Typically left at 99.0.")
    norm_upper_layout.addWidget(cellpose_norm_percentile_upper_spin)
    right_settings_column.addLayout(norm_upper_layout)
    
    # Niter dynamics
    niter_layout = QHBoxLayout()
    niter_layout.addWidget(QLabel("niter dynamics:"))
    cellpose_niter_dynamics_spin = QSpinBox()
    cellpose_niter_dynamics_spin.setMinimum(1)
    cellpose_niter_dynamics_spin.setMaximum(10000)
    cellpose_niter_dynamics_spin.setValue(200)
    cellpose_niter_dynamics_spin.setToolTip("Number of iterations used to move pixels along flow fields to form cell masks. Higher values improve convergence for large or elongated cells but increase runtime.")
    niter_layout.addWidget(cellpose_niter_dynamics_spin)
    right_settings_column.addLayout(niter_layout)
    
    # Batch size
    batch_size_layout = QHBoxLayout()
    batch_size_layout.addWidget(QLabel("batch size:"))
    cellpose_batch_size_spin = QSpinBox()
    cellpose_batch_size_spin.setMinimum(1)
    cellpose_batch_size_spin.setMaximum(32)
    cellpose_batch_size_spin.setValue(4)
    cellpose_batch_size_spin.setToolTip("Number of images to process simultaneously in a single batch. Higher values can speed up processing but use more memory. Recommended: 4-8 for CPU, 8-16 for GPU.")
    batch_size_layout.addWidget(cellpose_batch_size_spin)
    right_settings_column.addLayout(batch_size_layout)
    
    # Add both columns to the horizontal layout
    additional_settings_layout.addLayout(left_settings_column)
    additional_settings_layout.addLayout(right_settings_column)
    
    cellpose_additional_settings_group.setLayout(additional_settings_layout)
    left_column.addWidget(cellpose_additional_settings_group)
    
    left_column.addStretch()
    
    # Add left column to grid layout
    left_widget = QWidget()
    left_widget.setLayout(left_column)
    
    layout.addWidget(left_widget, 0, 0, 1, 2)  # Span both columns since right column is removed
    
    tab1.setLayout(layout)
    
    # Store references on parent for access elsewhere
    parent.cellpose_use_gpu_check = cellpose_use_gpu_check
    parent.cellpose_model_combo = cellpose_model_combo
    parent.cellpose_rois_label = cellpose_rois_label
    parent.pushButton0 = pushButton0
    parent.cellpose_defaults_button = cellpose_defaults_button
    parent.cellpose_fill_holes_button = cellpose_fill_holes_button
    parent.cellpose_hole_filling_group = cellpose_hole_filling_group
    parent.cellpose_max_hole_size_spin = cellpose_max_hole_size_spin
    parent.cellpose_additional_settings_group = cellpose_additional_settings_group
    parent.cellpose_diameter_spin = cellpose_diameter_spin
    parent.cellpose_flow_threshold_spin = cellpose_flow_threshold_spin
    parent.cellpose_cellprob_threshold_spin = cellpose_cellprob_threshold_spin
    parent.cellpose_norm_percentile_lower_spin = cellpose_norm_percentile_lower_spin
    parent.cellpose_norm_percentile_upper_spin = cellpose_norm_percentile_upper_spin
    parent.cellpose_niter_dynamics_spin = cellpose_niter_dynamics_spin
    parent.cellpose_batch_size_spin = cellpose_batch_size_spin
    parent.tab1 = tab1
    
    return tab1_scroll
