"""Tissue trajectory analysis implementation."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from gui.analysis.base_analysis import BaseAnalysis
from gui.analysis.analysis_results import AnalysisResults
from gui.handlers.analysis_handler import (
    detect_master_db,
    detect_frame_db_base,
    list_frames_from_db,
    get_centroids_over_time,
)
from utils.logger import TA_logger

logger = TA_logger()


def create_trajectory_plot(
    all_trajectories: List[Tuple[np.ndarray, np.ndarray, int, List[int]]],
    px2micron: float = 1.0,
    normalize: bool = True,
) -> plt.Figure:
    """
    Create a line plot of cell trajectories in XY coordinate system.
    
    Args:
        all_trajectories: List of (x_coords, y_coords, track_id, frames) tuples
        px2micron: Conversion factor from pixels to microns
        normalize: If True, normalize trajectories to start at (0, 0)
    
    Returns:
        matplotlib Figure
    """
    if not all_trajectories:
        # Return empty figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No trajectory data to plot", ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Mark origin (0, 0) as reference point (only if normalized)
    if normalize:
        ax.plot(0, 0, "ko", markersize=10, markeredgecolor='white', markeredgewidth=2, 
               zorder=12, label="Origin (start)")
    
    # Use a colormap to assign different colors to different cells
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trajectories)))
    
    # Store all trajectories for averaging
    all_x_series = []
    all_y_series = []
    
    # Plot individual cell trajectories
    for i, (x_coords, y_coords, track_id, frames) in enumerate(all_trajectories):
        if x_coords.size == 0 or y_coords.size == 0:
            continue
        
        # Convert to microns
        x_micron = x_coords * px2micron
        y_micron = y_coords * px2micron
        
        # Normalize to starting position if requested (subtract first point from all points)
        if normalize and len(x_micron) > 0 and len(y_micron) > 0:
            x_start = x_micron[0]
            y_start = y_micron[0]
            x_normalized = x_micron - x_start
            y_normalized = y_micron - y_start
        else:
            x_normalized = x_micron
            y_normalized = y_micron
        
        # Plot trajectory
        label = f"Cell {track_id}"
        ax.plot(x_normalized, y_normalized, "-o", color=colors[i], linewidth=2, 
                markersize=4, label=label, alpha=0.7)
        
        # Mark start point (should be at origin now, but mark it anyway)
        if len(x_normalized) > 0:
            ax.plot(x_normalized[0], y_normalized[0], "s", color=colors[i], 
                   markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=10)
        
        # Store for averaging (with frame information) - use normalized values
        all_x_series.append((x_normalized, track_id, frames))
        all_y_series.append((y_normalized, track_id, frames))
    
    # Calculate and plot average trajectory with SD
    if len(all_x_series) > 1:
        # Collect all unique frames across all trajectories
        all_frames = set()
        for x_vals, _, frames in all_x_series:
            all_frames.update(frames)
        all_frames = sorted(all_frames)
        
        # For each frame, collect positions from all cells that have data at that frame
        frame_x_values = {f: [] for f in all_frames}
        frame_y_values = {f: [] for f in all_frames}
        
        for (x_vals, _, frames), (y_vals, _, _) in zip(all_x_series, all_y_series):
            # Create a mapping from frame to index
            frame_to_idx = {f: i for i, f in enumerate(frames)}
            
            for frame in all_frames:
                if frame in frame_to_idx:
                    idx = frame_to_idx[frame]
                    if idx < len(x_vals) and idx < len(y_vals):
                        frame_x_values[frame].append(x_vals[idx])
                        frame_y_values[frame].append(y_vals[idx])
        
        # Calculate mean and std for each frame
        mean_x = []
        mean_y = []
        std_x = []
        std_y = []
        valid_frames = []
        
        for frame in all_frames:
            x_at_frame = frame_x_values[frame]
            y_at_frame = frame_y_values[frame]
            
            if len(x_at_frame) > 0 and len(y_at_frame) > 0:
                mean_x.append(np.mean(x_at_frame))
                mean_y.append(np.mean(y_at_frame))
                std_x.append(np.std(x_at_frame))
                std_y.append(np.std(y_at_frame))
                valid_frames.append(frame)
        
        if len(mean_x) > 0:
            mean_x = np.array(mean_x)
            mean_y = np.array(mean_y)
            std_x = np.array(std_x)
            std_y = np.array(std_y)
            
            # Plot average trajectory
            ax.plot(mean_x, mean_y, "-", color="black", linewidth=3, 
                   label="Average", zorder=11)
            
            # Plot SD bands
            ax.fill_betweenx(mean_y, mean_x - std_x, mean_x + std_x,
                           color="black", alpha=0.2, label="±1 SD (X)", zorder=9)
            ax.fill_between(mean_x, mean_y - std_y, mean_y + std_y,
                          color="gray", alpha=0.2, label="±1 SD (Y)", zorder=9)
    
    if normalize:
        xlabel = "X displacement (μm)"
        ylabel = "Y displacement (μm)"
        title = "Cell Trajectories in XY Coordinate System (Normalized to Start)"
    else:
        xlabel = "X position (μm)"
        ylabel = "Y position (μm)"
        title = "Cell Trajectories in XY Coordinate System"
    
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    # Move legend outside to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    
    # Add more padding around the plot to ensure axis labels, title, and legend are visible
    plt.tight_layout(pad=3.0, rect=[0, 0, 0.85, 1])  # Leave space on the right for legend
    return fig


class TissueTrajectoryAnalysis(BaseAnalysis):
    """Analysis of cell trajectories over time in XY coordinate system."""
    
    name = "Tissue trajectory"
    description = "Track the movement of centroids of selected cells across all frames. Output as a line plot on XY coordinate system."
    selection_mode = "group"
    selection_count = None  # Allow multiple cells
    
    def get_ui_widgets(self):
        """Return UI widgets for px2micron ratio input and normalization option."""
        try:
            from qtpy.QtWidgets import QLineEdit, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QCheckBox
            from qtpy.QtGui import QDoubleValidator
            from qtpy.QtCore import Qt
            
            # Main widget with vertical layout
            main_widget = QWidget()
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            # Pixels to microns ratio
            ratio_widget = QWidget()
            ratio_layout = QHBoxLayout()
            ratio_layout.setContentsMargins(0, 0, 0, 0)
            
            ratio_layout.addWidget(QLabel("Pixels to microns ratio:"))
            
            px2micron_edit = QLineEdit()
            px2micron_edit.setText("1.0")
            px2micron_edit.setToolTip("Conversion factor from pixels to microns (e.g., 0.1 means 1 pixel = 0.1 microns). Unlimited decimal places allowed.")
            
            # Set up validator for double values between 0.001 and 1000.0
            # Using QDoubleValidator with unlimited decimals
            validator = QDoubleValidator(0.001, 1000.0, 1000, px2micron_edit)  # 1000 decimals = effectively unlimited
            validator.setNotation(QDoubleValidator.StandardNotation)
            px2micron_edit.setValidator(validator)
            
            ratio_layout.addWidget(px2micron_edit)
            ratio_widget.setLayout(ratio_layout)
            main_layout.addWidget(ratio_widget)
            
            # Normalize starting positions checkbox
            normalize_checkbox = QCheckBox("Normalize starting positions")
            normalize_checkbox.setChecked(True)
            normalize_checkbox.setToolTip("If checked, all trajectories will start at (0, 0) to show relative movement")
            main_layout.addWidget(normalize_checkbox)
            
            main_widget.setLayout(main_layout)
            return {
                "px2micron_widget": main_widget,
                "px2micron": px2micron_edit,
                "normalize": normalize_checkbox
            }
        except Exception as e:
            logger.error(f"Error creating UI widgets: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def validate_selection(self, selected_cells):
        """Validate that at least one cell is selected."""
        # Handle different selection formats
        if not selected_cells:
            return False, "Please select at least one cell"
        
        # Flatten if it's a list of lists (group mode)
        if isinstance(selected_cells, list) and len(selected_cells) > 0:
            if isinstance(selected_cells[0], list):
                # It's a list of groups, flatten to get all track_ids
                flattened = [item for sublist in selected_cells for item in sublist]
                if not flattened:
                    return False, "Please select at least one cell"
            elif isinstance(selected_cells[0], (int, tuple)):
                # It's a list of track_ids or pairs
                if len(selected_cells) == 0:
                    return False, "Please select at least one cell"
            else:
                return False, "Invalid selection format"
        else:
            return False, "Please select at least one cell"
        
        return True, ""
    
    def run(
        self,
        selected_cells: List,
        ta_output_folder: str,
        frame_range: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> AnalysisResults:
        """Execute the tissue trajectory analysis."""
        # Flatten selection if it's a list of lists
        track_ids = []
        if isinstance(selected_cells, list) and len(selected_cells) > 0:
            if isinstance(selected_cells[0], list):
                # List of groups - flatten
                track_ids = [item for sublist in selected_cells for item in sublist]
            elif isinstance(selected_cells[0], (int, tuple)):
                # List of track_ids or pairs
                if isinstance(selected_cells[0], tuple):
                    # It's pairs, extract all track_ids
                    track_ids = [item for pair in selected_cells for item in pair]
                else:
                    # It's a list of track_ids
                    track_ids = list(selected_cells)
            else:
                track_ids = list(selected_cells)
        
        if not track_ids:
            raise ValueError("No cells selected for trajectory analysis")
        
        # Get px2micron ratio from kwargs
        px2micron = kwargs.get('px2micron', 1.0)
        if px2micron is None:
            px2micron = 1.0
        
        # Get normalize option from kwargs
        normalize = kwargs.get('normalize', True)
        if normalize is None:
            normalize = True
        
        logger.info(f"Running tissue trajectory analysis for {len(track_ids)} cells")
        logger.info(f"Using px2micron ratio: {px2micron}")
        logger.info(f"Normalize starting positions: {normalize}")
        
        # Detect databases
        master_db_path = detect_master_db(ta_output_folder)
        if not master_db_path:
            raise ValueError(f"Could not find master database in {ta_output_folder}")
        
        frame_db_base = detect_frame_db_base(ta_output_folder, master_db_path)
        
        # Get frame range from kwargs if not provided
        if frame_range is None:
            start_frame = kwargs.get('start_frame', 0)
            end_frame = kwargs.get('end_frame', 9999)
            if start_frame == 0 and end_frame == 9999:
                frame_range = None  # Use all frames
            else:
                available_frames = list_frames_from_db(master_db_path)
                if available_frames:
                    if end_frame == 9999:
                        end_frame = max(available_frames)
                    frame_range = (start_frame, end_frame)
        
        # Get centroids for each cell
        all_trajectories = []
        
        for track_id in track_ids:
            logger.info(f"Processing cell {track_id}")
            
            centroids = get_centroids_over_time(master_db_path, frame_db_base, track_id, frame_range)
            
            if not centroids:
                logger.warning(f"No centroids found for cell {track_id}")
                continue
            
            # Extract x and y coordinates in frame order
            frames = sorted(centroids.keys())
            x_coords = np.array([centroids[f][0] for f in frames])
            # Invert Y axis: in image coordinates Y increases downward, 
            # but in standard XY coordinates Y should increase upward
            # So we negate Y to make downward movement negative
            y_coords = np.array([-centroids[f][1] for f in frames])
            
            all_trajectories.append((x_coords, y_coords, track_id, frames))
        
        if not all_trajectories:
            raise ValueError("No trajectory data found for any selected cells")
        
        # Create plot
        fig = create_trajectory_plot(all_trajectories, px2micron, normalize)
        
        # Prepare CSV data
        csv_data = []
        for x_coords, y_coords, track_id, frames in all_trajectories:
            for i, frame_nb in enumerate(frames):
                if i < len(x_coords) and i < len(y_coords):
                    x_px = x_coords[i]
                    y_px = y_coords[i]
                    x_micron = x_px * px2micron
                    y_micron = y_px * px2micron
                    csv_data.append({
                        'track_id': int(track_id),
                        'frame_nb': int(frame_nb),
                        'x_pixels': float(x_px),
                        'y_pixels': float(y_px),
                        'x_microns': float(x_micron),
                        'y_microns': float(y_micron),
                    })
        
        return AnalysisResults(
            analysis_name=self.name,
            figures=[fig],
            images={},
            data={'trajectory_data': csv_data},
            metadata={
                'track_ids': track_ids,
                'frame_range': frame_range,
                'px2micron': px2micron,
                'ta_output_folder': ta_output_folder,
            }
        )
