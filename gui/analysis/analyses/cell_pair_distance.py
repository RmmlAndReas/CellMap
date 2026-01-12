"""Cell pair distance analysis implementation."""

import os
import math
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image, ImageDraw, ImageFont

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

FRAME_INTERVAL_SECONDS = 38  # Same as plot_t1_per_100_cells.py


def load_handcorrection_image(frame_db_base: str, frame_nb: int) -> Optional[np.ndarray]:
    """Load handCorrection.tif or fallback image for a given frame. Returns RGB uint8 array or None."""
    frame_name = f"Image{frame_nb:04d}"
    frame_dir = os.path.join(frame_db_base, frame_name)
    
    # Try multiple image sources in priority order (matching example scripts)
    image_paths = [
        os.path.join(frame_dir, "tracked_cells_resized.tif"),
        os.path.join(frame_dir, "cell_identity.tif"),
        os.path.join(frame_dir, "handCorrection.tif"),
        os.path.join(frame_dir, "handcorrection.tif"),
        os.path.join(frame_dir, "HandCorrection.tif"),
        os.path.join(frame_dir, "cells.tif"),
        os.path.join(frame_dir, "outlines.tif"),
    ]
    
    img = None
    used_path = None
    
    for path in image_paths:
        if os.path.exists(path):
            try:
                img = tifffile.imread(path)
                used_path = path
                break
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
                continue
    
    if img is None:
        return None
    
    # Convert to RGB uint8
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[-1] == 3:
            if img.dtype == np.uint8:
                rgb = img
            else:
                rgb = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        else:
            base = img[..., 0]
            if base.dtype != np.uint8:
                base = ((base - base.min()) / (base.max() - base.min() + 1e-10) * 255).astype(np.uint8)
            rgb = np.stack([base, base, base], axis=-1)
    else:
        return None
    
    return rgb


def calculate_distances(
    centroids_a: Dict[int, Tuple[float, float]],
    centroids_b: Dict[int, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance between centroids of two cells over time."""
    common_frames = sorted(set(centroids_a.keys()) & set(centroids_b.keys()))
    if not common_frames:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    frames = []
    time_seconds = []
    time_minutes = []
    distances = []

    for frame_nb in common_frames:
        xa, ya = centroids_a[frame_nb]
        xb, yb = centroids_b[frame_nb]
        d = math.hypot(xa - xb, ya - yb)
        t_sec = frame_nb * FRAME_INTERVAL_SECONDS

        frames.append(frame_nb)
        time_seconds.append(t_sec)
        time_minutes.append(t_sec / 60.0)
        distances.append(d)

    distances_arr = np.array(distances, dtype=float)
    
    if len(distances_arr) > 0 and distances_arr[0] > 0:
        relative_distances = distances_arr / distances_arr[0]
    else:
        relative_distances = np.full_like(distances_arr, np.nan)

    return (
        np.array(frames, dtype=int),
        np.array(time_seconds, dtype=float),
        np.array(time_minutes, dtype=float),
        distances_arr,
        relative_distances,
    )


def create_distance_plot(
    all_pairs_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]],
) -> plt.Figure:
    """Create a line plot of relative distance vs time. Returns matplotlib figure."""
    if not all_pairs_data:
        # Return empty figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No distance data to plot", ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_pairs_data)))
    all_time_series = []
    
    for i, (frames, time_minutes, distances, relative_distances, (track_a, track_b)) in enumerate(all_pairs_data):
        if frames.size == 0:
            continue
        
        label = f"Pair {i+1} ({track_a}, {track_b})"
        ax.plot(time_minutes, relative_distances, "-o", color=colors[i], linewidth=2, markersize=3, label=label, alpha=0.6)
        all_time_series.append((time_minutes, relative_distances))
    
    # Calculate average and standard deviation
    if len(all_time_series) > 1:
        all_times = set()
        for time_minutes, _ in all_time_series:
            all_times.update(time_minutes)
        all_times = sorted(all_times)
        
        interpolated_data = []
        for time_minutes, relative_distances in all_time_series:
            interp_values = np.interp(all_times, time_minutes, relative_distances)
            interpolated_data.append(interp_values)
        
        interpolated_array = np.array(interpolated_data)
        mean_values = np.nanmean(interpolated_array, axis=0)
        std_values = np.nanstd(interpolated_array, axis=0)
        
        ax.plot(all_times, mean_values, "-", color="black", linewidth=3, label="Average", zorder=10)
        ax.fill_between(all_times, mean_values - std_values, mean_values + std_values, 
                       color="black", alpha=0.2, label="±1 SD", zorder=9)
    
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Relative distance (distance / distance at t=0)")
    ax.set_title("Relative distance between cell centroids over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return fig


def create_distance_timelapse(
    frame_db_base: str,
    frames_all: List[int],
    pairs: List[Tuple[int, int]],
    all_centroids: List[Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]],
    output_tif: str,
    frame_range: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Create TIF timelapse highlighting multiple cell pairs and linking their centroids.
    
    Args:
        frame_db_base: Base path for frame databases
        frames_all: List of all frame numbers to include
        pairs: List of (track_a, track_b) tuples
        all_centroids: List of (centroids_a, centroids_b) tuples for each pair
        output_tif: Path to save the output TIF file
        frame_range: Optional tuple of (start_frame, end_frame) to filter frames
    """
    images: List[np.ndarray] = []
    
    # Filter frames if frame_range is specified
    if frame_range is not None:
        start_frame, end_frame = frame_range
        frames_all = [f for f in frames_all if start_frame <= f <= end_frame]
    
    total_frames = len(frames_all)
    
    if total_frames == 0:
        logger.warning("No frames to process for timelapse")
        return
    
    # Color palette for pairs (using matplotlib tab10 colors converted to RGB)
    colors_rgb = [
        (31, 119, 180),   # blue
        (255, 127, 14),   # orange
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # gray
        (188, 189, 34),   # olive
        (23, 190, 207),   # cyan
    ]
    
    # Load fonts
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for i, frame_nb in enumerate(frames_all):
        base = load_handcorrection_image(frame_db_base, frame_nb)
        if base is None:
            continue

        rgb = base.copy()
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        # Draw all pairs
        r = 4  # radius for centroid markers
        
        for pair_idx, ((track_a, track_b), (centroids_a, centroids_b)) in enumerate(zip(pairs, all_centroids)):
            color = colors_rgb[pair_idx % len(colors_rgb)]
            
            has_a = frame_nb in centroids_a
            has_b = frame_nb in centroids_b
            
            if has_a:
                xa, ya = centroids_a[frame_nb]
                draw.ellipse((xa - r, ya - r, xa + r, ya + r), outline=color, width=2)
            
            if has_b:
                xb, yb = centroids_b[frame_nb]
                draw.ellipse((xb - r, yb - r, xb + r, yb + r), outline=color, width=2)
            
            if has_a and has_b:
                xa, ya = centroids_a[frame_nb]
                xb, yb = centroids_b[frame_nb]
                draw.line((xa, ya, xb, yb), fill=color, width=2)

        # Timestamp
        t_sec = frame_nb * FRAME_INTERVAL_SECONDS
        t_min = t_sec / 60.0
        text = f"Frame {frame_nb} | {t_min:.1f} min"
        # outline
        for dx, dy in [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]:
            draw.text((10 + dx, 10 + dy), text, font=font, fill=(0, 0, 0))
        draw.text((10, 10), text, font=font, fill=(255, 255, 255))
        
        # Add legend at top right corner
        h, w = rgb.shape[:2]
        legend_spacing = 20
        box_size = 12
        text_width = 60
        legend_width = box_size + 5 + text_width
        legend_x = w - legend_width
        legend_y = 0
        legend_height = len(pairs) * legend_spacing
        
        # Draw legend background (darken)
        legend_bg_array = np.array(pil_img)
        bg_region = legend_bg_array[legend_y:legend_y + legend_height, legend_x:w]
        if bg_region.size > 0:
            legend_bg_array[legend_y:legend_y + legend_height, legend_x:w] = (bg_region * 0.5).astype(np.uint8)
            pil_img = Image.fromarray(legend_bg_array)
            draw = ImageDraw.Draw(pil_img)
        
        # Draw legend border
        draw.rectangle(
            [legend_x, legend_y, w, legend_y + legend_height],
            outline=(255, 255, 255),
            width=2
        )
        
        # Draw legend entries
        for pair_idx, (track_a, track_b) in enumerate(pairs):
            color = colors_rgb[pair_idx % len(colors_rgb)]
            y_pos = legend_y + pair_idx * legend_spacing
            
            # Draw color box
            draw.rectangle(
                [legend_x, y_pos, legend_x + box_size, y_pos + box_size],
                fill=color,
                outline=(255, 255, 255),
                width=1
            )
            
            # Draw label with outline
            label = f"Pair {pair_idx + 1}"
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                draw.text((legend_x + box_size + 5 + dx, y_pos - 1 + dy), label, font=font_small, fill=(0, 0, 0))
            draw.text((legend_x + box_size + 5, y_pos - 1), label, font=font_small, fill=(255, 255, 255))

        rgb = np.array(pil_img)
        images.append(rgb)

        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i + 1}/{total_frames} frames for distance timelapse...")

    if not images:
        logger.warning("No frames processed, skipping TIF timelapse.")
        return

    stack = np.stack(images, axis=0)
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    tifffile.imwrite(output_tif, stack, bigtiff=True)
    logger.info(f"Saved distance timelapse TIF to: {output_tif}")
    logger.info(f"  Shape: {stack.shape}")


def create_first_last_collage(
    frame_db_base: str,
    pairs: List[Tuple[int, int]],
    all_centroids: List[Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]],
    frame_range: Optional[Tuple[int, int]] = None,
    master_db_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Create first/last frame collage. Returns RGB uint8 array or None."""
    if not pairs or not all_centroids:
        return None
    
    # Determine first and last frames
    if frame_range is not None:
        first_frame = frame_range[0]
        last_frame = frame_range[1]
    else:
        # Use centroids to determine frames (most reliable since we already have the data)
        all_frames = set()
        for centroids_a, centroids_b in all_centroids:
            all_frames.update(centroids_a.keys())
            all_frames.update(centroids_b.keys())
        if not all_frames:
            return None
        first_frame = min(all_frames)
        last_frame = max(all_frames)
    
    base_first = load_handcorrection_image(frame_db_base, first_frame)
    base_last = load_handcorrection_image(frame_db_base, last_frame)
    if base_first is None or base_last is None:
        return None
    
    rgb_first = base_first.copy()
    rgb_last = base_last.copy()
    
    colors_rgb = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
        (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    ]
    
    # Draw on first frame
    pil_img_first = Image.fromarray(rgb_first)
    draw_first = ImageDraw.Draw(pil_img_first)
    
    for pair_idx, ((track_a, track_b), (centroids_a, centroids_b)) in enumerate(zip(pairs, all_centroids)):
        if first_frame in centroids_a and first_frame in centroids_b:
            color = colors_rgb[pair_idx % len(colors_rgb)]
            xa, ya = centroids_a[first_frame]
            xb, yb = centroids_b[first_frame]
            draw_first.line((xa, ya, xb, yb), fill=color, width=2)
            r = 4
            draw_first.ellipse((xa - r, ya - r, xa + r, ya + r), outline=color, width=2)
            draw_first.ellipse((xb - r, yb - r, xb + r, yb + r), outline=color, width=2)
    
    rgb_first = np.array(pil_img_first)
    
    # Draw on last frame
    pil_img_last = Image.fromarray(rgb_last)
    draw_last = ImageDraw.Draw(pil_img_last)
    
    for pair_idx, ((track_a, track_b), (centroids_a, centroids_b)) in enumerate(zip(pairs, all_centroids)):
        if last_frame in centroids_a and last_frame in centroids_b:
            color = colors_rgb[pair_idx % len(colors_rgb)]
            xa, ya = centroids_a[last_frame]
            xb, yb = centroids_b[last_frame]
            draw_last.line((xa, ya, xb, yb), fill=color, width=2)
            r = 4
            draw_last.ellipse((xa - r, ya - r, xa + r, ya + r), outline=color, width=2)
            draw_last.ellipse((xb - r, yb - r, xb + r, yb + r), outline=color, width=2)
    
    rgb_last = np.array(pil_img_last)
    
    # Create collage
    h1, w1 = rgb_first.shape[:2]
    h2, w2 = rgb_last.shape[:2]
    max_w = max(w1, w2)
    total_h = h1 + h2
    collage = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    
    w1_offset = (max_w - w1) // 2
    collage[:h1, w1_offset:w1_offset+w1] = rgb_first
    
    w2_offset = (max_w - w2) // 2
    collage[h1:h1+h2, w2_offset:w2_offset+w2] = rgb_last
    
    # Add labels
    pil_collage = Image.fromarray(collage)
    draw_collage = ImageDraw.Draw(pil_collage)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    label_first = f"Frame {first_frame}"
    label_last = f"Frame {last_frame}"
    
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        draw_collage.text((10 + dx, 10 + dy), label_first, font=font, fill=(0, 0, 0))
    draw_collage.text((10, 10), label_first, font=font, fill=(255, 255, 255))
    
    y_pos_last = h1 + 10
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        draw_collage.text((10 + dx, y_pos_last + dy), label_last, font=font, fill=(0, 0, 0))
    draw_collage.text((10, y_pos_last), label_last, font=font, fill=(255, 255, 255))
    
    # Add legend
    legend_spacing = 25
    box_size = 15
    text_width = 80
    legend_width = box_size + 5 + text_width
    legend_x = max_w - legend_width
    legend_y = 0
    legend_height = len(pairs) * legend_spacing
    
    legend_bg = Image.new('RGBA', (max_w, total_h), (0, 0, 0, 0))
    legend_draw = ImageDraw.Draw(legend_bg)
    legend_draw.rectangle(
        [legend_x, legend_y, max_w, legend_y + legend_height],
        fill=(0, 0, 0, 180),
        outline=(255, 255, 255),
        width=1
    )
    pil_collage = Image.alpha_composite(pil_collage.convert('RGBA'), legend_bg).convert('RGB')
    draw_collage = ImageDraw.Draw(pil_collage)
    
    draw_collage.rectangle(
        [legend_x, legend_y, max_w, legend_y + legend_height],
        outline=(255, 255, 255),
        width=2
    )
    
    for pair_idx, (track_a, track_b) in enumerate(pairs):
        color = colors_rgb[pair_idx % len(colors_rgb)]
        y_pos = legend_y + pair_idx * legend_spacing
        draw_collage.rectangle(
            [legend_x, y_pos, legend_x + box_size, y_pos + box_size],
            fill=color,
            outline=(255, 255, 255),
            width=1
        )
        label = f"Pair {pair_idx + 1}"
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            draw_collage.text((legend_x + box_size + 5 + dx, y_pos - 2 + dy), label, font=font_small, fill=(0, 0, 0))
        draw_collage.text((legend_x + box_size + 5, y_pos - 2), label, font=font_small, fill=(255, 255, 255))
    
    return np.array(pil_collage)


class CellPairDistanceAnalysis(BaseAnalysis):
    """Analysis of distance between cell pairs over time."""
    
    name = "Cell pair distance analysis"
    description = "Analyze the distance between pairs of cells over time. Select pairs of cells to track their centroids and compute distance."
    selection_mode = "pair"
    selection_count = 2
    
    def get_ui_widgets(self):
        """Return optional UI widgets for frame range and output options."""
        try:
            from qtpy.QtWidgets import QSpinBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QCheckBox
            
            # Main widget with vertical layout
            main_widget = QWidget()
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            # Frame range widget
            frame_range_widget = QWidget()
            frame_range_layout = QHBoxLayout()
            frame_range_layout.setContentsMargins(0, 0, 0, 0)
            
            frame_range_layout.addWidget(QLabel("Frame range:"))
            
            start_spin = QSpinBox()
            start_spin.setMinimum(0)
            start_spin.setMaximum(9999)
            start_spin.setValue(0)
            start_spin.setToolTip("Start frame (leave 0 for first frame)")
            frame_range_layout.addWidget(QLabel("Start:"))
            frame_range_layout.addWidget(start_spin)
            
            end_spin = QSpinBox()
            end_spin.setMinimum(0)
            end_spin.setMaximum(9999)
            end_spin.setValue(9999)
            end_spin.setToolTip("End frame (leave 9999 for last frame)")
            frame_range_layout.addWidget(QLabel("End:"))
            frame_range_layout.addWidget(end_spin)
            
            frame_range_widget.setLayout(frame_range_layout)
            main_layout.addWidget(frame_range_widget)
            
            # Output options checkboxes
            show_collage_checkbox = QCheckBox("Show first/last frame collage")
            show_collage_checkbox.setChecked(True)
            show_collage_checkbox.setToolTip("Display first and last frames with highlighted cell pairs in the results")
            main_layout.addWidget(show_collage_checkbox)
            
            create_movie_checkbox = QCheckBox("Create movie/timelapse")
            create_movie_checkbox.setChecked(False)
            create_movie_checkbox.setToolTip("Create a TIF timelapse movie showing cell pairs over time (saved to output folder)")
            main_layout.addWidget(create_movie_checkbox)
            
            main_widget.setLayout(main_layout)
            return {
                "frame_range": main_widget, 
                "start_frame": start_spin, 
                "end_frame": end_spin,
                "show_collage": show_collage_checkbox,
                "create_movie": create_movie_checkbox
            }
        except Exception as e:
            logger.error(f"Error creating UI widgets: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def validate_selection(self, selected_cells):
        """Validate that at least one pair is selected."""
        if not selected_cells:
            return False, "Please select at least one pair of cells"
        if not isinstance(selected_cells, list):
            return False, "Invalid selection format"
        if len(selected_cells) == 0:
            return False, "Please select at least one pair of cells"
        return True, ""
    
    def run(
        self,
        selected_cells: List[Tuple[int, int]],
        ta_output_folder: str,
        frame_range: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> AnalysisResults:
        """Execute the cell pair distance analysis."""
        pairs = selected_cells
        
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
        
        # Process all pairs
        all_pairs_data = []
        all_centroids = []
        
        for pair_idx, (track_a, track_b) in enumerate(pairs):
            logger.info(f"Processing pair {pair_idx + 1}/{len(pairs)}: ({track_a}, {track_b})")
            
            centroids_a = get_centroids_over_time(master_db_path, frame_db_base, track_a, frame_range)
            centroids_b = get_centroids_over_time(master_db_path, frame_db_base, track_b, frame_range)
            
            if not centroids_a or not centroids_b:
                logger.warning(f"No centroids found for pair ({track_a}, {track_b})")
                continue
            
            all_centroids.append((centroids_a, centroids_b))
            
            frames, time_seconds, time_minutes, distances, relative_distances = calculate_distances(
                centroids_a, centroids_b
            )
            
            if frames.size == 0:
                logger.warning(f"No overlapping frames for pair {pair_idx + 1}")
                continue
            
            all_pairs_data.append((frames, time_minutes, distances, relative_distances, (track_a, track_b)))
        if not all_pairs_data:
            raise ValueError("No valid distance data for any pair")
        
        # Create results
        fig = create_distance_plot(all_pairs_data)
        
        # Get checkbox states from kwargs
        show_collage = kwargs.get('show_collage', True)
        create_movie = kwargs.get('create_movie', False)
        
        # Create collage if checkbox is checked
        collage_fig = None
        if show_collage:
            collage_array = create_first_last_collage(
                frame_db_base, pairs, all_centroids, frame_range, master_db_path
            )
            if collage_array is not None:
                try:
                    # Convert collage array to matplotlib figure so it shows next to the plot
                    collage_fig, collage_ax = plt.subplots(figsize=(10, 8))
                    collage_ax.imshow(collage_array)
                    collage_ax.axis('off')
                    collage_ax.set_title("First and Last Frame Collage")
                    plt.tight_layout()
                except Exception as e:
                    logger.error(f"Error converting collage to figure: {e}")
        
        # Create movie if checkbox is checked
        if create_movie:
            # Get all frames for the movie
            available_frames = list_frames_from_db(master_db_path, frame_db_base)
            if available_frames:
                # Create output path for movie
                movie_output_path = os.path.join(ta_output_folder, "Analysis", "distance_timelapse.tif")
                os.makedirs(os.path.dirname(movie_output_path), exist_ok=True)
                
                create_distance_timelapse(
                    frame_db_base,
                    available_frames,
                    pairs,
                    all_centroids,
                    movie_output_path,
                    frame_range
                )
                logger.info(f"Movie saved to: {movie_output_path}")
        
        # Prepare CSV data
        csv_data = []
        for pair_idx, (frames, time_minutes, distances, relative_distances, (track_a, track_b)) in enumerate(all_pairs_data):
            for fr, t_min, d, rel_d in zip(frames, time_minutes, distances, relative_distances):
                csv_data.append({
                    'pair_id': pair_idx + 1,
                    'track_id_a': track_a,
                    'track_id_b': track_b,
                    'frame_nb': int(fr),
                    'time_seconds': int(fr * FRAME_INTERVAL_SECONDS),
                    'time_minutes': float(t_min),
                    'distance_pixels': float(d),
                    'relative_distance': float(rel_d),
                })
        
        # Add figures to list (plot first, then collage if available)
        figures = [fig]
        if collage_fig is not None:
            figures.append(collage_fig)
        
        return AnalysisResults(
            analysis_name=self.name,
            figures=figures,
            images={},  # No images, everything is in figures
            data={'distance_data': csv_data},
            metadata={
                'pairs': pairs,
                'frame_range': frame_range,
                'frame_interval_seconds': FRAME_INTERVAL_SECONDS,
                'ta_output_folder': ta_output_folder,
            }
        )
