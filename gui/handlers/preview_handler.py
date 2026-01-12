"""Preview generation and management handler."""

import os
import traceback
from utils.image_io import Img
from utils.image_utils import blend, mask_colors
from utils.luts import apply_lut, PaletteCreator
from database.advanced_sql_plotter import plot_as_any
from database.sqlite_db import createMasterDB
from utils.logger import TA_logger

logger = TA_logger()


def create_preview_from_db(main_window, preview_selected, file):
    """Create preview from database query."""
    if hasattr(main_window, 'groupBox_color_coding') and main_window.groupBox_color_coding.isChecked():
        if main_window.radioButton3.isChecked():
            if main_window.master_db is None:
                print('creating master db')
                main_window.master_db = createMasterDB(main_window.get_full_list())

    try:
        table, column = preview_selected[1:].split('.')
        if table == 'properties':
            return

        plot_type = _determine_plot_type(table, column)
        if plot_type == 'nematics':
            if ('Q1' in column or 'Q2' in column) and 'ch' in column:
                column = _build_nematic_column(column)
            elif column.lower() in ['s1', 's2']:
                column = ' S1, S2, 10 AS SCALING_FACTOR'

        SQL_command = f'SELECT local_id,{column} FROM {table}'
        extras = _build_extras(main_window)
        
        selected_tab_idx, _ = main_window.get_cur_tab_index_and_name()
        list_idx = main_window._tab_idx_to_list_idx(selected_tab_idx)
        current_idx_in_list = main_window.list.get_list(list_idx).get_selection_index()

        mask, SQL_plot = plot_as_any(file, SQL_command, plot_type=plot_type, return_mask=True,
                                     db=main_window.master_db if main_window.radioButton3.isChecked() else None,
                                     current_frame=current_idx_in_list, **extras)

        return _apply_overlay(main_window, file, SQL_plot, mask) if hasattr(main_window, 'groupBox_overlay') and main_window.groupBox_overlay.isChecked() else SQL_plot
    except:
        traceback.print_exc()
        logger.error(f'failed to plot from the database: {preview_selected}')
        return None


def create_preview_from_file(main_window, preview_selected, file, TA_path):
    """Create preview from file path."""
    if TA_path is None:
        return None

    try:
        full_path = os.path.join(TA_path, preview_selected)
        SQL_plot = Img(full_path)
        palette = _get_palette(main_window) if hasattr(main_window, 'groupBox_color_coding') and main_window.groupBox_color_coding.isChecked() else None
        mask = None

        if len(SQL_plot.shape) != 3:
            if palette is not None:
                mask = mask_colors(SQL_plot, 0)
                SQL_plot = apply_lut(SQL_plot, palette, True)
        else:
            if preview_selected == 'tracked_cells_resized.tif':
                # Check if completeness overlay is enabled
                if (hasattr(main_window, 'track_completeness_overlay_enabled') and 
                    main_window.track_completeness_overlay_enabled and
                    hasattr(main_window, 'track_completeness_cache') and
                    main_window.track_completeness_cache):
                    
                    # Get database path and frame number for optimization
                    db_path = None
                    frame_nb = None
                    try:
                        from database.sqlite_db import get_master_db_path
                        file_list = main_window.get_full_list(warn_on_empty_list=False)
                        if file_list:
                            db_path = get_master_db_path(file_list)
                        
                        # Get current frame index (0-based, database frame_nb is also 0-based)
                        selected_tab_idx, _ = main_window.get_cur_tab_index_and_name()
                        list_idx = main_window._tab_idx_to_list_idx(selected_tab_idx)
                        current_idx = main_window.list.get_list(list_idx).get_selection_index()
                        # Database uses 0-based frame_nb (matches list index)
                        if current_idx is not None and current_idx >= 0:
                            frame_nb = current_idx
                    except Exception as e:
                        logger.debug(f'Could not get frame number for optimization: {e}')
                    
                    # Apply color coding overlay: green for complete, red for incomplete
                    SQL_plot = _apply_track_completeness_overlay(
                        SQL_plot, 
                        main_window.track_completeness_cache,
                        db_path=db_path,
                        frame_nb=frame_nb
                    )
                else:
                    # Default: mask background colors
                    mask = mask_colors(SQL_plot, [(255, 255, 255), (0, 0, 0)])
            if hasattr(main_window, 'groupBox_color_coding') and main_window.groupBox_color_coding.isChecked():
                logger.warning('Lut cannot be applied to RGB images --> ignoring')

        return _apply_overlay(main_window, file, SQL_plot, mask) if hasattr(main_window, 'groupBox_overlay') and main_window.groupBox_overlay.isChecked() else SQL_plot
    except:
        traceback.print_exc()
        logger.error(f'failed to create composite image for {preview_selected}')
        return None


def _apply_track_completeness_overlay(tracked_image_rgb, track_completeness_cache, 
                                      db_path=None, frame_nb=None):
    """
    Apply color coding overlay to tracked image: green for complete tracks, red for incomplete.
    Optimized version that queries database for track IDs in the frame.
    
    Args:
        tracked_image_rgb: RGB image with tracked cells (numpy array, shape: H, W, 3)
        track_completeness_cache: Dict {track_id: bool} - True if complete (green), False if incomplete (red)
        db_path: Path to Master.db (optional, for querying track IDs)
        frame_nb: Frame number (optional, for querying track IDs)
    
    Returns:
        RGB image with color-coded overlay
    """
    import numpy as np
    from utils.image_utils import RGB_to_int24
    
    try:
        # Convert RGB to int24 to identify track IDs (still needed for applying colors)
        tracked_image_int24 = RGB_to_int24(tracked_image_rgb)
        
        # Create output image (start with original colors)
        output = tracked_image_rgb.copy()
        
        # OPTIMIZATION: Query database for track IDs in this frame instead of scanning image
        tracks_to_process = None
        if db_path and frame_nb is not None:
            try:
                from database.sqlite_db import TAsql
                db = TAsql(filename_or_connection=db_path)
                
                # Query track IDs present in this frame
                if db.exists('tracked_cells'):
                    # Try with frame_nb in tracked_cells first
                    try:
                        results = db.run_SQL_command_and_get_results(
                            f"""
                            SELECT DISTINCT tc.track_id_cells
                            FROM tracked_cells tc
                            JOIN cells c
                              ON tc.local_id_cells = c.local_id_cells
                             AND tc.frame_nb = c.frame_nb
                            WHERE c.frame_nb = {int(frame_nb)}
                            """
                        )
                    except:
                        # Fallback: simpler join if frame_nb doesn't exist in tracked_cells
                        results = db.run_SQL_command_and_get_results(
                            f"""
                            SELECT DISTINCT tc.track_id_cells
                            FROM tracked_cells tc
                            JOIN cells c ON tc.local_id_cells = c.local_id_cells
                            WHERE c.frame_nb = {int(frame_nb)}
                            """
                        )
                    
                    if results:
                        # Get track IDs that are both in the frame AND in completeness cache
                        frame_track_ids = {int(row[0]) for row in results if row[0] is not None}
                        tracks_to_process = [tid for tid in frame_track_ids if tid in track_completeness_cache]
                
                db.close()
            except Exception as e:
                logger.debug(f'Could not query database for track IDs: {e}')
                # Fall back to image scanning
        
        # Fallback: if database query failed, scan image for track IDs
        if tracks_to_process is None:
            unique_track_ids = np.unique(tracked_image_int24)
            unique_track_ids = unique_track_ids[(unique_track_ids != 0xFFFFFF) & (unique_track_ids != 0)]
            tracks_to_process = [int(tid) for tid in unique_track_ids if int(tid) in track_completeness_cache]
        
        if not tracks_to_process:
            return output  # No tracks to process
        
        # Separate complete and incomplete tracks
        complete_tracks = [tid for tid in tracks_to_process if track_completeness_cache[tid]]
        incomplete_tracks = [tid for tid in tracks_to_process if not track_completeness_cache[tid]]
        
        # Vectorized operation: create masks for all complete tracks at once
        if complete_tracks:
            complete_mask = np.isin(tracked_image_int24, complete_tracks)
            output[complete_mask, 0] = 0    # R = 0
            output[complete_mask, 1] = 255  # G = 255
            output[complete_mask, 2] = 0    # B = 0
        
        # Vectorized operation: create masks for all incomplete tracks at once
        if incomplete_tracks:
            incomplete_mask = np.isin(tracked_image_int24, incomplete_tracks)
            output[incomplete_mask, 0] = 255  # R = 255
            output[incomplete_mask, 1] = 0    # G = 0
            output[incomplete_mask, 2] = 0    # B = 0
        
        return output
    except Exception as e:
        logger.error(f'Error applying track completeness overlay: {e}')
        traceback.print_exc()
        return tracked_image_rgb  # Return original on error


def _determine_plot_type(table, column):
    """Determine plot type from table and column names."""
    if 'bonds' in table:
        return 'bonds'
    elif 'vertices' in table:
        return 'vertices'
    elif column in ['nb_of_vertices_or_neighbours', 'nb_of_vertices_or_neighbours_cut_off']:
        return 'packing'
    elif ('Q1' in column or 'Q2' in column) and 'ch' in column:
        return 'nematics'
    elif column.lower() in ['s1', 's2']:
        return 'nematics'
    else:
        return 'cells'


def _build_nematic_column(column):
    """Build nematic column command from column name."""
    nematic_root, channel_nb = column.split('ch')
    nematic_default_command = nematic_root.replace('Q1', 'Q#').replace('Q2', 'Q#')
    default_nematic_command = (nematic_default_command.replace('Q#', 'Q1') + 'ch' + channel_nb + ', ' +
                              nematic_default_command.replace('Q#', 'Q2') + 'ch' + channel_nb)
    if 'normalized' in column:
        default_nematic_command += ', 60 AS SCALING_FACTOR'
    else:
        default_nematic_command += ', 0.06 AS SCALING_FACTOR'
    return default_nematic_command


def _build_extras(main_window):
    """Build extras dict for plot_as_any."""
    extras = {}
    if hasattr(main_window, 'groupBox_color_coding') and main_window.groupBox_color_coding.isChecked():
        if hasattr(main_window, 'lut_combo'):
            extras['LUT'] = main_window.lut_combo.currentText()
    if hasattr(main_window, 'excluder_label') and main_window.excluder_label.isChecked():
        extras['freq'] = (main_window.lower_percent_spin.value(), main_window.upper_percent_spin.value())
    return extras


def _get_palette(main_window):
    """Get color palette from LUT."""
    if not hasattr(main_window, 'lut_combo'):
        from utils.luts import PaletteCreator
        lutcreator = PaletteCreator()
        return lutcreator.create3(lutcreator.list['GRAY'])
    lut = main_window.lut_combo.currentText()
    lutcreator = PaletteCreator()
    luts = lutcreator.list
    try:
        return lutcreator.create3(luts[lut])
    except:
        if lut is not None:
            logger.error(f'could not load the specified lut ({lut}) a gray lut is loaded instead')
        return lutcreator.create3(luts['GRAY'])


def _apply_overlay(main_window, file, foreground, mask):
    """Apply overlay blending."""
    img = Img(file)
    if len(img.shape) > 2:
        if hasattr(main_window, 'overlay_bg_channel_combo'):
            channel = main_window.overlay_bg_channel_combo.currentIndex() - 1
            if channel != -1:
                img = img[..., channel]
    alpha = main_window.overlay_fg_transparency_spin.value() if hasattr(main_window, 'overlay_fg_transparency_spin') else 0.5
    return blend(img, foreground, alpha=alpha,
                 mask_or_forbidden_colors=mask)
