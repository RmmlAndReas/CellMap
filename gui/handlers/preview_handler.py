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
        # #region agent log
        import json
        import time
        import numpy as np
        log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":"log_preview_before_load","timestamp":int(time.time()*1000),"location":"preview_handler.py:60","message":"Before loading preview file","data":{"full_path":full_path,"preview_selected":preview_selected,"file_exists":os.path.exists(full_path)},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + "\n")
        except: pass
        # #endregion
        SQL_plot = Img(full_path)
        # #region agent log
        try:
            unique_colors = None
            if isinstance(SQL_plot, np.ndarray) and SQL_plot.size > 0:
                if len(SQL_plot.shape) == 3:
                    from utils.image_utils import RGB_to_int24
                    int24_img = RGB_to_int24(SQL_plot)
                    unique_colors = [f"{c:06x}" for c in np.unique(int24_img)[:20]]  # First 20 unique colors
                else:
                    unique_colors = [f"{c:06x}" for c in np.unique(SQL_plot)[:20]]
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":"log_preview_after_load","timestamp":int(time.time()*1000),"location":"preview_handler.py:61","message":"After loading preview file","data":{"full_path":full_path,"shape":list(SQL_plot.shape) if isinstance(SQL_plot, np.ndarray) else None,"unique_colors_sample":unique_colors},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + "\n")
        except: pass
        # #endregion
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
                        file_path=file,
                        TA_path=TA_path,
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


def _load_segmentation_mask(file_path, TA_path):
    """
    Load and label segmentation mask (handCorrection.tif, cells.tif, or cell_identity.tif).
    
    Args:
        file_path: Path to current file
        TA_path: Path to frame directory (e.g., Image0001)
    
    Returns:
        Labeled mask array (contains local_id_cells) or None if not found
    """
    try:
        from skimage.measure import label
        from tracking.utils.tools import smart_name_parser, get_mask_file
        
        # Try to get mask file (handCorrection.tif, outlines.tif, or handCorrection.png)
        filename_without_ext = smart_name_parser(file_path, ordered_output='full_no_ext')
        mask_file_path = get_mask_file(filename_without_ext)
        
        # Also try handCorrection.png as fallback
        if not os.path.exists(mask_file_path):
            handCorrection1, _ = smart_name_parser(file_path, ordered_output=['handCorrection.png', 'handCorrection.tif'])
            if os.path.isfile(handCorrection1):
                mask_file_path = handCorrection1
        
        if not os.path.exists(mask_file_path):
            # Try cells.tif or cell_identity.tif in TA_path (frame directory)
            if TA_path and os.path.isdir(TA_path):
                cells_path = os.path.join(TA_path, 'cells.tif')
                cell_identity_path = os.path.join(TA_path, 'cell_identity.tif')
                if os.path.exists(cells_path):
                    mask_file_path = cells_path
                elif os.path.exists(cell_identity_path):
                    mask_file_path = cell_identity_path
        
        if not os.path.exists(mask_file_path):
            logger.debug(f'Segmentation mask not found for {file_path}')
            return None
        
        # Load the mask image
        cell_id_image = Img(mask_file_path)
        if cell_id_image is None:
            return None
        
        # Convert to grayscale if needed
        if len(cell_id_image.shape) >= 3:
            cell_id_image = cell_id_image[..., 0]
        
        # Label the mask (background=255 for handCorrection format)
        labeled_mask = label(cell_id_image, connectivity=1, background=255)
        
        return labeled_mask
    except Exception as e:
        logger.debug(f'Error loading segmentation mask: {e}')
        return None


def _get_local_to_track_mapping(TA_path, frame_nb):
    """
    Get local_id → track_id mapping from frame-specific database.
    Uses cell_tracks table with columns local_id and track_id (created by tracking).
    
    Args:
        TA_path: Path to frame directory (e.g., Image0001) or parent directory
        frame_nb: Frame number (0-based)
    
    Returns:
        Dict {local_id: track_id} or None if database/mapping unavailable
    """
    try:
        from database.sqlite_db import TAsql
        
        if not TA_path or frame_nb is None:
            return None
        
        # Determine frame directory path
        # Check if TA_path already ends with Image#### format
        frame_name = f"Image{frame_nb:04d}"
        if os.path.basename(TA_path) == frame_name:
            # TA_path is already the frame directory
            frame_dir = TA_path
        else:
            # TA_path is parent directory, need to join with frame name
            frame_dir = os.path.join(TA_path, frame_name)
        
        # Try TA.db first, fallback to pyTA.db
        frame_db_path = os.path.join(frame_dir, "TA.db")
        if not os.path.exists(frame_db_path):
            frame_db_path = os.path.join(frame_dir, "pyTA.db")
        
        if not os.path.exists(frame_db_path):
            logger.debug(f'Frame database not found: {frame_db_path}')
            return None
        
        # Query database for local_id → track_id mapping
        db = TAsql(filename_or_connection=frame_db_path)
        
        try:
            if not db.exists('cell_tracks'):
                logger.debug(f'cell_tracks table not found in {frame_db_path}')
                db.close()
                return None
            
            # Get all local_id → track_id mappings from cell_tracks
            results = db.run_SQL_command_and_get_results(
                "SELECT local_id, track_id FROM cell_tracks WHERE track_id IS NOT NULL"
            )
            
            if not results:
                logger.debug(f'No mappings found in cell_tracks table in {frame_db_path}')
                db.close()
                return None
            
            # Build mapping dictionary
            local_to_track_map = {}
            for row in results:
                if row[0] is not None and row[1] is not None:
                    local_to_track_map[int(row[0])] = int(row[1])
            
            db.close()
            return local_to_track_map if local_to_track_map else None
            
        except Exception as e:
            logger.debug(f'Error querying frame database: {e}')
            traceback.print_exc()
            db.close()
            return None
            
    except Exception as e:
        logger.debug(f'Error getting local to track mapping: {e}')
        traceback.print_exc()
        return None


def _apply_track_completeness_overlay(tracked_image_rgb, track_completeness_cache, 
                                      file_path=None, TA_path=None, db_path=None, frame_nb=None):
    """
    Apply color coding overlay to tracked image: green for complete tracks, red for incomplete.
    ID-based method: uses segmentation mask + database mapping (avoids color collisions).
    REQUIRES: file_path, TA_path, and frame_nb. If any are missing, returns original image with error.
    
    Args:
        tracked_image_rgb: RGB image with tracked cells (numpy array, shape: H, W, 3)
        track_completeness_cache: Dict {track_id: bool} - True if complete (green), False if incomplete (red)
        file_path: Path to current file (REQUIRED, for loading segmentation mask)
        TA_path: Path to frame directory (REQUIRED, for loading segmentation mask and database)
        db_path: Path to Master.db (optional, not used but kept for compatibility)
        frame_nb: Frame number (REQUIRED, for database queries)
    
    Returns:
        RGB image with color-coded overlay, or original image if ID-based method cannot be used
    """
    import numpy as np
    
    # Validate required parameters
    if not file_path:
        logger.error('Track completeness overlay requires file_path. Cannot apply overlay.')
        return tracked_image_rgb
    
    if not TA_path:
        logger.error('Track completeness overlay requires TA_path. Cannot apply overlay.')
        return tracked_image_rgb
    
    if frame_nb is None:
        logger.error('Track completeness overlay requires frame_nb. Cannot apply overlay.')
        return tracked_image_rgb
    
    try:
        # Load segmentation mask
        labeled_mask = _load_segmentation_mask(file_path, TA_path)
        
        if labeled_mask is None:
            logger.error(f'Track completeness overlay: Segmentation mask not found for {file_path}. '
                        f'Required files: handCorrection.tif, cells.tif, or cell_identity.tif')
            return tracked_image_rgb
        
        # Get local_id → track_id mapping from frame database
        local_to_track_map = _get_local_to_track_mapping(TA_path, frame_nb)
        
        if not local_to_track_map:
            logger.error(f'Track completeness overlay: No local_id → track_id mapping found for frame {frame_nb}. '
                        f'Frame database or tracked_cells table may be missing.')
            return tracked_image_rgb
        
        # Check shape compatibility
        if labeled_mask.shape[:2] != tracked_image_rgb.shape[:2]:
            logger.error(f'Track completeness overlay: Shape mismatch - mask {labeled_mask.shape[:2]} vs image {tracked_image_rgb.shape[:2]}. '
                        f'Cannot apply overlay.')
            return tracked_image_rgb
        
        # ID-based method: use segmentation mask + database mapping
        output = tracked_image_rgb.copy()
        
        # Get unique local_ids in the mask (exclude background 0)
        unique_local_ids = np.unique(labeled_mask)
        unique_local_ids = unique_local_ids[unique_local_ids != 0]
        
        # Build masks for complete and incomplete tracks
        complete_mask = np.zeros(labeled_mask.shape, dtype=bool)
        incomplete_mask = np.zeros(labeled_mask.shape, dtype=bool)
        
        for local_id in unique_local_ids:
            if local_id not in local_to_track_map:
                continue  # Skip if no track_id mapping
            
            track_id = local_to_track_map[local_id]
            
            if track_id not in track_completeness_cache:
                continue  # Skip if not in completeness cache
            
            # Get pixels for this local_id
            cell_mask = (labeled_mask == local_id)
            
            # Apply completeness color based on track_id
            if track_completeness_cache[track_id]:  # Complete track
                complete_mask |= cell_mask
            else:  # Incomplete track
                incomplete_mask |= cell_mask
        
        # Apply colors
        if np.any(complete_mask):
            output[complete_mask, 0] = 0    # R = 0
            output[complete_mask, 1] = 255  # G = 255 (green)
            output[complete_mask, 2] = 0    # B = 0
        
        if np.any(incomplete_mask):
            output[incomplete_mask, 0] = 255  # R = 255 (red)
            output[incomplete_mask, 1] = 0    # G = 0
            output[incomplete_mask, 2] = 0    # B = 0
        
        return output
        
    except Exception as e:
        logger.error(f'Error applying track completeness overlay (ID-based method): {e}')
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
