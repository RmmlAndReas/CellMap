"""Tab management and preview update handler."""

import os
import traceback
from utils.image_io import Img
from tracking.tools import smart_name_parser
from utils.logger import TA_logger

logger = TA_logger()


def update_preview_for_tab(main_window):
    """Update preview based on selected tab and file."""
    selected_tab_idx = main_window.tabs.currentIndex()
    
    # Map tab indices to list indices: all tabs use list 0
    list_idx = main_window._tab_idx_to_list_idx(selected_tab_idx)
    
    main_window.list.set_list(list_idx)
    list_widget = main_window.list.get_list(list_idx).list
    selected_items = list_widget.selectedItems()
    selected_tab_name = main_window.tabs.tabText(selected_tab_idx).lower()

    logger.debug(f'selected_tab_name "{selected_tab_name}" {selected_tab_idx}')

    if 'analysis' not in selected_tab_name:
        _close_master_db(main_window)

    _show_paint_tab(main_window)

    if 'analysis' in selected_tab_name or 'tracking' in selected_tab_name:
        _setup_preview_mode(main_window, selected_items)
        return

    _setup_segmentation_mode(main_window, selected_tab_name, selected_items, list_widget)


def _close_master_db(main_window):
    """Close master database if open."""
    try:
        if main_window.master_db is not None:
            print('closing master db')
            main_window.master_db.close()
    except:
        pass
    finally:
        main_window.master_db = None


def _show_paint_tab(main_window):
    """Show paint widget tab."""
    main_window.list.setDisabled(False)
    main_window.Stack.setCurrentIndex(0)


def _setup_preview_mode(main_window, selected_items):
    """Setup preview mode."""
    main_window.list.freeze(True)
    # Channels dropdown was removed - always use merge mode, so no need to set channel index
    main_window.paint.freeze(True, level=2)

    if selected_items:
        main_window._update_channels(selected_items[0].toolTip())
    else:
        main_window._update_channels(None)

    selected_tab_idx, selected_tab_name = main_window.get_cur_tab_index_and_name()
    
    # For tracking tab, automatically load tracked_cells_resized.tif
    if 'tracking' in selected_tab_name:
        _load_tracked_cells_preview(main_window, selected_items)
    else:
        # For analysis tab, preview combo is no longer used
        # Load tracked cells preview similar to tracking tab
        _load_tracked_cells_preview(main_window, selected_items)


def _setup_segmentation_mode(main_window, selected_tab_name, selected_items, list_widget):
    """Setup segmentation/editing mode."""
    main_window.list.freeze(False)

    is_seg_tab = (selected_tab_name.startswith('seg') or
                  'cellpose' in selected_tab_name)
    main_window.paint.freeze(not is_seg_tab, level=1 if not is_seg_tab else 0)

    if is_seg_tab:
        # Clear selected cells when entering segmentation tab
        if hasattr(main_window, 'selected_track_ids'):
            main_window.selected_track_ids.clear()
        # Clear selection manager state
        if hasattr(main_window, 'selection_manager'):
            main_window.selection_manager.clear_all()
        # Disable cell selection mode in segmentation tab
        if hasattr(main_window.paint, 'paint'):
            main_window.paint.paint.selection_mode = False
            main_window.paint.paint.track_view_mode = False  # Disable track view mode to enable brush cursor
            main_window.paint.paint.track_merge_mode = False  # Disable track merge mode
            # Clear any hover state
            main_window.paint.paint.hovered_track_id = None
            # Ensure brush tool (drawing) is enabled
            main_window.paint.paint.drawing_enabled = True
            # Clear tracked image (not needed in segmentation tab)
            main_window.paint.paint.tracked_image_int24 = None
            main_window.paint.paint.tracked_image_rgb = None
            main_window.paint.paint._tracked_image_qimage = None
            main_window.paint.paint.current_file = None
            # Clear box selection state if any
            if hasattr(main_window.paint.paint, 'box_selection_start'):
                main_window.paint.paint.box_selection_start = None
            if hasattr(main_window.paint.paint, 'box_selection_end'):
                main_window.paint.paint.box_selection_end = None
            # Ensure cursor is properly initialized for brush drawing
            if (main_window.paint.paint.image is not None and 
                (main_window.paint.paint.cursor is None or 
                 main_window.paint.paint.cursor.size() != main_window.paint.paint.image.size())):
                from qtpy.QtGui import QImage
                from qtpy.QtCore import Qt
                main_window.paint.paint.cursor = QImage(main_window.paint.paint.image.size(), QImage.Format_ARGB32)
                main_window.paint.paint.cursor.fill(Qt.transparent)
            # Force repaint to clear any visual selections
            main_window.paint.paint.update()
        # Disable cell selection mode in main window if it was active
        if hasattr(main_window, 'selection_mode_active'):
            if main_window.selection_mode_active:
                main_window.selection_mode_active = False
                # Uncheck the button if it exists
                if hasattr(main_window, 'select_cells_button'):
                    main_window.select_cells_button.setChecked(False)
        # Ensure brush tool (drawing) is enabled
        main_window.paint.enableMouseTracking()

    if selected_items:
        selected_file = selected_items[0].toolTip()
        main_window.paint.set_image(selected_file)
        main_window.last_opened_file_name.setText(selected_file)

        _update_file_icon(main_window, list_widget)

        if is_seg_tab:
            _load_mask(main_window, selected_file)
        else:
            main_window.paint.maskVisible = False
            main_window.paint.disableMouseTracking()
    else:
        main_window.paint.set_image(None)
        main_window.last_opened_file_name.setText('')


def _update_file_icon(main_window, list_widget):
    """Update file icon in list."""
    try:
        if list_widget.currentItem() and list_widget.currentItem().icon().isNull():
            logger.debug('Updating icon')
            from qtpy.QtGui import QIcon, QPixmap
            icon = QIcon(QPixmap.fromImage(main_window.paint.paint.image))
            pixmap = icon.pixmap(24, 24)
            icon = QIcon(pixmap)
            list_widget.currentItem().setIcon(icon)
    except:
        traceback.print_exc()
        logger.warning('failed creating an icon for the selected file')


def _load_mask(main_window, selected_file):
    """Load mask file if available."""
    main_window.paint.maskVisible = True
    main_window.paint.enableMouseTracking()
    # Load outlines.tif (cellpose output)
    TA_path_outlines = smart_name_parser(
        selected_file,
        ordered_output='outlines.tif'
    )
    if TA_path_outlines and os.path.isfile(TA_path_outlines):
        main_window.paint.set_mask(TA_path_outlines)

def _reparent_preview_combo(main_window):
    """Reparent preview combo box to the current active tab."""
    if not hasattr(main_window, 'image_preview_combo'):
        return
    
    selected_tab_idx, selected_tab_name = main_window.get_cur_tab_index_and_name()
    combo = main_window.image_preview_combo
    
    # Get target layout based on current tab
    target_layout = None
    if 'tracking' in selected_tab_name and hasattr(main_window, 'tracking_preview_layout'):
        target_layout = main_window.tracking_preview_layout
    # Analysis tab no longer has a preview layout
    
    # If we found a target layout, ensure combo is in it
    if target_layout:
        # Check if combo is already in this layout
        combo_in_layout = False
        for i in range(target_layout.count()):
            item = target_layout.itemAt(i)
            if item and item.widget() == combo:
                combo_in_layout = True
                break
        
        if not combo_in_layout:
            # Remove combo from its current parent layout if any
            current_parent = combo.parent()
            if current_parent:
                parent_layout = current_parent.layout()
                if parent_layout:
                    # Try to find and remove from any layout
                    _remove_widget_from_layout(combo, parent_layout)
            
            # Add to target layout at position (0, 1) - after the label
            existing_item = target_layout.itemAtPosition(0, 1)
            if existing_item and existing_item.widget() and existing_item.widget() != combo:
                target_layout.removeWidget(existing_item.widget())
            target_layout.addWidget(combo, 0, 1)


def _remove_widget_from_layout(widget, layout):
    """Recursively remove widget from layout."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item:
            if item.widget() == widget:
                layout.removeWidget(widget)
                return True
            elif item.layout():
                if _remove_widget_from_layout(widget, item.layout()):
                    return True
    return False


def _load_tracked_cells_preview(main_window, selected_items):
    """Load tracked_cells_resized.tif preview for tracking tab."""
    if not selected_items:
        main_window.paint.set_image(None)
        return
    
    selected_file = selected_items[0].toolTip()
    TA_path = smart_name_parser(selected_file, ordered_output='TA')
    tracked_cells_path = os.path.join(TA_path, 'tracked_cells_resized.tif')
    if os.path.isfile(tracked_cells_path):
        # Use the preview handler to load the image with proper masking
        from gui.handlers.preview_handler import create_preview_from_file
        preview_image = create_preview_from_file(main_window, 'tracked_cells_resized.tif', selected_file, TA_path)
        if preview_image is not None:
            main_window.paint.set_image(preview_image)
            # Also load tracked image for click detection
            if hasattr(main_window.paint, 'paint'):
                main_window.paint.paint.load_tracked_image(selected_file)
                # Enable track_view_mode for hover and click functionality
                # Enable if selection mode is not active, OR if completeness overlay is enabled
                overlay_enabled = (hasattr(main_window, 'track_completeness_overlay_enabled') and 
                                  main_window.track_completeness_overlay_enabled)
                if (not hasattr(main_window, 'selection_mode_active') or 
                    not main_window.selection_mode_active or 
                    overlay_enabled):
                    main_window.paint.paint.track_view_mode = True
        else:
            # Fallback: load directly
            from utils.image_io import Img
            img = Img(tracked_cells_path)
            main_window.paint.set_image(img)
            if hasattr(main_window.paint, 'paint'):
                main_window.paint.paint.load_tracked_image(selected_file)
                # Enable track_view_mode for hover and click functionality
                # Enable if selection mode is not active, OR if completeness overlay is enabled
                overlay_enabled = (hasattr(main_window, 'track_completeness_overlay_enabled') and 
                                  main_window.track_completeness_overlay_enabled)
                if (not hasattr(main_window, 'selection_mode_active') or 
                    not main_window.selection_mode_active or 
                    overlay_enabled):
                    main_window.paint.paint.track_view_mode = True
    else:
        # File doesn't exist yet, show nothing or the original image
        main_window.paint.set_image(selected_file)