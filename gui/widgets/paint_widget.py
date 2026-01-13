"""Paint widget for image editing and annotation.

This module provides a paint widget for manual image editing, annotation,
and mask correction with drawing tools and segmentation support.
"""

import os
from utils.qt_settings import set_UI
set_UI()
import traceback
from qtpy.QtGui import QKeySequence
from skimage.measure import label
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from gui.utils.watershed_utils import apply_watershed, manually_reseeded_wshed
from gui.utils.membrane_editing_utils import extract_line_segment_between_points
from qtpy.QtCore import QRect, QTimer, Qt
from qtpy.QtWidgets import QWidget, QApplication, QMessageBox
from gui.dialogs.file_dialogs import saveFileDialog
from qtpy.QtWidgets import QMenu
from qtpy import QtCore, QtGui, QtWidgets
from utils.image_io import Img
from utils.image_utils import toQimage, get_white_bounds, RGB_to_int24, is_binary, auto_scale, save_as_tiff
from measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
# Replaced custom wshed with scikit-image watershed
from selections.selection import get_colors_drawn_over, convert_selection_color_to_coords
from tracking.utils.tools import smart_name_parser, get_mask_file
import numpy as np
from utils.logger import TA_logger # logging

logger = TA_logger()

class Createpaintwidget(QWidget):
    default_mask_name = 'handCorrection.tif'

    def __init__(self, enable_shortcuts=False):
        super().__init__()
        self.save_path = None
        self.raw_image = None # the original unprocessed image
        self.image = None # the displayed image (a qimage representation of raw_image can be a subset of raw_image_also)
        self.raw_mask = None  # the original unprocessed mask
        self.imageDraw = None # the mask image drawn over the image
        self.raw_user_drawing = None # contains just the user drawing without image mask
        self.cursor = None
        self.maskVisible = True
        self.scale = 1.0
        self.drawing = False
        self.brushSize = 1
        self.minimal_cell_size = 10
        self._clear_size = 30
        self.drawColor = QtGui.QColor(QtCore.Qt.red) # blue green cyan
        self.eraseColor = QtGui.QColor(QtCore.Qt.black)
        self.eraseColor_visualizer = QtGui.QColor(QtCore.Qt.green)
        self.cursorColor = QtGui.QColor(QtCore.Qt.green)
        self.lastPoint = QtCore.QPointF()
        self.change = False
        self.propagate_mouse_move = False
        # KEEP IMPORTANT required to track mouse even when not clicked
        self.setMouseTracking(True)  # KEEP IMPORTANT
        self.scrollArea = None
        self.statusBar = None
        self.drawing_enabled = True
        self.channel = None
        self.force_cursor_visible = False # required to force activate cursor to be always visible
        self.save_file_name=None
        self._saving = False  # Flag to prevent multiple simultaneous saves

        self.auto_convert_float_to_binary_threshold = 0.4

        # Cell selection mode state
        self.selection_mode = False
        self.track_view_mode = False  # Mode for viewing track completeness and clicking to fix
        self.track_correction_mode = False  # Mode for correcting tracks - click cells across frames, then apply
        self.track_correction_selected_cell_id = None  # Currently highlighted cell ID (first clicked)
        self.track_correction_marked_cells = {}  # {frame_idx: [(x, y, track_id), ...]} - cells marked for correction
        self.track_correction_circles = {}  # {frame_idx: [(x, y), ...]} - circle positions for visual markers
        self.track_correction_circle_size = 3  # Circle radius in pixels (adjustable)
        self.box_selection_start = None
        self.box_selection_end = None
        self.hovered_track_id = None
        self.tracked_image_int24 = None  # Cached tracked image in int24 format
        self.tracked_image_rgb = None  # Cached tracked image in RGB format (for getting colors)
        self._tracked_image_qimage = None  # Cached QImage of tracked_image_rgb for fast rendering
        self.current_file = None  # Current file path for loading tracked image
        self.main_window = None  # Reference to main window (set by main_window)

        # Undo system for segmentation correction
        self.undo_stack = []  # Stack to store mask states for undo
        self.max_undo_history = 50  # Maximum number of undo states to keep

        # Enable keyboard focus to receive key events
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        # Install event filter to catch Enter keys at a lower level
        self.installEventFilter(self)

        if enable_shortcuts:
            self.add_shortcuts()
    
    def eventFilter(self, obj, event):
        """Event filter to catch S keys in track correction mode before they're processed."""
        if obj == self and event.type() == QtCore.QEvent.KeyPress:
            if self.track_correction_mode:
                key = event.key()
                modifiers = event.modifiers()
                # Check for S key without modifiers (Ctrl+S is for save)
                if key == QtCore.Qt.Key_S and modifiers == QtCore.Qt.NoModifier:
                    # #region agent log
                    import json
                    import time
                    log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
                    try:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"id":"log_eventfilter_s","timestamp":int(time.time()*1000),"location":"paint_widget.py:eventFilter","message":"Event filter caught S key","data":{"key_code":key,"has_main_window":self.main_window is not None},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
                    except: pass
                    # #endregion
                    if self.main_window is not None:
                        self.main_window.apply_track_correction()
                        return True  # Event handled
        return super().eventFilter(obj, event)

    def force_cursor_to_be_visible(self, boolean):
        self.force_cursor_visible = boolean

        # can be used for saving
    def set_save_path(self, path):
        self.save_path = path

    def get_save_path(self):
        return self.save_path

    def set_scale(self, scale):
        # Make sure scale is bounded to avoid issues and non-sense scalings
        if scale<0.01:
            scale = 0.01
        if scale>50:
            scale=50

        self.scale = scale

    def get_scale(self):
        return self.scale

    def set_draw_color(self, color):
        """Set the overlay drawing color.
        
        Args:
            color: Can be a QColor object, a color name string (e.g., 'red', 'blue', 'green'),
                   or an RGB tuple (r, g, b) with values 0-255.
        """
        if isinstance(color, QtGui.QColor):
            self.drawColor = color
        elif isinstance(color, str):
            # Try to parse as a color name
            self.drawColor = QtGui.QColor(color)
        elif isinstance(color, (tuple, list)) and len(color) >= 3:
            # RGB tuple
            self.drawColor = QtGui.QColor(int(color[0]), int(color[1]), int(color[2]))
        else:
            raise ValueError(f"Invalid color format: {color}. Expected QColor, color name string, or RGB tuple.")
        
        # Only update the mask display if there's no active drawing to avoid interference
        # Check if there's any user drawing in progress
        has_active_drawing = False
        if self.raw_user_drawing is not None:
            try:
                all_channels = self.convert_qimage_to_numpy(self.raw_user_drawing)
                # Check if there are any drawn pixels (red or blue channels, excluding green which is for erasing)
                has_active_drawing = (np.count_nonzero(all_channels[..., 2] > 0) > 0 or 
                                    np.count_nonzero(all_channels[..., 0] > 0) > 0)
            except:
                pass
        
        # Only update mask display if no active drawing exists
        # This prevents interference with membrane addition/removal operations
        if not has_active_drawing and self.imageDraw is not None and self.raw_mask is not None:
            self.imageDraw = toQimage(Img(self.createRGBA(self.raw_mask), dimensions='hwc'), preserve_alpha=True)
            self.update()

    def get_draw_color(self):
        """Get the current overlay drawing color as a QColor object."""
        return self.drawColor

    def choose_draw_color(self):
        """Open a color picker dialog to choose the overlay color."""
        from qtpy.QtWidgets import QColorDialog
        color = QColorDialog.getColor(self.drawColor, self, "Choose Overlay Color")
        if color.isValid():
            self.set_draw_color(color)
            return color
        return None

    def _handle_enter_key_paint_widget(self):
        """Handle Enter key in paint widget - check track correction mode first."""
        # #region agent log
        import json
        import time
        log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
        try:
            track_correction_mode = getattr(self, 'track_correction_mode', False)
            has_main_window = hasattr(self, 'main_window') and self.main_window is not None
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":"log_paint_enter_handler","timestamp":int(time.time()*1000),"location":"paint_widget.py:_handle_enter_key_paint_widget","message":"Paint widget Enter handler called","data":{"track_correction_mode":track_correction_mode,"has_main_window":has_main_window},"sessionId":"debug-session","runId":"run1","hypothesisId":"E"}) + "\n")
        except: pass
        # #endregion
        
        if hasattr(self, 'track_correction_mode') and self.track_correction_mode:
            if hasattr(self, 'main_window') and self.main_window is not None:
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":"log_paint_enter_calling_apply","timestamp":int(time.time()*1000),"location":"paint_widget.py:_handle_enter_key_paint_widget","message":"Calling apply_track_correction from paint widget","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"E"}) + "\n")
                except: pass
                # #endregion
                self.main_window.apply_track_correction()
                return
        # Otherwise, use the normal apply method
        self.apply()
    
    def add_shortcuts(self):
        padEnterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)
        padEnterShortcut.activated.connect(self._handle_enter_key_paint_widget)
        padEnterShortcut.setContext(QtCore.Qt.ApplicationShortcut)

        enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        enterShortcut.activated.connect(self._handle_enter_key_paint_widget)
        enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)

        enterShortcut2 = QtWidgets.QShortcut('Shift+Return', self)
        enterShortcut2.activated.connect(self.apply)
        enterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)

        enterShortcut3 = QtWidgets.QShortcut('Shift+Enter', self)
        enterShortcut3.activated.connect(self.apply)
        enterShortcut3.setContext(QtCore.Qt.ApplicationShortcut)

        self.shrtM = QtWidgets.QShortcut("M", self)
        self.shrtM.activated.connect(self.m_apply)
        self.shrtM.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrl_shift_S_grab_screen_shot = QtWidgets.QShortcut('Ctrl+Shift+S', self)
        self.ctrl_shift_S_grab_screen_shot.activated.connect(self.grab_screen_shot)
        self.ctrl_shift_S_grab_screen_shot.setContext(QtCore.Qt.ApplicationShortcut)

        self.increase_contrastC = QtWidgets.QShortcut('C', self)
        self.increase_contrastC.activated.connect(self.increase_contrast)
        self.increase_contrastC.setContext(QtCore.Qt.ApplicationShortcut)
    def get_selection_coords_based_on_current_mask(self, bg_color=None):
        coords_of_selection = []
        img_to_analyze = self.get_raw_image()

        if is_binary(img_to_analyze):
            if len(img_to_analyze.shape)==3:
                img_to_analyze = img_to_analyze[...,0]
            img_to_analyze = label(img_to_analyze, connectivity=1, background=255)
            bg_color = 0

        mask=self.get_user_drawing()

        selected_cells =  get_colors_drawn_over(mask,  img_to_analyze)
        if not selected_cells:
            logger.warning('No selection found --> nothing to do')
            return coords_of_selection

        self.set_mask(np.zeros_like(mask))
        self.update()

        tmp = img_to_analyze
        if len(tmp.shape) == 3:
            tmp = RGB_to_int24(tmp)

        if bg_color is None:
            if 0xFFFFFF in tmp:
                bg_color = 0xFFFFFF
            else:
                bg_color = 0

        return convert_selection_color_to_coords(tmp, selected_cells=selected_cells, bg_color=bg_color)
    def get_colors_drawn_over(self, forbidden_colors=[0,0xFFFFFF]):
        mask = self.get_user_drawing()
        if mask is None:
            return

        selected_colors = get_colors_drawn_over(mask, self.get_raw_image(), forbidden_colors=forbidden_colors)
        self.set_mask(np.zeros_like(mask))
        self.update()
        return selected_colors

    def increase_contrast(self):
        try:
            print('auto-increase contrast')
            copy = auto_scale(np.copy(self.raw_image))
            meta = None
            try:
                meta=self.raw_image.metadata
            except:
                pass
            self.set_display(copy, metadata=meta)
        except:
            traceback.print_exc()

    def grab_screen_shot(self):
            output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.png);;All Files (*)",
                                         default_ext='.png')
            if output_file is not None:
                try:
                    self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
                    self.cursor.fill(QtCore.Qt.transparent)
                    self.update()
                    screenshot=self.grab()
                    screenshot.save(output_file, 'png')
                except:
                    logger.error('Could not grab a screenshot...')

    def _safe_save(self):
        """Wrapper for save() that catches all exceptions to prevent crashes."""
        try:
            self.save()
        except Exception as e:
            logger.error(f'Critical error in save: {e}')
            print(f'Critical error in save: {e}')
            traceback.print_exc()
            # Don't re-raise - prevent window from closing
    
    def save(self):
        """Save the current mask to file (default behavior)."""
        return self.save_mask()

    def apply(self):
        print('apply method to override')

    def shift_apply(self):
        print('shift apply method to override')

    def ctrl_m_apply(self):
        print('ctrl m apply method to override')

    def suppr_pressed(self):
        print('Suppr method to override')

    def m_apply(self):
        self.maskVisible = not self.maskVisible
        self.update()

    def set_display(self, display, metadata=None):
        if isinstance(display, np.ndarray):
            display = toQimage(display, metadata=metadata)
        self.image = display
        self.update()

    def set_image(self, img):
        # #region agent log
        import json
        import time
        import numpy as np
        log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
        try:
            img_info = None
            if isinstance(img, np.ndarray):
                unique_colors = None
                if img.size > 0:
                    if len(img.shape) == 3:
                        from utils.image_utils import RGB_to_int24
                        int24_temp = RGB_to_int24(img)
                        unique_colors = [f"{c:06x}" for c in np.unique(int24_temp)[:20]]
                    else:
                        unique_colors = [f"{c:06x}" for c in np.unique(img)[:20]]
                img_info = {"type":"ndarray","shape":list(img.shape),"unique_colors_sample":unique_colors}
            elif isinstance(img, str):
                img_info = {"type":"str","path":img}
            else:
                img_info = {"type":str(type(img))}
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":"log_set_image_entry","timestamp":int(time.time()*1000),"location":"paint_widget.py:296","message":"set_image entry","data":img_info,"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + "\n")
        except: pass
        # #endregion
        if img is None:
            self.save_file_name = None
            self.raw_image = None
            self.image = None
            self.imageDraw = None
            self.raw_mask = None
            self.raw_user_drawing = None
            self.undo_stack = []  # Clear undo stack when image is cleared
            self.unsetCursor()  # restore cursor
            self.update()
            return
        else:
            self.setCursor(Qt.BlankCursor)

        if isinstance(img, str):
            self.save_file_name = smart_name_parser(img, ordered_output=self.default_mask_name)
            img = Img(img)

        if isinstance(img, np.ndarray):
            self.raw_image = img
            # #region agent log
            try:
                unique_colors_raw = None
                if img.size > 0:
                    if len(img.shape) == 3:
                        from utils.image_utils import RGB_to_int24
                        int24_temp = RGB_to_int24(img)
                        unique_colors_raw = [f"{c:06x}" for c in np.unique(int24_temp)[:20]]
                    else:
                        unique_colors_raw = [f"{c:06x}" for c in np.unique(img)[:20]]
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_set_image_raw","timestamp":int(time.time()*1000),"location":"paint_widget.py:317","message":"set_image raw_image set","data":{"shape":list(img.shape),"unique_colors_sample":unique_colors_raw},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + "\n")
            except: pass
            # #endregion
            self.image = toQimage(img)
            if self.image is None:
                logger.error('Image could not be displayed...')
                self.unsetCursor()
                self.update()
                return
            self.imageDraw = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
            self.imageDraw.fill(QtCore.Qt.transparent)
            self.raw_user_drawing = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
            self.raw_user_drawing.fill(QtCore.Qt.transparent)
        else:
            self.image = toQimage(img[0])
            self.raw_image = img[0]
            self.imageDraw = toQimage(Img(self.createRGBA(img[1]), dimensions='hwc'), preserve_alpha=True)
            self.raw_user_drawing = QtGui.QImage(self.imageDraw.size(), QtGui.QImage.Format_ARGB32)
            self.raw_user_drawing.fill(QtCore.Qt.transparent)

        if self.force_cursor_visible:
            self.unsetCursor()
        width = self.image.size().width()
        height = self.image.size().height()
        top = self.geometry().x()
        left = self.geometry().y()
        self.setGeometry(top, left, int(width*self.scale), int(height*self.scale))

        self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
        self.cursor.fill(QtCore.Qt.transparent)
        self.update()

    def get_raw_image(self):
        return self.raw_image

    def binarize(self, mask, auto_convert_float_to_binary=None, force=False):
        if auto_convert_float_to_binary is None:
            auto_convert_float_to_binary = self.auto_convert_float_to_binary_threshold
        if auto_convert_float_to_binary and (mask.max() <= 1 or force):
            mask = mask > auto_convert_float_to_binary
        return mask
    def set_mask(self, mask, auto_convert_float_to_binary=None):
        if auto_convert_float_to_binary is None:
            auto_convert_float_to_binary = self.auto_convert_float_to_binary_threshold
        if isinstance(mask, str):
            self.save_file_name = mask
            mask = Img(mask)
            self.raw_mask = mask
            if mask.has_c():
                if self.channel is not None:
                    mask=mask[...,self.channel]
                else:
                    mask = mask[..., 0]
            # Clear undo stack when loading a new mask from file
            self.undo_stack = []
        if mask is None:
            self.imageDraw = None
            self.raw_user_drawing = None
            self.raw_mask = None
            self.undo_stack = []  # Clear undo stack when mask is cleared
        else:
            mask = self.binarize(mask, auto_convert_float_to_binary=auto_convert_float_to_binary)
            # Update raw_mask to keep it in sync with the current mask state
            # This ensures apply_drawing() uses the latest mask, not the original loaded one
            # Convert to numpy array if needed (handles both Img objects and numpy arrays)
            if isinstance(mask, np.ndarray):
                self.raw_mask = np.copy(mask)
            else:
                # If it's an Img object or other type, convert to numpy array
                try:
                    self.raw_mask = np.array(mask)
                except Exception:
                    self.raw_mask = mask  # Fallback to original if conversion fails
            self.imageDraw = toQimage(Img(self.createRGBA(mask), dimensions='hwc'), preserve_alpha=True)
            self.raw_user_drawing = QtGui.QImage(self.imageDraw.size(), QtGui.QImage.Format_ARGB32)
            self.raw_user_drawing.fill(QtCore.Qt.transparent)
        self.update()
    def reset_user_drawing(self):
        if self.imageDraw is None:
            self.raw_user_drawing = None
        else:
            self.raw_user_drawing = QtGui.QImage(self.imageDraw.size(), QtGui.QImage.Format_ARGB32)
            self.raw_user_drawing.fill(QtCore.Qt.transparent)

    def get_mask(self):
        if self.imageDraw is not None:
            mask = self.convert_qimage_to_numpy(self.imageDraw)[..., 2]
            mask[mask!=255]=0
            return mask
        else:
            return None

    def get_user_drawing(self, show_erased=False):
        if self.raw_user_drawing is not None:
            if self.raw_user_drawing.width() == 0 or self.raw_user_drawing.height() == 0:
                return None

            all = self.convert_qimage_to_numpy(self.raw_user_drawing)

            # QImage Format_RGB32 is actually BGRA format:
            # Channel 0 = Blue
            # Channel 1 = Green (used for erasing)
            # Channel 2 = Red (used for drawing)
            # Channel 3 = Alpha
            drawn = all[..., 2].copy()  # Red channel for drawn lines
            drawn[drawn != 255] = 0

            if show_erased:
                erased = all[..., 1].copy()  # Green channel for erased lines
                erased[erased != 255] = 0
                return [erased, drawn]

            return drawn
        else:
            return None

    def convert_qimage_to_numpy(self, qimage):
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        image_pointer = qimage.bits()
        try:
            image_pointer.setsize(qimage.sizeInBytes())
        except:
            image_pointer.setsize(qimage.byteCount())

        arr = np.array(image_pointer).reshape(height, width, 4)
        return arr

    def createRGBA(self, handCorrection):
        RGBA = np.zeros((handCorrection.shape[0], handCorrection.shape[1], 4), dtype=np.uint8)

        red = self.drawColor.red()
        green = self.drawColor.green()
        blue = self.drawColor.blue()

        RGBA[handCorrection != 0, 0] = blue
        RGBA[handCorrection != 0, 1] = green
        RGBA[handCorrection != 0, 2] = red
        RGBA[..., 3] = 255
        RGBA[handCorrection == 0, 3] = 0

        return RGBA
    def channelChange(self, i, skip_update_display=False):
        if self.raw_image is not None:
                if i == 0 and not skip_update_display:
                    self.set_display(self.raw_image)
                    self.channel = None
                else:
                    if not skip_update_display:
                        meta = None
                        try:
                            meta = self.raw_image
                        except:
                            pass
                        channel_img = self.raw_image.imCopy(c=i - 1)
                        self.set_display(channel_img, metadata=meta)
                    self.channel = i-1
                self.update()
        else:
            self.channel = None


    def get_colors_drawn_over(self):
        mask = self.get_user_drawing()
        if mask is None:
            return []

        selected_colors =self.get_raw_image()[mask!=0]
        selected_colors = RGB_to_int24(selected_colors)

        selected_colors =set(selected_colors.ravel().tolist())
        if 0xFFFFFF in selected_colors:
            selected_colors.remove(0xFFFFFF)
        selected_colors = list(selected_colors)
        self.set_mask(np.zeros_like(mask))

        self.update()
        return selected_colors

    def edit_drawing(self):
        drawn_mask = self.get_mask()
        if drawn_mask is None:
            return
        drawn_mask = np.copy(drawn_mask)
        erased, drawn = self.get_user_drawing(show_erased=True)

        drawn_mask[erased!=0]=0
        drawn_mask[drawn!=0]=drawn[drawn!=0]

        self.set_mask(Img(drawn_mask, dimensions='hw'))


    # Wrapper methods for backward compatibility
    def _apply_watershed(self, lab_mask, min_seed_area=0, guide_mask=None):
        """Wrapper for apply_watershed from paint_watershed module."""
        return apply_watershed(lab_mask, min_seed_area, guide_mask)
    
    def _detect_membrane_connection_points(self, drawn, drawn_mask):
        """Wrapper for detect_membrane_crossing_points from paint_membrane_editing module."""
        from gui.utils.membrane_editing_utils import detect_membrane_crossing_points
        return detect_membrane_crossing_points(drawn, drawn_mask)
    
    def _find_membrane_segment_to_remove(self, drawn, drawn_mask):
        """Wrapper for find_membrane_segment_to_remove from paint_membrane_editing module."""
        from gui.utils.membrane_editing_utils import find_membrane_segment_to_remove
        return find_membrane_segment_to_remove(drawn, drawn_mask)
    
    def _extract_line_segment_between_points(self, drawn, endpoint1, endpoint2, drawn_mask):
        """Wrapper for extract_line_segment_between_points from paint_membrane_editing module."""
        return extract_line_segment_between_points(drawn, endpoint1, endpoint2, drawn_mask)

    def apply_drawing(self, minimal_cell_size=0):
        # NOTE: Membrane editing (addition/removal) intentionally follows the test scripts in `test/scripts/`.
        # The legacy watershed/guide-mask approach is bypassed. `minimal_cell_size` is ignored here by design.

        def _normalize_outline_mask(mask):
            if mask is None:
                return None
            if isinstance(mask, np.ndarray) and mask.ndim > 2:
                mask = mask[..., 0]
            if isinstance(mask, np.ndarray) and mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            # Normalize to 0/255 (255 = membrane, 0 = cell interior)
            if isinstance(mask, np.ndarray) and mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            elif isinstance(mask, np.ndarray):
                mask = (mask > 127).astype(np.uint8) * 255
            return mask

        def _create_line_mask(shape, start_point, end_point, thickness=2):
            # Same helper behavior as `test/scripts/test_membrane_addition.py`
            from skimage.draw import line
            from scipy.ndimage import binary_dilation
            m = np.zeros(shape, dtype=np.uint8)
            rr, cc = line(int(start_point[0]), int(start_point[1]), int(end_point[0]), int(end_point[1]))
            valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            m[rr[valid], cc[valid]] = 255
            if thickness > 1:
                structure = np.ones((thickness, thickness), dtype=bool)
                m = binary_dilation(m > 0, structure=structure).astype(np.uint8) * 255
            return m

        # Prefer the underlying mask data (self.raw_mask) instead of get_mask()
        drawn_mask = self.raw_mask
        if drawn_mask is not None:
            try:
                drawn_mask = np.array(drawn_mask)
            except Exception:
                drawn_mask = None
        if drawn_mask is None:
            drawn_mask = self.get_mask()
        drawn_mask = _normalize_outline_mask(drawn_mask)

        if drawn_mask is None:
            logger.warning('apply_drawing: No mask available')
            return
        if self.raw_user_drawing is None:
            logger.warning('apply_drawing: No user drawing available (raw_user_drawing is None)')
            return

        user_drawing_result = self.get_user_drawing(show_erased=True)
        if user_drawing_result is None:
            logger.warning('apply_drawing: get_user_drawing returned None')
            return

        erased, drawn = user_drawing_result
        drawn_count = np.count_nonzero(drawn) if drawn is not None else 0
        erased_count = np.count_nonzero(erased) if erased is not None else 0
        if drawn_count == 0 and erased_count == 0:
            logger.warning('apply_drawing: No drawing detected in user drawing (all zeros)')
            return

        # Save current mask state to undo stack before modifying
        if self.raw_mask is not None:
            try:
                undo_state = np.copy(self.raw_mask)
                self.undo_stack.append(undo_state)
                # Limit undo history size
                if len(self.undo_stack) > self.max_undo_history:
                    self.undo_stack.pop(0)
            except Exception as e:
                logger.warning(f'Failed to save undo state: {e}')

        original_mask = np.copy(drawn_mask)
        new_mask = np.copy(drawn_mask)

        # Right-click removal: mirror `test/scripts/test_membrane_removal.py`
        # CRITICAL FIX: Only process erased pixels if there are NO drawn pixels.
        # When both are present (e.g., user drawing has both red and green channels),
        # we should prioritize addition over removal to avoid removing membranes that
        # should be preserved.
        if erased is not None and erased_count > 0 and (drawn is None or drawn_count == 0):
            ep1, ep2 = self._detect_membrane_connection_points(erased, original_mask)
            if ep1 is None or ep2 is None:
                logger.warning('apply_drawing: Membrane removal skipped (could not detect two crossing points)')
            else:
                segment_to_remove = self._find_membrane_segment_to_remove(erased, original_mask)
                if segment_to_remove is not None:
                    new_mask[segment_to_remove == 255] = 0
                else:
                    logger.warning('apply_drawing: No membrane segment found to remove; erasing drawn pixels only')
                    new_mask[erased != 0] = 0
        elif erased is not None and erased_count > 0 and drawn_count > 0:
            logger.info('apply_drawing: Both drawn and erased pixels present - prioritizing addition, skipping removal')

        # Left-click addition: mirror `test/scripts/test_membrane_addition.py`
        if drawn is not None and drawn_count > 0:
            ep1, ep2 = self._detect_membrane_connection_points(drawn, original_mask)
            if ep1 is None or ep2 is None:
                # Error: No crossing points detected, cancel the step and clear user drawing
                # Remove the undo state we just added since operation is cancelled
                if self.undo_stack:
                    self.undo_stack.pop()
                logger.error('apply_drawing: Membrane addition failed - could not detect two crossing points. Operation cancelled.')
                self.reset_user_drawing()
                self.update()
                return  # Cancel the operation, don't modify the mask
            
            drawn_segment = self._extract_line_segment_between_points(drawn, ep1, ep2, original_mask)
            if drawn_segment is None or np.count_nonzero(drawn_segment) == 0:
                # Error: Could not extract segment, cancel the step and clear user drawing
                # Remove the undo state we just added since operation is cancelled
                if self.undo_stack:
                    self.undo_stack.pop()
                logger.error('apply_drawing: Membrane addition failed - could not extract line segment between crossing points. Operation cancelled.')
                self.reset_user_drawing()
                self.update()
                return  # Cancel the operation, don't modify the mask
            
            # Add the segment while preserving existing membrane pixels
            # CRITICAL FIX: The segment extraction returns a 1-pixel skeleton path that includes
            # overlap pixels (p1, p2) which are already in the existing membrane. When we add
            # the segment, we must preserve ALL existing membrane pixels, not just those in the segment.
            # Ensure shapes match and use logical OR to combine: new_mask OR drawn_segment (both are 0/255)
            segment_pixels = int(np.count_nonzero(drawn_segment != 0))
            # Ensure drawn_segment has the same shape and dtype as new_mask
            if drawn_segment.shape != new_mask.shape:
                logger.warning(f'apply_drawing: Segment shape {drawn_segment.shape} != mask shape {new_mask.shape}, resizing segment')
                from scipy.ndimage import zoom
                zoom_factors = (new_mask.shape[0] / drawn_segment.shape[0], new_mask.shape[1] / drawn_segment.shape[1])
                drawn_segment = zoom(drawn_segment, zoom_factors, order=0)
                drawn_segment = (drawn_segment > 127).astype(new_mask.dtype) * 255
            else:
                drawn_segment = drawn_segment.astype(new_mask.dtype)
            overlap_pixels = int(np.count_nonzero((new_mask == 255) & (drawn_segment != 0)))
            new_mask_before_maximum = int(np.count_nonzero(new_mask == 255))
            # Use logical OR to ensure we preserve ALL existing membrane pixels
            # This is equivalent to: new_mask = new_mask | drawn_segment (for binary masks)
            new_mask = np.maximum(new_mask, drawn_segment)

        self.set_mask(new_mask)
        # Clear user drawing after applying
        self.reset_user_drawing()
        self.update()
        # Note: Auto-save removed - user must save manually with Ctrl+S or save icon

        return

    def force_consistent_range(self, img, auto_convert_float_to_binary=None):
        if auto_convert_float_to_binary is None:
            auto_convert_float_to_binary = self.auto_convert_float_to_binary_threshold
        # for ch in range(img.shape[-1]):
        #     tmp = img[...,ch]
        #     min = tmp.min()
        #     max = tmp.max()
        #     # if min!=max:
        #         # change range of the stuff
        #         # print('error in range')
        #     if max<=1.0:
        #         print('error in range')
        #     print(min, max)
        #     print(min, max)
        # img = 
        return self.binarize(img, auto_convert_float_to_binary=auto_convert_float_to_binary, force=True)

    def save_mask(self):
        """Save the current mask to `self.save_file_name`."""
        # Prevent multiple simultaneous save operations
        if getattr(self, "_saving", False):
            logger.debug('Save already in progress, skipping...')
            return False

        self._saving = True
        try:
            if self.save_file_name is None:
                logger.warning('Cannot save: no save file name specified')
                print('Cannot save: no save file name specified')
                return False

            mask = self.get_mask()
            if mask is None:
                logger.warning('Cannot save: no mask available')
                print('Cannot save: no mask available')
                return False

            if not isinstance(mask, np.ndarray):
                logger.error('Cannot save: mask is not a numpy array')
                print('Cannot save: mask is not a numpy array')
                return False

            Img(mask, dimensions='hw').save(self.save_file_name)
            logger.info(f'State saved to {self.save_file_name}')
            print(f'State saved to {self.save_file_name}')
            return True
        except Exception as e:
            logger.error(f'Failed to save mask: {e}')
            print(f'Failed to save mask: {e}')
            traceback.print_exc()
            return False
        finally:
            self._saving = False
    # this is the local seeded watershed à la TA
    def manually_reseeded_wshed(self):
        """Wrapper for manually_reseeded_wshed from paint_watershed module."""
        return manually_reseeded_wshed(self)


    def undo(self):
        """Undo the last mask modification."""
        if not self.undo_stack:
            logger.info('No undo history available')
            return
        
        try:
            # Restore previous mask state
            previous_mask = self.undo_stack.pop()
            self.set_mask(previous_mask)
            # Clear user drawing
            self.reset_user_drawing()
            self.update()
            logger.info('Undo: Restored previous mask state')
        except Exception as e:
            logger.error(f'Failed to undo: {e}')
            traceback.print_exc()

    def keyPressEvent(self, event):
        """Handle keyboard events, especially Ctrl+S for saving and Ctrl+Z for undo."""
        # #region agent log
        import json
        import time
        log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
        try:
            key_code = event.key()
            key_name = None
            is_return = (key_code == QtCore.Qt.Key_Return)
            is_enter = (key_code == QtCore.Qt.Key_Enter)
            if is_return:
                key_name = "Key_Return"
            elif is_enter:
                key_name = "Key_Enter"
            modifiers_value = int(event.modifiers()) if hasattr(event.modifiers(), '__int__') else str(event.modifiers())
            with open(log_path, 'a') as f:
                f.write(json.dumps({"id":"log_keypress_entry","timestamp":int(time.time()*1000),"location":"paint_widget.py:790","message":"keyPressEvent called","data":{"key_code":key_code,"key_name":key_name,"is_return":is_return,"is_enter":is_enter,"track_correction_mode":self.track_correction_mode,"has_focus":self.hasFocus(),"modifiers":modifiers_value,"Key_Return_value":int(QtCore.Qt.Key_Return),"Key_Enter_value":int(QtCore.Qt.Key_Enter)},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
        except Exception as e:
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_keypress_entry_error","timestamp":int(time.time()*1000),"location":"paint_widget.py:790","message":"Error in keyPressEvent logging","data":{"error":str(e)},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
            except: pass
        # #endregion
        
        if event.key() == QtCore.Qt.Key_S and event.modifiers() == QtCore.Qt.ControlModifier:
            event.accept()  # Accept the event to prevent default behavior
            self._safe_save()
            return  # Don't call super() to prevent default behavior
        elif event.key() == QtCore.Qt.Key_Z and event.modifiers() == QtCore.Qt.ControlModifier:
            event.accept()  # Accept the event to prevent default behavior
            self.undo()
            return  # Don't call super() to prevent default behavior
        elif self.track_correction_mode and event.key() == QtCore.Qt.Key_S and event.modifiers() == QtCore.Qt.NoModifier:
            # S key in track correction mode (without Ctrl) - apply correction
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_keypress_s_matched","timestamp":int(time.time()*1000),"location":"paint_widget.py:848","message":"S key matched in track correction mode","data":{"key_code":event.key(),"has_main_window":self.main_window is not None},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
            except: pass
            # #endregion
            event.accept()
            if self.main_window is not None:
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":"log_keypress_calling_apply_s","timestamp":int(time.time()*1000),"location":"paint_widget.py:853","message":"Calling apply_track_correction from keyPressEvent (S key)","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
                except: pass
                # #endregion
                self.main_window.apply_track_correction()
            return
        elif self.track_correction_mode and (event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter):
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_keypress_checking_enter","timestamp":int(time.time()*1000),"location":"paint_widget.py:817","message":"Checking Enter key condition","data":{"key_code":event.key(),"Key_Return":int(QtCore.Qt.Key_Return),"Key_Enter":int(QtCore.Qt.Key_Enter),"matches_return":event.key() == QtCore.Qt.Key_Return,"matches_enter":event.key() == QtCore.Qt.Key_Enter},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
            except: pass
            # #endregion
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_keypress_enter_matched","timestamp":int(time.time()*1000),"location":"paint_widget.py:817","message":"Enter key matched in track correction mode","data":{"key_code":event.key(),"has_main_window":self.main_window is not None},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
            except: pass
            # #endregion
            # Enter key in track correction mode - apply correction
            event.accept()
            if self.main_window is not None:
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":"log_keypress_calling_apply","timestamp":int(time.time()*1000),"location":"paint_widget.py:833","message":"Calling apply_track_correction from keyPressEvent","data":{},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
                except: pass
                # #endregion
                self.main_window.apply_track_correction()
            return
        else:
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_keypress_no_match","timestamp":int(time.time()*1000),"location":"paint_widget.py:806","message":"Key did not match any handler, calling super","data":{"key_code":key_code,"track_correction_mode":self.track_correction_mode},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + "\n")
            except: pass
            # #endregion
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if not self.hasMouseTracking() or not self.drawing_enabled:
            return
        self.clickCount = 1

        if self.selection_mode:
            # Handle selection mode
            if event.buttons() == QtCore.Qt.LeftButton:
                # Get raw widget coordinates
                widget_pos = event.position()
                # Convert to image coordinates for box selection storage
                zoom_corrected_pos = widget_pos / self.scale
                self.lastPoint = zoom_corrected_pos
                
                # Start box selection (store in image coordinates)
                self.box_selection_start = zoom_corrected_pos
                self.box_selection_end = zoom_corrected_pos
                self.drawing = True
                
                # Get track ID using raw widget coordinates (function handles conversion)
                track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
                if track_id is not None:
                    # Toggle selection - need to access main window's selected_track_ids
                    # This will be handled in mouseReleaseEvent to avoid issues
                    pass
        elif self.track_correction_mode and event.buttons() == QtCore.Qt.LeftButton:
            # Handle track correction mode - click cells across frames to mark them for correction
            if self.main_window is None:
                return
            
            # Ensure tracked image is loaded
            if self.tracked_image_int24 is None:
                current_file = self.main_window.get_selection()
                if current_file:
                    if not self.load_tracked_image(current_file):
                        if self.statusBar:
                            self.statusBar.showMessage('Track correction mode: Tracked cells image not found. Please track cells first.')
                        logger.warning('Track correction mode: Tracked cells image not found.')
                        return
                else:
                    if self.statusBar:
                        self.statusBar.showMessage('Track correction mode: No file selected.')
                    return
            
            widget_pos = event.position()
            clicked_track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
            
            if clicked_track_id is None:
                # Clicked on background - provide feedback
                if self.statusBar:
                    self.statusBar.showMessage('Track correction mode: Click on a cell to mark it.')
                return
            
            # Get current frame index
            current_frame_idx = self.main_window.get_selection_index()
            if current_frame_idx is None:
                current_frame_idx = 0
            
            # Convert widget position to image coordinates for circle storage
            img_x = int(widget_pos.x() / self.scale)
            img_y = int(widget_pos.y() / self.scale)
            
            # Initialize frame data structures if needed
            if current_frame_idx not in self.track_correction_marked_cells:
                self.track_correction_marked_cells[current_frame_idx] = []
            if current_frame_idx not in self.track_correction_circles:
                self.track_correction_circles[current_frame_idx] = []
            
            # First click: set selected cell and highlight it
            if self.track_correction_selected_cell_id is None:
                self.track_correction_selected_cell_id = clicked_track_id
                logger.info(f'Track correction mode: Selected cell {clicked_track_id:06x} at frame {current_frame_idx + 1}')
                
                # Mark this cell
                self.track_correction_marked_cells[current_frame_idx].append((img_x, img_y, clicked_track_id))
                self.track_correction_circles[current_frame_idx].append((img_x, img_y))
                
                if self.statusBar:
                    self.statusBar.showMessage(f'Track correction: Cell {clicked_track_id:06x} selected. Navigate frames and click cells to mark them. Press Enter when done.')
            else:
                # Subsequent clicks: add circle marker (brush mode)
                # Check if this cell is already marked in this frame
                already_marked = any(
                    abs(x - img_x) < 5 and abs(y - img_y) < 5 
                    for x, y, _ in self.track_correction_marked_cells[current_frame_idx]
                )
                
                if not already_marked:
                    self.track_correction_marked_cells[current_frame_idx].append((img_x, img_y, clicked_track_id))
                    self.track_correction_circles[current_frame_idx].append((img_x, img_y))
                    logger.info(f'Track correction mode: Marked cell {clicked_track_id:06x} at ({img_x}, {img_y}) in frame {current_frame_idx + 1}')
                    
                    if self.statusBar:
                        marked_count = sum(len(cells) for cells in self.track_correction_marked_cells.values())
                        self.statusBar.showMessage(f'Track correction: {marked_count} cell(s) marked. Press Enter when done.')
            
            self.update()  # Trigger repaint to show circle
            # Ensure widget maintains focus to receive keyboard events
            if not self.hasFocus():
                self.setFocus()
        # Handle track highlighting in tracking tab - simple toggle on click
        elif (event.buttons() == QtCore.Qt.LeftButton and 
              not self.selection_mode and 
              not self.track_correction_mode and
              self.tracked_image_int24 is not None and
              self.main_window is not None):
            # Check if we're in tracking tab
            current_tab_idx, tab_name = self.main_window.get_cur_tab_index_and_name()
            if tab_name == 'tracking':
                widget_pos = event.position()
                clicked_track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
                
                if clicked_track_id is not None:
                    # Initialize highlighted_track_ids if needed
                    if not hasattr(self.main_window, 'highlighted_track_ids'):
                        self.main_window.highlighted_track_ids = set()
                    
                    # Toggle highlight: if already highlighted, remove it; otherwise add it
                    if clicked_track_id in self.main_window.highlighted_track_ids:
                        self.main_window.highlighted_track_ids.discard(clicked_track_id)
                        logger.info(f'Unhighlighted track {clicked_track_id:06x}')
                    else:
                        self.main_window.highlighted_track_ids.add(clicked_track_id)
                        logger.info(f'Highlighted track {clicked_track_id:06x}')
                    
                    # Trigger repaint to show/hide highlight
                    self.update()
        # Disabled: clicking to show confidence data in tracking tab
        # elif self.track_view_mode and event.buttons() == QtCore.Qt.LeftButton:
        #     # Handle track view mode - click to show confidence and/or fix dialog
        #     widget_pos = event.position()
        #     track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
        #     if track_id is not None and self.main_window is not None:
        #         file_list = self.main_window.get_full_list(warn_on_empty_list=False) or []
        #         
        #         # Load confidence data for this track
        #         confidence_data = self.main_window.get_track_confidence(track_id, file_list)
        #         
        #         # Show confidence dialog for all tracks (complete or incomplete)
        #         if confidence_data:
        #             from gui.dialogs.track_confidence_dialog import show_track_confidence_dialog
        #             navigate_callback = lambda frame_idx: self.main_window.navigate_to_frame(frame_idx)
        #             show_track_confidence_dialog(
        #                 self.main_window, 
        #                 track_id, 
        #                 file_list, 
        #                 confidence_data,
        #                 navigate_callback
        #             )
        #         else:
        #             # No confidence data available - show message
        #             from qtpy.QtWidgets import QMessageBox
        #             QMessageBox.information(
        #                 self,
        #                 "No Confidence Data",
        #                 f"No confidence data available for track {track_id}.\n"
        #                 f"Re-run tracking to generate confidence scores."
        #             )
        elif event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton:
            self.drawing = True
            zoom_corrected_pos = event.position() / self.scale
            self.lastPoint = zoom_corrected_pos
            self.drawOnImage(event)
        else:
            self.drawing = False

    def mouseMoveEvent(self, event):
        if not self.hasMouseTracking() or not self.drawing_enabled:
            return
        else:
            if self.propagate_mouse_move == True:
                self.propagate_mouse_event(event)

        widget_pos = event.position()
        zoom_corrected_pos = widget_pos / self.scale
        
        if self.statusBar:
            self.statusBar.showMessage('x=' + str(zoom_corrected_pos.x()) + ' y=' + str(
                zoom_corrected_pos.y()))
        
        if self.selection_mode:
            # Update hovered cell - use raw widget coordinates
            track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
            if track_id != self.hovered_track_id:
                self.hovered_track_id = track_id
                self.update()
            
            # Update box selection if dragging (store in image coordinates)
            if self.drawing and self.box_selection_start is not None:
                self.box_selection_end = zoom_corrected_pos
                self.update()
        elif self.track_view_mode:
            # Update hovered cell in track view mode for better UX
            track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
            if track_id != self.hovered_track_id:
                self.hovered_track_id = track_id
                self.update()
        elif self.track_correction_mode:
            # Update hovered cell in track correction mode for better UX
            track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
            if track_id != self.hovered_track_id:
                self.hovered_track_id = track_id
                self.update()
            # Ensure widget has focus to receive keyboard events
            if not self.hasFocus():
                self.setFocus()
        else:
            self.drawOnImage(event)

    def mouseReleaseEvent(self, event):
        if not self.hasMouseTracking() or not self.drawing_enabled:
            return
        
        if self.selection_mode and event.button() == QtCore.Qt.LeftButton:
            widget_pos = event.position()
            zoom_corrected_pos = widget_pos / self.scale
            
            # Check if this was a box selection or single click
            if (self.box_selection_start is not None and self.box_selection_end is not None and
                abs(self.box_selection_end.x() - self.box_selection_start.x()) > 5 and
                abs(self.box_selection_end.y() - self.box_selection_start.y()) > 5):
                # Box selection
                # box_selection_start/end are in image coordinates (already divided by scale)
                # get_track_ids_in_rect expects image coordinates, so we can pass them directly
                from qtpy.QtCore import QRectF
                rect = QRectF(
                    min(self.box_selection_start.x(), self.box_selection_end.x()),
                    min(self.box_selection_start.y(), self.box_selection_end.y()),
                    abs(self.box_selection_end.x() - self.box_selection_start.x()),
                    abs(self.box_selection_end.y() - self.box_selection_start.y())
                )
                # Pass as tuple since get_track_ids_in_rect handles both QRectF and tuple
                track_ids = self.get_track_ids_in_rect((rect.x(), rect.y(), rect.width(), rect.height()))
                self._handle_cell_selection(track_ids)
            else:
                # Single click selection - use raw widget coordinates
                track_id = self.get_track_id_at_position(widget_pos.x(), widget_pos.y())
                if track_id is not None:
                    self._handle_cell_selection({track_id})
            
            # Reset box selection
            self.box_selection_start = None
            self.box_selection_end = None
            self.drawing = False
            self.update()
        else:
            self.drawing = False

        if self.clickCount == 1:
            QTimer.singleShot(QApplication.instance().doubleClickInterval(),
                              self.updateButtonCount)
    
    def _handle_cell_selection(self, track_ids):
        """
        Handle cell selection/deselection and update main window state.
        
        Args:
            track_ids (set): Set of track IDs to toggle selection for
        """
        if not track_ids:
            return
        
        # Get main window reference
        main_window = self.main_window
        if main_window is None:
            logger.warning('Main window reference not set in paint widget')
            return
        
        # Check if we're in analysis mode
        if hasattr(main_window, 'current_analysis') and main_window.current_analysis:
            # Analysis mode - use selection manager
            for track_id in track_ids:
                main_window.handle_analysis_cell_selection(track_id)
            # Update visual feedback
            buffer = main_window.selection_manager.get_current_buffer()
            main_window.selected_track_ids = set(buffer)
            self.update()
            return
        
        # Normal selection mode - toggle selection
        newly_selected = set()
        newly_deselected = set()
        
        for track_id in track_ids:
            if track_id in main_window.selected_track_ids:
                # Deselect
                main_window.selected_track_ids.discard(track_id)
                newly_deselected.add(track_id)
            else:
                # Select
                main_window.selected_track_ids.add(track_id)
                newly_selected.add(track_id)
        
        # Update database
        if newly_selected:
            main_window.update_selected_cells_in_db(newly_selected, selected=True)
        if newly_deselected:
            main_window.update_selected_cells_in_db(newly_deselected, selected=False)
        
        # Trigger repaint to show updated selection
        self.update()
    
    def propagate_mouse_event(self, event):
        print(self,event)
        pass

    def drawOnImage(self, event):
        if self.imageDraw is None:
            return
        # Ensure cursor is initialized
        if self.cursor is None or (self.image is not None and self.cursor.size() != self.image.size()):
            if self.image is not None:
                self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
                self.cursor.fill(QtCore.Qt.transparent)
            else:
                return
        zoom_corrected_pos = event.position() / self.scale
        if self.drawing and (event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton):
            self._draw_on_image(self.imageDraw, event, zoom_corrected_pos)
            if event.buttons() == QtCore.Qt.RightButton:
                self._draw_on_image(self.raw_user_drawing, event, zoom_corrected_pos, erase_color=self.eraseColor_visualizer)
            else:
                self._draw_on_image(self.raw_user_drawing, event, zoom_corrected_pos)
        painter = QtGui.QPainter(self.cursor)
        try:
            r = QtCore.QRect(QtCore.QPoint(), self._clear_size * QtCore.QSize() * self.brushSize)
            painter.save()
            r.moveCenter(self.lastPoint.toPoint())
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.eraseRect(r)
            painter.restore()

            stroke_size = 2
            if self.brushSize<6:
                stroke_size = 1

            painter.setPen(QtGui.QPen(self.cursorColor, stroke_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                      QtCore.Qt.RoundJoin))
            if self.brushSize >1:
                painter.drawEllipse(zoom_corrected_pos, int(self.brushSize / 2.),
                                int(self.brushSize / 2.))
            else:
                painter.drawPoint(zoom_corrected_pos)
        except:
            traceback.print_exc()
        finally:
            painter.end()
        try:
            region = self.scrollArea.widget().visibleRegion()
        except:
            region =self.visibleRegion()

        self.update(region)
        self.lastPoint = zoom_corrected_pos

    def _draw_on_image(self, image_to_be_drawn, event, zoom_corrected_pos, erase_color=None):
        painter = QtGui.QPainter(image_to_be_drawn)
        try:
            if event.buttons() == QtCore.Qt.LeftButton and not (event.modifiers() == QtCore.Qt.ControlModifier or event.modifiers() == QtCore.Qt.ShiftModifier):
                painter.setPen(QtGui.QPen(self.drawColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            else:
                if erase_color is None:
                    erase_color = self.eraseColor
                painter.setPen(QtGui.QPen(erase_color, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            if self.lastPoint != zoom_corrected_pos:
                painter.drawLine(self.lastPoint, zoom_corrected_pos)
            else:
                painter.drawPoint(zoom_corrected_pos)
        except:
            traceback.print_exc()
        finally:
            painter.end()
    def contextMenuEvent(self, event):
        pass

    def updateButtonCount(self):
        self.clickCount = 1

    def mouseDoubleClickEvent(self, event):
        self.clickCount = 2

    def load_tracked_image(self, file_path):
        """
        Load the tracked cells image for the given file path.
        
        Args:
            file_path (str): Path to the current image file
            
        Returns:
            bool: True if tracked image was loaded successfully, False otherwise
        """
        try:
            # #region agent log
            import json
            import time
            log_path = r"c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log"
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_load_tracked_entry","timestamp":int(time.time()*1000),"location":"paint_widget.py:1154","message":"load_tracked_image entry","data":{"file_path":file_path,"current_file":self.current_file,"has_tracked_int24":self.tracked_image_int24 is not None},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
            except: pass
            # #endregion
            if file_path == self.current_file and self.tracked_image_int24 is not None:
                # #region agent log
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"id":"log_load_tracked_early_return","timestamp":int(time.time()*1000),"location":"paint_widget.py:1166","message":"Early return - already loaded","data":{"file_path":file_path,"current_file":self.current_file},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
                except: pass
                # #endregion
                return True  # Already loaded
            
            tracked_image_path = smart_name_parser(file_path, ordered_output='tracked_cells_resized.tif')
            # #region agent log
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_load_tracked_path","timestamp":int(time.time()*1000),"location":"paint_widget.py:1168","message":"Tracked image path resolved","data":{"tracked_image_path":tracked_image_path,"file_exists":os.path.exists(tracked_image_path) if tracked_image_path else False},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
            except: pass
            # #endregion
            if not os.path.exists(tracked_image_path):
                self.tracked_image_int24 = None
                self.tracked_image_rgb = None
                self._tracked_image_qimage = None
                self.current_file = None
                return False
            
            tracked_image_rgb = Img(tracked_image_path)
            # #region agent log
            try:
                import numpy as np
                unique_colors_before = None
                if isinstance(tracked_image_rgb, np.ndarray) and tracked_image_rgb.size > 0:
                    if len(tracked_image_rgb.shape) == 3:
                        from utils.image_utils import RGB_to_int24
                        int24_temp = RGB_to_int24(tracked_image_rgb)
                        unique_colors_before = [f"{c:06x}" for c in np.unique(int24_temp)[:20]]
                    else:
                        unique_colors_before = [f"{c:06x}" for c in np.unique(tracked_image_rgb)[:20]]
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_load_tracked_loaded","timestamp":int(time.time()*1000),"location":"paint_widget.py:1176","message":"Tracked image loaded from file","data":{"shape":list(tracked_image_rgb.shape) if isinstance(tracked_image_rgb, np.ndarray) else None,"unique_colors_sample":unique_colors_before},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
            except: pass
            # #endregion
            if len(tracked_image_rgb.shape) == 3 and tracked_image_rgb.shape[2] == 3:
                self.tracked_image_rgb = tracked_image_rgb.copy()
                self.tracked_image_int24 = RGB_to_int24(tracked_image_rgb)
            else:
                # If it's already a single channel, assume it's int24
                self.tracked_image_int24 = tracked_image_rgb.astype(np.uint32)
                # Convert to RGB for color extraction
                from utils.image_utils import int24_to_RGB
                self.tracked_image_rgb = int24_to_RGB(self.tracked_image_int24)
            
            # Invalidate cached QImage so it will be regenerated on next paint
            self._tracked_image_qimage = None
            
            self.current_file = file_path
            # #region agent log
            try:
                unique_colors_after = None
                if self.tracked_image_int24 is not None:
                    unique_colors_after = [f"{c:06x}" for c in np.unique(self.tracked_image_int24)[:20]]
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"id":"log_load_tracked_complete","timestamp":int(time.time()*1000),"location":"paint_widget.py:1190","message":"load_tracked_image complete","data":{"current_file":self.current_file,"unique_colors_sample":unique_colors_after},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + "\n")
            except: pass
            # #endregion
            return True
        except Exception as e:
            logger.error(f'Error loading tracked image: {e}')
            traceback.print_exc()
            self.tracked_image_int24 = None
            self.tracked_image_rgb = None
            self._tracked_image_qimage = None
            self.current_file = None
            return False

    def get_track_id_at_position(self, x, y):
        """
        Get the track ID (int24 color) at the given pixel position.
        Accounts for zoom and scroll transformations.
        
        Args:
            x (float): X coordinate in widget space (from event.position())
            y (float): Y coordinate in widget space (from event.position())
            
        Returns:
            int or None: Track ID (int24 color) at position, or None if no track or error
        """
        if self.tracked_image_int24 is None:
            return None
        
        try:
            # event.position() gives widget-relative coordinates
            # The widget's coordinate system is already the content coordinate system
            # (the scroll area handles viewport transformation, not the widget)
            # So we just need to convert from widget coordinates to image coordinates by dividing by scale
            img_x = int(x / self.scale)
            img_y = int(y / self.scale)
            
            # Check bounds
            if (img_y < 0 or img_y >= self.tracked_image_int24.shape[0] or
                img_x < 0 or img_x >= self.tracked_image_int24.shape[1]):
                return None
            
            track_id = int(self.tracked_image_int24[img_y, img_x])
            
            # Filter out background (0xFFFFFF is white/background)
            if track_id == 0xFFFFFF or track_id == 0:
                return None
            
            return track_id
        except Exception as e:
            logger.error(f'Error getting track ID at position: {e}')
            return None

    def get_track_ids_in_rect(self, rect):
        """
        Get all unique track IDs within a rectangular region.
        
        Args:
            rect (QRectF or tuple): Rectangle as QRectF or (x, y, width, height) in image coordinates
            
        Returns:
            set: Set of unique track IDs (int24 colors) in the region
        """
        if self.tracked_image_int24 is None:
            return set()
        
        try:
            # Convert QRectF to coordinates
            # rect is already in image coordinates (from mousePressEvent which divides by scale)
            if hasattr(rect, 'x'):
                x1, y1 = int(rect.x()), int(rect.y())
                x2, y2 = int(rect.x() + rect.width()), int(rect.y() + rect.height())
            else:
                x1, y1, w, h = rect
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x1 + w), int(y1 + h)
            
            # Note: No need to account for scroll offset here because rect is already in image coordinates
            # The scroll offset is already accounted for when converting from widget to image coords in mousePressEvent
            
            # Ensure coordinates are within bounds
            h, w = self.tracked_image_int24.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x1 >= x2 or y1 >= y2:
                return set()
            
            # Extract region and get unique track IDs
            region = self.tracked_image_int24[y1:y2, x1:x2]
            unique_ids = np.unique(region)
            
            # Filter out background
            track_ids = set(int(tid) for tid in unique_ids if tid != 0xFFFFFF and tid != 0)
            return track_ids
        except Exception as e:
            logger.error(f'Error getting track IDs in rect: {e}')
            traceback.print_exc()
            return set()

    def paintEvent(self, event):
        # super(Createpaintwidget, self).paintEvent(event) # KEEP somehow tjis causes bugs --> do not uncomment it
        canvasPainter = QtGui.QPainter(self)
        try:

            # the scrollpane visible region
            try:
                visibleRegion = self.scrollArea.widget().visibleRegion()
            except:
                #assume no scroll region --> visible region is self visible region
                visibleRegion = self.visibleRegion()
            # the corresponding rect
            visibleRect = visibleRegion.boundingRect()
            # the visibleRect taking zoom into account
            scaledVisibleRect = QRect(int(visibleRect.x() / self.scale), int(visibleRect.y() / self.scale),
                                      int(visibleRect.width() / self.scale), int(visibleRect.height() / self.scale))
            

            if self.image is None:
                canvasPainter.eraseRect(visibleRect)
                # canvasPainter.end()
                return

            # In track correction mode:
            # - If no cell selected: show colored tracked cells image
            # - If cell selected: show handCorrection.tif as background
            if self.track_correction_mode:
                if self.track_correction_selected_cell_id is None:
                    # No cell selected yet - show colored tracked cells image
                    if self.tracked_image_rgb is not None:
                        # Convert tracked_image_rgb to QImage if not already cached
                        if not hasattr(self, '_tracked_image_qimage') or self._tracked_image_qimage is None:
                            self._tracked_image_qimage = toQimage(self.tracked_image_rgb)
                        canvasPainter.drawImage(visibleRect, self._tracked_image_qimage, scaledVisibleRect)
                    else:
                        # Fallback to regular image
                        canvasPainter.drawImage(visibleRect, self.image, scaledVisibleRect)
                else:
                    # Cell selected - show handCorrection.tif as background (clean, no additions)
                    current_file = None
                    if self.main_window is not None:
                        current_file = self.main_window.get_selection()
                    if current_file:
                        try:
                            filename_without_ext = smart_name_parser(current_file, ordered_output='full_no_ext')
                            handcorrection_path = get_mask_file(filename_without_ext)
                            if os.path.exists(handcorrection_path):
                                handcorrection_img = Img(handcorrection_path)
                                if handcorrection_img is not None:
                                    # Convert grayscale to RGB if needed (handCorrection.tif is often grayscale)
                                    handcorrection_array = np.asarray(handcorrection_img)
                                    if len(handcorrection_array.shape) == 2:
                                        # Grayscale - convert to RGB
                                        handcorrection_rgb = np.stack([handcorrection_array, handcorrection_array, handcorrection_array], axis=-1)
                                        handcorrection_qimage = toQimage(handcorrection_rgb)
                                    else:
                                        handcorrection_qimage = toQimage(handcorrection_img)
                                    canvasPainter.drawImage(visibleRect, handcorrection_qimage, scaledVisibleRect)
                                else:
                                    canvasPainter.fillRect(visibleRect, QtGui.QColor(0, 0, 0))  # Black fallback
                            else:
                                canvasPainter.fillRect(visibleRect, QtGui.QColor(0, 0, 0))  # Black fallback
                        except Exception as e:
                            logger.warning(f'Error loading handCorrection.tif: {e}')
                            canvasPainter.fillRect(visibleRect, QtGui.QColor(0, 0, 0))  # Black fallback
                    else:
                        canvasPainter.fillRect(visibleRect, QtGui.QColor(0, 0, 0))  # Black fallback
            # In selection mode, show colorcoded tracked image instead of regular image
            # BUT: if completeness overlay is enabled, use self.image (which has the overlay) instead
            # IMPORTANT: Only draw if NOT in track correction mode (to avoid covering handCorrection.tif)
            elif not self.track_correction_mode:
                overlay_enabled = (self.main_window is not None and 
                                 hasattr(self.main_window, 'track_completeness_overlay_enabled') and 
                                 self.main_window.track_completeness_overlay_enabled)
                if self.selection_mode and self.tracked_image_rgb is not None and not overlay_enabled:
                    # Convert tracked_image_rgb to QImage if not already cached
                    if not hasattr(self, '_tracked_image_qimage') or self._tracked_image_qimage is None:
                        self._tracked_image_qimage = toQimage(self.tracked_image_rgb)
                    canvasPainter.drawImage(visibleRect, self._tracked_image_qimage, scaledVisibleRect)
                else:
                    # Use self.image which has the completeness overlay applied (if enabled)
                    canvasPainter.drawImage(visibleRect, self.image, scaledVisibleRect)
            
            # Don't draw mask or cursor in track correction mode (keep it clean - black background only)
            if not self.track_correction_mode:
                if self.maskVisible and self.imageDraw is not None:
                    canvasPainter.drawImage(visibleRect, self.imageDraw, scaledVisibleRect)
                canvasPainter.drawImage(visibleRect, self.cursor, scaledVisibleRect)
            
            # Draw selection mode overlays
            if self.selection_mode and self.tracked_image_int24 is not None:
                self._draw_selection_overlays(canvasPainter, visibleRect, scaledVisibleRect)
            
            # Draw hover overlay in track view mode
            if self.track_view_mode and self.tracked_image_int24 is not None and self.hovered_track_id is not None:
                self._draw_hover_overlay(canvasPainter, visibleRect, scaledVisibleRect)
            
            # Draw track highlight overlay (works in both normal mode and when completeness overlay is active)
            # BUT: Skip in track correction mode to avoid covering handCorrection.tif
            if (not self.track_correction_mode and
                self.tracked_image_int24 is not None and 
                self.main_window is not None and 
                hasattr(self.main_window, 'highlighted_track_ids') and 
                self.main_window.highlighted_track_ids):
                self._draw_track_highlight_overlay(canvasPainter, visibleRect, scaledVisibleRect)
            
            # Draw track correction mode overlays (highlighting and circles)
            if self.track_correction_mode and self.tracked_image_int24 is not None:
                self._draw_track_correction_overlays(canvasPainter, visibleRect, scaledVisibleRect)
        except:
            traceback.print_exc()
        finally:
            canvasPainter.end()
    
    def _draw_selection_overlays(self, painter, visibleRect, scaledVisibleRect):
        """
        Draw selection overlays: selected cells, hovered cell, and selection box.
        
        Args:
            painter: QPainter object
            visibleRect: Visible rectangle in widget coordinates
            scaledVisibleRect: Visible rectangle in image coordinates
        """
        try:
            if self.main_window is None or self.tracked_image_int24 is None:
                return
            
            img_h, img_w = self.tracked_image_int24.shape[:2]
            from qtpy.QtGui import QImage
            
            # Draw background mask (darken non-selected areas, keep selected cells in full color)
            if self.main_window.selected_track_ids:
                painter.save()
                # Use stronger opacity for better contrast
                painter.setOpacity(0.7)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))  # Black mask for background
                painter.setPen(QtCore.Qt.NoPen)
                
                # Create a mask for selected cells
                selected_mask = np.zeros((img_h, img_w), dtype=bool)
                for track_id in self.main_window.selected_track_ids:
                    selected_mask |= (self.tracked_image_int24 == track_id)
                
                # Invert mask to get background (non-selected areas)
                background_mask = ~selected_mask
                
                # Convert background mask to QImage
                mask_data = background_mask.astype(np.uint8) * 255
                bytesPerLine = img_w
                mask_img = QImage(mask_data.tobytes(), 
                                 img_w, img_h,
                                 bytesPerLine,
                                 QImage.Format_Grayscale8)
                
                # Draw background mask using EXACTLY the same source/destination as main image
                painter.drawImage(visibleRect, mask_img, scaledVisibleRect)
                painter.restore()
            
            # Draw hovered cell with yellow fill (only in selection mode, track view mode uses separate method)
            if self.selection_mode and self.hovered_track_id is not None:
                painter.save()
                painter.setOpacity(0.5)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))  # Yellow fill
                painter.setPen(QtCore.Qt.NoPen)
                
                # Create a mask for hovered cell
                hovered_mask = (self.tracked_image_int24 == self.hovered_track_id)
                
                # Convert to QImage
                hovered_data = hovered_mask.astype(np.uint8) * 255
                bytesPerLine = img_w
                hovered_img = QImage(hovered_data.tobytes(), 
                                    img_w, img_h,
                                    bytesPerLine,
                                    QImage.Format_Grayscale8)
                
                # Draw hovered cell
                painter.drawImage(visibleRect, hovered_img, scaledVisibleRect)
                painter.restore()
            
            # Draw selection box
            if (self.box_selection_start is not None and self.box_selection_end is not None and
                self.drawing):
                painter.save()
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2, QtCore.Qt.DashLine))  # Red dashed
                painter.setBrush(QtCore.Qt.NoBrush)
                
                x1 = min(self.box_selection_start.x(), self.box_selection_end.x())
                y1 = min(self.box_selection_start.y(), self.box_selection_end.y())
                x2 = max(self.box_selection_start.x(), self.box_selection_end.x())
                y2 = max(self.box_selection_start.y(), self.box_selection_end.y())
                
                # box_selection coordinates are already in image space
                # Convert to widget coordinates using the same formula as drawImage
                scale_x = visibleRect.width() / scaledVisibleRect.width() if scaledVisibleRect.width() > 0 else 1.0
                scale_y = visibleRect.height() / scaledVisibleRect.height() if scaledVisibleRect.height() > 0 else 1.0
                rect_x = visibleRect.x() + (x1 - scaledVisibleRect.x()) * scale_x
                rect_y = visibleRect.y() + (y1 - scaledVisibleRect.y()) * scale_y
                rect_w = (x2 - x1) * scale_x
                rect_h = (y2 - y1) * scale_y
                
                painter.drawRect(int(rect_x), int(rect_y), int(rect_w), int(rect_h))
                painter.restore()
        except Exception as e:
            logger.error(f'Error drawing selection overlays: {e}')
            traceback.print_exc()
    
    def _draw_hover_overlay(self, painter, visibleRect, scaledVisibleRect):
        """
        Draw hover overlay for track view mode: highlight hovered cell.
        
        Args:
            painter: QPainter object
            visibleRect: Visible rectangle in widget coordinates
            scaledVisibleRect: Visible rectangle in image coordinates
        """
        try:
            if self.hovered_track_id is None or self.tracked_image_int24 is None:
                return
            
            img_h, img_w = self.tracked_image_int24.shape[:2]
            from qtpy.QtGui import QImage
            
            painter.save()
            painter.setOpacity(0.4)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))  # Yellow fill
            painter.setPen(QtCore.Qt.NoPen)
            
            # Create a mask for hovered cell
            hovered_mask = (self.tracked_image_int24 == self.hovered_track_id)
            
            # Convert to QImage
            hovered_data = hovered_mask.astype(np.uint8) * 255
            bytesPerLine = img_w
            hovered_img = QImage(hovered_data.tobytes(), 
                                img_w, img_h,
                                bytesPerLine,
                                QImage.Format_Grayscale8)
            
            # Draw hovered cell
            painter.drawImage(visibleRect, hovered_img, scaledVisibleRect)
            painter.restore()
        except Exception as e:
            logger.error(f'Error drawing hover overlay: {e}')
            traceback.print_exc()
    
    def _draw_track_highlight_overlay(self, painter, visibleRect, scaledVisibleRect):
        """
        Draw highlight overlay for highlighted tracks: show outlines and fill in track's original color.
        
        Args:
            painter: QPainter object
            visibleRect: Visible rectangle in widget coordinates
            scaledVisibleRect: Visible rectangle in image coordinates
        """
        try:
            if (self.main_window is None or 
                self.tracked_image_int24 is None or 
                self.tracked_image_rgb is None or
                not hasattr(self.main_window, 'highlighted_track_ids') or 
                not self.main_window.highlighted_track_ids):
                return
            
            img_h, img_w = self.tracked_image_int24.shape[:2]
            from qtpy.QtGui import QImage
            
            # Create overlay image for highlights
            highlight_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            
            # Process each highlighted track
            for track_id in self.main_window.highlighted_track_ids:
                track_mask = (self.tracked_image_int24 == track_id)
                
                if not np.any(track_mask):
                    continue  # Track not present in this frame
                
                # Get the RGB color for this track from the tracked image
                # Find first pixel of this track to get its color
                track_pixels = np.where(track_mask)
                if len(track_pixels[0]) > 0:
                    first_y, first_x = track_pixels[0][0], track_pixels[1][0]
                    track_color = self.tracked_image_rgb[first_y, first_x]
                    
                    # Draw filled cell in track's color (with some transparency handled by opacity)
                    highlight_overlay[track_mask] = track_color
                    
                    # Draw outline in track's color
                    # Create a labeled image with just this track for boundary detection
                    track_labeled = track_mask.astype(np.uint8)
                    track_boundaries = find_boundaries(track_labeled, mode='inner', connectivity=1)
                    highlight_overlay[track_boundaries] = track_color
            
            if not np.any(highlight_overlay):
                return  # No highlighted tracks in this frame
            
            painter.save()
            painter.setOpacity(0.7)  # Slight transparency to see underlying image
            
            # Convert overlay to QImage
            highlight_data = highlight_overlay.tobytes()
            bytesPerLine = img_w * 3  # RGB = 3 bytes per pixel
            highlight_img = QImage(highlight_data, 
                                  img_w, img_h,
                                  bytesPerLine,
                                  QImage.Format_RGB888)
            
            # Draw highlighted tracks
            painter.drawImage(visibleRect, highlight_img, scaledVisibleRect)
            painter.restore()
        except Exception as e:
            logger.error(f'Error drawing track highlight overlay: {e}')
            traceback.print_exc()
    
    def _draw_track_correction_overlays(self, painter, visibleRect, scaledVisibleRect):
        """
        Draw track correction mode overlays: highlight selected cell, show outlines for others, and draw circles.
        
        Args:
            painter: QPainter object
            visibleRect: Visible rectangle in widget coordinates
            scaledVisibleRect: Visible rectangle in image coordinates
        """
        try:
            if self.tracked_image_int24 is None:
                return
            
            img_h, img_w = self.tracked_image_int24.shape[:2]
            from qtpy.QtGui import QImage
            
            # Get current frame index
            current_frame_idx = None
            if self.main_window is not None:
                current_frame_idx = self.main_window.get_selection_index()
            if current_frame_idx is None:
                current_frame_idx = 0
            
            # Draw selected track cells as gray (all cells belonging to the selected track)
            # NOTE: This is drawn on top of handCorrection.tif - user may want to disable this
            if self.track_correction_selected_cell_id is not None:
                # Find all cells with the same track ID as the selected cell
                selected_track_mask = (self.tracked_image_int24 == self.track_correction_selected_cell_id)
                
                if np.any(selected_track_mask):
                    # Create gray overlay for selected track (gray cells)
                    gray_value = 128  # Medium gray
                    overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8, order='C')
                    overlay[selected_track_mask] = [gray_value, gray_value, gray_value]  # Gray color
                    
                    # Ensure array is contiguous for QImage
                    overlay = np.ascontiguousarray(overlay)
                    
                    # Convert to QImage (RGB888 format: width, height, bytesPerLine)
                    overlay_qimage = QImage(overlay.tobytes(), 
                                          img_w, img_h,
                                          img_w * 3,
                                          QImage.Format_RGB888)
                    
                    # Draw gray overlay with some transparency
                    painter.save()
                    painter.setOpacity(0.7)  # Semi-transparent so background shows through
                    painter.drawImage(visibleRect, overlay_qimage, scaledVisibleRect)
                    painter.restore()
                    
                    # Highlight the selected cell more prominently (bright yellow highlight)
                    # Get the clicked cell position from the current frame
                    if current_frame_idx in self.track_correction_marked_cells:
                        for img_x, img_y, marked_track_id in self.track_correction_marked_cells[current_frame_idx]:
                            if marked_track_id == self.track_correction_selected_cell_id:
                                # This is the selected cell - the area highlight is drawn separately after outlines
                                break
                    else:
                        # Try to find the selected cell in the tracked image directly
                        # Find the centroid of the selected track in current frame
                        selected_cell_mask = (self.tracked_image_int24 == self.track_correction_selected_cell_id)
                        if np.any(selected_cell_mask):
                            from skimage.measure import regionprops
                            labeled_mask = label(selected_cell_mask.astype(int), connectivity=1)
                            props = regionprops(selected_cell_mask.astype(int))
                            if props:
                                # Get centroid of the selected cell
                                centroid = props[0].centroid
                                img_y, img_x = int(centroid[0]), int(centroid[1])
                                painter.save()
                                # Draw a very visible highlight - bright yellow with thick border
                                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 8))  # Bright yellow, very thick
                                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0, 200)))  # More opaque yellow fill
                                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)  # Smooth edges
                                
                                # Convert image coordinates to widget coordinates
                                scale_x = visibleRect.width() / scaledVisibleRect.width() if scaledVisibleRect.width() > 0 else 1.0
                                scale_y = visibleRect.height() / scaledVisibleRect.height() if scaledVisibleRect.height() > 0 else 1.0
                                widget_x = visibleRect.x() + (img_x - scaledVisibleRect.x()) * scale_x
                                widget_y = visibleRect.y() + (img_y - scaledVisibleRect.y()) * scale_y
                                
                                # Draw a larger circle to highlight the selected cell
                                highlight_radius = self.track_correction_circle_size * 2.0 * scale_x  # Make it even larger
                                painter.drawEllipse(int(widget_x - highlight_radius), int(widget_y - highlight_radius), 
                                                   int(highlight_radius * 2), int(highlight_radius * 2))
                                painter.restore()
            
            # Draw white outlines for all cells (only when a cell is selected)
            if self.track_correction_selected_cell_id is not None:
                # Create outline mask for all cells
                all_cells_mask = (self.tracked_image_int24 != 0xFFFFFF) & (self.tracked_image_int24 != 0)
                
                # Use find_boundaries to get outlines
                if np.any(all_cells_mask):
                    # Create a labeled mask for all cells
                    labeled = label(all_cells_mask, connectivity=1)
                    
                    # Find boundaries between regions
                    outline = find_boundaries(labeled, mode='inner', connectivity=1)
                    
                    # Skeletonize to make outlines thinner (1-pixel width)
                    outline_binary = outline.astype(bool)
                    outline_skeleton = skeletonize(outline_binary)
                    
                    # Convert skeletonized outline to QImage (white)
                    outline_data = outline_skeleton.astype(np.uint8) * 255
                    bytesPerLine = img_w
                    outline_img = QImage(outline_data.tobytes(), 
                                        img_w, img_h,
                                        bytesPerLine,
                                        QImage.Format_Grayscale8)
                    
                    # Draw white outline
                    painter.save()
                    painter.setOpacity(1.0)  # Full opacity for white outlines
                    painter.drawImage(visibleRect, outline_img, scaledVisibleRect)
                    painter.restore()
            
            # Draw selected cell area highlight (AFTER outlines so it's on top)
            if self.track_correction_selected_cell_id is not None:
                # Find the selected cell mask and highlight the entire cell area
                selected_cell_mask = (self.tracked_image_int24 == self.track_correction_selected_cell_id)
                if np.any(selected_cell_mask):
                    # Create yellow overlay for the selected cell area
                    highlight_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8, order='C')
                    highlight_overlay[selected_cell_mask] = [255, 255, 0]  # Bright yellow
                    highlight_overlay = np.ascontiguousarray(highlight_overlay)
                    
                    # Convert to QImage
                    highlight_qimage = QImage(highlight_overlay.tobytes(), 
                                            img_w, img_h,
                                            img_w * 3,
                                            QImage.Format_RGB888)
                    
                    # Draw highlighted cell area with transparency
                    painter.save()
                    painter.setOpacity(0.5)  # Semi-transparent so we can see the cell
                    painter.drawImage(visibleRect, highlight_qimage, scaledVisibleRect)
                    painter.restore()
            
            # Draw white circles for marked cells from all frames (to help trace the track)
            if self.track_correction_circles:
                circle_radius = self.track_correction_circle_size  # pixels in image coordinates (adjustable)
                scale_x = visibleRect.width() / scaledVisibleRect.width() if scaledVisibleRect.width() > 0 else 1.0
                scale_y = visibleRect.height() / scaledVisibleRect.height() if scaledVisibleRect.height() > 0 else 1.0
                
                # Draw circles from all frames (white, with different opacity for current vs other frames)
                for frame_idx, circles in self.track_correction_circles.items():
                    painter.save()
                    if frame_idx == current_frame_idx:
                        # Current frame: bright white, thicker
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3))
                        painter.setOpacity(1.0)
                    else:
                        # Other frames: dimmer white, thinner (to show track history)
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                        painter.setOpacity(0.6)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    
                    for img_x, img_y in circles:
                        # Convert image coordinates to widget coordinates
                        widget_x = visibleRect.x() + (img_x - scaledVisibleRect.x()) * scale_x
                        widget_y = visibleRect.y() + (img_y - scaledVisibleRect.y()) * scale_y
                        
                        # Draw circle (radius in widget coordinates)
                        widget_radius = circle_radius * scale_x
                        painter.drawEllipse(int(widget_x - widget_radius), int(widget_y - widget_radius), 
                                           int(widget_radius * 2), int(widget_radius * 2))
                    
                    painter.restore()
        except Exception as e:
            logger.error(f'Error drawing track correction overlays: {e}')
            traceback.print_exc()
    
if __name__ == '__main__':
    import sys
    from qtpy.QtWidgets import QApplication
    from matplotlib import pyplot as plt

    app = QApplication(sys.argv)
    class overriding_apply(Createpaintwidget):
        def apply(self):
            plt.imshow(self.get_mask())
            plt.show()

    w = overriding_apply(enable_shortcuts=True)
    w.set_image('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')

    w.show()
    sys.exit(app.exec())


