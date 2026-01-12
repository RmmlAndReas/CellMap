"""Main GUI window for CellMap."""

import gc
import os
import sys
import traceback
import numpy as np
from functools import partial
from pathlib import Path

from utils.qt_settings import set_UI
set_UI()

from qtpy.QtWidgets import (QWidget, QComboBox, QProgressBar, QVBoxLayout, QLabel, QTextBrowser,
                             QGroupBox, QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup,
                             QDialogButtonBox, QDialog, QHBoxLayout, QScrollArea, QMessageBox,
                             QSpinBox, QLineEdit, QStackedWidget, QGridLayout, QPushButton,
                             QFrame, QTabWidget, QApplication)
from qtpy.QtGui import (QPalette, QPixmap, QColor, QPainter, QBrush, QPen, QFontMetrics,
                        QTextCursor, QTextCharFormat, QIcon)
from qtpy.QtSvg import QSvgRenderer
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt, QTimer, QThreadPool

from gui.dialogs.file_dialogs import saveFileDialog
from utils.image_io import Img
from utils.image_utils import blend, to_stack, mask_colors
from gui.widgets.overlay_widget import Overlay
from gui.dialogs.tracking_static_dialog import TrackingDialog
from tracking.last_tracking_based_on_matching_or_on_translation_from_mermaid_warp import match_by_max_overlap_lst
from utils.qthandler import XStream, QtHandler
from gui.utils.blinker import Blinker
from utils.list_utils import create_list
from database.sqlite_db import populate_table_content, createMasterDB, save_track_completeness, load_track_completeness, get_master_db_path
from natsort import natsorted


from gui.widgets.dual_list_widget import dualList
from gui.widgets.tissue_analyzer_paint_widget import tascrollablepaint
from gui.tabs.analysis_tab import create_analysis_tab
from gui.tabs.tracking_tab import create_tracking_tab
from gui.tabs.cellpose_tab import create_cellpose_tab
from tracking.local_to_track_correspondance import add_localID_to_trackID_correspondance_in_DB
from tracking.tools import smart_name_parser
import qtawesome as qta
from utils.early_stopper import early_stop
import logging
from tracking.tracking_yet_another_approach_pyramidal_registration_n_neo_swapping_correction import track_cells_dynamic_tissue
from utils.logger import TA_logger
from utils.fake_worker import FakeWorker
from utils.worker import Worker

from gui.handlers.preview_handler import create_preview_from_db, create_preview_from_file
from gui.handlers.tab_handler import update_preview_for_tab
from gui.handlers.logger_handler import set_html_red, set_html_black
from gui.handlers.drag_drop_handler import handle_drag_enter, handle_drag_move, handle_drop

DEBUG = False
logger = TA_logger()

__MAJOR__ = 1
__MINOR__ = 0
__MICRO__ = 6
__RELEASE__ = 'a'
__VERSION__ = '1'
__NAME__ = 'CellMap'


class TissueAnalyzer(QtWidgets.QMainWindow):
    """Main application window for CellMap tissue analysis."""

    def __init__(self, parent=None):
        try:
            super().__init__(parent)
            self.initUI()
            self.setAcceptDrops(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    def initUI(self):
        """Initialize all UI components and layouts."""
        try:
            delayed_preview_update = QTimer()
            delayed_preview_update.setSingleShot(True)
            delayed_preview_update.timeout.connect(self.preview_changed)

            self.master_db = None
            self.is_dark_theme = False  # Track current theme state
            self._setup_window()
            self._setup_widgets()
            self._setup_tabs(delayed_preview_update)
            self._setup_logging()
            self._setup_layout()
            self._setup_statusbar()
            self._setup_menu_and_shortcuts()
            self._setup_threading()
        except Exception as e:
            logger.error(f"Error in initUI: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_window(self):
        """Configure main window properties."""
        try:
            screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
            centerPoint = QApplication.desktop().screenGeometry(screen).center()
        except:
            from qtpy.QtGui import QGuiApplication
            centerPoint = QGuiApplication.primaryScreen().geometry().center()

        self.setGeometry(QtCore.QRect(centerPoint.x() - 450, centerPoint.y() - 350, 900, 700))
        # Set window icon from logo file with multiple sizes for Windows taskbar and title bar
        logo_path = Path(__file__).parent.parent / "logo" / "logo.svg"
        if logo_path.exists():
            icon = QIcon()
            # Add multiple sizes including larger ones for title bar display
            sizes = [16, 32, 48, 64, 128, 256, 512]
            renderer = QSvgRenderer(str(logo_path))
            if renderer.isValid():
                for size in sizes:
                    pixmap = QPixmap(size, size)
                    pixmap.fill(Qt.transparent)
                    painter = QPainter(pixmap)
                    painter.setRenderHint(QPainter.Antialiasing)
                    painter.setRenderHint(QPainter.SmoothPixmapTransform)
                    renderer.render(painter)
                    painter.end()
                    icon.addPixmap(pixmap)
            self.setWindowIcon(icon)
            # Also set on QApplication to ensure it's used everywhere
            app = QApplication.instance()
            if app:
                app.setWindowIcon(icon)
        else:
            # Fallback to qtawesome icon if logo not found
            fallback_icon = qta.icon('mdi.hexagon-multiple-outline', color=QColor(200, 200, 200))
            self.setWindowIcon(fallback_icon)
            app = QApplication.instance()
            if app:
                app.setWindowIcon(fallback_icon)
        self.setWindowTitle(f'{__NAME__} v{__VERSION__}')

        self.scale = 1.0
        self.min_scaling_factor = 0.1
        self.max_scaling_factor = 20
        self.zoom_increment = 0.05

    def _setup_widgets(self):
        """Create and configure main widgets."""
        self.paint = tascrollablepaint()
        # Store main window reference in paint widget for selection mode
        if hasattr(self.paint, 'paint'):
            self.paint.paint.main_window = self
        
        self.Stack = QStackedWidget(self)
        self.last_opened_file_name = QLabel('')
        self.list = dualList()
        self.list.setToolTip("Drag and drop your images over this list")
        for lst in self.list.lists:
            lst.list.selectionModel().selectionChanged.connect(self.selectionChanged)
        
        # Create preview combo box for analysis tab
        self.image_preview_combo = QComboBox()
        self.image_preview_combo.setToolTip("Select which image to preview in the analysis tab")
        
        # Cell selection mode state
        self.selection_mode_active = False
        self.selected_track_ids = set()  # Set of track IDs (int24 colors) that are selected
        self.highlighted_track_ids = set()  # Set of track IDs (int24 colors) that are highlighted
        self.track_completeness_cache = {}  # Cache: {track_id: bool} - True if complete (green), False if incomplete (red)
        self.track_completeness_overlay_enabled = False  # Toggle for showing completeness overlay
        
        # Analysis framework
        from gui.analysis.registry import AnalysisRegistry
        from gui.handlers.selection_manager import SelectionManager
        self.analysis_registry = AnalysisRegistry()
        self.selection_manager = SelectionManager()
        self.current_analysis = None  # Name of currently active analysis

    def _setup_tabs(self, delayed_preview_update):
        """Create and configure all tabs."""
        self.tabs = QTabWidget(self)
        self.tabs.setFixedHeight(300)
        
        self.tab1_scroll = create_cellpose_tab(self)
        self.tab2b_scroll = create_tracking_tab(self)
        self.tab2_scroll = create_analysis_tab(self, delayed_preview_update)

        tab_configs = [
            (self.tab1_scroll, 'Segmentation', 'Segment epithelial images using CellPose.'),
            (self.tab2b_scroll, 'Tracking', 'Track cells (Segmentation is required)'),
            (self.tab2_scroll, 'Analysis', 'Analyze cell properties and create publication-ready visualizations.'),
        ]
        
        for idx, (tab, name, tooltip) in enumerate(tab_configs):
            self.tabs.addTab(tab, name)
            if tooltip:
                self.tabs.setTabToolTip(idx, tooltip)

        self.tabs.setCurrentIndex(0)
        self.tabs.currentChanged.connect(self.onTabChange)
        self.list.set_list(0)

    def _setup_logging(self):
        """Setup logging console."""
        self.logger_console = QTextBrowser(self)
        self.logger_console.setReadOnly(True)
        self.logger_console.textCursor().movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, 1)
        self.logger_console.setHtml('<html>')
        self.logger_console.ensureCursorVisible()
        self.logger_console.document().setMaximumBlockCount(1000)
        
        if not DEBUG:
            # Redirect sys.stdout and sys.stderr to XStream for capturing print statements
            import sys
            sys.stdout = XStream.stdout()
            sys.stderr = XStream.stderr()
            
            # Use Qt.QueuedConnection for thread-safe logging from worker threads
            XStream.stdout().messageWritten.connect(
                lambda text: set_html_black(self.logger_console, text),
                Qt.QueuedConnection
            )
            XStream.stderr().messageWritten.connect(
                lambda text: set_html_red(self.logger_console, text),
                Qt.QueuedConnection
            )
            self.handler = QtHandler()
            self.handler.setFormatter(logging.Formatter(TA_logger.default_format))
            TA_logger.setHandler(self.handler)

    def _setup_layout(self):
        """Setup main layout."""
        self.table_widget = QWidget()
        table_widget_layout = QGridLayout()
        table_widget_layout.addWidget(self.tabs, 0, 0)

        self.groupBox_logging = QGroupBox('Log')
        self.groupBox_logging.setToolTip("Shows TA current status, method progress, warnings and errors")
        self.groupBox_logging.setMinimumWidth(250)
        groupBox_layout = QGridLayout()
        groupBox_layout.addWidget(self.logger_console, 0, 0)
        self.groupBox_logging.setLayout(groupBox_layout)
        table_widget_layout.addWidget(self.groupBox_logging, 0, 1)
        # Set column stretch: tabs (2/3) and log (1/3)
        table_widget_layout.setColumnStretch(0, 2)
        table_widget_layout.setColumnStretch(1, 1)
        self.table_widget.setLayout(table_widget_layout)

        self.Stack.addWidget(self.paint)

        self.grid = QGridLayout()
        self.grid.setSpacing(0)  # Remove spacing between rows/columns to eliminate gap
        self.grid.addWidget(self.Stack, 0, 0, 2, 1)  # Image spans rows 0 and 1
        self.grid.addWidget(self.list, 0, 1, 2, 1)  # File list spans rows 0 and 1 (removed track highlight list)
        self.grid.setRowStretch(0, 50)
        self.grid.setRowStretch(1, 50)
        self.grid.setRowStretch(2, 25)
        self.grid.setColumnStretch(0, 80)  # Image column - 80% of width
        self.grid.setColumnStretch(1, 20)  # File selection column - 20% of width
        self.grid.addWidget(self.table_widget, 2, 0, 1, 2)

        self.setCentralWidget(QFrame())
        self.centralWidget().setLayout(self.grid)

    def _setup_statusbar(self):
        """Setup status bar with progress and controls."""
        statusBar = self.statusBar()
        self.paint.statusBar = statusBar

        self.pbar = QProgressBar(self)
        self.pbar.setToolTip('Segmentation progress indicator showing the percentage of completed computation.')
        self.pbar.setGeometry(200, 80, 250, 20)
        statusBar.addWidget(self.pbar)

        self.stop_all_threads_button = QPushButton('Stop')
        self.stop_all_threads_button.setToolTip("Stops the running function or deep learning prediction as soon as possible.")
        self.stop_all_threads_button.clicked.connect(self.stop_threads_immediately)
        statusBar.addWidget(self.stop_all_threads_button)

        self.about = QPushButton()
        self.about.setIcon(qta.icon('mdi.information-variant', options=[{'scale_factor': 1.5}]))
        self.about.clicked.connect(self.about_dialog)
        self.about.setToolTip('About...')
        statusBar.addWidget(self.about)
        statusBar.addWidget(self.last_opened_file_name)
        
        # Add theme toggle button to the right side of status bar
        self.theme_button = QPushButton()
        # Initialize with moon icon (since we start with light theme, button shows dark mode icon)
        self.theme_button.setIcon(qta.icon('mdi.weather-night', options=[{'scale_factor': 1.5}]))
        self.theme_button.setToolTip('Toggle dark/light theme')
        self.theme_button.clicked.connect(self.toggle_theme)
        statusBar.addPermanentWidget(self.theme_button)

    def _setup_menu_and_shortcuts(self):
        """Setup menu bar and keyboard shortcuts."""
        self.mainMenu = self.menuBar()
        fileMenu = self.mainMenu.addMenu('&File')
        
        restartAction = QtWidgets.QAction('&Restart', self)
        restartAction.setShortcut('Ctrl+R')
        restartAction.setStatusTip('Restart the application to apply code changes')
        restartAction.triggered.connect(self.restart_application)
        fileMenu.addAction(restartAction)
        
        fileMenu.addSeparator()
        
        exitAction = QtWidgets.QAction('E&xit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit the application')
        exitAction.triggered.connect(self.exit_application)
        fileMenu.addAction(exitAction)

        shortcuts = [
            (QtCore.Qt.Key_Delete, self.down),
            (QtCore.Qt.Key_F, self.fullScreen),
            (QtCore.Qt.Key_F12, self.fullScreen),
            (QtCore.Qt.Key_Escape, self.escape),
            (QtCore.Qt.Key_Space, self.nextFrame),
            (QtCore.Qt.Key_Backspace, self.prevFrame),
        ]
        
        for key, handler in shortcuts:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(handler)
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)

    def _setup_threading(self):
        """Setup threading infrastructure."""
        self.blinker = Blinker()
        self.to_blink_after_worker_execution = None
        self.threading_enabled = True
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(self.threadpool.maxThreadCount() - 1)
        self.overlay = Overlay(self.centralWidget())
        self.overlay.hide()

        self.oldvalue_CUDA_DEVICE_ORDER = os.environ.get("CUDA_DEVICE_ORDER", '')
        self.oldvalue_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", '')

    def dragEnterEvent(self, event):
        handle_drag_enter(self, event)

    def dragMoveEvent(self, event):
        handle_drag_move(self, event)

    def dropEvent(self, event):
        handle_drop(self, event)

    def set_html_red(self, text):
        set_html_red(self.logger_console, text)

    def set_html_black(self, text):
        set_html_black(self.logger_console, text)

    def resizeEvent(self, event):
        self.overlay.resize(event.size())
        event.accept()

    def progress_fn(self, current_progress):
        """Update progress bar with current progress percentage."""
        print(f"{current_progress}% done")
        self.pbar.setValue(current_progress)

    def _update_channels(self, img):
        """Update channel combo box based on image."""
        if not hasattr(self, 'overlay_bg_channel_combo'):
            return
        if isinstance(img, str):
            img = Img(img)

        selection = self.overlay_bg_channel_combo.currentIndex()
        self.overlay_bg_channel_combo.disconnect()
        self.overlay_bg_channel_combo.clear()
        comboData = ['merge']
        
        if img is not None:
            try:
                if img.has_c():
                    comboData.extend([str(i) for i in range(img.get_dimension('c'))])
            except:
                pass

        self.overlay_bg_channel_combo.addItems(comboData)
        self.overlay_bg_channel_combo.setCurrentIndex(selection if 0 <= selection < len(comboData) else 0)
        self.overlay_bg_channel_combo.currentIndexChanged.connect(self.preview_changed)

    def update_preview_depending_on_selected_tab(self):
        """Update preview based on selected tab."""
        update_preview_for_tab(self)

    def onTabChange(self):
        """Handle tab change event."""
        if self.tabs.currentIndex() == 0:
            self.update_cellpose_rois_label()
        
        current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
        
        # Reset track completeness overlay when switching between tracking and analysis tabs
        if tab_name == 'tracking' or tab_name == 'analysis':
            # Reset both completeness overlay buttons to unchecked
            if hasattr(self, 'completeness_overlay_button'):
                self.completeness_overlay_button.setChecked(False)
            if hasattr(self, 'analysis_completeness_overlay_button'):
                self.analysis_completeness_overlay_button.setChecked(False)
            # Disable the overlay state
            if hasattr(self, 'track_completeness_overlay_enabled') and self.track_completeness_overlay_enabled:
                self.toggle_completeness_overlay(False)
        
        # Clear selections when switching to tracking tab (from analysis or other tabs)
        if tab_name == 'tracking':
            # Clear selected cells
            self.selected_track_ids.clear()
            # Clear selection manager state
            if hasattr(self, 'selection_manager'):
                self.selection_manager.clear_all()
            # Disable selection mode if active
            if self.selection_mode_active:
                self.selection_mode_active = False
                if hasattr(self, 'select_cells_button'):
                    self.select_cells_button.setChecked(False)
            # Always update paint widget to refresh display (clear any visual selections)
            if hasattr(self.paint, 'paint'):
                self.paint.paint.selection_mode = False
                self.paint.paint.hovered_track_id = None
                self.paint.paint.update()
        
        # Clear selections when switching to analysis tab (from tracking or other tabs)
        elif tab_name == 'analysis':
            # Clear selected cells
            self.selected_track_ids.clear()
            # Clear highlighted tracks (used for visual highlighting)
            if hasattr(self, 'highlighted_track_ids'):
                self.highlighted_track_ids.clear()
            # Clear selection manager state
            if hasattr(self, 'selection_manager'):
                self.selection_manager.clear_all()
            # Disable selection mode if active
            if self.selection_mode_active:
                self.selection_mode_active = False
                if hasattr(self, 'select_cells_button'):
                    self.select_cells_button.setChecked(False)
            # Always update paint widget to refresh display (clear any visual selections)
            if hasattr(self.paint, 'paint'):
                self.paint.paint.selection_mode = False
                self.paint.paint.hovered_track_id = None
                self.paint.paint.update()
        
        # Clear selections and reset to brush mode when switching to segmentation tab
        elif tab_name.startswith('seg') or 'cellpose' in tab_name:
            # Clear selected cells
            self.selected_track_ids.clear()
            # Clear selection manager state
            if hasattr(self, 'selection_manager'):
                self.selection_manager.clear_all()
            # Disable selection mode if active
            if self.selection_mode_active:
                self.selection_mode_active = False
                if hasattr(self, 'select_cells_button'):
                    self.select_cells_button.setChecked(False)
            # Always update paint widget to refresh display (clear any visual selections)
            # and ensure brush tool (drawing mode) is enabled
            if hasattr(self.paint, 'paint'):
                self.paint.paint.selection_mode = False
                self.paint.paint.track_view_mode = False  # Disable track view mode to enable brush cursor
                self.paint.paint.track_merge_mode = False  # Disable track merge mode
                self.paint.paint.hovered_track_id = None
                self.paint.paint.drawing_enabled = True  # Ensure brush tool is enabled
                # Clear tracked image (not needed in segmentation tab)
                self.paint.paint.tracked_image_int24 = None
                self.paint.paint.tracked_image_rgb = None
                self.paint.paint._tracked_image_qimage = None
                self.paint.paint.current_file = None
                # Clear box selection state if any
                if hasattr(self.paint.paint, 'box_selection_start'):
                    self.paint.paint.box_selection_start = None
                if hasattr(self.paint.paint, 'box_selection_end'):
                    self.paint.paint.box_selection_end = None
                # Ensure cursor is properly initialized for brush drawing
                if (self.paint.paint.image is not None and 
                    (self.paint.paint.cursor is None or 
                     self.paint.paint.cursor.size() != self.paint.paint.image.size())):
                    from qtpy.QtGui import QImage
                    from qtpy.QtCore import Qt
                    self.paint.paint.cursor = QImage(self.paint.paint.image.size(), QImage.Format_ARGB32)
                    self.paint.paint.cursor.fill(Qt.transparent)
                # Force repaint to clear any visual selections
                self.paint.paint.update()
        
        # Disable selection mode if switching away from Tracking tab
        elif tab_name != 'tracking' and self.selection_mode_active:  # tab_name is lowercase
            self.selection_mode_active = False
            if hasattr(self, 'select_cells_button'):
                self.select_cells_button.setChecked(False)
            if hasattr(self.paint, 'paint'):
                self.paint.paint.selection_mode = False
                self.paint.paint.hovered_track_id = None
                self.paint.paint.update()
        
        self.update_preview_depending_on_selected_tab()
        
        # Clear selections again after preview update (in case they were reloaded)
        # This ensures selections are cleared when switching to analysis tab
        if tab_name == 'analysis':
            self.selected_track_ids.clear()
            # Clear highlighted tracks again (in case they were reloaded)
            if hasattr(self, 'highlighted_track_ids'):
                self.highlighted_track_ids.clear()
            if hasattr(self, 'selection_manager'):
                self.selection_manager.clear_all()
            if hasattr(self.paint, 'paint'):
                self.paint.paint.update()
        
        # Try to load track completeness when switching to tracking or analysis tab
        if tab_name in ('tracking', 'analysis'):
            self._load_track_completeness_if_needed()

    def selectionChanged(self):
        """Handle selection change event."""
        try:
            self.update_preview_depending_on_selected_tab()
            # Try to load track completeness from database when selection changes
            self._load_track_completeness_if_needed()
        except:
            traceback.print_exc()

    def clearlayout(self, layout):
        """Remove all widgets from a layout."""
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    def escape(self):
        """Exit fullscreen mode."""
        if self.Stack.isFullScreen():
            self.fullScreen()

    def fullScreen(self):
        """Toggle fullscreen mode for the image viewer."""
        if not self.Stack.isFullScreen():
            self.Stack.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowStaysOnTopHint)
            self.Stack.showFullScreen()
        else:
            self.Stack.setWindowFlags(QtCore.Qt.Widget)
            self.grid.addWidget(self.Stack, 0, 0)
            for widget in [self.grid, self.Stack, self.centralWidget(), self]:
                widget.update()
            for widget in [self.Stack, self.centralWidget()]:
                widget.repaint()
            self.show()

    def about_dialog(self):
        """Show about dialog."""
        msg = QMessageBox(parent=self)
        msg.setIcon(QMessageBox.Information)
        msg.setText("CellMap")
        msg.setInformativeText("Please check the third party licenses (press 'show details')")
        msg.setWindowTitle("About...")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def exit_application(self):
        """Exit the application."""
        reply = QMessageBox.question(self, 'Exit Application', 'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()

    def restart_application(self):
        """Restart the application to apply code changes."""
        reply = QMessageBox.question(self, 'Restart Application', 'Are you sure you want to restart the application?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()
            python = sys.executable
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.abspath(os.path.join(script_dir, '..', 'cellmap_main.py'))
            if os.path.exists(script_path):
                os.execl(python, python, script_path)
            else:
                os.execl(python, python, '-m', 'cellmap_main')

    def down(self):
        print('down')

    def nextFrame(self):
        print('next frame pressed')

    def prevFrame(self):
        print('prev frame pressed')

    def get_current_TA_path(self):
        """Get current TA path from selection."""
        selection = self.get_selection()
        return smart_name_parser(selection, ordered_output='TA') if selection else None

    def get_selection(self):
        """Get currently selected file."""
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        list_idx = self._tab_idx_to_list_idx(selected_tab_idx)
        self.list.set_list(list_idx)
        return self.list.get_list(list_idx).get_selection()

    def get_selection_multiple(self):
        """Get all currently selected files."""
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        list_idx = self._tab_idx_to_list_idx(selected_tab_idx)
        self.list.set_list(list_idx)
        return self.list.get_list(list_idx).get_selection(mode='multiple')

    def get_selection_index(self):
        """Get index of currently selected file."""
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        list_idx = self._tab_idx_to_list_idx(selected_tab_idx)
        self.list.set_list(list_idx)
        return self.list.get_list(list_idx).get_selection_index()

    def update_cellpose_rois_label(self):
        """Update the ROIs count label in the CellPose tab."""
        try:
            # Show count of selected images, or full list if none selected
            selected_images = self.get_selection_multiple()
            if selected_images:
                count = len(selected_images)
            else:
                lst = self.get_full_list(warn_on_empty_list=False)
                count = len(lst) if lst else 0
            self.cellpose_rois_label.setText(f"{count} ROIs")
        except:
            self.cellpose_rois_label.setText("0 ROIs")
    
    def navigate_to_frame(self, frame_idx):
        """
        Navigate to a specific frame by selecting it in the file list and updating preview.
        
        Args:
            frame_idx: Frame index (0-based) to navigate to
        """
        try:
            selected_tab_idx, _ = self.get_cur_tab_index_and_name()
            list_idx = self._tab_idx_to_list_idx(selected_tab_idx)
            self.list.set_list(list_idx)
            list_widget = self.list.get_list(list_idx)
            
            # Get the QListWidget
            qlist_widget = list_widget.list
            
            # Check if frame_idx is valid
            if 0 <= frame_idx < qlist_widget.count():
                # Select the item at frame_idx
                qlist_widget.setCurrentRow(frame_idx)
                # Scroll to make it visible
                qlist_widget.scrollToItem(qlist_widget.item(frame_idx))
                # Trigger preview update
                from gui.handlers.tab_handler import update_preview_for_tab
                update_preview_for_tab(self)
                logger.info(f'Navigated to frame {frame_idx + 1}')
            else:
                logger.warning(f'Frame index {frame_idx} is out of range (0-{qlist_widget.count() - 1})')
        except Exception as e:
            logger.error(f'Error navigating to frame {frame_idx}: {e}')
            traceback.print_exc()

    def get_full_list(self, warn_on_empty_list=True):
        """Get full file list for current tab."""
        tab_idx = self.tabs.currentIndex()
        list_idx = self._tab_idx_to_list_idx(tab_idx)
        lst = self.list.get_list(list_idx).get_full_list()
        if warn_on_empty_list and (not lst or lst is None):
            logger.error('Empty list, please load files first')
            self.blinker.blink(self.list)
            return
        return lst

    def check_channel_is_selected(self):
        """Check if image and channel are selected."""
        if self.paint.paint.raw_image is None:
            logger.warning("Please select an image first")
            self.blinker.blink(self.list)
            return False
        selected_channel = self.paint.get_selected_channel()
        channels_count = self.paint.channels.count() if hasattr(self.paint, 'channels') else 0
        if selected_channel is None and channels_count > 1:
            logger.warning("Please select a channel first")
            if hasattr(self.paint, 'channel_label') and hasattr(self.paint, 'channels'):
                self.blinker.blink([self.paint.channel_label, self.paint.channels])
            return False
        return True

    def _get_worker(self, func, *args, **kwargs):
        """Create a worker for threaded or non-threaded execution."""
        return Worker(func, *args, **kwargs) if self.threading_enabled else FakeWorker(func, *args, **kwargs)

    def thread_complete(self):
        """Called when a worker thread completes. Resets UI and handles errors."""
        self.pbar.setValue(0)
        self.overlay.hide()
        self.update_preview_depending_on_selected_tab()
        if self.to_blink_after_worker_execution is not None:
            self.blinker.blink(self.to_blink_after_worker_execution)
            self.to_blink_after_worker_execution = None

    def launch_in_a_tread(self, func):
        """Launch a function in a worker thread with progress tracking."""
        early_stop.stop = False
        self.pbar.setValue(0)
        self.overlay.show()
        worker = self._get_worker(func)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        worker.run() if isinstance(worker, FakeWorker) else self.threadpool.start(worker)

    def export_stack(self):
        """Export all preview images as a stack."""
        list_of_files = self.get_full_list(warn_on_empty_list=True)
        preview_selected = self.image_preview_combo.currentText()
        if not list_of_files:
            return

        images = []
        for file in list_of_files:
            try:
                image = self.create_preview(preview_selected, file, TA_path=smart_name_parser(file, ordered_output='TA'))
                images.append(image)
            except:
                logger.warning(f'could not create file for image "{file}"')
                images.append(None)

        images = [Img(image) if isinstance(image, str) else image for image in images]
        stack = to_stack(images)
        if stack is not None:
            output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.tif);;All Files (*)", default_ext='.tif')
            if output_file is not None:
                dimensions = 'hw'
                if stack.shape[-1] == 3:
                    dimensions += 'c'
                if len(stack.shape) >= (3 if stack.shape[-1] == 3 else 2):
                    dimensions = 'd' + dimensions
                Img(stack, dimensions=dimensions).save(output_file)

    def export_image(self):
        """Export the current preview image to file."""
        img_to_save = self.paint.paint.get_raw_image()
        if img_to_save is None:
            logger.error('Nothing to save')
            return
        output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.tif);;All Files (*)", default_ext='.tif')
        if output_file is not None:
            Img(img_to_save).save(output_file, mode='raw')

    def upper_or_lower_limit_changed(self):
        """Enable/disable percentile spin boxes."""
        if not hasattr(self, 'excluder_label'):
            return
        enabled = self.excluder_label.isChecked()
        if hasattr(self, 'lower_percent_spin'):
            self.lower_percent_spin.setEnabled(enabled)
        if hasattr(self, 'upper_percent_spin'):
            self.upper_percent_spin.setEnabled(enabled)
        self.preview_changed()

    def lut_changed(self):
        """Handle LUT change."""
        if hasattr(self, 'groupBox_color_coding') and self.groupBox_color_coding.isChecked():
            self.preview_changed()

    def preview_changed(self):
        """Update preview when selection or settings change."""
        if not hasattr(self, 'image_preview_combo'):
            return
        preview_selected = self.image_preview_combo.currentText()
        out = self.create_preview(preview_selected, self.get_selection(), self.get_current_TA_path())
        if out is not None:
            self.paint.set_image(out)
            # Load tracked image if in selection mode
            if self.selection_mode_active and hasattr(self.paint, 'paint'):
                current_file = self.get_selection()
                if current_file:
                    self.paint.paint.load_tracked_image(current_file)
                    self.load_selected_cells_from_db()

    def create_preview(self, preview_selected, file, TA_path=None):
        """Create preview image from database query or file path."""
        if not preview_selected:
            return None
        return (create_preview_from_db(self, preview_selected, file) if preview_selected.startswith('#')
                else create_preview_from_file(self, preview_selected, file, TA_path))

    def get_cur_tab_index_and_name(self):
        """Get current tab index and name."""
        selected_tab_idx = self.tabs.currentIndex()
        return selected_tab_idx, self.tabs.tabText(selected_tab_idx).lower()
    
    def _tab_idx_to_list_idx(self, tab_idx):
        """Map tab index to list index. All tabs (Segmentation, Tracking, Analysis) use list 0."""
        return 0

    def populate_preview_combo(self):
        """Populate preview combo box with available images and database columns."""
        if not hasattr(self, 'image_preview_combo'):
            return
        
        _, cur_tab_name = self.get_cur_tab_index_and_name()
        if 'analysis' not in cur_tab_name:
            return

        cur_sel_value = self.image_preview_combo.currentText()
        selection = self.get_selection()
        # Disconnect the specific signal instead of all signals
        try:
            self.image_preview_combo.currentTextChanged.disconnect()
        except TypeError:
            # No connections exist, which is fine
            pass
        self.image_preview_combo.clear()
        
        if not selection:
            self.image_preview_combo.currentTextChanged.connect(self.preview_changed)
            return

        TA_path = smart_name_parser(selection, ordered_output='TA')
        list_of_images_in_TA_folder = create_list(TA_path)

        for img in list_of_images_in_TA_folder:
            self.image_preview_combo.addItem(smart_name_parser(img, ordered_output='short'))

        database_entries = populate_table_content(os.path.join(TA_path, 'pyTA.db'))
        if database_entries:
            self.image_preview_combo.addItems(database_entries)

        if cur_sel_value:
            index = self.image_preview_combo.findText(cur_sel_value, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.image_preview_combo.setCurrentIndex(index)

        self.image_preview_combo.currentTextChanged.connect(self.preview_changed)

    def track_cells_static(self):
        """Track cells using static matching algorithm."""
        lst = self.get_full_list(warn_on_empty_list=True)
        if not lst or not self.check_channel_is_selected():
            return

        values, ok = TrackingDialog.getValues(parent=self)
        if ok:
            self.launch_in_a_tread(partial(self._track_cells_static, lst=lst, channel_of_interest=self.paint.get_selected_channel(),
                                          recursive_assignment=values[0], warp_using_mermaid_if_map_is_available=values[1]))

    def _track_cells_static(self, progress_callback, lst=None, channel_of_interest=None, recursive_assignment=False, warp_using_mermaid_if_map_is_available=True):
        """Track cells using static matching algorithm."""
        match_by_max_overlap_lst(lst, channel_of_interest=channel_of_interest, recursive_assignment=recursive_assignment,
                                 warp_using_mermaid_if_map_is_available=warp_using_mermaid_if_map_is_available,
                                 pre_register=True, progress_callback=progress_callback)
        logger.info('Creating correspondence between local cell id and track/global cell id')
        add_localID_to_trackID_correspondance_in_DB(lst, progress_callback)

    def track_cells_dynamic(self):
        """Track cells using dynamic tissue tracking algorithm."""
        lst = self.get_full_list(warn_on_empty_list=True)
        if not lst:
            return
        # Sort the file list alphabetically before running tracking
        tab_idx = self.tabs.currentIndex()
        list_idx = self._tab_idx_to_list_idx(tab_idx)
        list_widget = self.list.get_list(list_idx)
        # Preserve selection before sorting (to restore after)
        selected_paths = [item.toolTip() for item in list_widget.list.selectedItems()]
        # Block signals during sorting to prevent selectionChanged from clearing the image
        list_widget.list.blockSignals(True)
        list_widget.natsort_list()
        # Restore selection after sorting
        if selected_paths:
            list_widget.list.clearSelection()
            for i in range(list_widget.list.count()):
                item = list_widget.list.item(i)
                if item.toolTip() in selected_paths:
                    item.setSelected(True)
        # Unblock signals after sorting and selection restoration
        list_widget.list.blockSignals(False)
        # Check channel after sorting (when selection is restored)
        if not self.check_channel_is_selected():
            return
        
        # Get tracking parameters from UI widgets
        pyramidal_depth = self.pyramidal_depth_spinbox.value()
        max_iter = self.max_iter_spinbox.value()
        self.launch_in_a_tread(partial(self._track_cells_dynamic, pyramidal_depth=pyramidal_depth, max_iter=max_iter))

    def _track_cells_dynamic(self, progress_callback, pyramidal_depth=3, max_iter=15):
        """Track cells using dynamic tissue tracking algorithm."""
        lst = self.get_full_list(warn_on_empty_list=False)
        if not lst or not self.check_channel_is_selected():
            return
        # Ensure the list is sorted alphabetically before tracking
        lst = natsorted(lst)
        track_cells_dynamic_tissue(lst, channel=self.paint.get_selected_channel(), 
                                   PYRAMIDAL_DEPTH=pyramidal_depth, MAX_ITER=max_iter, 
                                   progress_callback=progress_callback)
        logger.info('Creating correspondence between local cell id and track/global cell id')
        add_localID_to_trackID_correspondance_in_DB(lst, progress_callback)
        # Calculate track completeness for all tracks after tracking completes
        if progress_callback:
            progress_callback.emit(95)
        logger.info('Calculating track completeness for all tracks...')
        self.calculate_all_tracks_completeness(lst)
        if progress_callback:
            progress_callback.emit(100)
        

    def toggle_cell_selection_mode(self):
        """Toggle cell selection mode on/off."""
        # Check if we're in Analysis or Tracking tab
        current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
        if tab_name not in ['tracking', 'analysis']:  # tab_name is lowercase from get_cur_tab_index_and_name()
            logger.warning('Cell selection mode is only available in the Analysis or Tracking tab')
            if hasattr(self, 'select_cells_button'):
                self.select_cells_button.setChecked(False)
            return
        
        # Check if tracked image exists before enabling
        if not self.selection_mode_active:
            current_file = self.get_selection()
            if current_file:
                if hasattr(self.paint, 'paint'):
                    if not self.paint.paint.load_tracked_image(current_file):
                        logger.warning('Tracked cells image not found. Please track cells first.')
                        if hasattr(self, 'select_cells_button'):
                            self.select_cells_button.setChecked(False)
                        return
        
        self.selection_mode_active = not self.selection_mode_active
        
        if hasattr(self, 'select_cells_button'):
            self.select_cells_button.setChecked(self.selection_mode_active)
        
        # Update paint widget selection mode
        if hasattr(self.paint, 'paint'):
            self.paint.paint.selection_mode = self.selection_mode_active
            if not self.selection_mode_active:
                # Clear hover state when disabling
                self.paint.paint.hovered_track_id = None
                self.paint.paint.update()
                # Enable click-to-view-lost-frames mode
                if hasattr(self.paint, 'paint'):
                    self.paint.paint.track_view_mode = True
                # Clear analysis context if active
                if self.current_analysis:
                    self.selection_manager.clear_analysis_context()
                    self.current_analysis = None
            else:
                # Preserve track_view_mode if completeness overlay is enabled
                # Otherwise disable track view mode when in selection mode
                if hasattr(self.paint, 'paint'):
                    # Only disable track_view_mode if completeness overlay is not enabled
                    if not (hasattr(self, 'track_completeness_overlay_enabled') and 
                            self.track_completeness_overlay_enabled):
                        self.paint.paint.track_view_mode = False
                # Load tracked image when enabling selection mode
                current_file = self.get_selection()
                if current_file:
                    self.paint.paint.load_tracked_image(current_file)
                    # If completeness overlay is enabled, ensure track_view_mode is enabled and refresh preview
                    if (hasattr(self, 'track_completeness_overlay_enabled') and 
                        self.track_completeness_overlay_enabled):
                        self.paint.paint.track_view_mode = True
                        # Refresh preview to show overlay
                        self._refresh_tracking_preview()
                    # Load selected cells from database (only if not in analysis mode)
                    if not self.current_analysis:
                        self.load_selected_cells_from_db()
        
        if self.selection_mode_active:
            if self.current_analysis:
                status = self.selection_manager.get_status_message()
                logger.info(f'Cell selection mode enabled for {self.current_analysis}. {status}')
            else:
                logger.info('Cell selection mode enabled. Draw a box or click cells to select them.')
        else:
            logger.info('Cell selection mode disabled.')
    
    def toggle_track_merge_mode(self):
        """Toggle track merge mode on/off."""
        # Check if we're in Tracking tab
        current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
        if tab_name != 'tracking':
            logger.warning('Track merge mode is only available in the Tracking tab')
            if hasattr(self, 'track_merge_button'):
                self.track_merge_button.setChecked(False)
            return
        
        # Check if tracked image exists before enabling
        if not hasattr(self, 'track_merge_mode_active'):
            self.track_merge_mode_active = False
        
        if not self.track_merge_mode_active:
            current_file = self.get_selection()
            if current_file:
                if hasattr(self.paint, 'paint'):
                    if not self.paint.paint.load_tracked_image(current_file):
                        logger.warning('Tracked cells image not found. Please track cells first.')
                        if hasattr(self, 'track_merge_button'):
                            self.track_merge_button.setChecked(False)
                        return
        
        self.track_merge_mode_active = not self.track_merge_mode_active
        
        if hasattr(self, 'track_merge_button'):
            self.track_merge_button.setChecked(self.track_merge_mode_active)
        
        # Update paint widget merge mode
        if hasattr(self.paint, 'paint'):
            self.paint.paint.track_merge_mode = self.track_merge_mode_active
            if not self.track_merge_mode_active:
                # Clear merge state when disabling
                self.paint.paint.track_merge_source_track_id = None
                self.paint.paint.track_merge_source_frame_idx = None
                self.paint.paint.hovered_track_id = None
                self.paint.paint.update()
                # Re-enable track view mode if completeness overlay is enabled
                if hasattr(self, 'track_completeness_overlay_enabled') and self.track_completeness_overlay_enabled:
                    if (not hasattr(self, 'selection_mode_active') or not self.selection_mode_active):
                        self.paint.paint.track_view_mode = True
            else:
                # Disable other modes when enabling merge mode
                self.paint.paint.selection_mode = False
                self.paint.paint.track_view_mode = False
                # Load tracked image when enabling merge mode
                current_file = self.get_selection()
                if current_file:
                    self.paint.paint.load_tracked_image(current_file)
        
        if self.track_merge_mode_active:
            logger.info('Track merge mode enabled. Click first cell, then navigate and click second cell.')
        else:
            logger.info('Track merge mode disabled.')
    
    def load_selected_cells_from_db(self):
        """Load selected cells from database for the current file."""
        # Don't load selections when in segmentation or analysis tab
        current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
        if tab_name.startswith('seg') or 'cellpose' in tab_name or tab_name == 'analysis':
            return
        
        try:
            current_file = self.get_selection()
            if not current_file:
                return
            
            from tracking.tools import smart_name_parser
            from database.sqlite_db import TAsql, ensure_selected_for_correction_column
            
            db_path = smart_name_parser(current_file, ordered_output='pyTA.db')
            if not os.path.exists(db_path):
                return
            
            # Ensure column exists
            ensure_selected_for_correction_column(db_path)
            
            db = TAsql(filename_or_connection=db_path)
            if not db.exists('cell_tracks'):
                db.close()
                return
            
            # Query selected track IDs
            query = 'SELECT DISTINCT track_id FROM cell_tracks WHERE selected_for_correction = 1 AND track_id IS NOT NULL'
            results = db.run_SQL_command_and_get_results(query)
            db.close()
            
            if results:
                # Update selected_track_ids set
                self.selected_track_ids = {int(row[0]) for row in results if row[0] is not None}
                logger.info(f'Loaded {len(self.selected_track_ids)} selected cells from database')
                # Trigger repaint
                if hasattr(self.paint, 'paint'):
                    self.paint.paint.update()
        except Exception as e:
            logger.error(f'Error loading selected cells from database: {e}')
            traceback.print_exc()

    def select_all_cells(self):
        """Select all cells in the current frame for correction."""
        # Check if we're in Tracking tab
        current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
        if tab_name != 'tracking':
            logger.warning('Select all cells is only available in the Tracking tab')
            return
        
        # Check if tracked image is loaded
        if not hasattr(self.paint, 'paint') or self.paint.paint.tracked_image_int24 is None:
            logger.warning('Tracked cells image not loaded. Please track cells first.')
            return
        
        try:
            import numpy as np
            # Get all unique track IDs from the tracked image
            tracked_image = self.paint.paint.tracked_image_int24
            unique_ids = np.unique(tracked_image)
            
            # Filter out background (0xFFFFFF is white/background, 0 is also background)
            all_track_ids = {int(tid) for tid in unique_ids if tid != 0xFFFFFF and tid != 0}
            
            if not all_track_ids:
                logger.warning('No cells found in tracked image')
                return
            
            # Select all cells
            self.selected_track_ids = all_track_ids.copy()
            
            # Update database
            self.update_selected_cells_in_db(all_track_ids, selected=True)
            
            # Trigger repaint
            if hasattr(self.paint, 'paint'):
                self.paint.paint.update()
            
            logger.info(f'Selected {len(all_track_ids)} cells for correction')
        except Exception as e:
            logger.error(f'Error selecting all cells: {e}')
            traceback.print_exc()

    def select_none_cells(self):
        """Deselect all cells in the current frame."""
        # Check if we're in Tracking tab
        current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
        if tab_name != 'tracking':
            logger.warning('Select none cells is only available in the Tracking tab')
            return
        
        # Get currently selected cells
        if not self.selected_track_ids:
            logger.info('No cells selected to deselect')
            return
        
        try:
            # Store track IDs to deselect
            track_ids_to_deselect = self.selected_track_ids.copy()
            
            # Clear selection
            self.selected_track_ids.clear()
            
            # Update database
            self.update_selected_cells_in_db(track_ids_to_deselect, selected=False)
            
            # Trigger repaint
            if hasattr(self.paint, 'paint'):
                self.paint.paint.update()
            
            logger.info(f'Deselected {len(track_ids_to_deselect)} cells')
        except Exception as e:
            logger.error(f'Error deselecting all cells: {e}')
            traceback.print_exc()

    def update_selected_cells_in_db(self, track_ids, selected=True):
        """
        Update the selected_for_correction flag in the database for given track IDs.
        
        Args:
            track_ids (set or list): Set or list of track IDs (int24 colors) to update
            selected (bool): True to mark as selected, False to deselect
        """
        if not track_ids:
            return
        
        try:
            # Get current file and its database
            current_file = self.get_selection()
            if not current_file:
                logger.warning('No file selected, cannot update database')
                return
            
            from tracking.tools import smart_name_parser
            from database.sqlite_db import TAsql, ensure_selected_for_correction_column
            
            db_path = smart_name_parser(current_file, ordered_output='pyTA.db')
            if not os.path.exists(db_path):
                logger.warning(f'Database not found: {db_path}')
                return
            
            # Ensure column exists
            ensure_selected_for_correction_column(db_path)
            
            db = TAsql(filename_or_connection=db_path)
            if not db.exists('cell_tracks'):
                logger.warning('cell_tracks table does not exist')
                db.close()
                return
            
            # Convert track IDs to list for SQL query
            track_id_list = list(track_ids)
            # SQLite doesn't support tuples directly, so we'll use a different approach
            # Update all rows where track_id matches any of the track IDs
            value = 1 if selected else 0
            
            # Build query with IN clause
            placeholders = ','.join(['?' for _ in track_id_list])
            query = f'UPDATE cell_tracks SET selected_for_correction = ? WHERE track_id IN ({placeholders})'
            db.cur.execute(query, [value] + track_id_list)
            db.con.commit()
            
            logger.info(f'Updated {len(track_id_list)} cells in database (selected={selected})')
            db.close()
        except Exception as e:
            logger.error(f'Error updating selected cells in database: {e}')
            traceback.print_exc()


    def calculate_all_tracks_completeness(self, file_list):
        """Calculate track completeness for all tracks across all frames.
        
        Optimized to load each frame image only once (O(M) instead of O(N×M)),
        where M is the number of frames and N is the number of tracks.
        """
        try:
            if not file_list:
                return
            
            logger.info(f'Calculating track completeness for all tracks across {len(file_list)} frames')
            
            from utils.image_io import Img
            from utils.image_utils import RGB_to_int24
            from tracking.tools import smart_name_parser
            import numpy as np
            
            # OPTIMIZATION: Load each frame only once, then build presence matrix
            # Track presence: {track_id: set of frame indices where present}
            track_presence = {}
            total_frames = len(file_list)
            
            # Pass 1: Load each frame once and collect all track IDs
            for frame_idx, frame_file in enumerate(file_list):
                tracked_image_path = smart_name_parser(frame_file, ordered_output='tracked_cells_resized.tif')
                if not os.path.exists(tracked_image_path):
                    continue
                
                try:
                    tracked_image_rgb = Img(tracked_image_path)
                    if len(tracked_image_rgb.shape) == 3 and tracked_image_rgb.shape[2] == 3:
                        tracked_image_int24 = RGB_to_int24(tracked_image_rgb)
                    else:
                        tracked_image_int24 = tracked_image_rgb.astype(np.uint32)
                    
                    # Get all unique track IDs in this frame
                    unique_ids = np.unique(tracked_image_int24)
                    # Filter out background
                    track_ids = {int(tid) for tid in unique_ids if tid != 0xFFFFFF and tid != 0}
                    
                    # Record which tracks are present in this frame
                    for track_id in track_ids:
                        if track_id not in track_presence:
                            track_presence[track_id] = set()
                        track_presence[track_id].add(frame_idx)
                    
                    if (frame_idx + 1) % 10 == 0:
                        logger.debug(f'Processed {frame_idx + 1}/{total_frames} frames...')
                        
                except Exception as e:
                    logger.warning(f'Error loading track IDs from {frame_file}: {e}')
            
            if not track_presence:
                logger.warning('No tracks found to calculate completeness')
                return
            
            # Pass 2: Determine completeness (present in ALL frames)
            completeness = {}
            total_tracks = len(track_presence)
            
            for idx, (track_id, present_frames) in enumerate(track_presence.items()):
                # Track is complete if present in ALL frames
                is_complete = len(present_frames) == total_frames
                completeness[track_id] = is_complete
                
                if (idx + 1) % 100 == 0:
                    logger.debug(f'Processed {idx + 1}/{total_tracks} tracks...')
            
            self.track_completeness_cache = completeness
            
            complete_count = sum(1 for v in completeness.values() if v)
            incomplete_count = len(completeness) - complete_count
            logger.info(f'Track completeness calculated: {complete_count} complete (green), {incomplete_count} incomplete (red)')
            
            # Save to Master.db in the main folder
            db_path = get_master_db_path(file_list)
            if db_path:
                save_track_completeness(completeness, db_path)
                logger.info(f'Saved track completeness to {db_path}')
            else:
                logger.warning('Could not determine Master.db path to save track completeness')
            
        except Exception as e:
            logger.error(f'Error calculating all tracks completeness: {e}')
            traceback.print_exc()

    def calculate_track_completeness(self, file_list, track_ids):
        """
        Calculate track completeness for specific track IDs only.
        This is much faster than recalculating all tracks.
        
        Optimized for batch operations: when multiple track IDs are provided,
        uses a frame-centric approach to minimize image loads.
        
        Args:
            file_list: List of file paths
            track_ids: Single track ID (int) or list of track IDs to recalculate
        """
        try:
            if not file_list or not track_ids:
                return
            
            # Convert single track ID to list
            if not isinstance(track_ids, (list, set, tuple)):
                track_ids = [track_ids]
            track_ids = set(track_ids)
            
            logger.info(f'Calculating track completeness for {len(track_ids)} specific track(s)')
            
            from utils.image_io import Img
            from utils.image_utils import RGB_to_int24
            from tracking.tools import smart_name_parser
            import numpy as np
            
            # Initialize or update cache
            if not hasattr(self, 'track_completeness_cache') or self.track_completeness_cache is None:
                self.track_completeness_cache = {}
            
            total_frames = len(file_list)
            
            # Optimization: For multiple tracks, use frame-centric approach
            # For single track, the simple approach is fine, but we'll use the optimized path for consistency
            if len(track_ids) > 1:
                # Frame-centric approach: load each frame once
                track_presence = {tid: set() for tid in track_ids}
                
                for frame_idx, frame_file in enumerate(file_list):
                    tracked_image_path = smart_name_parser(frame_file, ordered_output='tracked_cells_resized.tif')
                    if not os.path.exists(tracked_image_path):
                        continue
                    
                    try:
                        tracked_image_rgb = Img(tracked_image_path)
                        if len(tracked_image_rgb.shape) == 3 and tracked_image_rgb.shape[2] == 3:
                            tracked_image_int24 = RGB_to_int24(tracked_image_rgb)
                        else:
                            tracked_image_int24 = tracked_image_rgb.astype(np.uint32)
                        
                        # Get all unique track IDs in this frame
                        unique_ids = np.unique(tracked_image_int24)
                        frame_track_ids = {int(tid) for tid in unique_ids if tid != 0xFFFFFF and tid != 0}
                        
                        # Check which of our requested tracks are present
                        present_tracks = track_ids & frame_track_ids
                        for track_id in present_tracks:
                            track_presence[track_id].add(frame_idx)
                            
                    except Exception as e:
                        logger.warning(f'Error checking tracks in frame {frame_file}: {e}')
                
                # Determine completeness for each track
                for track_id in track_ids:
                    is_complete = len(track_presence[track_id]) == total_frames
                    self.track_completeness_cache[track_id] = is_complete
            else:
                # Single track: simple approach (still efficient for one track)
                track_id = next(iter(track_ids))
                present_in_frames = []
                
                for frame_file in file_list:
                    tracked_image_path = smart_name_parser(frame_file, ordered_output='tracked_cells_resized.tif')
                    if not os.path.exists(tracked_image_path):
                        present_in_frames.append(False)
                        continue
                    
                    try:
                        tracked_image_rgb = Img(tracked_image_path)
                        if len(tracked_image_rgb.shape) == 3 and tracked_image_rgb.shape[2] == 3:
                            tracked_image_int24 = RGB_to_int24(tracked_image_rgb)
                        else:
                            tracked_image_int24 = tracked_image_rgb.astype(np.uint32)
                        
                        # Check if track ID is present in this frame
                        is_present = np.any(tracked_image_int24 == track_id)
                        present_in_frames.append(is_present)
                    except Exception as e:
                        logger.warning(f'Error checking track {track_id} in frame {frame_file}: {e}')
                        present_in_frames.append(False)
                
                # Track is complete if present in ALL frames
                is_complete = all(present_in_frames)
                self.track_completeness_cache[track_id] = is_complete
            
            # Save updated completeness to Master.db
            db_path = get_master_db_path(file_list)
            if db_path:
                # Load existing completeness data
                existing_completeness = load_track_completeness(db_path)
                if existing_completeness is None:
                    existing_completeness = {}
                
                # Update with new values
                existing_completeness.update(self.track_completeness_cache)
                
                # Save back to database
                save_track_completeness(existing_completeness, db_path)
                logger.info(f'Updated track completeness for {len(track_ids)} track(s) in {db_path}')
            else:
                logger.warning('Could not determine Master.db path to save track completeness')
            
        except Exception as e:
            logger.error(f'Error calculating track completeness: {e}')
            traceback.print_exc()

    def toggle_completeness_overlay(self, enabled):
        """Toggle the track completeness overlay on/off."""
        self.track_completeness_overlay_enabled = enabled
        # Try to load completeness data if not already loaded
        if enabled:
            self._load_track_completeness_if_needed()
        # Enable track_view_mode when overlay is enabled
        # Note: track_view_mode can work alongside selection_mode_active
        if hasattr(self.paint, 'paint'):
            if enabled:
                self.paint.paint.track_view_mode = True
                # Ensure tracked image is loaded for hover and overlay to work
                current_file = self.get_selection()
                if current_file:
                    self.paint.paint.load_tracked_image(current_file)
            elif not enabled:
                # Only disable track_view_mode when overlay is disabled
                # But keep it enabled if we're in selection mode (for cell selection)
                if not (hasattr(self, 'selection_mode_active') and self.selection_mode_active):
                    self.paint.paint.track_view_mode = False
        # Refresh preview to show/hide overlay
        self._refresh_tracking_preview()
    
    def _load_track_completeness_if_needed(self):
        """Load track completeness from Master.db if not already loaded."""
        try:
            # Only load if cache is empty and we're in tracking or analysis tab
            if self.track_completeness_cache:
                return  # Already loaded
            
            current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
            if tab_name not in ('tracking', 'analysis'):
                return  # Only needed in tracking or analysis tab
            
            # Get file list to find Master.db
            file_list = self.get_full_list(warn_on_empty_list=False)
            if not file_list:
                return
            
            # Load from Master.db in the main folder
            db_path = get_master_db_path(file_list)
            if not db_path or not os.path.exists(db_path):
                print('Tracking has not been run yet. Master.db does not exist. Please run tracking first.')
                return
            
            completeness = load_track_completeness(db_path)
            if completeness:
                self.track_completeness_cache = completeness
                complete_count = sum(1 for v in completeness.values() if v)
                incomplete_count = len(completeness) - complete_count
                logger.info(f'Loaded track completeness from {db_path}: {complete_count} complete, {incomplete_count} incomplete')
        except Exception as e:
            logger.debug(f'Could not load track completeness from Master.db: {e}')
            # Don't log as error - it's okay if it doesn't exist yet

    def get_track_presence_info(self, track_id, file_list):
        """
        Get track presence information: first frame, last frame, and all frames where present/missing.
        
        Args:
            track_id: Track ID to check
            file_list: List of file paths
            
        Returns:
            dict with keys:
                - first_frame: First frame index where track appears (0-based)
                - last_frame: Last frame index where track appears (0-based)
                - present_frames: List of frame indices where track is present
                - missing_frames: List of frame indices where track is missing
        """
        from utils.image_io import Img
        from utils.image_utils import RGB_to_int24
        from tracking.tools import smart_name_parser
        import numpy as np
        
        present_frames = []
        missing_frames = []
        
        if not file_list:
            return {
                'first_frame': None,
                'last_frame': None,
                'present_frames': [],
                'missing_frames': []
            }
        
        for frame_idx, frame_file in enumerate(file_list):
            tracked_image_path = smart_name_parser(frame_file, ordered_output='tracked_cells_resized.tif')
            if not os.path.exists(tracked_image_path):
                missing_frames.append(frame_idx)
                continue
            
            try:
                tracked_image_rgb = Img(tracked_image_path)
                if len(tracked_image_rgb.shape) == 3 and tracked_image_rgb.shape[2] == 3:
                    tracked_image_int24 = RGB_to_int24(tracked_image_rgb)
                else:
                    tracked_image_int24 = tracked_image_rgb.astype(np.uint32)
                
                # Check if track ID is present in this frame
                is_present = np.any(tracked_image_int24 == track_id)
                if is_present:
                    present_frames.append(frame_idx)
                else:
                    missing_frames.append(frame_idx)
            except Exception as e:
                logger.warning(f'Error checking track {track_id} in frame {frame_file}: {e}')
                missing_frames.append(frame_idx)
        
        first_frame = present_frames[0] if present_frames else None
        last_frame = present_frames[-1] if present_frames else None
        
        return {
            'first_frame': first_frame,
            'last_frame': last_frame,
            'present_frames': present_frames,
            'missing_frames': missing_frames
        }

    def merge_tracks(self, track_id_x, track_id_y, file_list, merge_start_frame=0):
        """
        Merge two tracks: merge track_id_y into track_id_x (track_id_x is kept).
        
        Args:
            track_id_x: The track ID to keep (first selected track)
            track_id_y: The track ID to merge into track_id_x (second clicked track)
            file_list: List of file paths
            merge_start_frame: Frame index where merge should start (where second cell was clicked). Defaults to 0.
        """
        try:
            from tracking.track_correction import connect_tracks
            from tracking.local_to_track_correspondance import add_localID_to_trackID_correspondance_in_DB
            
            if not file_list:
                logger.error('No file list provided for track merge')
                return
            
            if merge_start_frame < 0 or merge_start_frame >= len(file_list):
                logger.error(f'Invalid merge start frame {merge_start_frame}')
                return
            
            logger.info(f'Merging track {track_id_y:06x} into track {track_id_x:06x} starting from frame {merge_start_frame + 1}')
            
            # Connect tracks from merge_start_frame onwards
            connect_tracks(file_list, merge_start_frame, track_id_x, track_id_y, __preview_only=False)
            
            # Update database to reflect the changes - only update affected frames from merge_start_frame onwards
            affected_frames = file_list[merge_start_frame:]
            logger.info(f'Updating database for {len(affected_frames)} affected frame(s)...')
            add_localID_to_trackID_correspondance_in_DB(affected_frames)
            
            # Recalculate track completeness only for the merged track
            # After connection, track_id_y is merged into track_id_x
            logger.info('Recalculating track completeness for merged track...')
            
            # Remove old track_id_y from cache since it's been merged into track_id_x
            if hasattr(self, 'track_completeness_cache') and self.track_completeness_cache is not None:
                if track_id_y in self.track_completeness_cache:
                    del self.track_completeness_cache[track_id_y]
            
            # Recalculate only the merged track (track_id_x now includes track_id_y)
            self.calculate_track_completeness(file_list, track_id_x)
            
            # Refresh preview to show updated tracks
            self._refresh_tracking_preview()
            
            logger.info(f'Track {track_id_y:06x} successfully merged into track {track_id_x:06x}')
            
            # Show success message
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Tracks Merged",
                f"Track {track_id_y:06x} has been successfully merged into track {track_id_x:06x}."
            )
        except Exception as e:
            logger.error(f'Error merging tracks: {e}')
            traceback.print_exc()
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to merge tracks: {e}"
            )


    def _refresh_tracking_preview(self):
        """Refresh the tracking or analysis tab preview to show updated track colors."""
        try:
            current_tab_idx, tab_name = self.get_cur_tab_index_and_name()
            if tab_name in ('tracking', 'analysis'):
                # Trigger preview update by calling the tab handler
                from gui.handlers.tab_handler import update_preview_for_tab
                update_preview_for_tab(self)
        except Exception as e:
            logger.error(f'Error refreshing tracking preview: {e}')
            traceback.print_exc()

    def reset_cellpose_defaults(self):
        """Reset all Cellpose parameters to their default values."""
        self.cellpose_use_gpu_check.setChecked(False)
        self.cellpose_model_combo.setCurrentText("cyto")
        self.cellpose_diameter_spin.setValue(15)
        self.cellpose_flow_threshold_spin.setValue(1.0)
        self.cellpose_cellprob_threshold_spin.setValue(0.0)
        self.cellpose_norm_percentile_lower_spin.setValue(1.0)
        self.cellpose_norm_percentile_upper_spin.setValue(99.0)
        self.cellpose_niter_dynamics_spin.setValue(200)
        self.cellpose_batch_size_spin.setValue(4)
        self.cellpose_additional_settings_group.setChecked(True)

    def reset_tracking_defaults(self):
        """Reset all tracking parameters to their default values."""
        self.pyramidal_depth_spinbox.setValue(3)
        self.max_iter_spinbox.setValue(15)

    def cellpose_seg(self):
        """Launch Cellpose segmentation in a thread."""
        # Check if any images are selected
        selected_images = self.get_selection_multiple()
        if not selected_images:
            logger.warning('No images selected. Please select images in the selection pane to run Cellpose.')
            self.blinker.blink(self.list)
            return
        
        if not self.check_channel_is_selected():
            return
        self.update_cellpose_rois_label()
        self.launch_in_a_tread(self._cellpose_seg)

    def _cellpose_seg(self, progress_callback):
        """Run Cellpose segmentation."""
        from segmentation.cellpose import segment_batch, get_logger
        
        # Ensure logger has GUI handler before starting
        # The handler should already be set up, but refresh logger to be sure
        get_logger()
        
        # Get selected images (already checked in cellpose_seg, but double-check here)
        lst = self.get_selection_multiple()
        if not lst:
            logger.warning('No images selected. Please select images in the selection pane to run Cellpose.')
            return
        
        if not self.check_channel_is_selected():
            return
        
        channel = self.paint.get_selected_channel()
        try:
            segment_batch(lst, model_type=self.cellpose_model_combo.currentText(),
                         diameter=None if self.cellpose_diameter_spin.value() == 0 else self.cellpose_diameter_spin.value(),
                         channels=[channel, channel] if channel is not None else [0, 0],
                         flow_threshold=self.cellpose_flow_threshold_spin.value(),
                         cellprob_threshold=self.cellpose_cellprob_threshold_spin.value(),
                         use_gpu=self.cellpose_use_gpu_check.isChecked(),
                         norm_percentiles=(self.cellpose_norm_percentile_lower_spin.value(),
                                         self.cellpose_norm_percentile_upper_spin.value()),
                         niter_dynamics=self.cellpose_niter_dynamics_spin.value(),
                         progress_callback=progress_callback,
                         batch_size=self.cellpose_batch_size_spin.value())
        except Exception as e:
            logger.error(f'Cellpose segmentation failed: {e}')
            traceback.print_exc()
        finally:
            gc.collect()

    def cellpose_fill_holes(self):
        """Fill holes in CellPose segmentation results for selected images."""
        # Check if any images are selected
        selected_images = self.get_selection_multiple()
        if not selected_images:
            logger.warning('No images selected. Please select images in the selection pane to fill holes.')
            self.blinker.blink(self.list)
            return
        
        logger.info(f'Starting CellPose hole filling for {len(selected_images)} image(s)')
        self.launch_in_a_tread(self._cellpose_fill_holes)

    def _cellpose_fill_holes(self, progress_callback):
        """Fill holes in CellPose segmentation results."""
        from segmentation.cellpose import (
            fill_holes_in_outline_mask, 
            fill_holes_using_seg_npy, 
            get_seg_npy_path, 
            get_output_path,
            _call_progress_callback
        )
        from utils.image_io import Img
        from tracking.tools import smart_name_parser
        import os
        
        # Get selected images
        selected_images = self.get_selection_multiple()
        if not selected_images:
            logger.warning('No images selected for CellPose hole filling')
            return
        
        total_images = len(selected_images)
        logger.info(f'Processing {total_images} image(s) for hole filling')
        
        success_count = 0
        error_count = 0
        
        for idx, image_path in enumerate(selected_images):
            if early_stop.stop:
                logger.info('CellPose hole filling stopped by user')
                break
            
            try:
                # Update progress
                if progress_callback:
                    progress = int((idx / total_images) * 100)
                    _call_progress_callback(progress_callback, progress)
                
                logger.info(f'Processing image {idx + 1}/{total_images}: {os.path.basename(image_path)}')
                
                # Find the outlines.tif file for this image
                outlines_path = get_output_path(image_path)
                
                if not os.path.exists(outlines_path):
                    logger.warning(f'No outlines.tif found for {image_path} at {outlines_path}, skipping')
                    error_count += 1
                    continue
                
                # Load the outlines mask
                outline_img = Img(outlines_path)
                if outline_img.has_c():
                    # Use first channel if multi-channel
                    outline_mask = outline_img[..., 0] if outline_img.ndim > 2 else outline_img
                else:
                    outline_mask = outline_img
                
                # Convert to numpy array if needed
                if not isinstance(outline_mask, np.ndarray):
                    outline_mask = np.array(outline_mask)
                
                # Get maximum hole size from UI (0 means fill all holes)
                max_hole_size_value = self.cellpose_max_hole_size_spin.value()
                max_region_size = max_hole_size_value if max_hole_size_value > 0 else None
                max_hole_size = None if max_hole_size_value == 0 else max_hole_size_value
                
                # Try to use _seg.npy file if available (more accurate)
                filled_outlines = None
                seg_npy_path = get_seg_npy_path(image_path)
                
                if seg_npy_path and os.path.exists(seg_npy_path):
                    logger.info(f'Using _seg.npy file for hole filling: {seg_npy_path}')
                    # For _seg.npy method, use max_region_size (None means fill all)
                    filled_outlines = fill_holes_using_seg_npy(seg_npy_path, max_region_size=max_region_size if max_region_size is not None else 1000000)
                    if filled_outlines is not None:
                        logger.info('Successfully filled holes using _seg.npy data')
                
                # Fall back to outline-based method if _seg.npy not available or failed
                if filled_outlines is None:
                    logger.info('Using outline-based hole filling method')
                    filled_outlines = fill_holes_in_outline_mask(
                        outline_mask, 
                        max_hole_size=max_hole_size,  # None means fill all holes
                        extend_membranes=True, 
                        extension_radius=1
                    )
                
                # Save the filled outlines back to outlines.tif
                try:
                    Img(filled_outlines, dimensions='hw').save(outlines_path)
                    logger.info(f'Filled outlines saved to {outlines_path}')
                    success_count += 1
                except Exception as e:
                    logger.error(f'Failed to save filled outlines for {image_path}: {e}')
                    error_count += 1
                    
            except Exception as e:
                logger.error(f'Failed to fill holes for {image_path}: {e}')
                traceback.print_exc()
                error_count += 1
        
        # Final progress update
        if progress_callback:
            _call_progress_callback(progress_callback, 100)
        
        # Log summary
        logger.info(f'CellPose hole filling completed: {success_count} succeeded, {error_count} failed out of {total_images} images')
        
        # Show completion message
        if success_count > 0:
            message = f"Successfully filled holes in {success_count} image(s)"
            if error_count > 0:
                message += f", {error_count} image(s) failed"
            logger.info(message)
        elif error_count > 0:
            logger.warning(f"Failed to fill holes in {error_count} image(s)")

    def fill_holes(self):
        """Fill holes in the current mask and extend membranes to ensure connectivity.
        If multiple files are selected, processes all of them in batch."""
        # Check if multiple files are selected
        selected_files = self.get_selection_multiple()
        
        if selected_files and len(selected_files) > 1:
            # Batch processing for multiple files
            logger.info(f'Filling holes for {len(selected_files)} selected files')
            self.launch_in_a_tread(self._fill_holes_batch)
        else:
            # Single file processing (current behavior)
            if self.paint.paint.imageDraw is None:
                QMessageBox.warning(self, "No Mask", "No mask loaded. Please load or create a mask first.")
                return
            
            # Get the current mask
            current_mask = self.paint.paint.get_mask()
            if current_mask is None:
                QMessageBox.warning(self, "No Mask", "Could not retrieve mask.")
                return
            
            try:
                from segmentation.cellpose import fill_holes_in_outline_mask, fill_holes_using_seg_npy, get_seg_npy_path
                
                # Try to use _seg.npy file if available (more accurate)
                filled_mask = None
                seg_npy_path = None
                
                if self.paint.paint.save_file_name is not None:
                    # Try to find corresponding _seg.npy file
                    seg_npy_path = get_seg_npy_path(self.paint.paint.save_file_name)
                    if seg_npy_path and os.path.exists(seg_npy_path):
                        logger.info(f'Using _seg.npy file for hole filling: {seg_npy_path}')
                        filled_mask = fill_holes_using_seg_npy(seg_npy_path, max_region_size=500)
                        if filled_mask is not None:
                            logger.info('Successfully filled holes using _seg.npy data')
                
                # Fall back to outline-based method if _seg.npy not available or failed
                if filled_mask is None:
                    logger.info('Using outline-based hole filling method')
                    filled_mask = fill_holes_in_outline_mask(
                        current_mask, 
                        max_hole_size=None,  # Fill all holes
                        extend_membranes=True, 
                        extension_radius=1
                    )
                
                # Update the mask in the paint widget
                from utils.image_io import Img
                self.paint.paint.set_mask(Img(filled_mask, dimensions='hw'))
                
                # Save the mask if there's a save file name
                if self.paint.paint.save_file_name is not None:
                    try:
                        Img(filled_mask, dimensions='hw').save(self.paint.paint.save_file_name)
                        logger.info(f'Filled mask saved to {self.paint.paint.save_file_name}')
                    except Exception as e:
                        logger.warning(f'Failed to save filled mask: {e}')
                
                method_used = "_seg.npy" if seg_npy_path and os.path.exists(seg_npy_path) else "outline-based"
                logger.info(f'Hole filling completed successfully using {method_used} method')
                QMessageBox.information(self, "Success", f"Holes filled successfully using {method_used} method.")
                
            except Exception as e:
                logger.error(f'Failed to fill holes: {e}')
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Failed to fill holes: {str(e)}")

    def _fill_holes_batch(self, progress_callback=None):
        """Fill holes for multiple selected files in batch."""
        from segmentation.cellpose import fill_holes_in_outline_mask, fill_holes_using_seg_npy, get_seg_npy_path, _call_progress_callback
        
        # Get selected files
        selected_files = self.get_selection_multiple()
        if not selected_files:
            logger.warning('No files selected for batch hole filling')
            return
        
        total_files = len(selected_files)
        logger.info(f'Starting batch hole filling for {total_files} files')
        
        success_count = 0
        error_count = 0
        
        for idx, file_path in enumerate(selected_files):
            if early_stop.stop:
                logger.info('Hole filling stopped by user')
                break
            
            try:
                # Update progress
                if progress_callback:
                    progress = int((idx / total_files) * 100)
                    _call_progress_callback(progress_callback, progress)
                
                logger.info(f'Processing file {idx + 1}/{total_files}: {os.path.basename(file_path)}')
                
                # Find corresponding mask file (handCorrection.tif)
                TA_path_alternative, TA_path = smart_name_parser(
                    file_path,
                    ordered_output=['handCorrection.png', 'handCorrection.tif']
                )
                mask_path = TA_path if os.path.isfile(TA_path) else (TA_path_alternative if os.path.isfile(TA_path_alternative) else None)
                
                if mask_path is None or not os.path.exists(mask_path):
                    logger.warning(f'No mask file found for {file_path}, skipping')
                    error_count += 1
                    continue
                
                # Load the mask
                mask_img = Img(mask_path)
                if mask_img.has_c():
                    # Use first channel if multi-channel
                    current_mask = mask_img[..., 0] if mask_img.ndim > 2 else mask_img
                else:
                    current_mask = mask_img
                
                # Convert to numpy array if needed (Img extends np.ndarray, but ensure we have a plain array)
                if not isinstance(current_mask, np.ndarray):
                    current_mask = np.array(current_mask)
                
                # Try to use _seg.npy file if available (more accurate)
                filled_mask = None
                seg_npy_path = get_seg_npy_path(mask_path)
                
                if seg_npy_path and os.path.exists(seg_npy_path):
                    logger.info(f'Using _seg.npy file for hole filling: {seg_npy_path}')
                    filled_mask = fill_holes_using_seg_npy(seg_npy_path, max_region_size=500)
                    if filled_mask is not None:
                        logger.info('Successfully filled holes using _seg.npy data')
                
                # Fall back to outline-based method if _seg.npy not available or failed
                if filled_mask is None:
                    logger.info('Using outline-based hole filling method')
                    filled_mask = fill_holes_in_outline_mask(
                        current_mask, 
                        max_hole_size=None,  # Fill all holes
                        extend_membranes=True, 
                        extension_radius=1
                    )
                
                # Save the filled mask
                try:
                    Img(filled_mask, dimensions='hw').save(mask_path)
                    logger.info(f'Filled mask saved to {mask_path}')
                    success_count += 1
                except Exception as e:
                    logger.error(f'Failed to save filled mask for {file_path}: {e}')
                    error_count += 1
                    
            except Exception as e:
                logger.error(f'Failed to fill holes for {file_path}: {e}')
                traceback.print_exc()
                error_count += 1
        
        # Final progress update
        if progress_callback:
            _call_progress_callback(progress_callback, 100)
        
        # Log summary
        logger.info(f'Batch hole filling completed: {success_count} succeeded, {error_count} failed out of {total_files} files')
        
        # Show completion message (schedule on main thread to avoid threading issues)
        if success_count > 0:
            message = f"Successfully filled holes in {success_count} file(s)"
            if error_count > 0:
                message += f", {error_count} file(s) failed"
            # Use QTimer to schedule the message box on the main thread
            QTimer.singleShot(0, lambda: QMessageBox.information(self, "Batch Hole Filling Complete", message))

    def stop_threads_immediately(self):
        """Stop all running worker threads."""
        early_stop.stop = True

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_theme(self):
        """Apply the current theme (dark or light) to the application."""
        app = QApplication.instance()
        if self.is_dark_theme:
            # Dark theme
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            app.setPalette(dark_palette)
            # Update button icon to sun (light mode icon)
            self.theme_button.setIcon(qta.icon('mdi.weather-sunny', options=[{'scale_factor': 1.5}]))
        else:
            # Light theme (default)
            light_palette = app.style().standardPalette()
            # Explicitly set tooltip colors for light theme
            light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            light_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            app.setPalette(light_palette)
            # Update button icon to moon (dark mode icon)
            self.theme_button.setIcon(qta.icon('mdi.weather-night', options=[{'scale_factor': 1.5}]))
    
    def run_analysis(self, analysis_name: str):
        """Run an analysis on selected cells.
        
        Args:
            analysis_name: Name of the analysis to run
        """
        try:
            # Get analysis from registry
            analysis = self.analysis_registry.get_analysis(analysis_name)
            if not analysis:
                logger.error(f"Analysis '{analysis_name}' not found")
                QMessageBox.warning(self, "Analysis Error", f"Analysis '{analysis_name}' not found")
                return
            
            # Check if this is the same analysis - if so, try to run
            if self.current_analysis == analysis_name:
                # Get selected cells from selection manager
                selected_cells = self.selection_manager.get_selection()
                
                # Validate selection
                is_valid, error_msg = analysis.validate_selection(selected_cells)
                if not is_valid:
                    logger.warning(f"Invalid selection: {error_msg}")
                    QMessageBox.warning(self, "Invalid Selection", error_msg)
                    return
                
                # Selection is valid, proceed to run
            else:
                # New analysis - set context and enable selection
                self.current_analysis = analysis_name
                self.selection_manager.set_analysis_context(
                    analysis_name,
                    analysis.selection_mode,
                    analysis.selection_count
                )
                
                # Enable selection mode if not already enabled
                if not self.selection_mode_active:
                    self.toggle_cell_selection_mode()
                
                # Ensure tracked image is loaded for selection and completeness overlay
                current_file = self.get_selection()
                if current_file and hasattr(self.paint, 'paint'):
                    self.paint.paint.load_tracked_image(current_file)
                    # Enable track_view_mode if completeness overlay is enabled
                    if (hasattr(self, 'track_completeness_overlay_enabled') and 
                        self.track_completeness_overlay_enabled):
                        self.paint.paint.track_view_mode = True
                
                # Refresh preview to ensure overlay is shown
                self._refresh_tracking_preview()
                
                # Get selected cells from selection manager
                selected_cells = self.selection_manager.get_selection()
                
                # Validate selection
                is_valid, error_msg = analysis.validate_selection(selected_cells)
                if not is_valid:
                    # Prompt user to select cells
                    status = self.selection_manager.get_status_message()
                    QMessageBox.information(
                        self, 
                        "Select Cells", 
                        f"{error_msg}\n\n{status}\n\nClick cells to select them, then click the analysis button again to run."
                    )
                    return
            
            # Get TA output folder from current selection
            current_file = self.get_selection()
            if not current_file:
                QMessageBox.warning(self, "No File Selected", "Please select a file first")
                return
            
            # Determine TA output folder by looking for master database
            # Start from the directory containing the selected file
            from gui.handlers.analysis_handler import detect_master_db
            ta_output_folder = None
            
            # Try the directory containing the file
            candidate_folder = os.path.dirname(current_file)
            if detect_master_db(candidate_folder):
                ta_output_folder = candidate_folder
            else:
                # Try parent directory (common structure: main_folder/Image0001/image.tif)
                parent_folder = os.path.dirname(candidate_folder)
                if detect_master_db(parent_folder):
                    ta_output_folder = parent_folder
                else:
                    # Try using smart_name_parser as fallback
                    from tracking.tools import smart_name_parser
                    ta_output_folder = smart_name_parser(current_file, ordered_output='TA')
                    # If smart_name_parser returns the file path, use its directory
                    if ta_output_folder == current_file or not os.path.isdir(ta_output_folder):
                        ta_output_folder = os.path.dirname(current_file)
            
            # Verify the folder exists and contains a master database
            if not ta_output_folder or not os.path.exists(ta_output_folder):
                QMessageBox.warning(self, "TA Folder Not Found", 
                                  f"Could not find TA output folder for {current_file}")
                return
            
            if not detect_master_db(ta_output_folder):
                QMessageBox.warning(self, "Master Database Not Found", 
                                  f"Could not find master database in TA output folder:\n{ta_output_folder}\n\n"
                                  f"Please ensure the selected file is from a tracked dataset.")
                return
            
            # Get frame range and other options from UI widgets if available
            frame_range = None
            kwargs = {}
            if hasattr(self, 'analysis_widgets') and analysis_name in self.analysis_widgets:
                widgets = self.analysis_widgets[analysis_name]
                if 'start_frame' in widgets and 'end_frame' in widgets:
                    start_frame = widgets['start_frame'].value()
                    end_frame = widgets['end_frame'].value()
                    if start_frame > 0 or end_frame < 9999:
                        frame_range = (start_frame, end_frame)
                    kwargs['start_frame'] = start_frame
                    kwargs['end_frame'] = end_frame
                
                # Get checkbox states
                if 'show_collage' in widgets:
                    kwargs['show_collage'] = widgets['show_collage'].isChecked()
                if 'create_movie' in widgets:
                    kwargs['create_movie'] = widgets['create_movie'].isChecked()
                
                # Get px2micron ratio for tissue trajectory analysis
                if 'px2micron' in widgets:
                    px2micron_widget = widgets['px2micron']
                    # Handle both QDoubleSpinBox and QLineEdit
                    if hasattr(px2micron_widget, 'value'):
                        kwargs['px2micron'] = px2micron_widget.value()
                    elif hasattr(px2micron_widget, 'text'):
                        try:
                            kwargs['px2micron'] = float(px2micron_widget.text())
                        except (ValueError, TypeError):
                            kwargs['px2micron'] = 1.0  # Default if invalid
                    else:
                        kwargs['px2micron'] = 1.0
                
                # Get normalize option for tissue trajectory analysis
                if 'normalize' in widgets:
                    normalize_widget = widgets['normalize']
                    if hasattr(normalize_widget, 'isChecked'):
                        kwargs['normalize'] = normalize_widget.isChecked()
                    else:
                        kwargs['normalize'] = True  # Default
            
            # Run analysis in worker thread for long operations
            def run_analysis_worker(progress_callback):
                try:
                    results = analysis.run(selected_cells, ta_output_folder, frame_range, **kwargs)
                    return results
                except Exception as e:
                    logger.error(f"Error running analysis: {e}")
                    traceback.print_exc()
                    raise
            
            def on_analysis_complete(results):
                if results:
                    from gui.dialogs.analysis_results_dialog import AnalysisResultsDialog
                    dialog = AnalysisResultsDialog(results, self)
                    dialog.exec()
                else:
                    QMessageBox.warning(self, "Analysis Error", "Analysis returned no results")
            
            def on_analysis_error(error):
                logger.error(f"Analysis error: {error}")
                QMessageBox.critical(self, "Analysis Error", f"Error running analysis: {error}")
            
            # Run in thread
            worker = Worker(run_analysis_worker)
            worker.signals.result.connect(on_analysis_complete)
            worker.signals.error.connect(on_analysis_error)
            self.threadpool.start(worker)
            
        except Exception as e:
            logger.error(f"Error in run_analysis: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error running analysis: {e}")
    
    def handle_analysis_cell_selection(self, track_id: int):
        """Handle cell selection for analysis mode.
        
        Args:
            track_id: Track ID of the selected cell
        """
        if not self.current_analysis:
            return
        
        # Add cell to selection manager
        completed = self.selection_manager.add_cell(track_id)
        
        # Update status
        status = self.selection_manager.get_status_message()
        logger.info(status)
        
        # If selection completed (for pair mode), update UI
        if completed:
            # Update selected_track_ids for visual feedback
            buffer = self.selection_manager.get_current_buffer()
            self.selected_track_ids = set(buffer)
            if hasattr(self.paint, 'paint'):
                self.paint.paint.update()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = TissueAnalyzer()
    w.show()
    sys.exit(app.exec())
