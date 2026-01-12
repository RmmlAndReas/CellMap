"""Scrollable paint widget for image viewing and editing.

This module provides a scrollable widget for displaying and editing images
with support for multiple channels, zoom, and drawing tools.
"""

import os
from utils.qt_settings import set_UI
set_UI()
import traceback

import matplotlib.pyplot as plt
import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QPalette, QKeySequence, QPainter
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget, QToolBar, QStatusBar, QLabel, \
    QHBoxLayout, QAction, QSlider
import qtawesome as qta
from qtpy.QtCore import Qt, QTimer
from gui.shapes.freehand2d import Freehand2D
from gui.shapes.image2d import Image2D
from gui.shapes.rect2d import Rect2D
from utils.image_io import Img
from utils.image_utils import int24_to_RGB, has_metadata
from gui.widgets.paint_widget import Createpaintwidget


class scrollable_paint(QWidget):

    def __init__(self, custom_paint_panel=None):
        super().__init__()
        if custom_paint_panel is None:
            self.paint = Createpaintwidget()
        else:
            self.paint = custom_paint_panel
        layout = QVBoxLayout()

        self.dimension_sliders = []
        self.is_width_for_alternating = True
        self.dimensions_container = QVBoxLayout()
        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.paint)
        self.paint.scrollArea = self.scrollArea
        layout.addLayout(self.dimensions_container)
        
        # Create horizontal layout for image area and toolbars
        main_content_layout = QHBoxLayout()
        
        # Left side: image area (scrollArea + status_bar)
        image_area_layout = QVBoxLayout()
        image_area_layout.addWidget(self.scrollArea)
        status_bar = QStatusBar()
        image_area_layout.addWidget(status_bar)
        self.paint.statusBar = status_bar
        
        # Add image area to main content layout (left side, stretchable)
        main_content_layout.addLayout(image_area_layout, 1)
        
        # Right side: toolbars will be added here in drawing_commands()
        self.toolbar_container = QVBoxLayout()
        main_content_layout.addLayout(self.toolbar_container, 0)
        
        # Add main content layout to main layout
        layout.addLayout(main_content_layout)
        self.setLayout(layout)
        self.add_shortcuts()
        self.drawing_commands()

    def add_shortcuts(self):
        zoomPlus = QtWidgets.QShortcut("Ctrl+Shift+=", self)
        zoomPlus.activated.connect(self.zoomIn)
        zoomPlus.setContext(QtCore.Qt.ApplicationShortcut)

        zoomPlus2 = QtWidgets.QShortcut("Ctrl++", self)
        zoomPlus2.activated.connect(self.zoomIn)
        zoomPlus2.setContext(QtCore.Qt.ApplicationShortcut)

        zoomMinus = QtWidgets.QShortcut("Ctrl+Shift+-", self)
        zoomMinus.activated.connect(self.zoomOut)
        zoomMinus.setContext(QtCore.Qt.ApplicationShortcut)

        zoomMinus2 = QtWidgets.QShortcut("Ctrl+-", self)
        zoomMinus2.activated.connect(self.zoomOut)
        zoomMinus2.setContext(QtCore.Qt.ApplicationShortcut)

        ctrl0 = QtWidgets.QShortcut("Ctrl+0", self)
        ctrl0.activated.connect(self.zoom_reset)
        ctrl0.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrlS = QtWidgets.QShortcut("Ctrl+S", self)
        # Use safe wrapper if available to avoid closing the app on save errors
        self.ctrlS.activated.connect(getattr(self.paint, "_safe_save", self.paint.save))
        self.ctrlS.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrlZ = QtWidgets.QShortcut("Ctrl+Z", self)
        self.ctrlZ.activated.connect(self.paint.undo)
        self.ctrlZ.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrlM = QtWidgets.QShortcut("Ctrl+M", self)
        self.ctrlM.activated.connect(self.paint.ctrl_m_apply)
        self.ctrlM.setContext(QtCore.Qt.ApplicationShortcut)

        self.shrtM = QtWidgets.QShortcut("M", self)
        self.shrtM.activated.connect(self.paint.m_apply)
        self.shrtM.setContext(QtCore.Qt.ApplicationShortcut)

        self.increase_contrastC = QtWidgets.QShortcut('C', self)
        self.increase_contrastC.activated.connect(self.paint.increase_contrast)
        self.increase_contrastC.setContext(QtCore.Qt.ApplicationShortcut)

        self.enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        self.enterShortcut.activated.connect(self.paint.apply)
        self.enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)

        self.enterShortcut2 = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)
        self.enterShortcut2.activated.connect(self.paint.apply)
        self.enterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)

        self.shiftEnterShortcut = QtWidgets.QShortcut("Shift+Enter", self)
        self.shiftEnterShortcut.activated.connect(self.paint.shift_apply)
        self.shiftEnterShortcut.setContext(QtCore.Qt.ApplicationShortcut)

        self.shiftEnterShortcut2 = QtWidgets.QShortcut("Shift+Return", self)
        self.shiftEnterShortcut2.activated.connect(self.paint.shift_apply)
        self.shiftEnterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrl_shift_S_grab_screen_shot = QtWidgets.QShortcut('Ctrl+Shift+S', self)
        self.ctrl_shift_S_grab_screen_shot.activated.connect(self.paint.grab_screen_shot)
        self.ctrl_shift_S_grab_screen_shot.setContext(QtCore.Qt.ApplicationShortcut)

        self.supr = QtWidgets.QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.supr.activated.connect(self.paint.suppr_pressed)
        self.supr.setContext(QtCore.Qt.ApplicationShortcut)

    def enable_shortcuts(self):
        self.disable_shortcuts()
        self.ctrlS.activated.connect(getattr(self.paint, "_safe_save", self.paint.save))
        self.ctrlM.activated.connect(self.paint.ctrl_m_apply)
        self.shrtM.activated.connect(self.paint.m_apply)
        self.enterShortcut.activated.connect(self.paint.apply)
        self.enterShortcut2.activated.connect(self.paint.apply)
        self.shiftEnterShortcut.activated.connect(self.paint.shift_apply)
        self.shiftEnterShortcut2.activated.connect(self.paint.shift_apply)

    def disable_shortcuts(self):
        try:
            self.ctrlS.disconnect()
        except:
            pass
        try:
            self.ctrlM.disconnect()
        except:
            pass
        try:
            self.shrtM.disconnect()
        except:
            pass
        try:
            self.enterShortcut.disconnect()
        except:
            pass
        try:
            self.enterShortcut2.disconnect()
        except:
            pass
        try:
            self.shiftEnterShortcut.disconnect()
        except:
            pass
        try:
            self.shiftEnterShortcut2.disconnect()
        except:
            pass

    def enableMouseTracking(self):
        self.paint.drawing_enabled = True

    def disableMouseTracking(self):
        self.paint.drawing_enabled = False

    def fit_to_width_or_height(self):
        if self.is_width_for_alternating:
            self.fit_to_width()
        else:
            self.fit_to_height()
        self.is_width_for_alternating = not self.is_width_for_alternating

    def fit_to_width(self):
        if self.paint.image is None:
            return
        width = self.scrollArea.width() - 2
        width -= self.scrollArea.verticalScrollBar().sizeHint().width()
        width_im = self.paint.image.width()
        scale = width / width_im
        self.scaleImage(scale)

    def fit_to_height(self, nb=0):
        if self.paint.image is None:
            return
        height = self.scrollArea.height() - 2
        height -= self.scrollArea.horizontalScrollBar().sizeHint().height()
        height_im = self.paint.image.height()
        scale = height / height_im
        self.scaleImage(scale)

    def fit_to_window(self):
        if self.paint.image is None:
            return

        width = self.scrollArea.width() - 2
        height = self.scrollArea.height() - 2

        height_im = self.paint.image.height()
        width_im = self.paint.image.width()
        scale = height / height_im
        if width / width_im < scale:
            scale = width / width_im
        self.scaleImage(scale)

    def zoomIn(self):
        self.scaleImage(self.paint.scale + (self.paint.scale*10.0/100.0))

    def zoomOut(self):
        self.scaleImage(self.paint.scale - (self.paint.scale*10.0/100.0))

    def zoom_reset(self):
        self.scaleImage(1.0)

    def scaleImage(self, scale):
        self.paint.set_scale(scale)
        if self.paint.image is not None:
            self.paint.resize(self.paint.scale * self.paint.image.size())
        else:
            self.paint.resize(QSize(0, 0))
        self.paint.update()

    def drawing_commands(self):
        # Create vertical layout for toolbars on the right side
        vlayout_of_toolbars = QVBoxLayout()
        vlayout_of_toolbars.setAlignment(Qt.AlignTop)
        self.tb1 = QToolBar()
        self.tb2 = QToolBar()
        
        # Set toolbars to be vertical (buttons stacked vertically)
        self.tb1.setOrientation(Qt.Vertical)
        self.tb2.setOrientation(Qt.Vertical)

        # Color palette button for overlay color
        color_palette_action = QAction(qta.icon('mdi.palette'), 'Choose overlay color', self)
        color_palette_action.setToolTip('Open color picker to choose the overlay drawing color')
        color_palette_action.triggered.connect(self.paint.choose_draw_color)
        self.tb2.addAction(color_palette_action)

        save_action = QAction(qta.icon('fa5.save'), 'Save mask', self)
        # Use safe wrapper if available to avoid closing the app on save errors
        save_action.triggered.connect(getattr(self.paint, "_safe_save", self.paint.save))
        self.tb2.addAction(save_action)

        undo_action = QAction(qta.icon('mdi.undo'), 'Undo (Ctrl+Z)', self)
        undo_action.setToolTip('Undo last mask modification (Ctrl+Z)')
        undo_action.triggered.connect(self.paint.undo)
        self.tb2.addAction(undo_action)

        show_hide_mask = QAction(qta.icon('mdi.target'), 'Show/hide mask (same as pressing "M")', self)
        show_hide_mask.triggered.connect(self.paint.m_apply)
        self.tb2.addAction(show_hide_mask)

        apply_rm_small_cells = QAction(qta.icon('mdi.check-underline'), 'Apply drawing and remove small cells (same as pressing "Shift+Enter")', self)
        apply_rm_small_cells.triggered.connect(self.paint.shift_apply)
        self.tb2.addAction(apply_rm_small_cells)

        zoom_plus = QAction(qta.icon('ei.zoom-in'), 'Zoom+', self)
        zoom_plus.triggered.connect(self.zoomIn)
        self.tb1.addAction(zoom_plus)

        zoom_minus = QAction(qta.icon('ei.zoom-out'), 'Zoom-', self)
        zoom_minus.triggered.connect(self.zoomOut)
        self.tb1.addAction(zoom_minus)

        zoom_width_or_height = QAction(qta.icon('mdi.fit-to-page-outline'), 'Alternates between best fit in width and height', self)
        zoom_width_or_height.triggered.connect(self.fit_to_width_or_height)
        self.tb1.addAction(zoom_width_or_height)

        zoom_0 = QAction(qta.icon('ei.resize-full'), 'Reset zoom', self)
        zoom_0.triggered.connect(self.zoom_reset)
        self.tb1.addAction(zoom_0)

        vlayout_of_toolbars.addWidget(self.tb2)
        vlayout_of_toolbars.addWidget(self.tb1)
        vlayout_of_toolbars.addStretch()  # Push toolbars to top

        # Add toolbars to the right side container
        self.toolbar_container.addLayout(vlayout_of_toolbars)

    def get_selected_channel(self):
        # Always return None (merge mode) since channels dropdown is removed
        return None

    # TODO also implement channel change directly within the display tool
    # if merge is applied --> apply on average of all channels --> maybe not so smart an idea but ok to start with and better than what I do in TA
    def channelChange(self, i):
        # update displayed image depending on channel
        # dqqsdqsdqsd
        # pass
        # try change channel if


        # tODO --> need at least to reactivate a bit that
        # pass
        # print('in channel change') # needs a fix
        meta = None
        try:
            meta = self.paint.raw_image.metadata
        except:
            pass
        self.paint.set_display(self.get_image_to_display_including_all_dims(), metadata=meta)
        self.paint.channelChange(i, skip_update_display=True)

        # print('in channel change !!!')
        # if self.img is not None:
        #     # print('in', self.img.metadata)
        #     if self.Stack.currentIndex() == 0:
        #         # need copy the image --> implement that
        #         # print(self.img[..., i].copy())
        #         # print(self.img[..., i])
        #         if i == 0:
        #             self.paint.setImage(self.img)
        #             # print('original', self.img.metadata)
        #         else:
        #             # print('modified0', self.img.metadata)
        #             channel_img = self.img.imCopy(c=i - 1)  # it's here that it is affected
        #             # print('modified1', self.img.metadata)
        #             # print('modified2', channel_img.metadata)
        #             self.paint.setImage(channel_img)
        #         self.paint.update()
        # else:
        #     # logger.error("Not implemented yet TODO add support for channels in 3D viewer")
        #     # sdqdqsdsqdqsd
        #     self.loadVolume()
        # or reimplement that for multi channels !!!




    def set_image(self, img):

        # probably I need store raw image to avoid issues --> TODO ???
        self.paint.set_image(img)# bug seems to be here

        img = self.paint.raw_image

        self.update_image_dimensions()

        # need also update the dimensions if any

        # print(type(img))

        # else I need also reset channels

        # if img is not None:
        #     if img.has_c():
        # Channels dropdown removed - always use merge mode
        # print('out')
        # channels = self.paint.get_nb_channels()
        # self._update_channels(channels)
        # make it update also the channels
    def _delete_layout_content(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)



    # centralized version of the thing
    def get_image_to_display_including_all_dims(self):
        # print('in get_image_to_display_including_all_dims')
        # returns the image to be displayed --> takes into account all the dimensions at once

        #
        # if self.raw_image is not None:
        #     if i == 0:
        #         self.set_display(self.raw_image)
        #         self.channel = None
        #         # print('original', self.img.metadata)
        #     else:
        #         # print('modified0', self.img.metadata)
        #         # I need a hack when the image is single channel yet I need several masks for it !!!
        #         if self.multichannel_mode and i - 1 >= self.raw_image.shape[-1]:
        #             channel_img = self.raw_image.imCopy(c=0)  # if out of bonds load the first channel
        #         else:
        #             channel_img = self.raw_image.imCopy(c=i - 1)  # it's here that it is affected
        #         self.channel = i - 1
        #         # print('modified1', self.img.metadata)
        #         # print('modified2', channel_img.metadata)
        #         self.set_display(
        #             channel_img)  # maybe do a set display instead rather --> easier to handle --> does a subest of the other
        image_to_display = None
        try:
            # need change just the displayed image
            if has_metadata(self.paint.raw_image) and self.paint.raw_image.metadata['dimensions']:
                # change the respective dim
                # need all the spinner values to be recovered in fact
                # and send the stuff
                # print('dimension exists', self.objectName(),'--> changing it')

                # need gather all the dimensions --> TODO
                dimensions = self.paint.raw_image.metadata['dimensions']
                position_h = dimensions.index('h')
                image_to_display = self.paint.raw_image
                if position_h != 0:
                    # loop for all the dimensions before
                    # print('changing stuff')
                    for pos_dim in range(0, position_h):
                        dim = dimensions[pos_dim]
                        value_to_set = self.get_dim_value_by_name(dim)
                        if value_to_set == None:
                            continue
                        else:
                            image_to_display = image_to_display[value_to_set]
                        # if not dimensions[pos_dim]==self.sender().objectName():
                        #     image_to_display = image_to_display[0]
                        # else:
                        #     image_to_display = image_to_display[self.sender().value()]
                # self.paint.set_display(image_to_display)
                channel_to_display = self.get_selected_channel()

                # if force dimensions --> I need a hack even if
                # hack for GT image editor --> where nb of channels are forced and do not necessarily match the channels of the image --> is that smart to put that here ???
                if channel_to_display is not None and 'c' in dimensions:
                    if channel_to_display >= self.paint.raw_image.shape[-1]:
                        image_to_display = image_to_display[..., 0]
                    else:
                        image_to_display=image_to_display[...,channel_to_display]
                    # print(image_to_display.shape)
        except:
            traceback.print_exc()
        return image_to_display

    def update_image_dimensions(self):
        self.dimension_sliders = []

        for i in reversed(range(self.dimensions_container.count())):
            # print(i)
            # print(self.dimensions_container.itemAt(i))
            # self.dimensions_container.itemAt(i).widget().setParent(None)
            self._delete_layout_content(self.dimensions_container.itemAt(i))

        if self.paint.raw_image is None:
            return

        # empty the content of the slider and refill it
        try:
            treat_channels_as_a_browsable_dimension = False
            if has_metadata(self.paint.raw_image):
                dimensions = self.paint.raw_image.metadata.get('dimensions', '')

                # if not self.paint.raw_image.has_c():
                #     nb_of_sliders = len(
                #         self.paint.raw_image.shape) - 2  # or -3 it depends whether I wanna show all the channels at once or not ???
                # else:
                #     nb_of_sliders = len(
                #         self.paint.raw_image.shape) - 2  # or -3 it depends whether I wanna show all the channels at once or not ???

                # print(nb_of_sliders)

                # create an image with plenty of sliders --> TODO
                # make the handling of the image to be displayed directly by the code !!!

                # then I need couple each slider to a dimension

                # dimensions that must have a slider
                # --> all dimensions but hw and maybe c must have a slider --> TODO
                for dim in dimensions:
                    if dim == 'h' or dim == 'y' or dim == 'w' or dim == 'x' or (
                            not treat_channels_as_a_browsable_dimension and dim == 'c'):
                        # we skip dimensions
                        continue
                    # print(dim, 'must have an assoicated slider')

                    # print(self.paint.raw_image.shape, ' toto ', )
                    self.dimensions_container.addLayout(self.create_dim_slider(dimension=dim, max_dim=self.paint.raw_image.shape[dimensions.index(dim)]))
        except:
            traceback.print_exc()



    def set_mask(self, mask):
        self.paint.set_mask(mask)



    def freeze(self, bool, level=1):

        self.tb1.setEnabled(True)
        self.tb2.setEnabled(True)

        if level == 1:
            self.tb2.setEnabled(not bool)
        elif level == 2:
            self.tb2.setEnabled(not bool)

        if bool:
            self.disable_shortcuts()
        else:
            self.enable_shortcuts()
        # remove draw mode maybe and freeze shortcuts

    def create_dim_slider(self, dimension=None, max_dim=1):
        dim_slider_with_label1 = QHBoxLayout()
        label_slider1 = QLabel()
        if dimension is not None:
            label_slider1.setText(dimension)
        fake_dim_slider = QSlider(Qt.Horizontal)
        fake_dim_slider.setMinimum(0)
        fake_dim_slider.setMaximum(max_dim-1)
        dim_slider_with_label1.addWidget(label_slider1)
        dim_slider_with_label1.addWidget(fake_dim_slider)
        # dim_slider_with_label1
        # add a partial?
        fake_dim_slider.valueChanged.connect(self.slider_dimension_value_changed)
        # fake_dim_slider.valueChanged.connect(lambda x: self.delayed_dimension_preview.start(600))
        fake_dim_slider.setObjectName(dimension)
        self.dimension_sliders.append(fake_dim_slider)
        return dim_slider_with_label1

    def get_dim_value_by_name(self, label):
        for slider in self.dimension_sliders:
            if slider.objectName() == label:
                return slider.value()
        return None

    # def get_dims_value_by_name(self, label):
    #     for slider in self.dimension_sliders:
    #         if slider.objectName() == label:
    #             return slider.value()
    #     return None

    # p
    def slider_dimension_value_changed(self):
        # TODO --> do delay only if necessary
        # self.delayed_dimension_preview.stop()
        # self.delayed_dimension_preview.start(600)

        # self.delayed_dimension_preview.timeout.connect

        # if self.delayed_dimension_preview.dis

        # TODO only allow this when the dimension changing is over
        # if the dimension exists --> print it
        # print(self.sender(), self.sender().objectName(), self.sender().value())
        # print(self.sender(), self.sender().objectName())
        # print(sender)
        #
        # try:
        #     # need change just the displayed image
        #     if self.paint.raw_image.metadata['dimensions']:
        #         # change the respective dim
        #         # need all the spinner values to be recovered in fact
        #         # and send the stuff
        #         # print('dimension exists', self.objectName(),'--> changing it')
        #
        #         # need gather all the dimensions --> TODO
        #         dimensions = self.paint.raw_image.metadata['dimensions']
        #         position_h =  dimensions.index('h')
        #         image_to_display = self.paint.raw_image
        #         if position_h!=0:
        #             # loop for all the dimensions before
        #             # print('changing stuff')
        #             for pos_dim in range(0,position_h):
        #                 dim = dimensions[pos_dim]
        #                 value_to_set = self.get_dim_value_by_name(dim)
        #                 if value_to_set == None:
        #                     continue
        #                 else:
        #                     image_to_display = image_to_display[value_to_set]
        #                 # if not dimensions[pos_dim]==self.sender().objectName():
        #                 #     image_to_display = image_to_display[0]
        #                 # else:
        #                 #     image_to_display = image_to_display[self.sender().value()]
        #         self.paint.set_display(image_to_display)
        # except:
        #     traceback.print_exc()

        try:
            meta = None
            try:
                meta = self.paint.raw_image.metadata
            except:
                pass
            # metadata is required to get the luts properly
            self.paint.set_display(self.get_image_to_display_including_all_dims(), metadata=meta)
        except:
            traceback.print_exc()

        # pass

    # then have a single code to handle dynamically all the dimension changes !!! --> TODO

        # TODO add a main method so it can be called directly
        # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!

