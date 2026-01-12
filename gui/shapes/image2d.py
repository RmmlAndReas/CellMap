"""Image2D shape class for CellMap."""
from utils.qt_settings import set_UI
set_UI()
import numpy as np
from qtpy import QtGui
from qtpy.QtGui import QPainter, QColor
from qtpy.QtCore import QRectF, QPointF, QSize, QRect
from gui.shapes.rect2d import Rect2D
from utils.image_io import Img
from utils.image_utils import toQimage
from utils.logger import TA_logger

logger = TA_logger()


class Image2D(Rect2D):
    """Image2D class for displaying images, extending Rect2D."""
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    CENTERED = 4

    def __init__(self, *args, x=None, y=None, width=None, height=None, data=None, dimensions=None, opacity=1., **kwargs):
        self.isSet = False
        self.scale = 1
        self.translation = QPointF()

        self.__crop_left = 0
        self.__crop_right = 0
        self.__crop_top = 0
        self.__crop_bottom = 0
        self.img = None
        self.annotation = []
        self.letter = None
        self.top_left_objects = []
        self.top_right_objects = []
        self.bottom_right_objects = []
        self.bottom_left_objects = []
        self.centered_objects = []

        self.fraction_of_parent_image_width_if_image_is_inset = 0.25
        self.border_size = None
        self.border_color = 0xFFFFFF

        if args:
            if len(args) == 1:
                if isinstance(args[0], str):
                    self.filename = args[0]
                elif isinstance(args[0], Img):
                    self.filename = None
                    self.img = args[0]
                    self.qimage = toQimage(self.img)
                    if x is None:
                        x = 0
                    if y is None:
                        y = 0
                    try:
                        super(Image2D, self).__init__(x, y, self.img.shape[1], self.img.shape[0])
                    except:
                        super(Image2D, self).__init__(x, y, self.img.shape[1], self.img.shape[0])
                    self.isSet = True
        else:
            self.filename = None

        if x is None and y is None and width is not None and height is not None:
            super(Image2D, self).__init__(0, 0, width, height)
            self.isSet = True
        elif x is None and y is None and width is None and height is None and self.filename is not None:
            try:
                self.img = Img(self.filename)
            except:
                logger.error('could not load image ' + str(self.filename))
                return
            self.qimage = toQimage(self.img)
            width = self.img.shape[1]
            height = self.img.shape[0]
            super(Image2D, self).__init__(0, 0, width, height)
            self.isSet = True
        elif x is not None and y is not None and width is not None and height is not None and self.img is None:
            self.img = None
            super(Image2D, self).__init__(x, y, width, height)
            self.isSet = True
        elif data is None:
            if self.filename is not None:
                self.img = Img(self.filename)
                self.qimage = toQimage(self.img)
                if x is None:
                    x = 0
                if y is None:
                    y = 0
                super(Image2D, self).__init__(x, y, self.img.shape[1], self.img.shape[0])
                self.isSet = True
        elif data is not None:
            self.img = Img(data, dimensions=dimensions)
            self.qimage = toQimage(self.img)
            if x is None:
                x = 0
            if y is None:
                y = 0
            super(Image2D, self).__init__(x, y, self.img.shape[1], self.img.shape[0])
            self.isSet = True
        self.opacity = opacity

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
            painter.setOpacity(self.opacity)
            rect_to_plot = self.boundingRect(scaled=True)

            if self.img is not None:
                x = 0
                y = 0
                try:
                    w = self.img.shape[1]
                    h = self.img.shape[0]
                except:
                    w = self.img.shape[1]
                    h = self.img.shape[0]

                if self.__crop_top is not None:
                    y = self.__crop_top
                    h -= self.__crop_top
                if self.__crop_left is not None:
                    x = self.__crop_left
                    w -= self.__crop_left
                if self.__crop_right is not None:
                    w -= self.__crop_right
                if self.__crop_bottom is not None:
                    h -= self.__crop_bottom
                qsource = QRectF(x, y, w, h)
                painter.drawImage(rect_to_plot, self.qimage, qsource)
            else:
                painter.drawRect(rect_to_plot)

            if self.annotation is not None and self.annotation:
                for annot in self.annotation:
                    annot.set_to_translation(rect_to_plot.topLeft())
                    annot.set_to_scale(self.scale)
                    annot.draw(painter=painter)

            painter.restore()

    def set_to_translation(self, translation):
        self.translation = translation

    def boundingRect(self, scaled=True):
        rect_to_plot = self.adjusted(0, 0, -self.__crop_right - self.__crop_left, -self.__crop_bottom - self.__crop_top)
        if self.scale is not None and self.scale != 1 and scaled:
            new_width = rect_to_plot.width() * self.scale
            new_height = rect_to_plot.height() * self.scale
            rect_to_plot.setWidth(new_width)
            rect_to_plot.setHeight(new_height)
        return rect_to_plot

    def get_P1(self):
        return self.boundingRect().topLeft()

    def width(self, scaled=True):
        return self.boundingRect(scaled=scaled).width()

    def height(self, scaled=True):
        return self.boundingRect(scaled=scaled).height()

    def setToWidth(self, width_in_px):
        pure_image_width = self.width(scaled=False)
        scale = width_in_px / pure_image_width
        self.scale = scale

    def setToHeight(self, height_in_px):
        pure_image_height = self.height(scaled=False)
        scale = height_in_px / pure_image_height
        self.scale = scale
