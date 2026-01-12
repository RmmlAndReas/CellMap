"""Freehand2D shape class for CellMap."""
from utils.qt_settings import set_UI
set_UI()
from qtpy import QtCore
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QPainter, QBrush, QPen, QImage, QColor
from gui.shapes.polygon2d import Polygon2D
from utils.logger import TA_logger

logger = TA_logger()


class Freehand2D(Polygon2D):
    """Freehand2D class for freehand drawing, extending Polygon2D."""

    def __init__(self, *args, color=0xFFFF00, opacity=1., stroke=0.65, line_style=None, theta=0, invert_coords=False, fill_color=None, **kwargs):
        super(Freehand2D, self).__init__()
        self.isSet = False
        if len(args) > 0:
            if isinstance(args[0], tuple):
                for i in range(0, len(args)):
                    if invert_coords:
                        self.append(QPointF(args[i][1], args[i][0]))
                    else:
                        self.append(QPointF(args[i][0], args[i][1]))
            else:
                for i in range(0, len(args), 2):
                    if invert_coords:
                        self.append(QPointF(args[i+1], args[i]))
                    else:
                        self.append(QPointF(args[i], args[i+1]))
            self.isSet = True
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.line_style = line_style
        self.theta = theta

    def setP1(self, point):
        self.append(point)

    def add(self, *args):
        self.append(args[1])
        self.isSet = True
