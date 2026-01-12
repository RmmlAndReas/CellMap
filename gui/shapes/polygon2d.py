"""Polygon2D shape class for CellMap."""
from utils.qt_settings import set_UI
set_UI()
from qtpy import QtCore
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QPolygonF, QTransform
from qtpy.QtGui import QPainter, QBrush, QPen, QImage, QColor
from utils.logger import TA_logger

logger = TA_logger()


class Polygon2D(QPolygonF):
    """Polygon2D class extending QPolygonF with drawing capabilities."""

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None, theta=0, **kwargs):
        super(Polygon2D, self).__init__()
        self.isSet = False
        if len(args) > 0:
            for i in range(0, len(args), 2):
                self.append(QPointF(args[i], args[i+1]))
            self.isSet = True

        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.scale = 1
        self.translation = QPointF()
        self.line_style = line_style
        self.theta = theta

    def set_rotation(self, theta):
        self.theta = theta

    def set_opacity(self, opacity):
        self.opacity = opacity

    def set_line_style(self, style):
        self.line_style = style

    def get_color(self):
        return self.color

    def get_fill_color(self):
        return self.fill_color

    def get_stroke_size(self):
        return self.stroke

    def get_points(self):
        points = []
        for point in self:
            points.append((point.x(), point.y()))
        return points

    def contains(self, *args):
        if len(args) != 1:
            point = QPointF(float(args[0]), float(args[1]))
            return self.containsPoint(point, Qt.OddEvenFill)
        else:
            return self.containsPoint(args[0], Qt.OddEvenFill)

    def draw(self, painter, draw=True):
        if self.color is None and self.fill_color is None:
            return

        if draw:
            painter.save()
            painter.setOpacity(self.opacity)
            if self.color is not None:
                pen = QPen(QColor(self.color))
                if self.stroke is not None:
                    pen.setWidthF(self.stroke)
                if self.line_style is not None:
                    if self.line_style in [Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine]:
                        pen.setStyle(self.line_style)
                    elif isinstance(self.line_style, list):
                        pen.setStyle(Qt.CustomDashLine)
                        pen.setDashPattern(self.line_style)
                painter.setPen(pen)
            else:
                painter.setPen(Qt.NoPen)
            if self.fill_color is not None:
                painter.setBrush(QBrush(QColor(self.fill_color)))
            else:
                painter.setBrush(Qt.NoBrush)
            polygon_to_draw = self.translated(0, 0)
            if self.scale is not None and self.scale != 1:
                polygon_to_draw = self.__scaled()

            if self.translation is not None:
                polygon_to_draw.translate(self.translation.x(), self.translation.y())

            if self.theta is not None and self.theta != 0:
                painter.translate(polygon_to_draw.boundingRect().center())
                painter.rotate(self.theta)
                painter.translate(-polygon_to_draw.boundingRect().center())

            painter.drawPolygon(polygon_to_draw)
            painter.restore()

    def get_P1(self):
        return self.boundingRect().topLeft()

    def set_P1(self, point):
        current_pos = self.boundingRect().topLeft()
        self.translate(point.x() - current_pos.x(), point.y() - current_pos.y())

    def add(self, *args, force=False):
        if self.count() > 1 and not force:
            self.remove(self.count() - 1)
        self.append(args[1])
        self.isSet = True

    def listVertices(self):
        return [point for point in self]

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

    def __scaled(self):
        vertices = self.listVertices()
        scaled_poly = QPolygonF()
        for vx in vertices:
            vx.setX(vx.x() * self.scale)
            vx.setY(vx.y() * self.scale)
            scaled_poly.append(vx)
        return scaled_poly

    def boundingRect(self):
        polygon_to_draw = self.translated(0, 0)
        try:
            if self.theta is not None and self.theta != 0:
                center = polygon_to_draw.boundingRect().center()
                t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(), -center.y())
                transformed = t.map(polygon_to_draw)
                return transformed.boundingRect()
        except:
            pass
        return polygon_to_draw.boundingRect()
