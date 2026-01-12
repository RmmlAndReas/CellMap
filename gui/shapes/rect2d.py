"""Rect2D shape class for CellMap."""
from utils.qt_settings import set_UI
set_UI()
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QBrush, QPen, QColor, QTransform
from utils.logger import TA_logger

logger = TA_logger()


class Rect2D(QRectF):
    """
    A custom class for 2D rectangles with additional features such as color, fill color, opacity, stroke, line style,
    rotation, scale, and translation.
    """

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None, theta=0, **kwargs):
        super(Rect2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.isSet = False
        self.scale = 1
        self.translation = QPointF()
        self.line_style = line_style
        self.theta = theta
        self.incompressible_width = 0
        self.incompressible_height = 0

    def set_rotation(self, theta):
        """Set the rotation angle of the rectangle."""
        self.theta = theta

    def set_opacity(self, opacity):
        """Set the opacity value of the rectangle."""
        self.opacity = opacity

    def set_line_style(self, style):
        """Set line style (dashed, dotted, or custom pattern)."""
        self.line_style = style

    def draw(self, painter, **kwargs):
        if self.color is None and self.fill_color is None:
            return

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
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.NoPen)

        if self.fill_color is not None:
            painter.setBrush(QBrush(QColor(self.fill_color)))

        rect_to_plot = self.adjusted(0, 0, 0, 0)
        if self.scale is not None and self.scale != 1:
            new_width = rect_to_plot.width() * self.scale
            new_height = rect_to_plot.height() * self.scale
            rect_to_plot.setX(rect_to_plot.x() * self.scale)
            rect_to_plot.setY(rect_to_plot.y() * self.scale)
            rect_to_plot.setWidth(new_width)
            rect_to_plot.setHeight(new_height)

        if self.translation is not None:
            rect_to_plot.translate(self.translation)

        if self.theta is not None and self.theta != 0:
            painter.translate(rect_to_plot.center())
            painter.rotate(self.theta)
            painter.translate(-rect_to_plot.center())

        painter.drawRect(rect_to_plot)
        painter.restore()

    def boundingRect(self, scaled=True):
        if scaled:
            try:
                if self.theta is not None and self.theta != 0:
                    center = self.center()
                    t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(), -center.y())
                    rotatedRect = t.mapRect(self)
                    return rotatedRect
            except:
                pass
        return self

    def add(self, *args):
        point = args[1]
        self.setWidth(point.x() - self.x())
        self.setHeight(point.y() - self.y())
        self.isSet = True

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

    def getIncompressibleWidth(self):
        return self.incompressible_width

    def getIncompressibleHeight(self):
        return self.incompressible_height

    def setTopLeft(self, *args):
        if args:
            if len(args) == 1:
                super().moveTopLeft(args[0])
            elif len(args) == 2:
                super().moveTopLeft(QPointF(args[0], args[1]))
            else:
                logger.error('invalid args for top left')

    def get_P1(self):
        return self.boundingRect().topLeft()

    def set_P1(self, *args):
        if not args:
            logger.error("no coordinate set...")
            return
        if len(args) == 1:
            self.moveTopLeft(args[0])
        else:
            self.moveTopLeft(QPointF(args[0], args[1]))
