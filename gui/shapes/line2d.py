"""Line2D shape class for CellMap."""
from utils.qt_settings import set_UI
set_UI()
from qtpy.QtCore import QPointF, QLineF, Qt
from qtpy.QtGui import QPen, QColor
from utils.logger import TA_logger

logger = TA_logger()


class Line2D(QLineF):
    """Line2D class for drawing lines, extending QLineF."""

    def __init__(self, *args, color=0xFFFF00, opacity=1., stroke=1.0, arrow=False, line_style=None, theta=0, **kwargs):
        super(Line2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.arrow = arrow
        self.color = color
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

    def draw(self, painter, draw=True):
        if self.color is None:
            return

        if draw:
            painter.save()

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
        painter.setOpacity(self.opacity)
        if draw:
            line_to_plot = self.translated(0, 0)
            if self.scale is not None and self.scale != 1:
                p1 = line_to_plot.p1()
                p2 = line_to_plot.p2()
                line_to_plot.setP1(QPointF(p1.x() * self.scale, p1.y() * self.scale))
                line_to_plot.setP2(QPointF(p2.x() * self.scale, p2.y() * self.scale))
            if self.translation is not None:
                line_to_plot.translate(self.translation)
            if self.theta is not None and self.theta != 0:
                painter.translate(line_to_plot.center())
                painter.rotate(self.theta)
                painter.translate(-line_to_plot.center())

            painter.drawLine(line_to_plot)
            painter.restore()

    def get_P1(self):
        return self.p1()

    def set_P1(self, *args):
        if not args:
            logger.error("no coordinate set...")
            return
        if len(args) == 1:
            p2 = self.p2()
            self.setP1(args[0])
            self.setP2(p2)
        else:
            p2 = self.p2()
            self.setP1(QPointF(args[0], args[1]))
            self.setP2(p2)

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation
