"""Qt logging handler utilities for CellMap."""
import sys
from utils.qt_settings import set_UI
set_UI()
from qtpy import QtCore
import logging


class QtHandler(logging.Handler):
    """
    Custom logging handler for Qt application.
    It emits log records to either stdout or stderr based on the log level.
    """

    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        """Formats and emits the log record."""
        record = self.format(record)
        if record:
            if record.startswith('ERROR') or record.startswith('CRITICAL') or record.startswith('WARNING'):
                XStream.stderr().write('%s' % record)
            else:
                XStream.stdout().write('%s' % record)


class XStream(QtCore.QObject):
    """
    Custom QObject class for capturing stdout and stderr.
    """

    _stdout = None
    _stderr = None
    messageWritten = QtCore.Signal(str)

    def flush(self):
        """Flushes the output."""
        pass

    def fileno(self):
        """Returns the file number."""
        return -1

    def write(self, msg):
        """Writes the message to the output stream."""
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        """Returns the singleton instance for stdout."""
        if XStream._stdout is None:
            XStream._stdout = XStream()
        return XStream._stdout

    @staticmethod
    def stderr():
        """Returns the singleton instance for stderr."""
        if XStream._stderr is None:
            XStream._stderr = XStream()
        return XStream._stderr


__all__ = ['XStream', 'QtHandler']
