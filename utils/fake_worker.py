"""Fake worker for synchronous execution in CellMap."""
import traceback
import sys
from utils.qt_settings import set_UI
set_UI()
from qtpy.QtCore import Signal, Slot, QObject
from utils.worker import WorkerSignals

__all__ = ['FakeWorker']


class FakeWorker(QObject):
    """
    FakeWorker class for synchronous execution (no threading).

    Args:
        fn (function): The function callback to run. Supplied args and kwargs will be passed through to the runner.
        args: Arguments to pass to the callback function
        kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(FakeWorker, self).__init__()

        # Store constructor arguments (reused for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        """
        Initialize the runner function with passed args and kwargs.
        """
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def start(self):
        """
        Start the worker by running the callback function.
        """
        self.run()
