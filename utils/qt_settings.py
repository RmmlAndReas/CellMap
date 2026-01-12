"""Qt settings utilities for CellMap.

This module provides utilities for configuring QtPy UI backend.
"""
import os

# Default UI backend
default_UI = 'pyqt6'


def default_qtpy_UI():
    """
    Returns the default QtPy UI.

    Returns:
        str: The default QtPy UI.

    Examples:
        >>> default_qtpy_UI()
        'pyqt6'
    """
    global default_UI
    if default_UI not in list_UIs():
        default_UI = list_UIs()[0]
    return default_UI


def list_UIs():
    """
    Returns a list of available UI options for QtPy.

    Returns:
        list: The list of available UI options.

    Examples:
        >>> list_UIs()
        ['pyqt5', 'pyside2', 'pyqt6', 'pyside6']
    """
    from qtpy import API_NAMES
    return list(API_NAMES.keys())


def set_UI(ignore_if_already_set=True, UI=default_UI):
    """
    Sets the QtPy UI environment variable.

    Args:
        ignore_if_already_set (bool, optional): Flag to ignore if the UI is already set. Defaults to True.
        UI (str, optional): The UI to set. Defaults to default_UI.

    Examples:
        >>> set_UI()
        >>> print(os.environ['QT_API'])
        pyqt6
    """
    if ignore_if_already_set and 'QT_API' in os.environ:
        return

    if UI is None:
        UI = default_qtpy_UI()
    os.environ['QT_API'] = UI


def print_default_qtpy_UI_really_in_use():
    """
    Prints the default QtPy UI and the version in use.

    Examples:
        >>> print_default_qtpy_UI_really_in_use()
        pyqt6 v6.0.0
    """
    try:
        UI_defined = None
        from qtpy.QtCore import PYQT_VERSION_STR
        try:
            UI_defined = os.environ['QT_API']
        except:
            pass
        print(UI_defined + ' v' + PYQT_VERSION_STR)
    except:
        pass


def force_qtpy_to_use_user_specified(force=True):
    """
    Forces QtPy to use the user-specified UI.

    Args:
        force (bool, optional): Flag to enable or disable forcing QtPy to use the user-specified UI. Defaults to True.
    """
    if force:
        os.environ['FORCE_QT_API'] = '1'
    else:
        try:
            del os.environ['FORCE_QT_API']
        except:
            pass
