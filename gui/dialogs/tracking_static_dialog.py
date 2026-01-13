import os
from utils.qt_settings import set_UI # set the UI to be used py qtpy
set_UI()
import sys


#https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDialogButtonBox.html
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QPushButton, QWidget, QHBoxLayout, QLabel, \
    QDoubleSpinBox, QCheckBox, QSpinBox, QApplication, QGroupBox


class TrackingDialog(QDialog):

    def __init__(self, parent=None, mode='static'):
        super(TrackingDialog, self).__init__(parent)
        self.mode = mode  # 'static' or 'dynamic'

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.setupUI()
        self.setWindowTitle('Tracking options')

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def setupUI(self):
        main_panel = QWidget()
        main_panel.setLayout(QVBoxLayout())

        if self.mode == 'static':
            # Static tracking parameters
            self.b1 = QCheckBox("Recursive error removal (loses less cells but algorithm becomes much slower)") # NB I may also ask for the nb of recursions
            self.b1.setChecked(False)
            main_panel.layout().addWidget(self.b1)

            self.c1 = QCheckBox("Warp using mermaid maps if available (tuto coming soon)")
            self.c1.setChecked(True)
            main_panel.layout().addWidget(self.c1)
        elif self.mode == 'dynamic':
            # Dynamic tracking parameters
            # Pyramidal depth parameter
            pyramidal_layout = QHBoxLayout()
            pyramidal_label = QLabel("Pyramidal depth:")
            pyramidal_label.setToolTip(
                "Depth of pyramidal registration (1-5).\n"
                "Higher values improve alignment but increase computation time.\n"
                "For large cells, lower values may be sufficient and faster.")
            self.pyramidal_depth = QSpinBox()
            self.pyramidal_depth.setMinimum(1)
            self.pyramidal_depth.setMaximum(5)
            self.pyramidal_depth.setValue(3)
            pyramidal_layout.addWidget(pyramidal_label)
            pyramidal_layout.addWidget(self.pyramidal_depth)
            pyramidal_layout.addStretch()
            main_panel.layout().addLayout(pyramidal_layout)

            # Threshold translation parameter
            threshold_translation_layout = QHBoxLayout()
            threshold_translation_label = QLabel("Translation threshold:")
            threshold_translation_label.setToolTip(
                "Maximum translation (in pixels) to accept during pyramidal registration (1-200).\n"
                "Translations larger than this are rejected as likely errors.\n"
                "Increase for tissues with large jumps between frames.\n"
                "Default: 20 pixels")
            self.threshold_translation = QSpinBox()
            self.threshold_translation.setMinimum(1)
            self.threshold_translation.setMaximum(200)
            self.threshold_translation.setValue(20)
            threshold_translation_layout.addWidget(threshold_translation_label)
            threshold_translation_layout.addWidget(self.threshold_translation)
            threshold_translation_layout.addStretch()
            main_panel.layout().addLayout(threshold_translation_layout)

            # Max iterations parameter
            max_iter_layout = QHBoxLayout()
            max_iter_label = QLabel("Max optimization iterations:")
            max_iter_label.setToolTip(
                "Maximum number of optimization iterations for swapping correction (5-30).\n"
                "The algorithm will stop early if no improvement is detected.\n"
                "Higher values may help with difficult tracking cases.")
            self.max_iter = QSpinBox()
            self.max_iter.setMinimum(5)
            self.max_iter.setMaximum(30)
            self.max_iter.setValue(15)
            max_iter_layout.addWidget(max_iter_label)
            max_iter_layout.addWidget(self.max_iter)
            max_iter_layout.addStretch()
            main_panel.layout().addLayout(max_iter_layout)
        else:
            # Both modes - show all parameters
            # Static tracking parameters
            static_group = QGroupBox("Static Tracking Parameters")
            static_layout = QVBoxLayout()
            
            self.b1 = QCheckBox("Recursive error removal (loses less cells but algorithm becomes much slower)")
            self.b1.setChecked(False)
            static_layout.addWidget(self.b1)

            self.c1 = QCheckBox("Warp using mermaid maps if available (tuto coming soon)")
            self.c1.setChecked(True)
            static_layout.addWidget(self.c1)
            
            static_group.setLayout(static_layout)
            main_panel.layout().addWidget(static_group)

            # Dynamic tracking parameters
            dynamic_group = QGroupBox("Dynamic Tracking Parameters")
            dynamic_layout = QVBoxLayout()
            
            # Pyramidal depth parameter
            pyramidal_layout = QHBoxLayout()
            pyramidal_label = QLabel("Pyramidal depth:")
            pyramidal_label.setToolTip(
                "Depth of pyramidal registration (1-5).\n"
                "Higher values improve alignment but increase computation time.\n"
                "For large cells, lower values may be sufficient and faster.")
            self.pyramidal_depth = QSpinBox()
            self.pyramidal_depth.setMinimum(1)
            self.pyramidal_depth.setMaximum(5)
            self.pyramidal_depth.setValue(3)
            pyramidal_layout.addWidget(pyramidal_label)
            pyramidal_layout.addWidget(self.pyramidal_depth)
            pyramidal_layout.addStretch()
            dynamic_layout.addLayout(pyramidal_layout)

            # Threshold translation parameter
            threshold_translation_layout = QHBoxLayout()
            threshold_translation_label = QLabel("Translation threshold:")
            threshold_translation_label.setToolTip(
                "Maximum translation (in pixels) to accept during pyramidal registration (1-200).\n"
                "Translations larger than this are rejected as likely errors.\n"
                "Increase for tissues with large jumps between frames.\n"
                "Default: 20 pixels")
            self.threshold_translation = QSpinBox()
            self.threshold_translation.setMinimum(1)
            self.threshold_translation.setMaximum(200)
            self.threshold_translation.setValue(20)
            threshold_translation_layout.addWidget(threshold_translation_label)
            threshold_translation_layout.addWidget(self.threshold_translation)
            threshold_translation_layout.addStretch()
            dynamic_layout.addLayout(threshold_translation_layout)

            # Max iterations parameter
            max_iter_layout = QHBoxLayout()
            max_iter_label = QLabel("Max optimization iterations:")
            max_iter_label.setToolTip(
                "Maximum number of optimization iterations for swapping correction (5-30).\n"
                "The algorithm will stop early if no improvement is detected.\n"
                "Higher values may help with difficult tracking cases.")
            self.max_iter = QSpinBox()
            self.max_iter.setMinimum(5)
            self.max_iter.setMaximum(30)
            self.max_iter.setValue(15)
            max_iter_layout.addWidget(max_iter_label)
            max_iter_layout.addWidget(self.max_iter)
            max_iter_layout.addStretch()
            dynamic_layout.addLayout(max_iter_layout)
            
            dynamic_group.setLayout(dynamic_layout)
            main_panel.layout().addWidget(dynamic_group)

        self.layout().addWidget(main_panel)

    # get user values
    def values(self):
        if self.mode == 'static':
            return (self.b1.isChecked(), self.c1.isChecked())
        elif self.mode == 'dynamic':
            return (self.pyramidal_depth.value(), self.threshold_translation.value(), self.max_iter.value())
        else:
            # Return all values: static first, then dynamic
            return (
                (self.b1.isChecked(), self.c1.isChecked()),
                (self.pyramidal_depth.value(), self.threshold_translation.value(), self.max_iter.value())
            )

    # static method to create the dialog and return (values, accepted)
    @staticmethod
    def getValues(parent=None, preview_enabled=False, mode='static'):
        dialog = TrackingDialog(parent=parent, mode=mode)
        result = dialog.exec()
        values = dialog.values()
        return (values, result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    values, ok = TrackingDialog.getValues()

    print(ok)

    if ok:
        print('just do preview')
    else:
        print('nothing todo')

    print(values, ok)

