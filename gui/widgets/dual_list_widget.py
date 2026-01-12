# e QStackedWidget and put all possible widgets into the stack. When selecting an item, just call setCurrentWidget

import os
from utils.qt_settings import set_UI # set the UI to be used py qtpy
set_UI()
# this is a class with two lists

from qtpy.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from utils.list_utils import loadlist
from gui.widgets.file_list_widget import ListGUI


# en fait ça marche pas il me faut un widget double en fait!!!
# class dualList(ListGUI):
class dualList(QWidget):

    # TODO define supported formats --> TODO
    def __init__(self, parent=None):
        super().__init__(parent)
        self.list1 = ListGUI()
        self.list2 = ListGUI()
        self.list1.setObjectName('list1')
        self.list2.setObjectName('list2')
        self.lists = [self.list1, self.list2]
        # if random.random() <0.5:
        #     self.current_list = self.list1
        # else:
        #     self.current_list = self.list2

        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.list1)
        self.Stack.addWidget(self.list2)

        # if random.random() <0.5:
        #     self.Stack.setCurrentIndex(0)
        # else:
        #     self.Stack.setCurrentIndex(1)


        layout = QVBoxLayout()
        # layout.addWidget(self.list1)
        layout.addWidget(self.Stack)
        self.setLayout(layout)

    # do not do that again just check which list needs be checked
    # def get_full_list(self, idx=None):
    #     if idx is None:
    #         print(self.Stack.currentIndex())
    #         print(self.lists[self.Stack.currentIndex()].objectName())
    #         self.lists[self.Stack.currentIndex()].get_full_list()
    #     else:
    #         self.lists[idx].get_full_list()
    #
    # def get_selection(self, mode='single', idx=None):
    #     if idx is None:
    #         self.lists[self.Stack.currentIndex()].get_selection(mode=mode)
    #     else:
    #         self.lists[idx].get_selection(mode=mode)

    def freeze(self, bool):

        # print('freeze called', bool)
        #
        for lst in self.lists:
            # print(list)
            lst.freeze(bool)
            # self.list1.freeze(bool)
            # self.list2.freeze(bool)

    # get list with given idx
    def get_list(self, idx, force_within_bounds=True):
        if force_within_bounds:
            if idx<0:
                result = self.lists[0]
            elif idx>=len(self.lists):
                result = self.lists[len(self.lists)-1]
            else:
                result = self.lists[idx]
        else:
            result = self.lists[idx]
        return result

    def set_list(self,idx, force_within_bounds=True):
        if force_within_bounds:
            if idx < 0:
                self.Stack.setCurrentIndex(0)
            if idx >= len(self.lists):
                self.Stack.setCurrentIndex(len(self.lists) - 1)
        result = self.Stack.setCurrentIndex(idx)
        return result


if __name__ == '__main__':
    # TODO add a main method so it can be called directly
    # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!
    import sys
    from qtpy.QtWidgets import QApplication, QWidget, QWidget

    # should probably have his own scroll bar embedded somewhere

    app = QApplication(sys.argv)

    w = dualList() #file_to_load='/E/Sample_images/sample_images_pyta/list.lst')

    w.list1.add_to_list(loadlist('/E/Sample_images/sample_images_pyta/list.lst'))

    # marche la mais pas ds l'autre --> pkoi

    w.freeze(True)
    w.freeze(False)
    w.freeze(True)

    w.show()

    print('get_full_list', w.get_list(0).get_full_list()) # very good and useful
    print('sel', w.get_list(0).get_selection(mode='all'))

    sys.exit(app.exec())



