import sys
from PyQt5 import QtWidgets
from ui_mainwindow import Ui_MainWindow
import pyqtgraph as pg
import tkinter as tk
from tkinter import filedialog
import os
import json
from read_data import read_ecg
from PyQt5.QtCore import Qt

event_dict = {
    0: 'P on',
    1: 'P off',
    2: 'R on',
    3: 'R off',
    4: 'T on',
    5: 'T off'
}


class MainWidget(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Annotation Tool")  # 设置窗口标题
        self._init_graph_item()
        # pg.setConfigOption('leftButtonPan', False)
        self.data = None
        self.annotation_xlsx = None
        self.annotation_state = -1

        self.current_file = ''
        self.current_dir = ''
        self.files_list = []
        self.current_file_index = -1
        self.openfileButton.clicked.connect(self.open_file)
        self.saveButton.clicked.connect(self.save_annotation)
        self.nextButton.clicked.connect(self.next_file)
        self.clearButton.clicked.connect(self.init_annotation)
        self.lastButton.clicked.connect(self.last_file)

        self.current_point = -1

        self.annotation_points = []

    def _init_graph_item(self):
        win = pg.GraphicsLayoutWidget()
        self.gridLayout.addWidget(win, 3, 0)
        x = range(0, 600000, 200)
        ticks = [(i, i / 50.) for i in x]
        strAxis = pg.AxisItem(orientation='bottom')
        strAxis.setTicks([ticks])
        win.nextRow()

        self.leader1 = win.addPlot(title='2')
        self.leader1.vb.setMenuEnabled(False)
        self.leader1.setMouseEnabled(x=True, y=False)
        self.leader1.setRange(xRange=[0, 5000])
        self.leader1.showGrid(x=True, y=True)
        self.leader1.vb.wheelEvent = self.wheelEvent
        self.move_slot_leader1 = pg.SignalProxy(self.leader1.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.vline1 = pg.InfiniteLine(angle=90, movable=False)
        self.leader1.addItem(self.vline1)

        win.nextRow()

        self.leader2 = win.addPlot(title='v2')
        self.leader2.vb.setMenuEnabled(False)
        self.leader2.setMouseEnabled(x=True, y=False)
        self.leader2.setRange(xRange=[0, 5000])
        self.leader2.showGrid(x=True, y=True)
        self.leader2.vb.wheelEvent = self.wheelEvent
        self.vline2 = pg.InfiniteLine(angle=90, movable=False)
        self.leader2.addItem(self.vline2)


        # self.leader2.vb.mouseDragEvent = self.mouseDragEvent_leader2
        # # self.clicked_slot_leader2 = pg.SignalProxy(self.leader2.scene().sigMouseClicked, rateLimit=60,
        #                                            slot=self.mouseClicked_leader2)

        win.nextRow()
        self.leader3 = win.addPlot(title='v4')
        self.leader3.vb.setMenuEnabled(False)
        self.leader3.setMouseEnabled(x=True, y=False)
        self.leader3.setRange(xRange=[0, 5000])
        self.leader3.showGrid(x=True, y=True)
        self.leader3.vb.wheelEvent = self.wheelEvent
        self.vline3 = pg.InfiniteLine(angle=90, movable=False)
        self.leader3.addItem(self.vline3)


    def mark(self, class_signal):
        self.annotation_points[class_signal].append(int(self.current_point))
        self.show_points()

    def show_points(self):
        msg = ''
        for index, points in enumerate(self.annotation_points):
            msg += event_dict[index]
            msg += ':  '
            msg += str(points)
            msg += '\n'
        self.label_annotation_points.setText(msg)

    def open_file(self):
        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename(filetypes=[('TXT', '*.txt'), ('All Files', '*')])
        if file_name != '':
            self.current_dir, tmp = os.path.split(file_name)
            self.current_file = file_name
            self.open_dir()
            self.read_data_display()

    def open_dir(self):
        files = os.listdir(self.current_dir)

        def sort_key(s):
            # 排序关键字匹配
            # 匹配开头数字序号
            return int(s.split('.')[0])

        files.sort(key=sort_key)
        data_list = []
        for file in files:
            if file.endswith('.txt'):
                self.files_list.append(file)
        tmp, file_name = os.path.split(self.current_file)
        for index, data in enumerate(self.files_list):
            if file_name == data:
                self.current_file_index = index
                print(self.current_file_index)

    def next_file(self):
        # print(self.current_dir, self.current_file_index)
        if self.current_dir != '' and self.current_file_index != -1:
            if self.data is not None:
                self.save_annotation()
            self.current_file_index += 1
            if len(self.files_list) > self.current_file_index >= 0:
                self.current_file = os.path.join(self.current_dir, self.files_list[self.current_file_index])
                self.read_data_display()
            if self.current_file_index >= len(self.files_list):
                self.current_file_index = len(self.files_list) - 1
            elif self.current_file_index < 0:
                self.current_file_index = 0

    def last_file(self):
        # print(self.current_dir, self.current_file_index)
        if self.current_dir != '' and self.current_file_index != -1:
            if self.data is not None:
                self.save_annotation()
            self.current_file_index -= 1
            if len(self.files_list) > self.current_file_index >= 0:
                self.current_file = os.path.join(self.current_dir, self.files_list[self.current_file_index])
                self.read_data_display()
            if self.current_file_index >= len(self.files_list):
                self.current_file_index = len(self.files_list) - 1
            elif self.current_file_index < 0:
                self.current_file_index = 0

    def on_refresh_view(self):
        self.leader1.setRange(xRange=[0, 5000])
        self.leader2.setRange(xRange=[0, 5000])
        self.leader3.setRange(xRange=[0, 5000])
        self.leader1.clear()
        self.leader2.clear()
        self.leader3.clear()
        if self.data is not None:
            self.leader1.plot(self.data[:, 1])
            self.leader2.plot(self.data[:, 3])
            self.leader3.plot(self.data[:, 5])
            self.leader1.addItem(self.vline1)
            self.leader2.addItem(self.vline2)
            self.leader3.addItem(self.vline3)

    def init_annotation(self):
        self.annotation_points = []
        for index in range(6):
            points_list = []
            self.annotation_points.append(points_list)

    def read_data_display(self):
        self.data = read_ecg(self.current_file)
        if self.data is not None:
            self.label_file_name.setText(self.current_file)
            self.init_annotation()
            self.on_refresh_view()
            self.show_points()

    def mouseMoved(self, evt):
        if self.data is not None:
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            # print(pos)
            if self.leader1.sceneBoundingRect().contains(pos):
                mousePoint = self.leader1.vb.mapSceneToView(pos)
                index = int(mousePoint.x())
                if 0 < index < self.data[:, 1].shape[0]:
                    self.label_pos.setText('X: {},  Y: {}'.format(mousePoint.x(), mousePoint.y()))
                    self.current_point = mousePoint.x()
                    self.vline1.setPos(mousePoint.x())
                    self.vline2.setPos(mousePoint.x())
                    self.vline3.setPos(mousePoint.x())
                    # self.label_pos.setText(
                    #     "<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>p=%0.1f</span>" % (
                    #         mousePoint.x(), mousePoint.y()))

    def mouseClicked_leader2(self, evt):
        if self.data is not None:
            pos = evt[0].scenePos()
            if self.leader2.sceneBoundingRect().contains(pos):
                mousePoint = self.leader2.vb.mapSceneToView(pos)
                # print(mousePoint)
                index = int(mousePoint.x())
                if 0 <= self.annotation_state < 5:
                    self.annotation_points[self.annotation_state].append(index)
                    self.annotation_state = -1
                    self.label_state.setText('')

    def save_annotation(self):
        if self.data is not None:
            save_file = self.current_file.replace('.txt', '.json')
            with open(save_file, 'w') as f:
                save_dict = {}
                save_dict[event_dict[0]] = self.annotation_points[0]
                save_dict[event_dict[1]] = self.annotation_points[1]
                save_dict[event_dict[2]] = self.annotation_points[2]
                save_dict[event_dict[3]] = self.annotation_points[3]
                save_dict[event_dict[4]] = self.annotation_points[4]
                save_dict[event_dict[5]] = self.annotation_points[5]
                save_json = json.dumps(save_dict)
                f.write(save_json)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            self.annotation_state = 0
            self.label_state.setText(event_dict[0])
            self.mark(0)
        if event.key() == Qt.Key_2:
            self.annotation_state = 1
            self.label_state.setText(event_dict[1])
            self.mark(1)
        if event.key() == Qt.Key_3:
            self.annotation_state = 2
            self.label_state.setText(event_dict[2])
            self.mark(2)
        if event.key() == Qt.Key_4:
            self.annotation_state = 3
            self.label_state.setText(event_dict[3])
            self.mark(3)
        if event.key() == Qt.Key_5:
            self.annotation_state = 4
            self.label_state.setText(event_dict[4])
            self.mark(4)
        if event.key() == Qt.Key_6:
            self.annotation_state = 5
            self.label_state.setText(event_dict[5])
            self.mark(5)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWidget()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
