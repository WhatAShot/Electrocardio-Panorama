# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.buttonLayout = QtWidgets.QGridLayout()
        self.gridLayout.addLayout(self.buttonLayout, 0, 0)

        self.openfileButton = QtWidgets.QPushButton(self.centralwidget)
        self.openfileButton.setObjectName("openfileButton")
        self.buttonLayout.addWidget(self.openfileButton, 0, 0, 1, 1)

        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setObjectName('saveButton')
        self.buttonLayout.addWidget(self.saveButton, 1, 0, 1, 1)

        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setObjectName('nextButton')
        self.buttonLayout.addWidget(self.nextButton, 2, 0, 1, 1)

        self.clearButton = QtWidgets.QPushButton(self.centralwidget)
        self.clearButton.setObjectName('clearButton')
        self.buttonLayout.addWidget(self.clearButton, 4, 0, 1, 1)

        self.lastButton = QtWidgets.QPushButton(self.centralwidget)
        self.lastButton.setObjectName('lastButton')
        self.buttonLayout.addWidget(self.lastButton, 3, 0, 1, 1)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.label_pos = QtWidgets.QLabel()
        self.buttonLayout.addWidget(self.label_pos, 0, 1)

        self.label_state = QtWidgets.QLabel()
        self.buttonLayout.addWidget(self.label_state, 1, 1)

        self.label_file_name = QtWidgets.QLabel()
        self.buttonLayout.addWidget(self.label_file_name, 2, 1)

        self.label_annotation_points = QtWidgets.QLabel()
        self.buttonLayout.addWidget(self.label_annotation_points, 4, 1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openfileButton.setText(_translate("MainWindow", "open"))
        self.saveButton.setText('save')
        self.nextButton.setText('next')
        self.clearButton.setText('clear')
        self.lastButton.setText('previous')
