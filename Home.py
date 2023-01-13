# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\home.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from AdminLogin import AdminLogin
from userlogin import UserLogin
from userhome import UserHome

class Ui_Dialog(object):

    def adminlogin(self, event):
        print("admin")
        self.al = QtWidgets.QDialog()
        self.ui = AdminLogin()
        self.ui.setupUi(self.al)
        self.al.show()

        event.accept()
    def userlogin(self, event):
        print("user")
        self.al2 = QtWidgets.QDialog()
        self.ui2 = UserLogin()
        self.ui2.setupUi(self.al2)
        self.al2.show()

        event.accept()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.WindowModal)
        Dialog.resize(659, 540)
        Dialog.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        Dialog.setAutoFillBackground(False)
        Dialog.setStyleSheet("image: url(../images/the-heart-stroke.jpeg);\n"
"")
        self.adminlogo = QtWidgets.QLabel(Dialog)
        self.adminlogo.setGeometry(QtCore.QRect(60, 150, 541, 111))
        self.adminlogo.setStyleSheet("image: url(../images/admin-logo.png);")
        self.adminlogo.setText("")
        self.adminlogo.setObjectName("adminlogo")

        self.adminlogo.mousePressEvent = self.adminlogin

        self.userlogo = QtWidgets.QLabel(Dialog)
        self.userlogo.setGeometry(QtCore.QRect(60, 280, 541, 101))
        self.userlogo.setStyleSheet("image: url(../images/user-logo.png);")
        self.userlogo.setText("")
        self.userlogo.setObjectName("userlogo")

        self.userlogo.mousePressEvent = self.userlogin

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

