# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\userlogin.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from UserRegister import UserRegister
from userloginaction import UserLoginCheck
from userhome import UserHome
import sys

class UserLogin(object):


    def ulogin(self):
        try:
            uidvar = self.uid.text()
            pwdvar = self.pwd.text()
            self.uid.setText("")
            self.pwd.setText("")
            al = UserLoginCheck()
            res = al.datacheck(uidvar, pwdvar)
            if res:
                self.showAlertBox("Alert", "Fill the Fields")
            elif UserLoginCheck.logincheck(uidvar, pwdvar):
                self.u = QtWidgets.QDialog()
                self.ui = UserHome()
                self.ui.setupUi(self.u)
                self.u.show()
                Dialog.hide()

            else:
                self.showAlertBox("Login Alert", "Login Fail")

        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)

    def userregister(self):
        try:
            self.Dialog21 = QtWidgets.QDialog()
            self.ui21 = UserRegister()
            self.ui21.setupUi(self.Dialog21)
            self.Dialog21.show()




        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)

    def showAlertBox(self, title, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle(title)
        msgBox.setText(message)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(637, 481)
        Dialog.setStyleSheet("background-color: rgb(214, 0, 0);")
        self.pwd = QtWidgets.QLineEdit(Dialog)
        self.pwd.setGeometry(QtCore.QRect(270, 270, 320, 30))
        self.pwd.setStyleSheet("font: 11pt \"Malgun Gothic\";\n"
"color: rgb(255, 255, 255);")
        self.pwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pwd.setObjectName("pwd")
        self.login = QtWidgets.QPushButton(Dialog)
        self.login.setGeometry(QtCore.QRect(270, 330, 75, 31))
        self.login.setStyleSheet("background-color: rgb(244, 244, 244);")
        self.login.setObjectName("login")
        ######################3
        self.login.clicked.connect(self.ulogin)
        #################

        self.uid = QtWidgets.QLineEdit(Dialog)
        self.uid.setGeometry(QtCore.QRect(270, 220, 320, 30))
        self.uid.setStyleSheet("font: 11pt \"Malgun Gothic\";\n"
"color: rgb(255, 255, 255);")
        self.uid.setText("")
        self.uid.setObjectName("uid")
        self.uregister = QtWidgets.QPushButton(Dialog)
        self.uregister.setGeometry(QtCore.QRect(400, 330, 75, 31))
        self.uregister.setStyleSheet("background-color: rgb(244, 244, 244);")
        self.uregister.setObjectName("uregister")
        ######################3
        self.uregister.clicked.connect(self.userregister)
        #################
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(170, 160, 171, 31))
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pwd.setPlaceholderText(_translate("Dialog", "Enter Password"))
        self.login.setText(_translate("Dialog", "Login"))
        self.uid.setPlaceholderText(_translate("Dialog", "Enter Email Id"))
        self.uregister.setText(_translate("Dialog", "Register"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600; color:#eaeaea;\">User Page</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = UserLogin()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

