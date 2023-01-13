# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\adminlogin.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from adminloginaction import AdminLoginCheck
from AdminHome import AdminHome



class AdminLogin(object):

    def adminlogin(self):
        try:
            auidvar = self.auid.text()
            apwdvar = self.apwd.text()
            self.auid.setText("")
            self.apwd.setText("")
            al = AdminLoginCheck()
            res = al.datacheck(auidvar, apwdvar)
            if res:
                self.showAlertBox("Alert", "Fill the Fields")
            elif AdminLoginCheck.logincheck(auidvar, apwdvar):
                self.u = QtWidgets.QDialog()
                self.ui = AdminHome()
                self.ui.setupUi(self.u)
                self.u.show()


            else:
                self.showAlertBox("Login Alert", "Login Fail")

        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)

    #
    ##Alert Winwow
    #

    def showAlertBox(self, title, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle(title)
        msgBox.setText(message)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(637, 519)
        Dialog.setStyleSheet("image: url(../images/the-heart-stroke.jpeg);")
        self.auid = QtWidgets.QLineEdit(Dialog)
        self.auid.setGeometry(QtCore.QRect(400, 290, 211, 21))
        self.auid.setObjectName("auid")
        self.apwd = QtWidgets.QLineEdit(Dialog)
        self.apwd.setGeometry(QtCore.QRect(400, 350, 211, 21))
        self.apwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.apwd.setObjectName("apwd")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(350, 270, 151, 16))
        self.label.setStyleSheet("selection-color: rgb(255, 255, 127);\n"
"font: 14pt \"MS Shell Dlg 2\";\n"
"color: rgb(35, 93, 144);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(350, 330, 151, 16))
        self.label_2.setStyleSheet("selection-color: rgb(255, 255, 127);\n"
"font: 14pt \"MS Shell Dlg 2\";\n"
"color: rgb(35, 93, 144);")
        self.label_2.setObjectName("label_2")
        self.login = QtWidgets.QPushButton(Dialog)
        self.login.setGeometry(QtCore.QRect(400, 400, 75, 23))
        self.login.setObjectName("login")


        ######################3
        self.login.clicked.connect(self.adminlogin)
        #################

        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(80, 230, 141, 31))
        self.label_3.setStyleSheet("font: 75 16pt \"MS Shell Dlg 2\";\n"
"color: rgb(241, 241, 241);")
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.auid.setText(_translate("Dialog", "admin"))
        self.label.setText(_translate("Dialog", "Enter Username"))
        self.label_2.setText(_translate("Dialog", "Enter Password"))
        self.login.setText(_translate("Dialog", "Login"))
        self.label_3.setText(_translate("Dialog", "Admin Login"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = AdminLogin()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

