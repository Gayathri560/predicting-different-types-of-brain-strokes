# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\userregister.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from registeraction import RegisterAction

class UserRegister(object):
    def signup(self):
        try:
            print("signup")
            namevar = self.name.text()
            emailvar = self.email.text()
            contactvar = self.contact.text()
            cityvar = self.city.text()
            pwdvar = self.pwd.text()
            al = RegisterAction()
            res = al.datacheck(namevar, emailvar, contactvar, cityvar, pwdvar)
            if res:
                self.showAlertBox("Alert", "Fill the Fields")
            elif RegisterAction.signup(namevar, emailvar, contactvar, cityvar, pwdvar):
                self.u = QtWidgets.QDialog()
                self.showAlertBox("Alert", "Done ")
                # self.ui = UserLogin()
                # self.ui.setupUi(self.u)
                # self.u.show()
                # Dialog.hide()

            else:
                self.showAlertBox("Login Alert", "Login Fail")

        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno);
            print(e);

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
        Dialog.resize(637, 481)
        Dialog.setStyleSheet("background-color: rgb(214, 0, 0);")
        self.pwd = QtWidgets.QLineEdit(Dialog)
        self.pwd.setGeometry(QtCore.QRect(260, 370, 211, 30))
        self.pwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pwd.setObjectName("pwd")
        self.name = QtWidgets.QLineEdit(Dialog)
        self.name.setGeometry(QtCore.QRect(260, 110, 211, 30))
        self.name.setText("")
        self.name.setObjectName("name")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(190, 340, 151, 31))
        self.label_2.setStyleSheet("selection-color: rgb(255, 255, 127);\n"
"font: 14pt \"MS Shell Dlg 2\";\n"
"color: rgb(35, 93, 144);")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(190, 70, 151, 31))
        self.label.setStyleSheet("selection-color: rgb(255, 255, 127);\n"
"color: rgb(255, 255, 255);\n"
"font: 14pt \"MS Shell Dlg 2\";\n"
"color: rgb(35, 93, 144);")
        self.label.setObjectName("label")
        self.uregister = QtWidgets.QPushButton(Dialog)
        self.uregister.setGeometry(QtCore.QRect(260, 420, 75, 31))
        self.uregister.setObjectName("uregister")
        ######################3
        self.uregister.clicked.connect(self.signup)
        #################
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(240, 10, 171, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(190, 140, 111, 31))
        self.label_4.setObjectName("label_4")
        self.email = QtWidgets.QLineEdit(Dialog)
        self.email.setGeometry(QtCore.QRect(259, 170, 211, 30))
        self.email.setObjectName("email")
        self.contact = QtWidgets.QLineEdit(Dialog)
        self.contact.setGeometry(QtCore.QRect(260, 240, 210, 30))
        self.contact.setText("")
        self.contact.setObjectName("contact")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(190, 210, 131, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(190, 270, 111, 31))
        self.label_6.setObjectName("label_6")
        self.city = QtWidgets.QLineEdit(Dialog)
        self.city.setGeometry(QtCore.QRect(260, 300, 210, 30))
        self.city.setObjectName("city")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#f1f1f1;\">Enter Password</span></p></body></html>"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#f1f1f1;\">Enter Name</span></p></body></html>"))
        self.uregister.setText(_translate("Dialog", "Register"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600; color:#eaeaea;\">User Registration</span><br/></p></body></html>"))
        self.label_4.setText(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600; color:#f8f8f8;\">Enter Email</span></p></body></html>"))
        self.email.setToolTip(_translate("Dialog", "<html><head/><body><p>contact</p></body></html>"))
        self.label_5.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#f4f4f4;\">Enter Contact</span></p></body></html>"))
        self.label_6.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#fefefe;\">Enter City</span></p></body></html>"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = UserRegister()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

