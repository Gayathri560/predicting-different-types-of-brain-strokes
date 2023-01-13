# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\userhome.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from Prediction import Prediction
from DBConnection import DBConnection
from PyQt5 import QtCore, QtGui, QtWidgets



class UserHome(object):
    def viewres(self):
        Bar.generate(self.email)

    def tprediction(self):
        try:
            #print(self.email)
            v0 = self.f0.text()
            v1 = self.f1.text()
            v2 = self.f2.text()
            v3 = self.f3.text()
            v4 = self.f4.text()
            v5 = self.f5.text()
            v6 = self.f6.text()
            v7 = self.f7.text()
            if True:
                row1 = ["f0","f1", "f2", "f3", "f4", "f5", "f6", "f7"]

                row = [v0, v1, v2, v3, v4, v5, v6, v7]

                print(row)

                import csv

                with open('test.csv', 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row1)
                    writer.writerow(row)

                csvFile.close()

                res = Prediction.predict_nv()
                self.showAlertBox("Alert", "Result is " + str(res))

                import datetime
                now = datetime.datetime.now()
                date = now.strftime("%Y-%m-%d")
                print(date)
                database = DBConnection.getConnection()
                cursor = database.cursor()
                query = "insert into uresults values(%s,%s,%s)"
                values = (self.email, date, res)
                print(values)
                cursor.execute(query, values)
                database.commit()

            else:
                self.showAlertBox("Alert", "Fill the Fields Properly")

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
        Dialog.resize(785, 606)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 810, 611))
        self.tabWidget.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.tabWidget.setStyleSheet("background-color: rgb(208, 0, 0);")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(30, 10, 731, 51))
        self.label_4.setStyleSheet("\n"
"font: 75 14pt \"Microsoft YaHei UI Light\";\n"
"\n"
"color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.f0 = QtWidgets.QLineEdit(self.tab_2)
        self.f0.setGeometry(QtCore.QRect(340, 160, 190, 30))
        self.f0.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f0.setText("")
        self.f0.setObjectName("f0")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(90, 160, 170, 30))
        self.label.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";\n"
"color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.f1 = QtWidgets.QLineEdit(self.tab_2)
        self.f1.setGeometry(QtCore.QRect(340, 200, 190, 30))
        self.f1.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f1.setText("")
        self.f1.setObjectName("f1")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(90, 200, 171, 30))
        self.label_2.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.f2 = QtWidgets.QLineEdit(self.tab_2)
        self.f2.setGeometry(QtCore.QRect(340, 240, 190, 30))
        self.f2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f2.setText("")
        self.f2.setObjectName("f2")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setGeometry(QtCore.QRect(90, 240, 171, 30))
        self.label_3.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.f3 = QtWidgets.QLineEdit(self.tab_2)
        self.f3.setGeometry(QtCore.QRect(340, 280, 190, 30))
        self.f3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f3.setText("")
        self.f3.setObjectName("f3")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(90, 280, 241, 30))
        self.label_5.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);font: 14pt \"Microsoft YaHei UI\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(90, 360, 241, 30))
        self.label_6.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);font: 14pt \"Microsoft YaHei UI\";")
        self.label_6.setObjectName("label_6")
        self.f4 = QtWidgets.QLineEdit(self.tab_2)
        self.f4.setGeometry(QtCore.QRect(339, 320, 190, 30))
        self.f4.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f4.setText("")
        self.f4.setObjectName("f4")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(50, 105, 341, 31))
        self.label_7.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);")
        self.label_7.setObjectName("label_7")
        self.prediction = QtWidgets.QPushButton(self.tab_2)
        self.prediction.setGeometry(QtCore.QRect(340, 490, 110, 30))
        self.prediction.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.prediction.setStyleSheet("\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.prediction.setObjectName("prediction")
        ######################3
        self.prediction.clicked.connect(self.tprediction)
        #################


        self.f5 = QtWidgets.QLineEdit(self.tab_2)
        self.f5.setGeometry(QtCore.QRect(340, 360, 190, 30))
        self.f5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f5.setText("")
        self.f5.setObjectName("f5")
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        self.label_9.setGeometry(QtCore.QRect(90, 400, 241, 30))
        self.label_9.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);font: 14pt \"Microsoft YaHei UI\";")
        self.label_9.setObjectName("label_9")
        self.f6 = QtWidgets.QLineEdit(self.tab_2)
        self.f6.setGeometry(QtCore.QRect(340, 400, 190, 30))
        self.f6.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f6.setText("")
        self.f6.setObjectName("f6")
        self.label_10 = QtWidgets.QLabel(self.tab_2)
        self.label_10.setGeometry(QtCore.QRect(90, 440, 241, 30))
        self.label_10.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);font: 14pt \"Microsoft YaHei UI\";")
        self.label_10.setObjectName("label_10")
        self.f7 = QtWidgets.QLineEdit(self.tab_2)
        self.f7.setGeometry(QtCore.QRect(340, 440, 190, 30))
        self.f7.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-image: url(:/newPrefix/C:/Users/sajid/Desktop/white.png);")
        self.f7.setText("")
        self.f7.setObjectName("f7")
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(30, 10, 731, 51))
        self.label_11.setStyleSheet("\n"
"font: 75 14pt \"Microsoft YaHei UI Light\";\n"
"\n"
"color: rgb(255, 255, 255);")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(90, 320, 241, 30))
        self.label_12.setStyleSheet("font: 14pt \"Microsoft YaHei UI\";color: rgb(255, 255, 255);font: 14pt \"Microsoft YaHei UI\";")
        self.label_12.setObjectName("label_12")
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_4.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">An Artificial Intelligence Approach for Predicting Different Types of Stroke</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Home"))
        self.f0.setPlaceholderText(_translate("Dialog", "1 to 100"))
        self.label.setText(_translate("Dialog", "Age"))
        self.f1.setPlaceholderText(_translate("Dialog", "Male- 1 ; Female -0"))
        self.label_2.setText(_translate("Dialog", "Gender"))
        self.f2.setPlaceholderText(_translate("Dialog", "Yes-1, No-0"))
        self.label_3.setText(_translate("Dialog", "Smoking"))
        self.f3.setPlaceholderText(_translate("Dialog", "70 to 210"))
        self.label_5.setText(_translate("Dialog", "Heart Rate"))
        self.label_6.setText(_translate("Dialog", "Cholesterol"))
        self.f4.setPlaceholderText(_translate("Dialog", "0 to 9"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p><span style=\" color:#f5f5f5;\">Stroke Detection </span></p></body></html>"))
        self.prediction.setText(_translate("Dialog", "Predection"))
        self.f5.setPlaceholderText(_translate("Dialog", "100 to 400"))
        self.label_9.setText(_translate("Dialog", "Blood Pressure"))
        self.f6.setPlaceholderText(_translate("Dialog", "90 to 200"))
        self.label_10.setText(_translate("Dialog", "Blood Sugar"))
        self.f7.setPlaceholderText(_translate("Dialog", "80 to 300"))
        self.label_11.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">An Artificial Intelligence Approach for Predicting Different Types of Stroke</span></p></body></html>"))
        self.label_12.setText(_translate("Dialog", "Chest Pain"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "Prediction"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = UserHome()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

