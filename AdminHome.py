# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\adminhome.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from DBConnection import DBConnection
from adminloginaction import AdminLoginCheck
import xlrd
from SVM import SVM
from ANN import ANN
from Bar import Bar


class AdminHome(object):
    def svmcall(self):
        try:
            self.su = QtWidgets.QDialog()
            self.sui = SVM()
            self.sui.setupUi(self.su)
            self.su.show()


        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)


    def graph(self):
        try:
            Bar.main()

        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)

    def anncall(self):
        try:
            self.eu = QtWidgets.QDialog()
            self.eui = ANN()
            self.eui.setupUi(self.eu)
            self.eu.show()


        except Exception as e:
            print(e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)

    def browsefile(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", "E://", "*.xlsx")

        print(fileName)
        self.FileText.setText(fileName)


    def uploadAction(self):
        try:
            fname = self.FileText.text()
            book = xlrd.open_workbook(fname)
            sheet = book.sheet_by_index(0)
            database = DBConnection.getConnection()
            cursor = database.cursor()
            cursor.execute("delete from dataset")
            database.commit()
            query = "insert into dataset values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            for r in range(1, sheet.nrows):
                sno = sheet.cell(r, 0).value
                age = sheet.cell(r, 1).value
                gender = sheet.cell(r, 2).value
                smoking = sheet.cell(r, 3).value
                heart_rate = sheet.cell(r, 4).value
                chest_pain = sheet.cell(r, 5).value
                cholesterol = sheet.cell(r, 6).value
                bloodpressure = sheet.cell(r, 7).value
                bloodsugar = sheet.cell(r, 8).value
                disease = sheet.cell(r, 9).value
                print(sno, age, gender)

                values = (
                    sno, age, gender, smoking, heart_rate, chest_pain, cholesterol, bloodpressure, bloodsugar, disease)
                print(values)
                cursor.execute(query, values)

                database.commit()
                columns = str(sheet.ncols)
                # rows=str(sheet.nrows)
                print("inserted")
            self.showAlertBox("Information", "DataSet Loaded Successfully")
            self.FileText.setText("")
        except Exception as e:
            print("Error=" + e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(799, 498)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(-10, 0, 811, 501))
        font = QtGui.QFont()
        font.setFamily("Segoe Script")
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tabWidget.setMouseTracking(True)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.tabWidget.setObjectName("tabWidget")
        self.Home = QtWidgets.QWidget()
        self.Home.setObjectName("Home")
        self.label_2 = QtWidgets.QLabel(self.Home)
        self.label_2.setGeometry(QtCore.QRect(40, 20, 731, 51))
        self.label_2.setStyleSheet("\n"
                                   "font: 75 14pt \"Microsoft YaHei UI Light\";\n"
                                   "\n"
                                   "color: rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.label_6 = QtWidgets.QLabel(self.Home)
        self.label_6.setGeometry(QtCore.QRect(70, 140, 241, 40))
        self.label_6.setStyleSheet("font: 75 12pt \"Microsoft Tai Le\";\n"
                                   "color: rgb(255, 255, 255);")
        self.label_6.setObjectName("label_6")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/C:/Users/sajid/Desktop/admin-logo.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.tabWidget.addTab(self.Home, icon, "")
        self.UploadDataset = QtWidgets.QWidget()
        self.UploadDataset.setObjectName("UploadDataset")

        self.FileText = QtWidgets.QLineEdit(self.UploadDataset)
        self.FileText.setGeometry(QtCore.QRect(330, 230, 311, 30))
        self.FileText.setObjectName("FileText")

        self.FileButton = QtWidgets.QPushButton(self.UploadDataset)
        self.FileButton.setGeometry(QtCore.QRect(680, 230, 90, 30))
        self.FileButton.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.FileButton.setObjectName("FileButton")
        #############################333333
        self.FileButton.clicked.connect(self.browsefile)
        #######################3
        self.label = QtWidgets.QLabel(self.UploadDataset)
        self.label.setGeometry(QtCore.QRect(250, 190, 241, 31))
        self.label.setObjectName("label")
        self.FileUpload = QtWidgets.QPushButton(self.UploadDataset)
        self.FileUpload.setGeometry(QtCore.QRect(330, 270, 110, 30))
        self.FileUpload.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.FileUpload.setObjectName("FileUpload")
        ################################33

        self.FileUpload.clicked.connect(self.uploadAction)
        ####################################33
        self.label_3 = QtWidgets.QLabel(self.UploadDataset)
        self.label_3.setGeometry(QtCore.QRect(40, 20, 731, 51))
        self.label_3.setStyleSheet("\n"
                                   "font: 75 14pt \"Microsoft YaHei UI Light\";\n"
                                   "\n"
                                   "color: rgb(255, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.tabWidget.addTab(self.UploadDataset, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.ann = QtWidgets.QPushButton(self.tab)
        self.ann.setGeometry(QtCore.QRect(320, 150, 160, 40))
        self.ann.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                               "font: 12pt \"Century Gothic\";")
        self.ann.setObjectName("ann")
        ######################3
        self.ann.clicked.connect(self.anncall)
        #################
        self.svm = QtWidgets.QPushButton(self.tab)
        self.svm.setGeometry(QtCore.QRect(320, 220, 160, 40))
        self.svm.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                               "font: 12pt \"Century Gothic\";")
        self.svm.setObjectName("svm")
        ######################3
        self.svm.clicked.connect(self.svmcall)
        #################
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(40, 20, 731, 51))
        self.label_4.setStyleSheet("\n"
                                   "font: 75 14pt \"Microsoft YaHei UI Light\";\n"
                                   "\n"
                                   "color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.performance = QtWidgets.QPushButton(self.tab_2)
        self.performance.setGeometry(QtCore.QRect(320, 160, 160, 40))
        self.performance.setStyleSheet("color: rgb(0, 0, 255);\n"
                                       "background-color: rgb(255, 255, 255);\n"
                                       "font: 12pt \"Microsoft YaHei UI\";")
        self.performance.setObjectName("performance")

        ######################3
        self.performance.clicked.connect(self.graph)
        #################

        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(40, 20, 731, 51))
        self.label_5.setStyleSheet("\n"
                                   "font: 75 14pt \"Microsoft YaHei UI Light\";\n"
                                   "\n"
                                   "color: rgb(255, 255, 255);")
        self.label_5.setObjectName("label_5")
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.tabWidget.setStatusTip(_translate("Dialog", "Admin Home page"))
        self.tabWidget.setWhatsThis(_translate("Dialog", "Admin Home"))
        self.label_2.setText(_translate("Dialog",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\"> Predicting Different Types of Brain Strokes</span></p></body></html>"))
        self.label_6.setText(_translate("Dialog", "Welcome Admin.."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Home), _translate("Dialog", "Home"))
        self.FileButton.setText(_translate("Dialog", "Browse"))
        self.label.setText(_translate("Dialog",
                                      "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                      "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                      "p, li { white-space: pre-wrap; }\n"
                                      "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                      "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; color:#f5f5f5;\">DataSet Upload</span></p></body></html>"))
        self.FileUpload.setText(_translate("Dialog", "Upload"))
        self.label_3.setText(_translate("Dialog",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Predicting Different Types of Brain Strokes</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.UploadDataset), _translate("Dialog", "Upload Dataset"))
        self.ann.setText(_translate("Dialog", "ANN"))
        self.svm.setText(_translate("Dialog", "SVM"))
        self.label_4.setText(_translate("Dialog",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Predicting Different Types of Brain Strokes</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Algorithms"))
        self.performance.setText(_translate("Dialog", "View Performance"))
        self.label_5.setText(_translate("Dialog",
                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Predicting Different Types of Brain Strokes</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "Performance"))


    def showAlertBox(self, title, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle(title)
        msgBox.setText(message)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = AdminHome()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())




