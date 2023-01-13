# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\Heart_Stroke\ann.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from DBConnection import DBConnection
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import sys
import time
from sklearn import metrics
import pickle

class ANN(object):

    def trainbrowsedef(self):
        try:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", "E://", "*.csv")
            print(fileName)
            self.lineEdit.setText(fileName)
        except Exception as e:
            print("Error=" + e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)
            print(e)


    def training(self):
        try:
            fname = self.lineEdit.text()
            print(fname)
            trainset = []
            # testset=[float('0'), float('1'), float('1'),float('1'),float('0'),float('0'),float('0'),float('1')]
            # testset=
            database = DBConnection.getConnection()
            cursor = database.cursor()
            cursor.execute(
                "select * from dataset")
            row = cursor.fetchall()
            y_train = []
            trainset.clear()
            y_train.clear()
            train=len(row)
            for r in row:

                x_train = []
                x_train.clear()
                x_train.append(float(r[1]))
                x_train.append(float(r[2]))
                x_train.append(float(r[3]))
                x_train.append(float(r[4]))
                x_train.append(float(r[5]))
                x_train.append(float(r[6]))
                x_train.append(float(r[7]))
                x_train.append(float(r[8]))
                y_train.append(r[9])
                trainset.append(x_train)
            print("y=", y_train)
            # print("trd=", trainset)
            trainset = np.array(trainset)
            print("trd=", trainset)
            # ANN

            # Train the model

            y_train = np.array(y_train)

            tf = pd.read_csv(fname)
            print("----------", type(tf))
           

            res = tf['Result']
            R1 = np.array(res)

            # print(res)

            tf = tf.drop(['Result'], 1)
            testdata = np.array(tf)
            print("td=", testdata)
            #testdata = testdata.reshape(len(testdata), -1)

            #ann = MLPClassifier()

            #ann.fit(trainset, y_train)
            filename = 'finalized_model.sav'

            ann = pickle.load(open(filename, 'rb'))
            result = ann.predict(testdata)

            s = time.clock()
            #result = ann.predict(testdata)  # Predicting
            e = time.clock()
            t = round(e - s, 5)
            print("ann:", t, "s")
            # print("pre=", result)
            # print(round(metrics.accuracy_score(y_train, result),3)*100)
            cursor = database.cursor()
            cursor.execute("delete from results")
            cursor = database.cursor()

            sql = "insert into results(predicate, actual) values(%s,%s)"

            x = len(result)
            print("-----------", x)
            i = 0
            for i in range(x):
                values = ( str(result[i]),str(R1[i]))
                print(values)
                cursor.execute(sql, values)
                database.commit()
                i = i + 1

            cursor.execute("update results set result='True' where predicate=actual")
            database.commit()
            cursor.execute("select count(*) from results")
            x = cursor.fetchall()
            for x1 in x:
                tot=x1[0]
            self.tot_lable.setText(str(tot) + " Records")
            cursor.execute("select count(*) from results where result='True' ")
            x = cursor.fetchall()
            for x1 in x:
                correct=x1[0]
            self.correct_label.setText(str(correct) + " Records")

            false=tot-correct
            self.false_label.setText(str(false)+ " Records")

            acc=(correct/tot)*100
            acc=round(acc)
            self.accuracy_label.setText(str(acc) + "%")
            self.train_lable.setText(str(train) + " Records")

            cursor.execute("update eval set ann=" + str(acc) + " ")

            database.commit()

            database.commit()

            #return result


        except Exception as e:
            print("Error=" + e.args[0])
            tb = sys.exc_info()[2]
            print(tb.tb_lineno)


    def setupUi(self, Dialog):
            Dialog.setObjectName("Dialog")
            Dialog.resize(489, 535)
            Dialog.setStyleSheet("background-color: rgb(222, 0, 0);")
            self.label = QtWidgets.QLabel(Dialog)
            self.label.setGeometry(QtCore.QRect(30, 10, 230, 51))
            self.label.setStyleSheet("font: 100 20pt \"Agency FB\";\n"
    "color: rgb(255, 255, 255);\n"
    "")
            self.label.setObjectName("label")
            self.lineEdit = QtWidgets.QLineEdit(Dialog)
            self.lineEdit.setGeometry(QtCore.QRect(20, 140, 350, 30))
            self.lineEdit.setObjectName("lineEdit")
            self.trainbrowse = QtWidgets.QPushButton(Dialog)
            self.trainbrowse.setGeometry(QtCore.QRect(380, 140, 80, 30))
            self.trainbrowse.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.trainbrowse.setObjectName("trainbrowse")
            ####################
            self.trainbrowse.clicked.connect(self.trainbrowsedef)
            ####################
            self.trainupload = QtWidgets.QPushButton(Dialog)
            self.trainupload.setGeometry(QtCore.QRect(20, 180, 80, 30))
            self.trainupload.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.trainupload.setObjectName("trainupload")
            ####################
            self.trainupload.clicked.connect(self.training)
            ####################

            self.label_2 = QtWidgets.QLabel(Dialog)
            self.label_2.setGeometry(QtCore.QRect(20, 100, 271, 41))
            self.label_2.setStyleSheet("font: 14pt \"Bahnschrift SemiLight\";")
            self.label_2.setObjectName("label_2")
            self.label_3 = QtWidgets.QLabel(Dialog)
            self.label_3.setGeometry(QtCore.QRect(10, 60, 460, 16))
            self.label_3.setStyleSheet("color: rgb(255, 85, 0);\n"
    "color: rgb(255, 255, 255);")
            self.label_3.setObjectName("label_3")
            self.label_4 = QtWidgets.QLabel(Dialog)
            self.label_4.setGeometry(QtCore.QRect(50, 280, 91, 16))
            self.label_4.setText("")
            self.label_4.setObjectName("label_4")
            self.label_7 = QtWidgets.QLabel(Dialog)
            self.label_7.setGeometry(QtCore.QRect(30, 290, 261, 31))
            self.label_7.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.label_7.setObjectName("label_7")
            self.tot_lable = QtWidgets.QLabel(Dialog)
            self.tot_lable.setGeometry(QtCore.QRect(220, 290, 81, 30))
            self.tot_lable.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.tot_lable.setObjectName("tot_lable")
            self.label_6 = QtWidgets.QLabel(Dialog)
            self.label_6.setGeometry(QtCore.QRect(50, 330, 91, 16))
            self.label_6.setText("")
            self.label_6.setObjectName("label_6")
            self.correct_label = QtWidgets.QLabel(Dialog)
            self.correct_label.setGeometry(QtCore.QRect(220, 330, 81, 30))
            self.correct_label.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.correct_label.setObjectName("correct_label")
            self.label_9 = QtWidgets.QLabel(Dialog)
            self.label_9.setGeometry(QtCore.QRect(30, 330, 181, 31))
            self.label_9.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.label_9.setObjectName("label_9")
            self.label_10 = QtWidgets.QLabel(Dialog)
            self.label_10.setGeometry(QtCore.QRect(30, 370, 181, 31))
            self.label_10.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.label_10.setObjectName("label_10")
            self.false_label = QtWidgets.QLabel(Dialog)
            self.false_label.setGeometry(QtCore.QRect(220, 370, 81, 30))
            self.false_label.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.false_label.setObjectName("false_label")
            self.label_12 = QtWidgets.QLabel(Dialog)
            self.label_12.setGeometry(QtCore.QRect(30, 410, 181, 31))
            self.label_12.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.label_12.setObjectName("label_12")
            self.accuracy_label = QtWidgets.QLabel(Dialog)
            self.accuracy_label.setGeometry(QtCore.QRect(220, 410, 30, 30))
            self.accuracy_label.setStyleSheet("font: 75 14pt \"Agency FB\";\n"
    "")
            self.accuracy_label.setObjectName("accuracy_label")
            self.label_8 = QtWidgets.QLabel(Dialog)
            self.label_8.setGeometry(QtCore.QRect(30, 250, 131, 31))
            self.label_8.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.label_8.setObjectName("label_8")
            self.train_lable = QtWidgets.QLabel(Dialog)
            self.train_lable.setGeometry(QtCore.QRect(220, 250, 81, 30))
            self.train_lable.setStyleSheet("font: 75 14pt \"Agency FB\";")
            self.train_lable.setObjectName("train_lable")

            self.retranslateUi(Dialog)
            QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "ANN"))
        self.trainbrowse.setText(_translate("Dialog", "Browse"))
        self.trainupload.setText(_translate("Dialog", "Upload"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#fcfcfc;\">Upload Test Dataset</span></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "-----------------------------------------------------------------------------------------------------------------"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p>Total No. of Testdata:</p></body></html>"))
        self.tot_lable.setText(_translate("Dialog", "NoN"))
        self.correct_label.setText(_translate("Dialog", "NoN"))
        self.label_9.setText(_translate("Dialog", "<html><head/><body><p>Total No. of Correct Predicates:</p></body></html>"))
        self.label_10.setText(_translate("Dialog", "<html><head/><body><p>Total No. of False Predicates:</p></body></html>"))
        self.false_label.setText(_translate("Dialog", "NoN"))
        self.label_12.setText(_translate("Dialog", "<html><head/><body><p>Accuracy of Testdata:</p></body></html>"))
        self.accuracy_label.setText(_translate("Dialog", "NoN"))
        self.label_8.setText(_translate("Dialog", "<html><head/><body><p>Total No. of Traindata</p></body></html>"))
        self.train_lable.setText(_translate("Dialog", "NoN"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = ANN()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

