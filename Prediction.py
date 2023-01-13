# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\QT\phishing\niave.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from DBConnection import DBConnection
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

import numpy as np
import pandas as pd
import sys
import time
from sklearn import metrics


class Prediction:
    def predict_nv():
        try:
            fn = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
            # list=[f1,f2,f3,f4,f5,f6,f8,f9,f10,f15]

            trainset = []
            database = DBConnection.getConnection()
            cursor = database.cursor()
            cursor.execute(
                "select * from dataset")
            row = cursor.fetchall()
            y_train = []
            trainset.clear()
            y_train.clear()
            train = len(row)
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

            # Train the model

            y_train = np.array(y_train)

            print("sssss", pd.read_csv("test.csv"))
            tf = pd.read_csv("test.csv")
            print(tf, "<----------")

            testdata = np.array(tf)
            print("td=", testdata)
            testdata = testdata.reshape(len(testdata), -1)

            ann = MLPClassifier()

            ann.fit(trainset, y_train)
            s = time.clock()
            result = ann.predict(testdata)  # Predicting Heart Disease
            e = time.clock()
            t = round(e - s, 5)
            print("elm:", t, "s")
            print("pre=", result[0])
            return result[0]


        except Exception as e:
            print("Error=" + e)

if __name__ == '__main__':
    Prediction.predict_nv()