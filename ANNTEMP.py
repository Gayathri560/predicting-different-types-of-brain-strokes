import sys
from DBConnection import DBConnection
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import sys
import time
from sklearn import metrics
import pickle


try:
    print("start")
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
    # ANN

    # Train the model

    y_train = np.array(y_train)


    ann = MLPClassifier()

    ann.fit(trainset, y_train)

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(ann, open(filename, 'wb'))

except Exception as e:
    print("Err=" + e.args[0])