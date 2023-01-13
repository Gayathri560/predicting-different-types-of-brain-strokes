import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from DBConnection import DBConnection

class Bar:
    @staticmethod
    def main():

        objects = ('SVM', 'ANN')
        y_pos = np.arange(len(objects))
        database = DBConnection.getConnection()
        cursor = database.cursor()
        cursor.execute("select * from eval")
        row = cursor.fetchall()
        performance=[]
        for r in row:
            performance.append(r[0])
            performance.append(r[1])


        print(performance)
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('%')
        plt.title('Performance of ANN and SVM')

        plt.show()

