import sys,os
sys.path.append(os.path.realpath('../'))
from EX07.mylinearregression import MyLinearRegression as MyLR
from EX10.polynomial_model import add_polynomial_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    X = np.array(data[['Micrograms']])
    Y = np.array(data[['Score']])
    my_lreg = {}
    cost = []
    for power in range(2, 11):
        new_x = add_polynomial_features(X, power)
        # print(X)
        # print(new_x)
        my_lreg[power] = MyLR(np.zeros(new_x.shape[1] + 1))
        my_lreg[power].fit_(new_x, Y)
        cost.append(my_lreg[power].cost_(new_x, Y))
        print("cost : ", cost)    

    plt.figure()
    plt.bar(range(2, 11), cost, bottom=0)
    plt.show()