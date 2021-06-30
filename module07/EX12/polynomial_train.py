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
    continuous_x = np.arange(1, 7, 0.1).reshape(-1, 1)
    my_lreg = {}
    cost = []
    plt.figure()
    for power in range(2, 6):
        new_x = add_polynomial_features(X, power)
        my_lreg[power] = MyLR(np.ones(new_x.shape[1] + 1))
        my_lreg[power].fit_(new_x, Y)
        cost.append(my_lreg[power].cost_(new_x, Y))
        print("cost : ", cost)    
        new_continuous_x = add_polynomial_features(continuous_x, power)
        plt.subplot(2, 2, power-1)
        plt.scatter(X, Y)
        plt.plot(continuous_x, my_lreg[power].predict_(new_continuous_x))

    plt.show()