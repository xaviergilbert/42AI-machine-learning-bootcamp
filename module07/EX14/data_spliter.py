import sys,os
sys.path.append(os.path.realpath('../'))
from EX07.mylinearregression import MyLinearRegression as MyLR
from EX10.polynomial_model import add_polynomial_features
from EX13.data_spliter import data_spliter

import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = np.array(data[['Micrograms']])
    y = np.array(data[['Score']])

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8, seed=1)
    print("x_train : ", x_train)
    print("x_test : ", x_test)
    print("y_train : ", y_train)
    print("y_test : ", y_test)
    print()

    model = {}
    cost = {}
    for power in range(2, 11):
        x_train_poly = add_polynomial_features(x_train, power)
        x_test_poly = add_polynomial_features(x_test, power)
        model[power] = MyLR(np.ones(x_train_poly.shape[1] + 1))
        model[power].fit_(x_train_poly, y_train, alpha=5e-8, n_cycle=500000)
        cost[power] = model[power].cost_(x_test_poly, y_test)

    print(cost)