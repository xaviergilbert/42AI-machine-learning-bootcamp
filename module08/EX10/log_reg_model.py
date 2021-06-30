import sys,os
sys.path.append(os.path.realpath('../'))
from EX01.data_spliter import data_spliter
from EX09.my_logistic_regression import MyLogisticRegression as MyLR

import numpy as np
import pandas as pd



if __name__ == "__main__":
    data = pd.read_csv('solar_system_census.csv')
    result = pd.read_csv('solar_system_census_planets.csv')
    x = np.array(data[['height', 'weight', 'bone_density']])
    y = np.array(result[['Origin']])

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8, seed=42)
    # print(x_train)
    print(data.describe())


    prediction = []
    for planet in range(len(np.unique(y))):
        y_train_dic = np.array([1 if value == planet else 0 for value in y_train])
        y_test_dic = np.array([1 if value == 0 else 0 for value in y_test])

    # print(venus_train)

        model = MyLR([np.ones(x.shape[1] + 1)])
        model.fit_(x_train, y_train_dic, alpha=0.001, n_cycle=100000)
        prediction.append(model.predict_(x_train))
        cost = model.cost_(x_test, y_test_dic)
        # print(model.theta)
        # print("predictions : ", prediction)
        print("cost : ", cost)
    
    prediction = np.array(prediction).T
    final_prediction = []
    for row in prediction:
        # print(row)
        final_prediction.append(np.argmax(row))
        # exit()

    print("\ny_hat / y")
    for final_predictioni, y_testi in zip(final_prediction, y_test):
        print(final_predictioni, y_testi)
    # print(final_prediction)