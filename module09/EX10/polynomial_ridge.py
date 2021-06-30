# from EX00.my_logistic_regression import MyLogisticRegression
from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features
from my_linear_regression import MyLinearRegression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv('spacecraft_data.csv')
    x = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    y = np.array(data[['Sell_price']])

    # PART 1 -- Data Splitting
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    # PART 2 -- Training
    # x_train = add_polynomial_features(x_train, 3)
    # x_test = add_polynomial_features(x_test, 3)
    # print(x_train)
    model = {}
    mse = {}
    thetas = np.ones(4)
    for lambda_ in np.arange(0.1, 1, 0.1):
        lambda_ = round(lambda_, 1)
        # print(lambda_)
        # print(thetas)
        model[lambda_] = MyLinearRegression(thetas)
        model[lambda_].fit_(x_train, y_train, alpha=5e-5, n_cycle=500000, lambda_=lambda_)
        # break
        print("Cost")
        mse[lambda_] = model[lambda_].cost_(x_test, y_test, lambda_)
        print("MSE " + str(lambda_) + " : ", mse[lambda_])
        # break
    # print(x.shape)

    plt.figure()
    plt.title("Cost value of models in regard to lambda (log)")
    plt.bar(list(mse.keys()), list(mse.values()), width=0.05, log=True)
    plt.show()

    for key in model.keys():
        
        plt.figure()
        plt.suptitle("Model relurarize with lambda = " + str(key))
        for i, feature in enumerate(['Age', 'Thrust_power', 'Terameters']):
            # print(i)

            plt.subplot(2, 2, i + 1)
            plt.title("Spacecraft price in regard to " + feature)
            plt.scatter(x_test[:, i], y_test)
            plt.plot(x_test[:, i], model[key].predict_(x_test))
            plt.xlabel(feature)
            plt.ylabel("price")
        
        plt.show()