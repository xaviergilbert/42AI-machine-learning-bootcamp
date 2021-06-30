import sys,os

from numpy.core.fromnumeric import size
sys.path.append(os.path.realpath('../..'))
from module07.EX07.mylinearregression import MyLinearRegression as MyLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mylinearregression import MyLinearRegression as MyLR

if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[['Age','Thrust_power','Terameters']])
    Y = np.array(data[['Sell_price']])
    my_lreg = MyLR([1.0, 1.0, 1.0, 1.0])

    print("MSE : ", my_lreg.mse_(X,Y))
    # Output:
    print("Output expected : ", 144044.877)

    my_lreg.fit_(X,Y, alpha = 5e-5, n_cycle = 600000)
    print("thetas : ", my_lreg.thetas)
    # Output:
    print("Output expected : ", np.array([[334.994],[-22.535],[5.857],[-2.586]]))

    print("MSE : ", my_lreg.mse_(X,Y))
    # Output:
    print("Output expected : ", 586.896999)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.scatter(np.array(data['Age']), np.array(data[['Sell_price']]), c="blue")
    plt.scatter(np.array(data['Age']), my_lreg.predict_(X), c="red", s=10)
    plt.xlabel("age")
    plt.ylabel("price")

    plt.subplot(2, 2, 2)
    plt.scatter(np.array(data['Thrust_power']), np.array(data[['Sell_price']]), c="green")
    plt.scatter(np.array(data['Thrust_power']), my_lreg.predict_(X), c="red", s=10)
    plt.xlabel("Thrust_power")
    plt.ylabel("price")

    plt.subplot(2, 2, 3)
    plt.scatter(np.array(data['Terameters']), np.array(data[['Sell_price']]), c="purple")
    plt.scatter(np.array(data['Terameters']), my_lreg.predict_(X), c="red", s=10)
    plt.xlabel("Terameters")
    plt.ylabel("price")

    plt.show()