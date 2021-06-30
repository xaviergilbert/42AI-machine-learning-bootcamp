# import sys,os
# sys.path.append(os.path.realpath('../..'))
# from module06.EX06.my_linear_regression import MyLinearRegression as MyLR
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
# from mylinearregression import MyLinearRegression as MyLR

class MyLinearRegression():
    def __init__(self,  theta):
        self.thetas = np.array(theta)
        if self.thetas.ndim == 2:
            self.thetas = self.thetas.flatten()
        self.x_intercept = None


    def fit_(self, x, y, alpha=5e-5, n_cycle=1000000):
        self.alpha = alpha
        self.max_iter = n_cycle
        for epoch in range(self.max_iter):
            gradient = self.gradient_(x, y)
            self.thetas = self.thetas - (self.alpha * gradient)
            # print("self.thetas : ", self.thetas)
            # sleep(1)

    def gradient_(self, x, y):
        self.x_intercept = x
        if self.thetas.shape[0] < x.shape[1]:
            x_intercept = self.add_intercept(x)
        y_pred = self.predict_(x)
        y = np.reshape(y, y_pred.shape)
        theta_ = (y_pred - y) * self.x_intercept.T / y.shape[0]
        return np.sum(theta_, axis=1)

    def predict_(self, x):
        self.x_intercept = self.add_intercept(x)
        res = np.sum(self.x_intercept * self.thetas, axis = 1)
        return res

    def cost_elem_(self, x, y):
        y_pred = self.predict_(x)
        y = np.reshape(y, y_pred.shape)
        res = np.square(y - y_pred) / (y.shape[0])
        return res

    def cost_(self, x, y):
        return np.sum(self.cost_elem_(x, y), axis = 0)

    def add_intercept(self, x):
        if isinstance(x, np.ndarray) == False or x.size == 0:
            return None
        b = np.ones((x.shape[0],1), dtype=int)
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0], 1))
        res = np.append(b, x, axis=1)
        return res

    def mse_(self, x, y):
        return self.cost_(x, y)


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

    # Example 0:
    print("Predictions : ", mylr.predict_(X))
    # Output:
    print("Expected output : ", np.array([[8.], [48.], [323.]]))
    print()

    # Example 1:
    print("cost elem : ", mylr.cost_elem_(X,Y) / 2)
    # Output:
    print("Expected output : ", np.array([[37.5], [0.], [1837.5]]))
    print()

    # Example 2:
    print("cost : ", mylr.cost_(X,Y) / 2)
    # Output:
    print("Expected output : ", 1875.0)
    print()

    # Example 3:
    mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
    print("thetas : ", mylr.thetas)
    # Output:
    print("Expected output : ", np.array([[18.023], [3.323], [-0.711], [1.605], [-0.1113]]))
    print()

    # Example 4:
    print("prediction : ", mylr.predict_(X))
    # Output:
    print("Expected output : ", np.array([[23.499], [47.385], [218.079]]))
    print()

    # Example 5:
    print("cost elem : ", mylr.cost_elem_(X,Y) / 2)
    # Output:
    print("Expected output : ", np.array([[0.041], [0.062], [0.001]]))
    print()

    # Example 6:
    print("cost : ", mylr.cost_(X,Y) / 2)
    # Output:
    print("Expected output : ", 0.1056)
    print()
