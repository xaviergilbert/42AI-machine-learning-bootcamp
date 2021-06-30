# import sys,os
# sys.path.append(os.path.realpath('../..'))
# from module06.EX06.my_linear_regression import MyLinearRegression as MyLR
import numpy as np
from matplotlib import pyplot as plt
# from mylinearregression import MyLinearRegression as MyLR

class MyLinearRegression():
    def __init__(self,  theta):
        self.thetas = np.array(theta)
        self.x_intercept = None


    def fit_(self, x, y, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        for epoch in range(self.max_iter):
            gradient = self.gradient_(x, y)
            self.thetas = self.thetas - (self.alpha * gradient)
            # sleep(1)

    def gradient_(self, x, y):
        y_pred = self.predict_(x)
        y = np.reshape(y, y_pred.shape)
        theta_ = (y_pred - y) * self.x_intercept.T / y.shape[0]
        return np.sum(theta_, axis=1)

    def predict_(self, x):
        if self.thetas.ndim == 2:
            self.thetas = self.thetas.flatten()
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
        return np.append(b, x, axis=1)


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    lr1 = MyLinearRegression([2, 0.7])
    # Example 0.0:
    print("predict : ", lr1.predict_(x))
    # Output:
    print("output wanted : ", np.array([[10.74695094],
        [17.05055804],
        [24.08691674],
        [36.24020866],
        [42.25621131]]))
    print()

    # Example 0.1:
    print("cost elem : ", lr1.cost_elem_(lr1.predict_(x),y) / 2)
    # Output:
    print("output wanted : ", np.array([[77.72116511],
        [49.33699664],
        [72.38621816],
        [37.29223426],
        [78.28360514]]))
    print()

    # Example 0.2:
    print("cost : ", lr1.cost_(lr1.predict_(x),y) / 2)
    # Output:
    print("output wanted : ", 315.0202193084312)
    print()

    # Example 1.0:
    lr2 = MyLinearRegression([0, 0])
    lr2.fit_(x, y, alpha=5e-8, n_cycle = 1500000)
    print("thetas : ", lr2.thetas)
    # Output:
    print("output wanted : ", np.array([[1.40709365],
        [1.1150909 ]]))


    # Example 1.1:
    print("prediction : ", lr2.predict_(x))
    # Output:
    print("output wanted : ", np.array([[15.3408728 ],
        [25.38243697],
        [36.59126492],
        [55.95130097],
        [65.53471499]]))
    print()

    # Example 1.2:
    print("cost elem : ", lr2.cost_elem_(lr2.predict_(x),y) / 2)
    # Output:
    print("output wanted : ", np.array([[35.6749755 ],
        [ 4.14286023],
        [ 1.26440585],
        [29.30443042],
        [22.27765992]]))
    print()

    # Example 1.3:
    print("cost : ", lr2.cost_(lr2.predict_(x),y) / 2)
    # Output:
    print("output wanted : ", 92.66433192085971)
    print()

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, x * 1.1150909 + 1.40709365, label="Expected")
    plt.plot(x, lr2.predict_(x), c="green", label="My prediction")
    plt.legend()
    plt.show()