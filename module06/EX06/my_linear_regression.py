import numpy as np
from time import sleep

class MyLinearRegression():
    def __init__(self,  thetas, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.thetas = np.array(thetas)
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
