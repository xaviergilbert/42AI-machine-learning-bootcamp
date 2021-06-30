import sys,os
sys.path.append(os.path.realpath('../'))
from EX00.my_logistic_regression import MyLogisticRegression
import numpy as np

class MyRidge(MyLogisticRegression):
    """
        Description:
            My personnal ridge regression class to fit like a boss.
    """
    def __init__(self,  thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
        self.set_params_(thetas, alpha, n_cycle, lambda_)

    def get_params_(self):
        return {'alpha': self.alpha, 'n_cycle': self.max_iter, 'thetas': self.thetas, 'lambda': self.lambda_}

    def set_params_(self, thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.thetas = self.thetas.flatten()
        self.lambda_ = lambda_

    def fit_(self, x, y):

        y = y.flatten()
        # X = self.add_intercept(x)
        for epoch in self.max_iter:
            # predict = self.sigmoid(np.sum(X * self.thetas, axis = 1))
            predict = self.predict_(x)
            
            regularisation = np.append(np.array([0]), self.thetas[1:]) * self.lambda_
            gradient = (np.sum(self.X.T * (predict - y), axis = 1) + regularisation) / len(y)
            self.thetas = self.thetas - (self.alpha * gradient)