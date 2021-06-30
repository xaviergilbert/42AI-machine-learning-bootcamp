
import numpy as np

class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, n_cycle=100000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = np.array(theta)
        self.X = None
        # Your code here

    def fit_(self, x, y, alpha=0.001, n_cycle=100000):
        self.alpha = alpha
        self.max_iter = n_cycle
        # self.X = self.add_intercept(x)
        y = y.flatten()
        self.theta = self.theta.flatten()
        # y_hat = y_hat.flatten()
        for epoch in range(self.max_iter):
            predict = self.predict_(x)
            gradient_ = np.sum(self.X.T * (predict - y) / len(y), axis = 1)
            self.theta = self.theta - (self.alpha * gradient_)
            

    def predict_(self, x):
        self.X = self.add_intercept(x)
        return self.sigmoid_(self.X.dot(self.theta))

    def cost_(self, x, y, eps=1e-15):
        y_hat = self.predict_(x)
        ones = np.ones(len(y))
        y = y.flatten()
        y_hat = y_hat.flatten()
        premiere_partie = y * np.log(y_hat + eps)
        deuxieme_partie = (ones - y) * np.log(ones - y_hat + eps)
        total = np.sum((premiere_partie + deuxieme_partie) / -len(y))
        return total

    def add_intercept(self, x):
        b = np.ones((x.shape[0],1), dtype=int)
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0], 1))
        res = np.append(b, x, axis=1)
        return res

    def sigmoid_(self, x):
        if type(x) != np.ndarray or x.size == 0:
            return None
        return (1 / (1 + np.exp(-x)))


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])

    # Example 0:
    print("Predict : ", mylr.predict_(X))
    # Output:
    print("Output expected : ", np.array([[0.99930437],
        [1.        ],
        [1.        ]]), "\n")

    # Example 1:
    print("Cost : ", mylr.cost_(X,Y))
    # Output:
    print("Output expected : ", 11.513157421577004, "\n")

    # Example 2:
    mylr.fit_(X, Y)
    print("Theta : ", mylr.theta)
    # Output:
    print("Output expected : ", np.array([[ 1.04565272],
        [ 0.62555148],
        [ 0.38387466],
        [ 0.15622435],
        [-0.45990099]]), "\n")

    # Example 3:
    print("Predict : ", mylr.predict_(X))
    # Output:
    print("Output expected : ", np.array([[0.72865802],
        [0.40550072],
        [0.45241588]]), "\n")

    # Example 4:
    print("Cost : ", mylr.cost_(X,Y))
    # Output:
    print("Output expected : ", 0.5432466580663214, "\n")