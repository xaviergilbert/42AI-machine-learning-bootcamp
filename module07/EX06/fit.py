import numpy as np
from time import sleep

def add_intercept(x):
    if isinstance(x, np.ndarray) == False or x.size == 0:
        return None

    b = np.ones((x.shape[0],1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.append(b, x, axis=1)

def predict_(x, thetas):
    if thetas.ndim == 2:
        thetas = thetas.flatten()
    x_intercept = add_intercept(x)
    res = np.sum(x_intercept * thetas, axis = 1)
    return res

def gradient_(x, y, thetas):
    x_intercept = x
    # print("x.shape : ", x.shape)
    # print("thetas.shape : ", thetas.shape)
    if thetas.shape[0] > x.shape[1]:
        x_intercept = add_intercept(x)
        # print("x_intercept : ", x_intercept)
    y_pred = np.sum(x_intercept * thetas, axis = 1)
    # print("y_pred : ", y_pred)
    y = np.reshape(y, y_pred.shape)
    theta_ = (y_pred - y) * x_intercept.T / y.shape[0]
    # print("theta_ : ", theta_)
    return np.sum(theta_, axis=1)
    # return theta_

def fit_(x, y, thetas, alpha, n_cycles):
    """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * n: (number of training examples, number of features).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            n_cycles: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
    """
    if thetas.ndim == 2:
        thetas = thetas.flatten()
    alpha = alpha
    max_iter = n_cycles
    for epoch in range(max_iter):
        gradient = gradient_(x, y, thetas)
        # print("gradient : ", gradient)
        thetas = thetas - (alpha * gradient)
        # print("Thetas : ", thetas)
        # sleep(0.1)
    return thetas

if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta,  alpha = 0.0005, n_cycles=42000)
    print("Theta : ", theta2)
    # Output:
    print("Expected output : ", np.array([[41.99],[0.97], [0.77], [-1.20]]))
    # exit()

    # Example 1:
    print("Predictions : ", predict_(x, theta2))
    # Output:
    print("Expected output : ", np.array([[19.5992], [-2.8003], [-25.1999], [-47.5996]]))