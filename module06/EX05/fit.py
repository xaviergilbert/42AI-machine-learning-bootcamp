import numpy as np
from time import sleep
import matplotlib.pyplot as plt

def add_intercept(x):
    """
        Adds a column of 1's to the non-empty numpy.ndarray x.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * 1.
        Returns:
            X as a numpy.ndarray, a vector of dimension m * 2.
            None if x is not a numpy.ndarray.
            None if x is a empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) == False or x.size == 0:
        return None
        
    b = np.ones((x.shape[0],1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.append(b, x, axis=1)

def predict(x, theta):
    """
        Computes the prediction vector y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimensions m * 1.
            theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
    """
    x_intercept = add_intercept(x)
    res = np.sum(x_intercept * theta, axis=1)
    return res

def simple_gradient(x, y, theta):
    """
      Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
      Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a 2 * 1 vector.
      Returns:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
      Raises:
        This function should not raise any Exception.
    """
    x_intercept = add_intercept(x)
    y_pred = np.sum(x_intercept * theta, axis=1)
    # theta_0_ = np.sum(y_pred - y) / y.shape[0]
    # theta_1_ = np.sum((y_pred - y) * x) / y.shape[0]
    theta_ = np.dot((y_pred - y), x_intercept) / y.shape[0]
    return np.sum(theta_, axis=0)

def fit_(x, y, theta, alpha, max_iter):
    """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
	"""
    for epoch in range(max_iter):
        gradient = simple_gradient(x, y, theta)
        # print("gradient : ", gradient)
        theta = theta - (alpha * gradient)
        # print("theta : ", theta)
        # sleep(1)
    return theta

if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1])

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=3000000)
    print("theta1 : ", theta1)
    # Output:
    # array([[1.40709365],
    #     [1.1150909 ]])

    # Example 1:
    y_pred = predict(x, theta1)
    print("Predict : ", y_pred)
    # Output:
    # array([[15.3408728 ],
    #     [25.38243697],
    #     [36.59126492],
    #     [55.95130097],
    #     [65.53471499]])

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()