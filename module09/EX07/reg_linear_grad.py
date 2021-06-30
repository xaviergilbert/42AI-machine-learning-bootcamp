import numpy as np

def add_intercept(x):
    b = np.ones((x.shape[0],1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    res = np.append(b, x, axis=1)
    return res

def l2(theta):
    """
        Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
        Args:
            theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        Returns:
            The L2 regularization as a float.
            None if theta in an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
    """  
    # print(np.sum(theta[1:] * theta[1:]))
    return np.sum(theta[1:] * theta[1:])

def reg_linear_grad(y, x, thetas, lambda_):
    """
        Computes the regularized linear gradient of three non-empty numpy.ndarray, with two for-loop. The three arrays must have compatible dimensions.
        Args:
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of dimension n * 1.
            lambda_: has to be a float.
        Returns:
            A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles dimensions.
        Raises:
            This function should not raise any Exception.
    """
    return vec_reg_linear_grad(y, x, theta, lambda_)

def vec_reg_linear_grad(y, x, theta, lambda_):
    """
        Computes the regularized linear gradient of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
        Args:
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of dimension n * 1.
            lambda_: has to be a float.
        Returns:
            A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles dimensions.
        Raises:
            This function should not raise any Exception.
    """
    theta = theta.flatten()
    y = y.flatten()
    regularisation = np.append(np.array([0]), theta[1:]) * lambda_
    X = add_intercept(x)
    predict = np.sum(X * theta, axis = 1)
    gradient = (np.sum(X.T * (predict - y), axis = 1) + regularisation) / len(y)
    return gradient


if __name__ == "__main__":
    x = np.array([
        [ -6,  -7,  -9],
        [ 13,  -2,  14],
        [ -7,  14,  -1],
        [ -8,  -4,   6],
        [ -5,  -9,   6],
        [  1,  -5,  11],
        [  9, -11,   8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    # Example 1.1:
    print("result : ", reg_linear_grad(y, x, theta, 1))
    # Output:
    print("Expected output : ", np.array([[ -60.99      ],
        [-195.64714286],
        [ 863.46571429],
        [-644.52142857]]))

    # Example 1.2:
    print("result : ", vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    print("Expected output : ", np.array([[ -60.99      ],
        [-195.64714286],
        [ 863.46571429],
        [-644.52142857]]))

    # Example 2.1:
    print("result : ", reg_linear_grad(y, x, theta, 0.5))
    # Output:
    print("Expected output : ", np.array([[ -60.99      ],
        [-195.86142857],
        [ 862.71571429],
        [-644.09285714]]))

    # Example 2.2:
    print("result : ", vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    print("Expected output : ", np.array([[ -60.99      ],
        [-195.86142857],
        [ 862.71571429],
        [-644.09285714]]))

    # Example 3.1:
    print("result : ", reg_linear_grad(y, x, theta, 0.0))
    # Output:
    print("Expected output : ", np.array([[ -60.99      ],
        [-196.07571429],
        [ 861.96571429],
        [-643.66428571]]))

    # Example 3.2:
    print("result : ", vec_reg_linear_grad(y, x, theta, 0.0))
    # Output:
    print("Expected output : ", np.array([[ -60.99      ],
        [-196.07571429],
        [ 861.96571429],
        [-643.66428571]]))