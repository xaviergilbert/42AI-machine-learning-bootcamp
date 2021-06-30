import numpy as np


def add_intercept(x):
    b = np.ones((x.shape[0],1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    res = np.append(b, x, axis=1)
    return res

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reg_logistic_grad(y, x, theta, lambda_):
    """
        Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three arrays must have compatible dimensions.
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
    return vec_reg_logistic_grad(y, x, theta, lambda_)

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """
        Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
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
    predict = sigmoid(np.sum(X * theta, axis = 1))
    gradient = (np.sum(X.T * (predict - y), axis = 1) + regularisation) / len(y)
    return gradient

if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4], 
                [2, 4, 5, 5], 
                [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # # Example 1.1:
    # reg_logistic_grad(y, x, theta, 1)
    # # Output:
    # print("Expected result : ", np.array([[-0.55711039],
    #     [-1.40334809],
    #     [-1.91756886],
    #     [-2.56737958],
    #     [-3.03924017]]))

    # Example 1.2:
    print("Result : ", vec_reg_logistic_grad(y, x, theta, 1))
    # Output:
    print("Expected result : ", np.array([[-0.55711039],
        [-1.40334809],
        [-1.91756886],
        [-2.56737958],
        [-3.03924017]]))

    # # Example 2.1:
    # reg_logistic_grad(y, x, theta, 0.5)
    # # Output:
    # print("Expected result : ", np.array([[-0.55711039],
    #     [-1.15334809],
    #     [-1.96756886],
    #     [-2.33404624],
    #     [-3.15590684]]))

    # Example 2.2:
    print("Result : ", vec_reg_logistic_grad(y, x, theta, 0.5))
    # Output:
    print("Expected result : ", np.array([[-0.55711039],
        [-1.15334809],
        [-1.96756886],
        [-2.33404624],
        [-3.15590684]]))

    # # Example 3.1:
    # reg_logistic_grad(y, x, theta, 0.0)
    # # Output:
    # print("Expected result : ", np.array([[-0.55711039],
    #     [-0.90334809],
    #     [-2.01756886],
    #     [-2.10071291],
    #     [-3.27257351]]))

    # Example 3.2:
    print("Result : ", vec_reg_logistic_grad(y, x, theta, 0.0))
    # Output:
    print("Expected result : ", np.array([[-0.55711039],
        [-0.90334809],
        [-2.01756886],
        [-2.10071291],
        [-3.27257351]]))