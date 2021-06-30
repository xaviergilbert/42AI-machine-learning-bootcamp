import numpy as np

def reg_log_cost_(y, y_hat, theta, lambda_):
    """
        Computes the regularized cost of a logistic regression model from two non-empty numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
            theta: has to be a numpy.ndarray, a vector of dimension n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized cost as a float.
            None if y, y_hat, or theta is empty numpy.ndarray.
            None if y and y_hat do not share the same dimensions.
        Raises:
            This function should not raise any Exception.
    """
    ones = np.ones(len(y))
    cost = (y * np.log(y_hat) + (ones - y) * np.log(ones - y_hat)) * (-1/len(y))
    regularisation = (lambda_ / (2 * len(y))) * np.sum(theta[1:] * theta[1:])
    res = np.sum(cost) + regularisation 
    return res


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0])
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
    theta = np.array([1, 2.5, 1.5, -0.9])

    # Example :
    print("Result : ", reg_log_cost_(y, y_hat, theta, .5))
    # Output:
    print("Expected output : ", 0.43377043716475955)

    # Example :
    print("Result : ", reg_log_cost_(y, y_hat, theta, .05))
    # Output:
    print("Expected output : ", 0.13452043716475953)

    # Example :
    print("Result : ", reg_log_cost_(y, y_hat, theta, .9))
    # Output:
    print("Expected output : ", 0.6997704371647596)