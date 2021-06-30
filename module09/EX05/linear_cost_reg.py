import numpy as np

def reg_cost_(y, y_hat, theta, lambda_):
    """
        Computes the regularized cost of a linear regression model from two non-empty numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
            theta: has to be a numpy.ndarray, a vector of dimension n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized cost as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same dimensions.
        Raises:
            This function should not raise any Exception.
    """
    regularisation = np.sum(theta[1:] * theta[1:])
    cost = np.sum((y_hat - y) ** 2) + regularisation * lambda_
    cost = cost / (len(y) * 2)
    return cost


if __name__ == "__main__":
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20])
    theta = np.array([1, 2.5, 1.5, -0.9])

    # Example :
    print("Result : ", reg_cost_(y, y_hat, theta, .5))
    # Output:
    print("Output expected : ", 0.8503571428571429)

    # Example :
    print("Result : ", reg_cost_(y, y_hat, theta, .05))
    # Output:
    print("Output expected : ", 0.5511071428571429)

    # Example :
    print("Result : ", reg_cost_(y, y_hat, theta, .9))
    # Output:
    print("Output expected : ", 1.116357142857143)