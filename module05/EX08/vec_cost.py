import numpy as np


def cost_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.ndarray.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.ndim == 2:
        y = np.reshape(y, (y.shape[0],))
    if y.ndim != y_hat.ndim or len(y) != len(y_hat):
        return None
    res = np.sum((y_hat - y)**2 / y_hat.shape[0] / 2)
    return res

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Example 1:
    print("Result exemple 1 :\n", cost_(X, Y))
    # Output:
    # 2.142857142857143

    # Example 2:
    print("Result exemple 2 :\n", cost_(X, X))
    # Output:
    # 0.0