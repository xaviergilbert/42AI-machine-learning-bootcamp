import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x.
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
    theta_ = np.dot((y - y_pred), x_intercept) / y.shape[0]
    return theta_

if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])

    # Example 0:
    theta1 = np.array([2, 0.7])
    print("Result : ", simple_gradient(x, y, theta1), "\n")
    # Output:
    # array([21.0342574, 587.36875564])

    # Example 1:
    theta2 = np.array([1, -0.4])
    print("Result : ", simple_gradient(x, y, theta2))
    # Output:
    # array([58.86823748, 2229.72297889])