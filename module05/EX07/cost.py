import numpy as np
from numpy.core.defchararray import add

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

def predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
      theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray.
      None if x or theta dimensions are not appropriate.
    Raises:
      This function should not raise any Exceptions.
    """
    if x.size == 0 or theta.size == 0:
        return None
    x = add_intercept(x)
    res = np.sum(x * theta.T, axis=1)
    # return np.reshape(res, (res.shape[0], 1))
    return res


def cost_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
    Raises:
        This function should not raise any Exception.
    """

    try:
        res = [((y_hati - yi) ** 2 * 1 / y_hat.shape[0] / 2) for y_hati, yi in zip(y_hat, y)]
    except:
        return None
    return np.array(res)

def cost_(y, y_hat):
    """
    Description:
        Calculates the value of cost function.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
    Raises:
        This function should not raise any Exception.
    """
    if y.ndim == 2:
        y = np.reshape(y, (y.shape[0],))
    if y.ndim != y_hat.ndim or len(y) != len(y_hat):
        return None
    res = np.sum((y_hat - y)**2 / y_hat.shape[0] / 2)
    return res


if __name__ == "__main__":
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    # Example 1:
    # print("y_hat1 :", y_hat1)
    # print("y1 :", y1)
    print("Result exemple 1 :\n", cost_elem_(y1, y_hat1))
    
    # Output:
    # array([[0.], [0.1], [0.4], [0.9], [1.6]])

    # Example 2:
    print("Result exemple 2 :\n", cost_(y1, y_hat1))

    # Output:
    3.0

    x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    theta2 = np.array([[0.05], [1.], [1.], [1.]])
    y_hat2 = predict(x2, theta2)
    y2 = np.array([[19.], [42.], [67.], [93.]])

    # Example 3:
    print("Result exmple 3 :\n", cost_elem_(y2, y_hat2))

    # Output:
    # array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])

    # Example 4:
    print("Result exemple 4 :\n", cost_(y2, y_hat2))

    # Output:
    # 4.238750000000004

    x3 = np.array([0, 15, -9, 7, 12, 3, -21])
    theta3 = np.array([[0.], [1.]])
    y_hat3 = predict(x3, theta3)
    # print(y_hat3)
    y3 = np.array([2, 14, -13, 5, 12, 4, -19])

    # Example 5:
    print("Result exemple 5 :\n", cost_(y3, y_hat3))

    # Output:
    # 2.142857142857143

    # Example 6:
    print("Result exemple 6 :\n", cost_(y3, y3))

    # Output:
    0.0