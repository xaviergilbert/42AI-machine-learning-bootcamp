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


def vec_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrice of dimension (m, n).
        y: has to be an numpy.ndarray, a vector of dimension (m, 1).
        theta: has to be an numpy.ndarray, a vector of dimension (n, 1).
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions (n, 1), containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    x_intercept = add_intercept(x)
    res = (np.sum(x_intercept * theta, axis=1) - y) * x.T / x.shape[0]
    return np.sum(res, axis = 1)

if __name__ == "__main__":
    X = np.array([
        [ -6,  -7,  -9],
            [ 13,  -2,  14],
            [ -7,  14,  -1],
            [ -8,  -4,   6],
            [ -5,  -9,   6],
            [  1,  -5,  11],
            [  9, -11,   8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta = np.array([0, 3, 0.5, -6])
    print("result : ", vec_gradient(X, Y, theta))
    # array([ -37.35714286, 183.14285714, -393.])

    theta = np.array([0, 0, 0, 0])
    print("result : ", vec_gradient(X, Y, theta))
    # array([  0.85714286, 23.28571429, -26.42857143])

    print("result : ", vec_gradient(X, add_intercept(X).dot(theta), theta))
    # array([0., 0., 0.])
    exit()