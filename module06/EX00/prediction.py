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

def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.ndarray.
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


if __name__ == "__main__":
    x = np.arange(1,6)

    # Example 1:
    theta1 = np.array([5, 0])
    print("Result : ", predict_(x, theta1))
    # Ouput:
    # array([5., 5., 5., 5., 5.])
    # Do you remember why y_hat contains only 5's here?  

    # Example 2:
    theta2 = np.array([0, 1])
    print("Result : ", predict_(x, theta2))
    # Output:
    # array([1., 2., 3., 4., 5.])
    # Do you remember why y_hat == x here?  


    # Example 3:
    theta3 = np.array([5, 3])
    print("Result : ", predict_(x, theta3))
    # Output:
    # array([ 8., 11., 14., 17., 20.])


    # Example 4:
    theta4 = np.array([-3, 1])
    print("Result : ", predict_(x, theta4))
    # Output:
    # array([-2., -1.,  0.,  1.,  2.])
    exit()