import numpy as np

def add_intercept(x):
    if isinstance(x, np.ndarray) == False or x.size == 0:
        return None

    b = np.ones((x.shape[0],1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.append(b, x, axis=1)

def simple_predict(x, thetas):
    """
        Computes the prediction vector y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
    """
    if thetas.ndim == 2:
        thetas = thetas.flatten()
    x_intercept = add_intercept(x)
    res = np.sum(x_intercept * thetas, axis = 1)
    return res

if __name__ == "__main__":
    x = np.arange(1,13).reshape((4,3))

    # Example 1:
    theta1 = np.array([5, 0, 0, 0])
    print("Predictions : ", simple_predict(x, theta1))
    # Ouput:
    print("Expected output : ", np.array([5., 5., 5., 5.]))
    # Do you understand why y_hat contains only 5's here?  
    print()

    # Example 2:
    theta2 = np.array([0, 1, 0, 0])
    print("Predictions : ", simple_predict(x, theta2))
    # Output:
    print("Expected output : ", np.array([ 1.,  4.,  7., 10.]))
    # Do you understand why y_hat == x[:,0] here?  
    print()

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98])
    print("Predictions : ", simple_predict(x, theta3))
    # Output:
    print("Expected output : ", np.array([ 9.64, 24.28, 38.92, 53.56]))
    print()

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5])
    print("Predictions : ", simple_predict(x, theta4))
    # Output:
    print("Expected output : ", np.array([12.5, 32. , 51.5, 71. ]))
    print()