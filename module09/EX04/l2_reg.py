from typing import Reversible
import numpy as np

def iterative_l2(thetas):
    """
        Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
        Args:
            theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        Returns:
            The L2 regularization as a float.
            None if theta in an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
    """
    res = 0
    for idx, theta in enumerate(thetas):
        if idx == 0:
            continue
        res += theta ** 2
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

if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19])

    # Example 1: 
    print("result : ", iterative_l2(x))
    # Output:
    print("Expected output : ", 911.0)

    # Example 2: 
    print("result : ", l2(x))
    # Output:
    print("Expected output : ", 911.0)

    y = np.array([3,0.5,-6])
    # Example 3: 
    print("result : ", iterative_l2(y))
    # Output:
    print("Expected output : ", 36.25)

    # Example 4: 
    print("result : ", l2(y))
    # Output:
    print("Expected output : ", 36.25)
