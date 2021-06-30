import numpy as np

def sigmoid_(x):
    if type(x) != np.ndarray or x.size == 0:
        return None
    return (1 / (1 + np.exp(-x)))

def logistic_predict_(x, theta):
    """
        Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
            theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))
    if theta.ndim == 2:
        theta = theta[:, 0]

    if type(x) != np.ndarray or type(theta) != np.ndarray:
        print("x array or theta array not np.ndarray")
        return None
    if x.size == 0 or theta.size == 0:
        print("x or theta size are 0")
        return None
    if x.shape[1] != theta.shape[0] - 1:
        print("x.shape[1] != theta.shape[0] + 1")
        return None

    x_intercept = np.append(np.ones((x.shape[0],1)), x, axis=1)
    predict = x_intercept.dot(theta)
    res = sigmoid_(predict)
    return res


if __name__ == "__main__":
    # Example 1
    x = np.array([4])
    theta = np.array([[2], [0.5]])
    print("Logistic prediction : ", logistic_predict_(x, theta))
    # Output: 
    print("Expected output : ", np.array([[0.98201379]]))

    # Example 1
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]]) 
    print("Logistic prediction : ", logistic_predict_(x2, theta2))
    # Output: 
    print("Expected output : ", np.array([[0.98201379],
        [0.99624161],
        [0.97340301],
        [0.99875204],
        [0.90720705]]))

    # Example 3
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print("Logistic prediction : ", logistic_predict_(x3, theta3))
    # Output: 
    print("Expected output : ", np.array([[0.03916572],
        [0.00045262],
        [0.2890505 ]]))