import numpy as np

def add_intercept(x):
    if isinstance(x, np.ndarray) == False or x.size == 0:
        return None

    b = np.ones((x.shape[0],1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.append(b, x, axis=1)

def gradient(x, y, thetas):
    """
        Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            theta: has to be an numpy.ndarray, a vector (n +1) * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
        Raises:
            This function should not raise any Exception.
    """
    x_intercept = x
    if thetas.shape[0] < x.shape[1]:
        x_intercept = add_intercept(x)
    y_pred = np.sum(x_intercept * thetas, axis = 1)
    y = np.reshape(y, y_pred.shape)
    theta_ = (y_pred - y) * x_intercept.T / y.shape[0]
    return np.sum(theta_, axis=1)

if __name__ == "__main__":
    x = np.array([
            [ -6,  -7,  -9],
            [ 13,  -2,  14],
            [ -7,  14,  -1],
            [ -8,  -4,   6],
            [ -5,  -9,   6],
            [  1,  -5,  11],
            [  9, -11,   8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta1 = np.array([3,0.5,-6])

    # Example :
    print("Gradient : ", gradient(x, y, theta1))
    # Output:
    print("Expected result : ", np.array([ -37.35714286,  183.14285714, -393.        ]))

    # Example :
    theta2 = np.array([0,0,0])
    print("Gradient : ", gradient(x, y, theta2))
    # Output:
    print("Expected result : ", np.array([  0.85714286,  23.28571429, -26.42857143]))