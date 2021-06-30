import numpy as np

def add_polynomial_features(x, power):
    """
        Add polynomial features to vector x by raising its values up to the power given in argument.  
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * 1.
            power: has to be an int, the power up to which the components of vector x are going to be raised.
        Returns:
            The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg he polynomial feature values for all training examples.
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
    """
    # print(x[:, 0])
    shape = x.shape[1]
    for idx in range(power):
        if idx >= 1:
            for col in range(shape):
            # x = np.append(x, np.reshape(np.array(x[:, 0] ** (idx + 1)), (-1, 1)), axis = 1)
                x = np.append(x, np.reshape(np.array(x[:, col] ** (idx + 1)), (-1, 1)), axis = 1)
        # print(x)
    return x

if __name__ == "__main__":
    x = np.arange(1,11).reshape(5, 2)
    # print(x)
    # exit()

    # Example 1:
    print("polynomial features : ", add_polynomial_features(x, 3))
    # Output:
    print("Expected output : ", np.array([[   1,    2,    1,    4,    1,    8],
       [   3,    4,    9,   16,   27,   64],
       [   5,    6,   25,   36,  125,  216],
       [   7,    8,   49,   64,  343,  512],
       [   9,   10,   81,  100,  729, 1000]]))
    print()

    # Example 2:
    print("polynomial features : ", add_polynomial_features(x, 4))
    # Output:
    print("Expected output : ", np.array([[    1,     2,     1,     4,     1,     8,     1,    16],
       [    3,     4,     9,    16,    27,    64,    81,   256],
       [    5,     6,    25,    36,   125,   216,   625,  1296],
       [    7,     8,    49,    64,   343,   512,  2401,  4096],
       [    9,    10,    81,   100,   729,  1000,  6561, 10000]]))
    exit()