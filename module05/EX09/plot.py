import numpy as np
import matplotlib.pyplot as plt


def add_intercept(x):
    """
      Adds a column of 1's to the non-empty numpy.ndarray x.
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

    b = np.ones((x.shape[0], 1), dtype=int)
    if x.ndim == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.append(b, x, axis=1)

def predict_(x, theta):
    """
      Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
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
    if x.size == 0 or theta.size == 0 or x.ndim != 1 or theta.ndim != 1:
        return None
    x = add_intercept(x)
    # print(x)
    # print(x * theta)

    return np.sum(x * theta, axis=1)

def plot_with_cost(x, y, theta):
    """
      Plot the data and prediction line from three non-empty numpy.ndarray.
      Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
      Returns:
          Nothing.
      Raises:
        This function should not raise any Exception.
    """
    y_pred = predict_(x, theta)
    plt.figure()
    plt.scatter(x, y, c='blue')
    plt.plot(x, y_pred, c="red")
    for idx, xi in enumerate(x):
        plt.plot([xi, xi], [y[idx], y_pred[idx]], c="red", ls="--")
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482,
                  13.14755699, 18.60682298, 14.14329568])

    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_cost(x, y, theta1)

    # Example 2:
    theta2 = np.array([14, 0])
    plot_with_cost(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_cost(x, y, theta3)
