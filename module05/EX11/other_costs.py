import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    result = np.sum((y - y_hat) ** 2) / y.shape[0]
    return result

def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    return np.sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    result = np.sum(np.abs(y - y_hat)) / y.shape[0]
    return result

def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.		
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    result = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return result

if __name__ == "__main__":
    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    ## your implementation
    print("mse :", mse_(x,y))
    ## Output:
    # 4.285714285714286
    ## sklearn implementation
    print("sklearn mse : ", mean_squared_error(x,y))
    ## Output:
    # 4.285714285714286

    # Root mean squared error
    ## your implementation
    print("rmse : ", rmse_(x,y))
    ## Output:
    # 2.0701966780270626
    ## sklearn implementation not available: take the square root of MSE
    print("sklearn rmse : ", sqrt(mean_squared_error(x,y)))
    ## Output:
    # 2.0701966780270626

    # Mean absolute error
    ## your implementation
    print("mae : ", mae_(x,y))
    # Output:
    # 1.7142857142857142
    ## sklearn implementation
    print("sklearn mae : ", mean_absolute_error(x,y))
    # Output:
    # 1.7142857142857142

    # R2-score
    ## your implementation
    print("r2 : ", r2score_(x,y))
    ## Output:
    # 0.9681721733858745
    ## sklearn implementation
    print("sklearn r2 : ", r2_score(x,y))
    ## Output:
    # 0.9681721733858745
    exit()