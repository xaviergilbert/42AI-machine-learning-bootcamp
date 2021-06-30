import numpy as np
import random
import time

def data_spliter(x, y, proportion, seed=None):
    """
        Shuffles and splits the dataset (given by x and y) into a training and a test set, while respecting the given proportion of examples to be kept in the traning set.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            proportion: has to be a float, the proportion of the dataset that will be assigned to the training set.
        Returns:
            (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray
            None if x or y is an empty numpy.ndarray.
            None if x and y do not share compatible dimensions.
        Raises:
            This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))
    matrix = np.append(x, y, axis = 1)
    
    if seed == None:
        seed = random.randint(0, 100)
    np.random.seed(seed)
    np.random.shuffle(matrix)

    split_idx = round(proportion * len(y))
    x_train = matrix[:split_idx, :x.shape[1]]
    x_test = matrix[split_idx:, :x.shape[1]]
    y_train = matrix[:split_idx, x.shape[1]:]
    y_test = matrix[split_idx:, x.shape[1]:]

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x1 = np.array([1, 42, 300, 10, 59])
    y = np.array([0,1,0,1,0])

    # Example 1:
    x_train, x_test, y_train, y_test = data_spliter(x1, y, 0.8)
    print("x_train : ", x_train)
    print("x_test : ", x_test)
    print("y_train : ", y_train)
    print("y_test : ", y_test)


    x2 = np.array([ [  1,  42],
                    [300,  10],
                    [ 59,   1],
                    [300,  59],
                    [ 10,  42]])
    y = np.array([0,1,0,1,0])

    # Example 3:
    x_train, x_test, y_train, y_test = data_spliter(x2, y, 0.8)
    # Output:
    # (np.array([[ 10,  42],
    #         [300,  59],
    #         [ 59,   1],
    #         [300,  10]]), np.array([[ 1, 42]]), np.array([0, 1, 0, 1]), np.array([0]))
    print("x_train : ", x_train)
    print("x_test : ", x_test)
    print("y_train : ", y_train)
    print("y_test : ", y_test)