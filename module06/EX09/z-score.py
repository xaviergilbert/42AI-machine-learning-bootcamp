import numpy as np
import math



def mean_(datas):
    return sum(datas) / len(datas)
    # for data in datas:

def var_(datas):
    datas = sorted(datas)
    mean = mean_(datas)
    diff = (datas - mean) ** 2
    return mean_(diff)

def std_(datas):
    return math.sqrt(var_(datas))

def zscore(x):
    res = (x - mean_(x)) / std_(x)
    return res


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print("zscore : ", zscore(X))
    # Output:
    # array([-0.08620324,  1.2068453 , -0.86203236,  0.51721942,  0.94823559,
    #         0.17240647, -1.89647119])

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print("zscore : ", zscore(Y))
    # Output:
    # array([ 0.11267619,  1.16432067, -1.20187941,  0.37558731,  0.98904659,
    #         0.28795027, -1.72770165])