import numpy as np
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    lr1 = MyLR([2, 0.7])

    # Example 0.0:
    prediction = lr1.predict_(x)
    print("prediction : ", prediction)
    # Output:
    # array([[10.74695094],
    #     [17.05055804],
    #     [24.08691674],
    #     [36.24020866],
    #     [42.25621131]])

    # Example 0.1:
    cost_elem = lr1.cost_elem_(lr1.predict_(x),y)
    print("cost_elem : ", cost_elem)
    # Output:
    # array([[77.72116511],
    #     [49.33699664],
    #     [72.38621816],
    #     [37.29223426],
    #     [78.28360514]])

    # Example 0.2:
    cost = lr1.cost_(lr1.predict_(x),y)
    print("cost : ", cost)
    # Output:
    # 315.0202193084312


    # Example 1.0:
    lr2 = MyLR([0, 0])
    lr2.fit_(x, y)
    thetas = lr2.thetas
    print("thetas : ", thetas)
    # Output:
    # array([[1.40709365],
    #     [1.1150909 ]])


    # Example 1.1:
    predict = lr2.predict_(x)
    print("predict : ", predict)
    # Output:
    output = np.array([[15.3408728 ],
        [25.38243697],
        [36.59126492],
        [55.95130097],
        [65.53471499]])
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, predict)
    plt.plot(x, output, c="red")
    plt.show()

    # Example 1.2:
    # print("thetas : ", lr2.thetas)
    cost_elem = lr2.cost_elem_(lr2.predict_(x),y)
    print("cost_elem : ", cost_elem)
    # Output:
    # array([[35.6749755 ],
    #     [ 4.14286023],
    #     [ 1.26440585],
    #     [29.30443042],
    #     [22.27765992]])

    # Example 1.3:
    cost = lr2.cost_(lr2.predict_(x),y)
    print("cost : ", cost)
    # Output:
    # 92.66433192085971
    exit()