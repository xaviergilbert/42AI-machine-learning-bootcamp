import sys,os
sys.path.append(os.path.realpath('..'))
from EX06.my_linear_regression import MyLinearRegression as MyLR
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

linear_model2.fit_(Xpill, Yscore)
predictions = linear_model2.predict_(Xpill)

plt.figure()
plt.scatter(Xpill, Yscore, c="blue")
plt.scatter(Xpill, predictions, c="green")
plt.plot(Xpill, predictions, c="green")
plt.show()

print("linear_model1.cost_ : ", linear_model1.cost_(Xpill, Yscore))
# # 57.60304285714282
print("mean_squared_error : ", mean_squared_error(Yscore, Y_model1))
# # 57.603042857142825
print("linear_model2.cost_ : ", linear_model2.cost_(Xpill, Yscore))
# # 232.16344285714285
print("mean_squared_error : ", mean_squared_error(Yscore, predictions))
# 232.16344285714285