from vector import Vector
import numpy as np

v1 = Vector([[1.0, 1.0, 2.0, 3.0]])
v2 = Vector([1.0, 1.0, 2.0, 3.0])
v3 = Vector(3)
v4 = Vector((10,15))

# print("Shape : ", v1.shape)
# print("Vecteur : ", v1.values)
# print((v1 + 5).values)

print((v1 + 4).values)
print((v1 - v1).values)


v1 = np.array([[0.0, 1.0, 2.0, 3.0]])
print(v1)