class Vector:
    def __init__(self, vector):
        # print("Type : ", type(vector))
        self.values = self.get_values(vector)
        # print("Values: ", self.values)
        self.shape = self.get_shape()
    
    def get_values(self, vector):
        if type(vector) == int:
            tmp = []
            for idx in range(vector):
                tmp.append([float(idx)])
            return tmp
        elif type(vector) == tuple:
            tmp = []
            for idx in range(vector[0], vector[1]):
                tmp.append(float(idx))
            return tmp
        else:
            return vector

    def get_shape(self):
        col = 0
        row = 0
        for idx, elem in enumerate(self.values):
            if type(elem) == list:
                for idx2, elem2 in enumerate(elem):
                    row = idx2
                    continue
            col = idx
        return (row + 1, col + 1)

    def __add__(self, other):

        if isinstance(other, Vector):
            if self.shape == other.shape:
                new_vector = []
                for row in range(self.shape[1]):
                    if self.shape[0] > 1:
                        for col in range(self.shape[0]):
                            new_vector.append(self.values[row][col] + other.values[row][col])
                    else:
                        new_vector.append(self.values[row] + other.values[row])
            else:
                raise ValueError("Vectors must be of the same shape")
        elif isinstance(other, (int, float)):
            new_vector = []
            for row in range(self.shape[1]):
                if self.shape[0] > 1:
                    for col in range(self.shape[0]):
                        new_vector.append(self.values[row][col] + other)
                else:
                    new_vector.append(self.values[row] + other)
        else:
            raise ValueError("Addition with type {} not supported".format(type(other)))
        
        return self.__class__(new_vector)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        if isinstance(other, Vector):
            if self.shape == other.shape:
                new_vector = []
                for row in range(self.shape[1]):
                    if self.shape[0] > 1:
                        for col in range(self.shape[0]):
                            new_vector.append(self.values[row][col] - other.values[row][col])
                    else:
                        new_vector.append(self.values[row] - other.values[row])
            else:
                raise ValueError("Vectors must be of the same shape")
        elif isinstance(other, (int, float)):
            new_vector = []
            for row in range(self.shape[1]):
                if self.shape[0] > 1:
                    for col in range(self.shape[0]):
                        new_vector.append(self.values[row][col] - other)
                else:
                    new_vector.append(self.values[row] - other)
        else:
            raise ValueError("Substraction with type {} not supported".format(type(other)))
        
        return self.__class__(new_vector)


        # new_vector = []
        # for row in range(self.shape[1]):
        #     for col in range(self.shape[0]):
        #         new_vector.append(self.values[col][row] - other)
        # return Vector(new_vector) 

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        new_vector = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                new_vector.append(self.values[col][row] * other)
        return Vector(new_vector)       

    def __truediv__(self, other):
        new_vector = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                new_vector.append(self.values[col][row] / other)
        return Vector(new_vector)     


# # Column vector of dimensions n * 1
# v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
# v1 * 5
# # Output
# Vector([[0.0], [5.0], [10.0], [15.0]])

# # Row vector of dimensions 1 * n
# v1 = Vector([0.0, 1.0, 2.0, 3.0])
# v2 = v1 * 5
# # Output
# Vector([0.0, 5.0, 10.0, 15.0])