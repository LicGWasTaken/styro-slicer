import numpy as np

class Vector3:
    def __init__(self, *args):
        self.decimals = (
            2  # Keep this relatively low, the slicing isn't the most accurate
        )
        if len(args) == 3:
            self.x = round(args[0], self.decimals) + 0  # +0 gets rid of -0
            self.y = round(args[1], self.decimals) + 0
            self.z = round(args[2], self.decimals) + 0
        elif len(args) == 1:
            if isinstance(args[0], list):
                arr = args[0]
                if len(arr) != 3 or isinstance(arr[0], list):
                    raise ValueError("Class Vector3 requires a 1D list of 3 elements")
                self.x = round(arr[0], self.decimals) + 0
                self.y = round(arr[1], self.decimals) + 0
                self.z = round(arr[2], self.decimals) + 0
            elif isinstance(args[0], np.ndarray):
                np_arr = args[0]
                if len(np_arr.shape) != 1 or len(np_arr) != 3:
                    raise ValueError(
                        "Class Vector3 requires a 1D np.array of 3 elements"
                    )
                self.x = round(np_arr[0].item(), self.decimals) + 0
                self.y = round(np_arr[1].item(), self.decimals) + 0
                self.z = round(np_arr[2].item(), self.decimals) + 0
            else:
                raise TypeError("Unsupported type for initialization")
        else:
            raise TypeError(
                "Class Vector3 requires either 3 arguments or a single list/np.array of 3 elements"
            )

    def __repr__(self):
        decimals = 3
        x = None if self.x == None else round(self.x, decimals) + 0
        y = None if self.y == None else round(self.y, decimals) + 0
        z = None if self.z == None else round(self.z, decimals) + 0
        return f"({x} | {y} | {z})"

    # Operator overloads
    def __key(self):
        return (self.x, self.y, self.z)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return self.__key() == other.__key()
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(np.add(self.to_list(), other.to_list()))
        else:
            return Vector3(np.add(self.to_list(), other))

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(np.subtract(self.to_list(), other.to_list()))
        else:
            return Vector3(np.subtract(self.to_list(), other))

    def __rsub__(self, other):
        return Vector3(other - self.x, other - self.y, other - self.z)

    def __mul__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector3(np.multiply(self.to_list(), other))
        
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Vector3):
            x = self.x if other.x == 0 else self.x / other.x
            y = self.y if other.y == 0 else self.y / other.y
            z = self.z if other.z == 0 else self.z / other.z
            return Vector3(x, y, z)
        else:
            if other == 0:
                print(Exception(ZeroDivisionError))
                return self
            return Vector3(np.divide(self.to_list(), other))

    def __rtruediv__(self, other):
        return Vector3(other / self.x, other / self.y, other / self.z)
    
    # Python functions
    def __abs__(self):
        return Vector3(abs(self.x), abs(self.y), abs(self.z))

    # Functions
    def to_list(self):
        return [self.x, self.y, self.z]

    def to_np_array(self):
        return np.array([self.x, self.y, self.z])
    
    def max(self):
            return max(self.to_list())

    def magnitude(self):
        return np.sqrt((np.square(self.x) + np.square(self.y) + np.square(self.z)))

    def normalized(self):
        return self / self.magnitude()

    def rotate_z(self, angle):
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        return Vector3(np.dot(R, self.to_list()))

    def zero():
        return Vector3(0, 0, 0)

