import numpy as np

class Vector3:
    def __init__(self, *args):
        if len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
        elif len(args) == 1:
            if isinstance(args[0], list):
                arr = args[0]
                if len(arr) != 3 or isinstance(arr[0], list):
                    raise ValueError('Class Vector3 requires a 1D list of 3 elements')
                self.x = arr[0]
                self.y = arr[1]
                self.z = arr[2]
            elif isinstance(args[0], np.ndarray):
                np_arr = args[0]
                if len(np_arr.shape) != 1 or len(np_arr) != 3:
                    raise ValueError('Class Vector3 requires a 1D np.array of 3 elements')
                self.x = np_arr[0].item()
                self.y = np_arr[1].item()
                self.z = np_arr[2].item()
            else:
                raise TypeError('Unsupported type for initialization')
        else:
            raise TypeError('Class Vector3 requires either 3 arguments or a single list/np.array of 3 elements')

    def __repr__(self):
        decimals = 2
        return f'({round(self.x, decimals)} | {round(self.y, decimals)} | {round(self.z, decimals)})'
    
    # Operator overloads
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(np.add(self.list(), other.list()))
        else:
            return Vector3(np.add(self.list(), other))
        
    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(np.subtract(self.list(), other.list()))
        else:
            return Vector3(np.subtract(self.list(), other))
        
    def __mul__(self, other):
        pass

    def __div__(self, other):
        pass

    # Functions
    def list(self):
        return [self.x, self.y, self.z]
    
    def np_array(self):
        return np.array([self.x, self.y, self.z])

    def magnitude(self):
        # return np.linalg.norm(vector)
        return np.sqrt((np.square(self.x) + np.square(self.y) + np.square(self.z)))
    
    def normalized(self):
        return self.list() / self.magnitude
    
    def rotate_z(self, angle):
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        return Vector3(np.dot(R, self.list()))

