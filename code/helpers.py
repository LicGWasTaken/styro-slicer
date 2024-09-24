import colorama
import numpy as np
from vector import Vector3

def print_bp(text: str):
    print("--> " + text)

def print_error(text: str):
    print(colorama.Fore.RED + text + colorama.Style.RESET_ALL)

def get_first_key(d: dict):
    for key in d:
        return key
    raise IndexError

def angle_between_vectors(v0: Vector3, v1: Vector3, deg: bool):
    dividend = (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z)
    divisor = np.sqrt(np.square(v0.x) + np.square(v0.y) + np.square(v0.z)) * np.sqrt(
        np.square(v1.x) + np.square(v1.y) + np.square(v1.z)
    )
    factor = 180 / np.pi if deg else 1
    return np.acos(np.round(dividend / divisor, 6)) * factor

def is_structured(l: list, s: str):
    values = []
    for c in s:
        if c == "n":
            values.append(0)
        elif c.isnumeric():
            values.append(int(c))
    
    for i in range(len(values)):
        if isinstance(l, list):
            if values[i] == 0:
                l = l[0]
                continue
            else:
                if values[i] == len(l):
                    l = l[0]
                    continue
                else:
                    return False
    return True

