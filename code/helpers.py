import colorama
import numpy as np


def print_bp(text):
    print("--> " + text)


def print_error(text):
    print(colorama.Fore.RED + text + colorama.Style.RESET_ALL)


def get_first_key(d):
    for key in d:
        return key
    raise IndexError


def angle_between_vectors(v0, v1, deg):
    dividend = (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z)
    divisor = np.sqrt(np.square(v0.x) + np.square(v0.y) + np.square(v0.z)) * np.sqrt(
        np.square(v1.x) + np.square(v1.y) + np.square(v1.z)
    )
    factor = 180 / np.pi if deg else 1
    return np.acos(np.round(dividend / divisor, 6)) * factor


# def sort_indices_descending(arr):
#     return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)
