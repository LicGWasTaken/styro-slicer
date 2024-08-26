import colorama
import numpy as np

def print_bp(text):
    print('--> ' + text)

def print_error(text):
    print(colorama.Fore.RED + text + colorama.Style.RESET_ALL)

def rotate_vector_z(vector, theta):
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R, vector)

# def sort_indices_descending(arr):
#     return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)
