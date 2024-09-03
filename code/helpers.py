import colorama
import numpy as np

def print_bp(text):
    print("--> " + text)

def print_error(text):
    print(colorama.Fore.RED + text + colorama.Style.RESET_ALL)

# def sort_indices_descending(arr):
#     return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)

