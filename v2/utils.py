import colorama
import numpy as np
import math
import prefs

# tmp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def msg(str, type="debug", end="\n"):
    if type == "error":
        print(colorama.Style.BRIGHT + colorama.Fore.RED + str, end=end)
    elif type =="warning":
        print(colorama.Style.NORMAL + colorama.Fore.YELLOW + str, end=end)
    elif type =="process":
        print(colorama.Style.NORMAL + str + "...", end=end)
    elif type =="info":
        print(colorama.Style.DIM + "--> " + str, end=end)
    elif type == "debug":
        print(colorama.Style.BRIGHT + colorama.Fore.LIGHTBLUE_EX + str, end=end)

    print(colorama.Style.RESET_ALL, end="")

def is_structured(list_: list, format: list):
    """Recursively compare the structure (dimensions and var types) of two lists"""

    # Variable length
    if len(format) < 2:
        for var in list_:
            if not is_structured(var, format[0]):
                return 0
        return 1

    # Fixed length
    else:
        if len(format) != len(list_):
            return 0

        for i, var in enumerate(list_):
            if not isinstance(var, list):
                if not isinstance(var, format[i]):
                    return 0
            elif not is_structured(var, format[i]):
                return 0
        return 1
    
def rotate_z_rad(point: np.ndarray, rad: float):
    try:
        p = np.asarray(
        [
            point[0] * math.cos(rad) - point[1] * math.sin(rad),
            point[0] * math.sin(rad) + point[1] * math.cos(rad),
            point[2]
        ])
    except:
        p = np.asarray(
        [
            point[0] * math.cos(rad) - point[1] * math.sin(rad),
            point[0] * math.sin(rad) + point[1] * math.cos(rad)
        ])
    return p

def magnitude(v):
    try:
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    except:
        return math.sqrt(v[0]**2 + v[1]**2)
    
def clamp(n, min, max):
    return min * (n < min) + max * (n > max) + n * (n >= min and n <= max)

def polar_angle(v):
    rad = math.atan2(v[1], v[0])
    return rad + 2 * math.pi * (rad < 0)

def minkowski(a: np.array, b: np.array):
    """Both arrays have to be sorted counterclockwise beginning from the bottom left angle"""
    # Calculate the minkoski sum
    i = 0
    j = 0
    exit = False
    out = []
    for n in range(len(a) + len(b)):
        out.append(np.asarray(a[i] + b[j]))

        if i + 1 >= len(a):
            rad_a = polar_angle(a[0] - a[i])
        else:
            rad_a = polar_angle(a[i + 1] - a[i])

        if j + 1 >= len(b):
            rad_b = polar_angle(b[0] - b[i])
        else:
            rad_b = polar_angle(b[j + 1] - b[j])

        if exit == False:
            if rad_a <= rad_b:
                i += 1
            if rad_a >= rad_b:
                j += 1
        else:
            if i == 0:
                j += 1
            else:
                i += 1

        if i >= len(a):
            i = 0
            exit = True
        if j >= len(b):
            j = 0
            exit = True

        if i == 0 and j == 0 and exit == True:
            break
        
    return np.asarray(out)

def plot(arr: np.ndarray, color:str = "hotpink"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color=color)
    except:
        ax.scatter(arr[:, 0], arr[:, 1], color=color)
    plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

# ----------- SDFs -----------

def box_SDF(
    origin: np.ndarray, point: np.ndarray, extents: np.ndarray, rotation_z: float
):
    """referencing https://iquilezles.org/articles/distfunctions/"""
    p = point - origin
    if rotation_z != 0:  # rotation in rad
        p = rotate_z_rad(p, -rotation_z)

    p = abs(p)
    r = extents / 2
    try:
        # 3D
        max_p = np.asarray(
            [max(p[0] - r[0], 0.0), max(p[1] - r[1], 0.0), max(p[2] - r[2], 0.0)]
        )
    except:
        # 2D
        max_p = np.asarray([max(p[0] - r[0], 0.0), max(p[1] - r[1], 0.0)])

    return magnitude(max_p)

def sphere_SDF(
    origin: np.ndarray, point: np.ndarray, radius: float 
):
    p = point - origin
    return magnitude(p) - radius

def vertical_capsule_SDF(
    origin: np.ndarray, point: np.ndarray, radius: float, height: float
):
    """Total capsule height = height + 2 * radius"""
    p = point - origin
    # Move the centerpoint from the bottom to the center of the capsule
    p[2] += height / 2
    p[2] -= clamp(p[2], 0.0, height)
    return magnitude(p) - radius

