import numpy as np
import math
import time
import utils as u

# TODO currently scales the gcode which leads to incorrect kerfs
# TODO add an alternative that cuts in alternating directions
scale = 1
decimals = 3
material_offset = 20

def to_gcode_axysimmetric(file_name, arr: list, origin: np.ndarray, material_size: np.ndarray, feed: float, slew: float):
    """axysimmetric: absolute
    E,X,Y,Z --> Z, -, (X, U), (Y, V)"""

    arr = set_rotation(arr, math.pi)
    arr = set_scale(arr, scale)
    arr = set_origin(arr, origin)

    print(approach(arr[0][1], 5))

    # Open 
    gcode_file_name = file_name_timestamp(file_name)
    file = open(f"{gcode_file_name}.gcode", mode="w")

    # Prefixes
    file.write(f"; O{origin}\n")
    # Homing, mm unit, Absolute positioning
    file.write(f"G28\nG21\nG90\nG1 F{feed}\n")
    file.write("\n")

    for i, (angle, points) in enumerate(arr):
        # Rotation
        s = f"G1 F{slew} Z{angle}\n"
        file.write(s)

        # Turn the current back on
        s = f"M42 P0 S1\n"
        file.write(s)

        # Move into the working area
        app = approach(points, offset=5)
        xu = round(app[0][1], decimals)
        yv = round(app[0][2], decimals)
        s = f"G1 F{slew} Y{yv} V{yv}\n"
        s += f"G1 F{feed} X{xu} U{xu}\n"
        file.write(s)

        s = f"G1 F{feed}\n"
        for p in points:
            xu = round(p[1], decimals)
            yv = round(p[2], decimals)
            s += f"G1 X{xu} U{xu} Y{yv} V{yv}\n"
        file.write(s)

        # Move outside of the working area
        xu = round(app[1][1], decimals)
        yv = round(app[1][2], decimals)
        s = f"G1 X{xu} U{xu}\n"
        file.write(s)

        # Calculate the size of the rotated block
        rad_1 = angle * math.pi / 180
        try:
            rad_2 = arr[i + 1][0] * math.pi / 180
        except IndexError:
            rad_2 = 0
        size_1 = material_length(material_size, rad_1)
        size_2 = material_length(material_size, rad_2)
        max_length = max(size_1, size_2)

        # Move as far outside of the working area as required
        # And back to the start
        xu = (max_length + material_offset) / 2
        xu = origin[1] - xu
        xu = round(xu, decimals)
        yv = round(app[0][2], decimals)
        s = f"G1 X{xu} U{xu}\n"
        s += f"M42 P0 S0\n"
        s += f"G1 F{slew} Y{yv} V{yv}\n"
        file.write(s)

    # Suffixes
    file.write("M18\n")  # Stepper off
    file.write("\n")
    # file.write(mcdonalds())
    # file.write("\n")

    file.close()
    return

def set_rotation(arr: list, rad: float):
    out = []

    for angle, points in arr:
        tmp = []
        for p in points:
            tmp.append(u.rotate_z_rad(p, rad))
        out.append([angle, np.asarray(tmp)])
    
    return out

def set_origin(arr: list, origin: np.ndarray):
    """Shift all coordinates to be around the new origin"""
    """origin -> np.ndarray([x, y, z])"""
    out = []

    for angle, points in arr:
        tmp = []
        for p in points:
            tmp.append(p + origin)
        out.append([angle, np.asarray(tmp)])
    
    return out

def set_scale(arr: list, scale: float):
    out = []

    for angle, points in arr:
        tmp = []
        for p in points:
            tmp.append(p * scale)
        out.append([angle, np.asarray(tmp)])
    
    return out

def approach(ps: np.ndarray, offset: float):
    """returns bottom and top approach coordinates"""
    b = ps[np.argmin(ps[:, 2])]
    t = ps[np.argmax(ps[:, 2])]
    b[2] -= offset
    t[2] += offset

    return np.asarray([b, t])

def file_name_timestamp(str):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month = months[int(time.strftime("%m")) - 1]
    time_str = time.strftime("%d-%Hh%Mm")
    file_name = str.rsplit('.', 1)[0]
    return file_name + "_" + month + time_str

def material_length(size: np.ndarray, rad: float):
    """Find the maximum y coordinate of a rotated 2d rectangle.
    This assumes the x axis to be aligned with the wire at the start of the gcode"""
    # TODO Image for parameter in ui
    x = size[0]
    y = size[1]
    verts = np.asarray([[x/2, y/2], [-x/2, y/2], [-x/2, -y/2], [x/2, -y/2]])
    rotated_verts = []
    for v in verts:
        new_vert = u.rotate_z_rad(np.asarray(v), rad)
        rotated_verts.append(new_vert)
    
    rotated_verts = np.asarray(rotated_verts)
    max_y = rotated_verts[np.argmax(rotated_verts[:, 1])][1]
    return max_y * 2

def mcdonalds():
    s = """
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P600 S0
G4 P600
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P600 S0
G4 P600
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P300 S0
G4 P300
M300 P150 S523
G4 P150
M300 P150 S698
G4 P150
M300 P150 S0
G4 P150
M300 P150 S698
G4 P150
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P150 S349
G4 P150
M300 P150 S523
G4 P150
M300 P300 S0
G4 P300
M300 P75 S831
G4 P75
M300 P75 S0
G4 P75
M300 P75 S831
G4 P75
M300 P75 S831
G4 P75
M300 P75 S831
G4 P75
M300 P225 S0
G4 P225
M300 P75 S831
G4 P75
M300 P75 S0
G4 P75
M300 P75 S831
G4 P75
M300 P75 S831
G4 P75
M300 P75 S831
G4 P75"""

    return s
    