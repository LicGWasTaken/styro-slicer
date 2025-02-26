import math
import numpy as np
import prefs
import utils as u

feed = 2500

def to_gcode(file_name: str, coords: list, rad: float, velocity: int):
    u.msg(f"runtime: {run_time(coords, velocity)}s")

    # Prefixes
    file = open(prefs.MESH_FOLDER + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    # Homing, mm unit, Absolute positioning, Set extruder = 0, Feedrate 500mm/min
    file.write(f"G28\nG21\nG90\nG92 E0\nG1 F{velocity}\nM92 E8.88\n")
    file.write("M302 S0\nG1 E10\nG1 E0\n")  # Always allow extrusion
    file.write("\n")
    file.write(f"G1 F{feed}\nG1 X10 Y10 Z0\nG1 F{velocity}\n")

    offset = 65
    # Main code
    for i, r in enumerate(coords):
        # Move into the working area
        s = f"G1 F{velocity}\n"
        s += f"G1 X{bottom_approach_z(r) + offset} Y{bottom_approach_z(r) + offset} Z{50}\n"
        s += f"G1 X{bottom_approach_z(r) + offset} Y{bottom_approach_z(r) + offset} Z{r[0][0] + 155}\n"
        file.write(s)

        for j, p in enumerate(r):
            p += np.asarray([155, 155, offset])
            if j % 2 == 0:
                continue
            s = f"G1 X{p[2]} Y{p[2]} Z{p[0]}\n"#\n"  # Machine z is software x#
            #ca = (rad * 180 / math.pi) * p[2] / (r[len(r) - 1][2] + 35)
            #cs += f"E{(rad * 180 / math.pi) * (i) + a}\n" # Makes spiralling cuts
            file.write(s)

        # Move outside of the working area
        s = f"G1 X{top_approach_z(r) + offset} Y{top_approach_z(r) + offset} Z{p[0]}\n"
        s += f"G1 X{top_approach_z(r) + offset} Y{top_approach_z(r) + offset} Z{50}\n"
        s += f"G1 F{feed}\n"
        s += "G1 X10 Y10 Z50\n"
        file.write(s)

        # Rotation
        s = f"G1 E{rad * (i + 1) * 180 / math.pi}\n"
        # s = f"G1 E{rad * 180 / math.pi}\n"
        file.write(s)

    # Suffixes
    file.write("M18\n")  # Stepper off
    file.write("\n")

    file.write("%\n")  # End of file
    file.close()

    return

def top_approach_z(ps: np.ndarray):
    # return max(ps[:, 2:]) - ps[len(ps)][2] + 5
    return max(ps[:, 2]) + 5

def bottom_approach_z(ps: np.ndarray):
    # return ps[0][2] - min(ps[:, 2:]) - 5
    return min(ps[:, 2]) - 5

def run_time(coords: list, velocity: int):
    """Not working, probaly because we are only calculating cutting sequences"""
    # Calculate the rough total circumference
    circumferece = 0
    for i, r in enumerate(coords):
        # circumferece += u.magnitude(r[0])
        # circumferece += u.magnitude(r[len(r) - 1])
        for j, p in enumerate(r):
            if j < len(r) - 1:
                circumferece += u.magnitude(r[j + 1] - p) 

    # Velocity in mm/min
    return circumferece / velocity

