import math
import numpy as np
import prefs

def to_gcode(file_name: str, coords: list, rad: float):
    # Prefixes
    file = open(prefs.MESH_FOLDER + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    # Homing, mm unit, Absolute positioning, Set extruder = 0, Feedrate 500mm/min
    file.write("G28\nG21\nG90\nG92 E0\nG1 F500\n")  
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")
    
    # Main code
    for i, r in enumerate(coords):
        if i % 2 == 0:
            continue
        for p in r:
            s = f"G1 X{p[2]} Y{p[2]} Z{p[0]}\n"  # Machine z is software x
            file.write(s)

        # Move outside of the working area 
        s = f"G1 X{p[2] + 5} Y{p[2] + 5} Z{p[0]}\n"
        file.write(s)
        s = f"G1 X{p[2] + 5} Y{p[2] + 5} Z{0}\n"
        file.write(s)

        # Rotation
        s = f"G1 E{rad * (i + 1) * 180 / math.pi}\n"
        file.write(s)

    # Suffixes
    file.write("M18\n")  # Stepper off
    file.write("\n")

    file.write("%\n")  # End of file
    file.close()

    return

