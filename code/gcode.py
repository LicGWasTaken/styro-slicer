import preferences as prefs
from vector import Vector3
import helpers as h
import plotter

def to_gcode(file_name: str, XY: list, UV: list):
    if not h.is_structured(XY, "(n, 2)") or not h.is_structured(UV, "(n, 2)"):
        raise ValueError("list not structured correctly")

    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    file.write("G28\nG21\nG90\n")  # Homing, mm unit, Absolute positioning
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")

    for xy, uv in zip(XY, UV):
        xy = [round(xy.x, 1), round(xy.z, 1)]
        uv = [round(uv.x, 1), round(uv.z, 1)]
        s = f"G1 X{xy[0]} Y{xy[1]} U{uv[0]} V{uv[1]}\n"
        file.write(s)

    file.write("\nM18")  # Stepper off
    file.write("%\n")  # End of file
    file.close()

    return 0

def to_test_gcode(file_name: str, Vs: list):
    if not h.is_structured(Vs, "(n, 3)"):
        raise ValueError("list not structured correctly")
    # boundaries = Vector3(300, 300, 400)

    # TODO
    # Make coordinates positive
    # for i in range(len(in_slice)):
    #     in_slice[i] += abs(min_coords)
    # max_coords += abs(min_coords)

    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    file.write("G28\nG21\nG90\n")  # Homing, mm unit, Absolute positioning
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")

    for v in Vs:
        s = f"G1 X{v.y} Y{v.y} Z{v.x}\n"  # Machine z is software x
        file.write(s)

    file.write("M18\n")  # Stepper off
    file.write("\n")

    file.write("%\n")  # End of file
    file.close()

    return 0

