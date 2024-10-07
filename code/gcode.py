import preferences as prefs
from vector import Vector3
import helpers as h
    
def to_gcode(file_name: str, XYs: list, UVs: list):
    if not h.is_structured(XYs, "(n, 1)") or not h.is_structured(UVs, "(n, 1)"):
        raise ValueError("list not structured correctly")

    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    file.write("G28\nG21\nG90\n")  # Homing, mm unit, Absolute positioning
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")

    for xy, uv in zip(XYs, UVs):
        xy = [round(xy.x, 1), round(xy.z, 1)]
        uv = [round(uv.x, 1), round(uv.z, 1)]
        s = f"G1 X{xy[0]} Y{xy[1]} U{uv[0]} V{uv[1]}\n"
        file.write(s)

    file.write("\nM18")  # Stepper off
    file.write("%\n")  # End of file
    file.close()

    return 0

def to_test_gcode(file_name: str, points: list):
    """Generate gcode fit for the test environment, using only XYZ and
    with the following bounds: (300, 300, 400)"""
    if not h.is_structured(points, "(n, 3)"):
        raise ValueError("list not structured correctly")

    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    file.write("G28\nG21\nG90\n")  # Homing, mm unit, Absolute positioning
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")

    for v in points:
        s = f"G1 X{v.z} Y{v.z} Z{v.x}\n"  # Machine z is software x
        file.write(s)

    file.write("M18\n")  # Stepper off
    file.write("\n")

    file.write("%\n")  # End of file
    file.close()

    return 0

