import preferences as prefs
from vector import Vector3
import plotter


def to_gcode(file_name, XY, UV):
    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    file.write("G28\nG21\nG90\n")  # Homing, mm unit, Absolute positioning
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")

    for xy, uv in zip(XY, UV):
        xy = [round(xy[0], 1), round(xy[2], 1)]
        uv = [round(uv[0], 1), round(uv[2], 1)]
        s = f"G1 X{xy[0]} Y{xy[1]} U{uv[0]} V{uv[1]}\n"
        file.write(s)

    file.write("\nM18")  # Stepper off
    file.write("%\n")  # End of file
    file.close()

    return 0


def to_test_gcode(file_name, XY):
    # Define printer boundaries
    boundaries = [300, 300, 400]

    # Scale down the coordinates TODO: this doesn't assure that the x and y coordinates are below 300
    maximum = 0
    for xy in XY:
        if max(xy) > maximum:
            maximum = max(xy)

    mult = max(boundaries) / maximum
    for i in range(len(XY)):
        XY[i] = (Vector3(XY[i]) * mult).to_list()

    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n")  # Start of file
    file.write("G28\nG21\nG90\n")  # Homing, mm unit, Absolute positioning
    file.write("M302 S0\n")  # Always allow extrusion
    file.write("\n")

    for xy in XY:
        xy = [round(xy[0], 1), round(xy[2], 1)]
        s = f"G1 X{xy[1]} Y{xy[1]} Z{xy[0]}\n"  # Machine z is software x
        file.write(s)

    file.write("M18\n")  # Stepper off
    file.write("\n")

    file.write("%\n")  # End of file
    file.close()

    return 0
