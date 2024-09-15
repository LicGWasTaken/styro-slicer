import preferences as prefs

def to_gcode(file_name, XY, UV):
    # Open the output in write mode
    file = open(prefs.OUTPUT_FOLDER_PATH + file_name + ".gcode", mode="w")
    file.write("%\n") # Start of file
    file.write("G90 G17 G21\n\n") # Absolute selection, TODO XY plane, mm unit

    file.write("G00 X0.0 Y0.0 U0.0 V0.0\n\n") # Move to starting position
    file.write("M80\n") # Wire on

    for xy, uv in zip(XY, UV):
        xy = [round(xy[0], 1), round(xy[1], 1)]
        uv = [round(uv[0], 1), round(uv[1], 1)]
        s = f"G01 X{xy[0]} Y{xy[1]} U{uv[0]} V{uv[1]}\n"
        file.write(s)

    file.write("\nG00 X0.0 Y0.0 U0.0 V0.0\n") # Move back to starting position
    file.write("M81\n\n") # Wire off

    file.write("M30\n") # End of program
    file.write("%\n") # End of file
    file.close()

    return 0

