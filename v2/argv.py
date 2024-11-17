import argparse
import ast
import json
import os
import prefs
import utils as u

# TODO 1: Add the saved settings to the passed argv dictionary

def get_arguments():
    """Get the file name/path and additional kwargs from the user"""
    u.msg("Getting passed arguments", "process")

    parser = argparse.ArgumentParser()

    # Define a positional argument
    parser.add_argument(
        "file_", type=str, help="file path or file name if within the default folder"
    )
    # Define the optional kwargs argument
    parser.add_argument("--kwargs", type=str, help="additional keyword arguments")

    args = parser.parse_args()

    # Convert string to dictionary and ensure safe evaluation
    if args.kwargs:
        try:
            kwargs = ast.literal_eval(args.kwargs)
            if not isinstance(kwargs, dict):
                u.msg("kwargs is not a dictionary", "error")
                return 1
        except (ValueError, SyntaxError):
            u.msg("Invalid kwargs format", "error")
            return 2
    else:
        kwargs = {}

    # Assure that the arguments are valid
    if not check_file_validity(args.file_):
        u.msg("invalid file path", "error")
        return 3

    if not check_kwargs_validity(kwargs):
        u.msg("invalid arguments", "error")
        return 4
    
    # Get the file path
    if os.path.isfile(args.file_):
        path = args.file_
    else:
        path = prefs.MESH_FOLDER + args.file_

    # TODO 1

    u.msg(f"{len(kwargs)} keyword arguments passed", "info")
    return path, kwargs

def check_file_validity(file_: str):
    u.msg("Checking file path", "process")

    # Make sure the string is a valid file path
    if not os.path.isfile(file_) and not os.path.isfile(prefs.MESH_FOLDER + file_):
        return 0
    else:
        if os.path.isfile(file_):
            path = file_
        else:
            path = prefs.MESH_FOLDER + file_

    # Make sure the format is supported
    tmp = 0
    for format in prefs.SUPPORTED_FORMATS:
        if format in file_:
            tmp = 1
            break

    if not tmp:
        u.msg("unsupported file format", "warning")
        return 0

    u.msg(f"file found at {path}", "info")
    return 1

def check_kwargs_validity(kwargs: dict):
    u.msg("Checking command line arguments", "process")

    # Return if no arguments were passed
    if len(kwargs) < 1:
        u.msg("no keyword arguments passed", "info")
        return 1

    # Make sure all arguments are allowed
    for key in kwargs.keys():
        value = kwargs[key]

        if key not in prefs.VALID_ARGVS.keys():
            u.msg(f"{key}: invalid argument", "error")
            return 0

        type_ = prefs.VALID_ARGVS[key]
        if isinstance(type_, type):
            try:
                value = type_(value)
            except ValueError:
                u.msg(f"{key}: invalid type", "error")
                return 0
        else:
            # Make sure the list structure is valid
            if not u.is_structured(value, type_):
                u.msg("invalid list structure", "error")
                return 0
    
    # Load settings json in read mode
    with open(prefs.JSON_FOLDER + "settings.json", "r") as file:
        settings = json.load(file)

    # Handle argument specific conditions
    if "projection-axis" in kwargs.keys():
        valid_values = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if kwargs["projection-axis"] not in valid_values:
            u.msg("invalid projection-axis", "error")
            return 0
        
    if "selected-material-size" in kwargs.keys():
        if kwargs["selected-material-size"] not in settings["material-sizes"]:
            u.msg("invalid selected-material-size", "error")
            return 0
        
        if "autoselect-material-size" in kwargs.keys():
            kwargs["autoselect-material-size"] = False

    return 1

