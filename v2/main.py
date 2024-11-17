import argv
import numpy as np
import open3d as o3d
import sys
import time
import trimesh
import utils as u

args = None

def main(file_, **kwargs):
    u.msg(f"args: {file_}, {kwargs}", "debug")
    u.msg("Running main", "process")

    return 0

if __name__ == "__main__":
    # Start timer
    timer = time.perf_counter()

    # Get the passed file and keyword arguments
    try:
        file_, kwargs = argv.get_arguments()
    except TypeError:
        sys.exit()

    # Run the program
    main(file_, **kwargs)

    # Stop timer and print results
    u.msg(f"time elapsed: {round(time.perf_counter() - timer, 3)}s", "debug")

