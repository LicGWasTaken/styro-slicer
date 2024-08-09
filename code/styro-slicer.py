import pymesh
import sys
import os

from mesh_cleanup import cleanup
import helpers

DEFAULT_MESH_PATH = '/workspace/obj/low.stl'
SUPPORTED_FORMATS = ['.stl']
VALID_ARGVS = ['cleanup']

def check_arguments():
    print('checking command line arguments...')

    # Check for proper usage
    argv_length = len(sys.argv)
    if argv_length < 2:
        helpers.print_bp('no arguments passed, running with default settings')
        return 1
    for arg in sys.argv: # Look for a valid path
        if os.path.isfile(arg) and not arg == 'styro-slicer.py':
            for format in SUPPORTED_FORMATS:
                if format not in arg:
                    helpers.print_error(arg + ': invalid file format')
                    return 2
            valid_path = True
            break
        valid_path = False
    if not valid_path:
        helpers.print_bp('no valid file path passed, using default mesh')
        return 1
    for arg in sys.argv:
        if arg not in VALID_ARGVS and not (arg == 'styro-slicer.py' or os.path.isfile(arg)):
            helpers.print_error('invalid arguments, check README for usage')
            return 2
    
    return 0

def main():
    # Check command line arguments
    var = check_arguments()
    if var == 2:
        return 1
    elif var == 1:
        mesh_path = DEFAULT_MESH_PATH
    else:
        mesh_path = sys.argv[1]

    # Load the passed mesh file
    print("loading mesh at " + mesh_path + "...")
    mesh = pymesh.load_mesh(mesh_path)

# Cleanup mesh
    if 'cleanup' in sys.argv:
        cleanup(mesh=mesh)

    return

if __name__ == '__main__':
    main()
    print()