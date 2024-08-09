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
        helpers.print_error('no arguments passed, running with default settings')
        return 1
    for arg in sys.argv:
        if arg not in VALID_ARGVS and not (arg == sys.argv[0] or arg == sys.argv[1]):
            helpers.print_error('invalid argument, check README for usage')
            return 2
    for format in SUPPORTED_FORMATS:
        if format not in sys.argv[1]:
            helpers.print_error('invalid file format')
            return 2
    if not os.path.isfile(sys.argv[1]):
        helpers.print_error('invalid file path')
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