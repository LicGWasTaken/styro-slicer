import pymesh
import sys

from mesh_cleanup import cleanup

default_mesh_path = '/workspace/obj/low.stl'
supported_formats = ['.stl']

def main():
    # Get mesh path
    mesh_path = default_mesh_path
    if len(sys.argv) != 1:
        for format in supported_formats:
            if format in sys.argv[1]:
                mesh_path = sys.argv[1]
            else:
                print('currently unsupported file format')
                return 1

    # Load the passed mesh file
    print("loading mesh at " + mesh_path + "...")
    mesh = pymesh.load_mesh(mesh_path)

    if 'cleanup' in sys.argv:
        # Cleanup mesh
        cleanup(mesh=mesh)

    return

if __name__ == '__main__':
    print()
    main()