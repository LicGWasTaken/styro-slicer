import sys, os
import numpy as np
import trimesh
import helpers, plotter

DEFAULT_MESH_PATH = '/workspace/obj/cube.stl'
MESH_FOLDER_PATH = '/workspace/obj/'
SUPPORTED_FORMATS = ['.stl']
VALID_ARGVS = []
MATERIAL_SIZES = [[60, 20, 100], [20, 20, 20]]

def check_arguments():
    print('checking command line arguments...')

    # Check for proper usage
    argv_length = len(sys.argv)
    if argv_length < 2:
        helpers.print_bp('no arguments passed, running with default settings')
        return 1
    for arg in sys.argv: # Look for a valid path
        if (os.path.isfile(arg) or os.path.isfile(MESH_FOLDER_PATH + arg)) and not arg == 'styro-slicer.py':
            for format in SUPPORTED_FORMATS:
                if format not in arg:
                    helpers.print_error(arg + ': invalid file format')
                    return 2
            break
    else:
        helpers.print_bp('no valid file path passed, using default mesh')
        return 1
    for arg in sys.argv:
        if arg not in VALID_ARGVS and not (arg == 'styro-slicer.py' or os.path.isfile(arg) or os.path.isfile(MESH_FOLDER_PATH + arg)):
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
        mesh_path = sys.argv[1] if os.path.isfile(sys.argv[1]) else MESH_FOLDER_PATH + sys.argv[1]

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)  
    if not mesh.is_watertight:
        helpers.print_error('mesh not watertight')
        return 2
    
    # Get mesh boundaries
    oriented_bounds = trimesh.bounds.oriented_bounds(mesh)

    # Check if mesh fits within available sizes
    extents = oriented_bounds[1]
    sizes = sorted(MATERIAL_SIZES, key=lambda x: np.prod(x))
    for size in sizes:
        if all(s < l for s, l in zip(sorted(extents, reverse=True), sorted(size, reverse=True))):
            helpers.print_bp('size ' + str(size) + ' fits')
            break
    else:
        helpers.print_error('mesh does not fit within available material sizes')
        # Slice mesh into submeshes TODO
        return 3
        
    # Center mesh to world origin
    to_origin = oriented_bounds[0]
    mesh.apply_transform(to_origin)

    # 'Slice' mesh
    plane_normal = (0, 0, 1)
    lines = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin=(0, 0, 1))
    plotter.plot_lines(lines, fmt='bo')

    return

if __name__ == '__main__':
    main()
    print()