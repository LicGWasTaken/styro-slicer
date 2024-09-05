import sys, os
import numpy as np
import trimesh
import time

from vector import Vector3
import helpers as h, plotter

MESH_FOLDER_PATH = "/workspace/obj/"
DEFAULT_MESH_PATH = MESH_FOLDER_PATH + "cube.stl"
SUPPORTED_FORMATS = [".stl"]
VALID_ARGVS = ["offset", "mat-size"]
MATERIAL_SIZES = [[60, 20, 100], [20, 30, 40]]
DEFAULT_OFFSET = 5

def check_arguments():
    print("checking command line arguments...")

    # Check for proper usage
    argv_length = len(sys.argv)
    if argv_length < 2:
        h.print_bp("no arguments passed, running with default settings")
        return 1
    # Look for a valid file path
    for arg in sys.argv:
        if (
            os.path.isfile(arg) or os.path.isfile(MESH_FOLDER_PATH + arg)
        ) and not arg == "styro-slicer.py":
            for format in SUPPORTED_FORMATS:
                if format not in arg:
                    h.print_error(arg + ": invalid file format")
                    return 2
            break
    else:
        h.print_bp("no valid file path passed, using default mesh")
        return 1
    # Make sure all arguments are allowed
    for arg in sys.argv:
        if arg not in VALID_ARGVS and not (
            arg == "styro-slicer.py"
            or os.path.isfile(arg)
            or os.path.isfile(MESH_FOLDER_PATH + arg)
        ):
            h.print_error("invalid arguments, check README for usage")
            return 2
    return 0

def sort_coords(in_coords):
    """Sort lines to form a closed loop"""
    out_coords = []
    d = {}
    for line in in_coords:
        for v in line:
            if v not in d:
                d[v] = [in_coords.index(line), None]
            else:
                d[v] = [d[v][0], in_coords.index(line)]
    
    l = list(d) # Instantiate dict into list to enable indexing
    key = l[0]
    idx = d[key][0]
    for i in range(len(d.keys())):
        v = in_coords[idx][0] if key != in_coords[idx][0] else in_coords[idx][1]
        out_coords.append([key, v])
        key = v
        idx = d[key][0] if idx != d[key][0] else d[key][1]

    return out_coords
    
def redefine_coords(in_coords):
    """Make points equidistant"""
    out_coords = []
    in_coords = sort_coords(in_coords)
    length = sum((line[1] - line[0]).magnitude() for line in in_coords)
    vertex_count = np.size(in_coords) / 2 # 2 verteces per line
    dist = length / vertex_count

    i = 0
    prev = in_coords[i][0]
    while True:
        # Get equidistant point along direction
        dir_vect = in_coords[i][1] - in_coords[i][0]
        new = prev + dir_vect * (dist / dir_vect.magnitude())

        # If past the next point, adjust to follow the outline correctly 
        scalar = (new - in_coords[i][0]).magnitude() / dir_vect.magnitude() # origin + dir_vect * scalar = new
        scalar /= dist
        
        if scalar > 1: 
            i += 1

            # Exit condition
            if i >= vertex_count:
                return out_coords
            
            tmp_dist = (scalar - 1) * dir_vect.magnitude()
            tmp_dir_vect = in_coords[i][1] - in_coords[i][0]

            # Keep sharp edges
            if h.angle_between_vectors(tmp_dir_vect, dir_vect, deg=True) >= 45:
                new = in_coords[i][0]
            else:
                new = in_coords[i][0] + dir_vect * (tmp_dist / dir_vect.magnitude())

        out_coords.append([prev, new])
        prev = new

def subdivide(resolution):
    pass

def main():
    timer = time.perf_counter()

    # Check command line arguments
    var = check_arguments()
    if var == 2:
        return 1
    elif var == 1:
        mesh_path = DEFAULT_MESH_PATH
    else:
        mesh_path = (
            sys.argv[1]
            if os.path.isfile(sys.argv[1])
            else MESH_FOLDER_PATH + sys.argv[1]
        )

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)
    if not mesh.is_watertight:
        h.print_error("mesh not watertight")
        return 2

    # Get mesh boundaries
    oriented_bounds = trimesh.bounds.oriented_bounds(mesh)

    # Check if mesh fits within available sizes
    extents = oriented_bounds[1]
    sizes = sorted(MATERIAL_SIZES, key=lambda x: np.prod(x))
    for size in sizes:
        # Sort extents and sizes descending and compare them
        if all(
            s < l
            for s, l in zip(sorted(extents, reverse=True), sorted(size, reverse=True))
        ):
            h.print_bp("size " + str(size) + " fits")
            break
    else:
        h.print_error("mesh does not fit within available material sizes")
        # return 3

    # Center mesh to world origin
    to_origin = oriented_bounds[0]
    mesh.apply_transform(to_origin)

    # Align longest extent to z axis
    extents = np.rint(extents) + DEFAULT_OFFSET
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    max_extent_idx = np.where(extents == np.max(extents))[0][0]
    mesh.apply_transform(trimesh.geometry.align_vectors(axes[max_extent_idx], axes[2]))

    # 'Slice' mesh
    steps = 360
    angle = 0 if steps == 0 else np.pi / steps
    lines = []
    plane_normal = Vector3(0, 1, 0)
    plane_origin = Vector3(0, 0, 0)
    for i in range(steps):
        plane_normal = plane_normal.rotate_z(angle)
        coords = trimesh.intersections.mesh_plane(
            mesh, plane_normal.list(), plane_origin.list()
        )
        tmp = []
        for line in coords:
            tmp.append([Vector3(line[0]), Vector3(line[1])])
        coords = redefine_coords(tmp)
        # coords = tmp
        for line in coords:
            lines.append(line)
    h.print_bp(f'generated {np.size(lines)} points')
    # plotter.plot_lines(lines, color="purple", marker="+")

    print(f'time elapsed: {round(time.perf_counter() - timer, 3)}s')
    return 0

if __name__ == "__main__":
    main()
    print()

