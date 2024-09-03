import sys, os
import numpy as np
import trimesh

from vector import Vector3
import helpers, plotter

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
        helpers.print_bp("no arguments passed, running with default settings")
        return 1
    # Look for a valid file path
    for arg in sys.argv:
        if (
            os.path.isfile(arg) or os.path.isfile(MESH_FOLDER_PATH + arg)
        ) and not arg == "styro-slicer.py":
            for format in SUPPORTED_FORMATS:
                if format not in arg:
                    helpers.print_error(arg + ": invalid file format")
                    return 2
            break
    else:
        helpers.print_bp("no valid file path passed, using default mesh")
        return 1
    # Make sure all arguments are allowed
    for arg in sys.argv:
        if arg not in VALID_ARGVS and not (
            arg == "styro-slicer.py"
            or os.path.isfile(arg)
            or os.path.isfile(MESH_FOLDER_PATH + arg)
        ):
            helpers.print_error("invalid arguments, check README for usage")
            return 2
    return 0

def redefine_coords(in_coords):
    """Make points equidistant"""
    pass

    length = sum(v.magnitude() for sub_list in in_coords for v in sub_list)
    vertex_count = np.size(in_coords) / 6  # 3 coordinates per vertex, 2 verteces per line
    dist = length / vertex_count

    # out_coords = []
    # i = 0
    # prev = in_coords[i][0]
    # while True:
    #     dir_vect = in_coords[i][1] - in_coords[i][0]

    #     # Get equidistant point along direction
    #     factor = dist / helpers.mag(dir_vect)
    #     new = prev + factor * dir_vect

    #     # If past the next point, adjust to follow the outline correctly
    #     # origin + dir_vect * x = new
    #     x = (new[0] - in_coords[0]) / dir_vect[0]

    #     pct = np.divide((new - in_coords[i][0]), dir_vect)  
    #     pct[np.isnan(pct)] = 0 # Replace NaN with 0s to still work when coordinate of dir_vect is 0
    #     print(pct)

        # if pct > 1:  # TODO this is an list and i need it to be a scalar
        #     i += 1
        #     if i > size:
        #         return out_coords
        #     tmp_dist = pct - 1 * helpers.mag(dir_vect)
        #     dir_vect = in_coords[i][1] - in_coords[i][0]
        #     factor = tmp_dist / helpers.mag(dir_vect)
        #     new = in_coords[i][0] + factor * dir_vect

        # arr = np.list([prev, new])
        # np.append(out_coords, arr)
        # prev = new

def subdivide(resolution):
    pass

def main():
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
        helpers.print_error("mesh not watertight")
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
            helpers.print_bp("size " + str(size) + " fits")
            break
    else:
        helpers.print_error("mesh does not fit within available material sizes")
        # return 3

    # Center mesh to world origin
    to_origin = oriented_bounds[0]
    mesh.apply_transform(to_origin)

    # Align longest extent to z axis
    extents = np.rint(extents) + DEFAULT_OFFSET
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    max_extent_idx = np.where(extents == np.max(extents))[0][0]
    mesh.apply_transform(
        trimesh.geometry.align_vectors(axes[max_extent_idx], axes[2])
    )

    # 'Slice' mesh
    steps = 3
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
    plotter.plot_lines(coords, color="purple", marker="+")
    return 0

if __name__ == "__main__":
    main()
    print()

