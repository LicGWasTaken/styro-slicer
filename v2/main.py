# import alphashape
import argv
import concave_hull as ch
import gcode
import math
import numpy as np
import open3d as o3d
import prefs as prefs
import scipy as sp
import sys
import time
import trimesh
import utils as u

"""TODO
X and Y axes have averaged speed because the machine thinks they are perpendicular. Therefore, increase the speed according to pythagoras.
run_time function doesn't work at all
Instead of moving out to a fixed position, move slightly outside the block's boundary.
Scale-to-machine and -material currently update all axes. It makes more sense for them to only scale the longest axis and keep everything else proportional.
Also probably smart to remove the 2 different scaling settings and just put in a scaling factor.
Clean up output by renaming the files and moving them to a different folder.
.step Support (r = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh('orn.stp')))
slicing for linear parts
GUI
"""

# x forward, y right, z up

PCD_SIZE = 50000  # The total amount of points in the point cloud
Z_SLICE_COUNT = 50  # The amount of subsections when remeshing along the z axis
ROTATIONAL_SLICE_COUNT = 120  # The amount of subsections when subdividing rotationally
SUB_PCD_PRECISION = (
    1 / 100
)  # Factor to calculate the bounding box width compared to the total extent

def main(file_, **kwargs):
    # Define global variables within main
    global PCD_SIZE
    global Z_SLICE_COUNT
    global ROTATIONAL_SLICE_COUNT
    global SUB_PCD_PRECISION

    u.msg(f"args: {file_}, {kwargs}", "debug")
    u.msg("Loading mesh data", "process")

    # Load mesh
    in_mesh = trimesh.load_mesh(file_)
    u.msg(f"loaded mesh at {file_}", "info")

    # # For issues with incorrect unit conversions
    # mm_to_inch = True
    # if mm_to_inch:
    #     tri_mesh.apply_scale(1/25.4)

    # Get mesh boundaries
    # trimesh.bounds.oriented_bounds(tri_mesh) also rotates the mesh, which isn't helpful in this scenario
    to_origin, extents = u.axis_oriented_extents(in_mesh)
    u.msg(f"mesh extents: {in_mesh.extents}", "info")

    # Select the smallest possible material size
    if (
        "selected-material-size" in kwargs.keys()
        and kwargs["selected-material-size"] != None
    ):
        selected_material_size = kwargs["selected-material-size"]
    else:
        if "material-sizes" not in kwargs.keys():
            u.msg("no material-sizes passed, skipping process", "warning")
        else:
            # Sort the materials by volume
            sorted_sizes = sorted(kwargs["material-sizes"], key=math.prod)

            for size in sorted_sizes:
                valid = True
                for i in range(3):
                    if np.sort(extents)[i] >= sorted(size)[i]:
                        valid = False

                if valid:
                    selected_material_size = size
                    u.msg(f"selected material size {selected_material_size}", "info")
                    break

            if not valid:
                u.msg("mesh doesn't fit within available sizes", "warning")

    to_origin, extents = u.axis_oriented_extents(in_mesh)
    in_mesh = in_mesh.apply_transform(to_origin)

    # Align the mesh
    if (
        "align-part" in kwargs
        and kwargs["align-part"]
        and selected_material_size != None
    ):
        # Mesh XYZ axes
        current_axes = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]

        if "mesh-alignment" in kwargs.keys() and kwargs["mesh-alignment"] != None:
            rotation_matrix = kwargs["mesh-alignment"]

            # Convert the axes to a 4 dimensional rotation matrix
            # The appended 0 refers to translation
            rotation_matrix = np.concatenate([rotation_matrix, [[0], [0], [0]]], axis=1)
        else:
            # The index of the biggest extent to be aligned with the z axis
            idx = np.where(extents == np.sort(extents)[2])[0][0]
            if idx == 0:  # x-axis
                rotation_matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
            elif idx == 1:  # y-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
            else:  # z-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        rotation_matrix = np.append(rotation_matrix, [[0, 0, 0, 1]], axis=0)
        in_mesh = in_mesh.apply_transform(rotation_matrix)
        to_origin, extents = u.axis_oriented_extents(in_mesh)

    # Scale it according to the kwargs
    if "scale-to-material" in kwargs.keys() and kwargs["scale-to-machine"]:
        scale = kwargs["machine-size"] / extents
        in_mesh.apply_scale(scale)
        u.msg(f"applied machine scaling", "info")
    elif (
        selected_material_size != None
        and "scale-to-material" in kwargs.keys()
        and kwargs["scale-to-material"]
    ):
        scale = selected_material_size / extents
        in_mesh.apply_scale(scale)
        u.msg("applied material scaling", "info")

    to_origin, extents = u.axis_oriented_extents(in_mesh)
    in_mesh = in_mesh.apply_transform(to_origin)

    # --------------- Axisymmetrical V3 ---------------------
    u.msg("Computing axisymmetrical coordinates", "process")
    _mesh = in_mesh
    to_origin, mesh_extents = u.axis_oriented_extents(_mesh)
    _mesh.apply_transform(to_origin)
    rad = 2 * math.pi / ROTATIONAL_SLICE_COUNT
    point_count = 100
    addend = mesh_extents[2] / point_count
    out = []
    for i in range(ROTATIONAL_SLICE_COUNT):
        tmp = []
        Intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(_mesh)
        # Temporary, change this to something with extents
        starting_distance = u.magnitude([mesh_extents[0], mesh_extents[1]])
        min_distance = 0.5
        z = -mesh_extents[2] / 2
        for z_ in range(point_count):
            # Cast a ray towards the mesh to find the minimum distance
            ray_origin = [[0, starting_distance, z]]
            ray_direction = [[0, -1, 0]]
            hit, _, _ = Intersector.intersects_location(ray_origin, ray_direction)
            if len(hit) == 0:
                z += addend
                continue
            # hit = np.min(hit) # The ray also intersects the back of the mesh

            top_boundary = starting_distance
            bottom_boundary = 0
            current_distance = starting_distance
            ray_directions = [[1, 0, 0], [-1, 0, 0]]
            while (top_boundary - bottom_boundary) / 4 > min_distance:
                # Shoot a perpendicular ray
                ray_origin = [0, current_distance, z]
                ray_origins = [ray_origin, ray_origin]
                hit = Intersector.intersects_any(ray_origins, ray_directions)
                if np.any(hit):
                    bottom_boundary = current_distance
                else:
                    top_boundary = current_distance
                current_distance = top_boundary - (top_boundary - bottom_boundary) / 2
            tmp.append(np.asarray([0, current_distance, z]))
            z += addend
    
        # Rotate the mesh
        rotation_matrix = trimesh.transformations.rotation_matrix(rad, [0, 0, 1])
        _mesh.apply_transform(rotation_matrix)
        for j, p in enumerate(tmp):
            tmp[j] = u.rotate_z_rad(p, rad * i)
        out.append(tmp)

        if i + 1 < ROTATIONAL_SLICE_COUNT:
            u.msg(f"finished {i + 1} / {ROTATIONAL_SLICE_COUNT} sections", "info", "\r")
        else:
            u.msg(f"finished {i + 1} / {ROTATIONAL_SLICE_COUNT} sections", "info")
    tmp = []
    for a in out:
        for b in a:
            tmp.append(b)
    tmp = np.asarray(tmp)
    u.plot(tmp)

    return 0

if __name__ == "__main__":
    # Start timer
    timer = time.perf_counter()

    # Get the passed file and keyword arguments
    try:
        file_, kwargs = argv.get_arguments()
    except TypeError:
        u.msg(
            "TypeError from get_arguments",
            "debug",
        )
        sys.exit()

    # Run the program
    main(file_, **kwargs)

    # Stop timer and print results
    u.msg(f"time elapsed: {round(time.perf_counter() - timer, 3)}s", "debug")

