import sys
import os
import numpy as np
import trimesh
import time

from vector import Vector3
import helpers as h
import plotter
import preferences as prefs
import gcode

def check_arguments():
    print("checking command line arguments...")

    # Check for proper usage
    argv_length = len(sys.argv)
    if argv_length < 2:
        h.print_bp("no arguments passed, running with default settings")
        return 1
    # Look for a valid file path TODO: this throws an error if it finds the same file with another format
    for arg in sys.argv:
        if (
            os.path.isfile(arg) or os.path.isfile(prefs.MESH_FOLDER_PATH + arg)
        ) and not arg == sys.argv[0]:
            for format in prefs.SUPPORTED_FORMATS:
                if format not in arg:
                    h.print_error(arg + ": invalid file format")
                    return 2
            break
    else:
        h.print_bp("no valid file path passed, using default mesh")
        return 1
    # Make sure all arguments are allowed
    for arg in sys.argv:
        if arg not in prefs.VALID_ARGVS and not (
            arg == sys.argv[0]
            or os.path.isfile(arg)
            or os.path.isfile(prefs.MESH_FOLDER_PATH + arg)
        ):
            h.print_error("invalid arguments, check README for usage")
            return 2
    return 0

def sort_segments(in_slice: list):
    """Sort segments to form a closed loop"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")

    out_slice = []
    d = {}
    for segment in in_slice:
        for v in segment:
            if v not in d:
                d[v] = [in_slice.index(segment), None]
            else:
                d[v] = [d[v][0], in_slice.index(segment)]

    l = list(d)  # Instantiate dict into list to enable indexing
    key = l[0]
    idx = d[key][0]
    for i in range(len(d.keys())):
        try:
            v = in_slice[idx][0] if key != in_slice[idx][0] else in_slice[idx][1]
        except TypeError:
            h.print_error(
                "TypeError in sort_segments, most likely a rounding error. Try decreasing the decimals in of the Vector3 class."
            )
            break
        out_slice.append([key, v])
        key = v
        idx = d[key][0] if idx != d[key][0] else d[key][1]

    return out_slice

def redefine_segments(in_slice: list):
    """ignore obsolete points"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")
        return 1

    out_slice = []

    # Make sure the input is sorted
    in_slice = sort_segments(in_slice)

    # Delete consecutive straight segments
    prev_segment = in_slice[0]
    prev_v = (prev_segment[1] - prev_segment[0]).normalized()
    for i in range(len(in_slice))[1:]:
        segment = in_slice[i]
        v = (segment[1] - segment[0]).normalized()

        # If the normalized vectors align, merge the segments
        if prev_v == v:
            prev_segment = [prev_segment[0], segment[1]]
        else:
            if prev_segment not in out_slice and i != 1:
                out_slice.append(prev_segment)
            prev_segment = segment

        prev_v = v

    # Close the loop
    segment = in_slice[0]
    v = (segment[1] - segment[0]).normalized()
    if prev_v == v:
        out_slice.append([prev_segment[0], segment[1]])
    else:
        out_slice.append(prev_segment)

    return out_slice

def subdivide(resolution):
    pass

def decimate(resolution):
    pass

def rotation_angle(steps: int, deg: float):
    if steps == 0:
        return 0
    elif not deg:
        return np.pi / steps
    else:
        return np.pi / steps * 180 / np.pi

def slice_mesh_axisymmetric(mesh: trimesh.base.Trimesh, steps: int):
    angle = rotation_angle(steps, deg=False)

    # Initialize required values
    out_coords = []
    out_slices = []
    plane_normal = Vector3(0, 1, 0)
    plane_origin = Vector3(0, 0, 0)

    # Slice the mesh
    for i in range(steps):
        plane_normal = plane_normal.rotate_z(angle)
        slice = trimesh.intersections.mesh_plane(
            mesh, plane_normal.to_list(), plane_origin.to_list()
        )

        # Reformat output as Vector3s
        tmp = []
        for segment in slice:
            v3_segment = [Vector3(segment[0]), Vector3(segment[1])]
            out_coords.append(v3_segment)
            tmp.append(v3_segment)
        out_slices.append(tmp)

    # Print results
    h.print_bp(f"generated {np.size(out_coords)} points")

    return out_slices

def get_normals(mesh: trimesh.base.Trimesh, in_coords: list):
    if not h.is_structured(in_coords, "(n, 2)"):
        raise ValueError("list not structured correctly")
        return 1

    out_normals = []
    vert_normals = mesh.vertex_normals
    for i in range(len(in_coords)):
        idxs = mesh.kdtree.query(in_coords[i][0].to_list(), k=2)[1]
        v = (vert_normals[idxs[0]] + vert_normals[idxs[1]]) / 2

        out_normals.insert(i, Vector3(v))

    return out_normals

def project_to_plane(in_slice: list, in_normals: list, plane_offset: float, angle_rad: float):
    """Sadly, trimesh.points.project_to_plane seems to be faulty
    Therefore, I'm forced to implement it myself T-T"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")
    # TODO: Update to use Vector2s
    # TODO allow different normals, requires changes in slice rotation

    out_points = []
    plane_origin = [0, plane_offset, 0]
    plane_normal =  Vector3(0, 1, 0).normalized().to_np_array() 

    # ------- OLD --------
    old_points = [segment[0].rotate_z(-angle_rad).to_list() for segment in in_slice]
    distances = trimesh.points.point_plane_distance(
        old_points, plane_normal, plane_origin
    )
    old_points = [
        Vector3(point - plane_normal * distances[i]) for i, point in enumerate(old_points)
    ]
    # --------------------
    # TODO make projections follow normal vector
    
    # x = trimesh.intersections.plane_lines(
    #     [0, 0, 0], [0, 0, 1], [[0, -1, -1], [1, 2, 2]]
    # )

    return old_points

def scale_to_fit(in_slice: list, bounds: Vector3, extents: Vector3):
    """Scale points down to fit within the boundaries"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")
    
    out_slice = []

    # TODO: add an option to do this as an argument that automatically takes the smallest given block size

    # max_coords = Vector3.zero()
    # for l in in_slice:
    #     max_coords.x = max(max_coords.x, l[0].x, l[1].x)
    #     max_coords.y = max(max_coords.y, l[0].y, l[1].y)
    #     max_coords.z = max(max_coords.z, l[0].z, l[1].z)
    
    # Scale down the coordinates
    if all(extent < bound for extent, bound in zip(extents.to_list(), bounds.to_list())):
        return in_slice

    f = 1 / (extents / 2 / bounds).max()
    for l in in_slice:
        out_slice.append([l[0] * f, l[1] * f])

    return out_slice

def main():
    timer = time.perf_counter()

    # Check command line arguments
    var = check_arguments()
    if var == 2:
        return 1
    elif var == 1:
        mesh_path = prefs.DEFAULT_MESH_PATH
    else:
        mesh_path = (
            sys.argv[1]
            if os.path.isfile(sys.argv[1])
            else prefs.MESH_FOLDER_PATH + sys.argv[1]
        )

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)
    if not mesh.is_watertight:
        h.print_error("mesh not watertight")
        return 2

    # Get mesh boundaries
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)

    # Check if mesh fits within available sizes
    sizes = sorted(prefs.MATERIAL_SIZES, key=lambda x: np.prod(x))
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
        h.print_bp(f"extents: {extents.tolist()}")
        # return 3

    # Center mesh to world origin
    mesh.apply_transform(to_origin)

    # Align longest extent to z axis TODO: also align the second axis
    extents = np.rint(extents) + prefs.DEFAULT_OFFSET
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    max_extent_idx = np.where(extents == np.max(extents))[0][0]
    mesh.apply_transform(trimesh.geometry.align_vectors(axes[max_extent_idx], axes[2]))

    # 'Slice' mesh
    coords = []
    XYs = []
    UVs = []
    slices = slice_mesh_axisymmetric(mesh, prefs.STEPS)
    for i in range(len(slices)):
        slice = slices[i]
        slice = redefine_segments(slice)
        slice = scale_to_fit(slice, Vector3(300, 300, 400), Vector3(extents))

        for line in slice:
            coords.append(line)

        normals = get_normals(mesh, coords)
        angle_rad = rotation_angle(prefs.STEPS, deg=False) * i
        plane_offset = sorted(extents)[1] / 2

        try:
            proj = project_to_plane(
                slice,
                normals,
                plane_offset,
                angle_rad
            )
            for point in proj:
                XYs.append(point)

            proj = project_to_plane(
                slice,
                normals,
                -plane_offset,
                angle_rad
            )
            for point in proj:
                UVs.append(point)
        except:
            h.print_error("error projecting points")
            return 1

        # # Make coordinates positive
        # for i in range(len(slice)):
        #     for j in range(2):
        #         slice[i][j] += abs(h.min_line_values(slice))

    # Plot points using matplotlib
    file_name = mesh_path[mesh_path.rindex("/") + 1 : mesh_path.rindex(".")]
    plotter.plot(
        lines=coords,
        points=XYs,
        # file_name=file_name,
        line_color="purple",
        line_marker="+",
        point_color="blue",
        point_marker=".",
        # subplots=True,
    )

    gcode.to_test_gcode("test", XYs)
    gcode.to_gcode("unnamed", XYs, UVs)

    print(f"time elapsed: {round(time.perf_counter() - timer, 3)}s")
    return 0

if __name__ == "__main__":
    main()
    print()

