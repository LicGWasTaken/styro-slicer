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


def sort_segments(in_slice):
    """Sort segments to form a closed loop"""
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


def redefine_segments(in_slice):
    """ignore obsolete points"""
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
        # out_slice.append(segment)

    return out_slice


def subdivide(resolution):
    pass


def decimate(resolution):
    pass


def rotation_angle(steps, deg):
    if steps == 0:
        return 0
    elif not deg:
        return np.pi / steps
    else:
        return np.pi / steps * 180 / np.pi


def slice_mesh_axisymmetric(mesh, steps, get_plane_normals):
    angle = rotation_angle(steps, deg=False)

    # Initialize required values
    out_coords = []
    out_slices = []
    out_plane_normals = []
    plane_normal = Vector3(0, 1, 0)
    plane_origin = Vector3(0, 0, 0)

    # Slice the mesh
    for i in range(steps):
        plane_normal = plane_normal.rotate_z(angle)
        if get_plane_normals:
            out_plane_normals.append(plane_normal)
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

    if get_plane_normals:
        return out_slices, out_plane_normals
    return out_slices


def get_normals(mesh, in_coords):
    out_normals = []
    vert_normals = mesh.vertex_normals

    for i in range(len(in_coords)):
        idxs = mesh.kdtree.query(in_coords[i][0].to_list(), k=2)[1]
        v = (vert_normals[idxs[0]] + vert_normals[idxs[1]]) / 2

        out_normals.insert(i, Vector3(v))

    return out_normals


def project_to_plane(in_slice, plane_offset, angle_rad):
    """Sadly, trimesh.points.project_to_plane seems to be faulty
    Therefore, I'm forced to implement it myself T-T"""
    plane_normal = Vector3(0, 1, 0).normalized().to_np_array()
    coords = [segment[0].rotate_z(-angle_rad).to_list() for segment in in_slice]
    distances = trimesh.points.point_plane_distance(
        coords, plane_normal, [0, plane_offset, 0]
    )
    coords = [point - plane_normal * distances[i] for i, point in enumerate(coords)]

    return coords


def detect_collisions(mesh, in_coords):
    for segment in in_coords:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        # intersector.intersects_location() TODO needs vector normals to work - does it though?
    return


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
        # return 3

    # Center mesh to world origin
    mesh.apply_transform(to_origin)

    # Align longest extent to z axis
    extents = np.rint(extents) + prefs.DEFAULT_OFFSET
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    max_extent_idx = np.where(extents == np.max(extents))[0][0]
    mesh.apply_transform(trimesh.geometry.align_vectors(axes[max_extent_idx], axes[2]))

    # 'Slice' mesh
    coords = []
    XY = []
    UV = []
    slices, plane_normals = slice_mesh_axisymmetric(
        mesh, prefs.STEPS, get_plane_normals=True
    )
    for i in range(len(slices)):
        slice = slices[i]
        slice = redefine_segments(slice)
        for segment in slice:
            coords.append(segment)

        plane_offset = sorted(extents)[1] / 2
        proj = project_to_plane(
            slice,
            plane_offset,
            angle_rad=rotation_angle(prefs.STEPS, deg=False) * i + 1,
        )
        for point in proj:
            point += extents / 2
            XY.append(point)

        proj = project_to_plane(
            slice,
            -plane_offset,
            angle_rad=rotation_angle(prefs.STEPS, deg=False) * i + 1,
        )
        for point in proj:
            # Make points positive
            point += extents / 2
            UV.append(point)

    # Plot points using matplotlib
    file_name = mesh_path[mesh_path.rindex("/") + 1 : mesh_path.rindex(".")]
    plotter.plot(
        lines=coords,
        # points=XY,
        # file_name=file_name,
        line_color="purple",
        line_marker="+",
        point_color="blue",
        point_marker=".",
        # subplots=True,
    )

    gcode.to_test_gcode("unnamed", XY)

    print(f"time elapsed: {round(time.perf_counter() - timer, 3)}s")
    return 0


if __name__ == "__main__":
    main()
    print()
