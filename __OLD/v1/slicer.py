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

def sort_segments(in_slice: list, in_face_slices: list):
    """Sort segments to form a closed loop"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")

    out_slice = []
    out_face_slices = []
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
        tmp = (
            in_slice.index([key, v])
            if [key, v] in in_slice
            else in_slice.index([v, key])
        )
        out_face_slices.append(in_face_slices[tmp])
        key = v
        idx = d[key][0] if idx != d[key][0] else d[key][1]

    return out_slice, out_face_slices

def redefine_segments(in_slice: list, in_face_indeces: list):
    """ignore obsolete points"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")

    out_slice = []
    out_face_indeces = []

    # Make sure the input is sorted
    in_slice, in_face_indeces = sort_segments(in_slice, in_face_indeces)

    # Delete consecutive straight segments TODO: could possibly be optimized using normal vectors
    prev_segment = in_slice[0]
    prev_v = (prev_segment[1] - prev_segment[0]).normalized()
    current_face_normal_index = in_face_indeces[0]
    for i in range(len(in_slice))[1:]:
        # breakpoint()
        segment = in_slice[i]
        v = (segment[1] - segment[0]).normalized()

        # If the normalized vectors align, merge the segments
        if prev_v == v:
            prev_segment = [prev_segment[0], segment[1]]
        else:
            if prev_segment not in out_slice and i != 1:
                out_slice.append(prev_segment)
                out_face_indeces.append(current_face_normal_index)

            prev_segment = segment
            current_face_normal_index = in_face_indeces[i]

        prev_v = v

    # Close the loop
    segment = in_slice[0]
    v = (segment[1] - segment[0]).normalized()
    if prev_v == v:
        out_slice.append([prev_segment[0], segment[1]])
    else:
        out_slice.append(prev_segment)
    out_face_indeces.append(current_face_normal_index)

    # breakpoint()
    # for out_line in out_slice:
    #     for i, in_line in enumerate(in_slice):
    #         if in_line[1] == out_line[0]:
    #             out_face_indeces.append(in_face_indeces[i])
    #             break

    # for out_line, out_index in zip(out_slice, out_face_indeces):
    #     print(out_line, out_index)
    return out_slice, out_face_indeces

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
    out_face_indeces = []
    plane_normal = Vector3(0, 1, 0)
    plane_origin = Vector3(0, 0, 0)

    # Slice the mesh
    for i in range(steps):
        slice, face_indeces = trimesh.intersections.mesh_plane(
            mesh, plane_normal.to_list(), plane_origin.to_list(), return_faces=True
        )

        # Reformat output as Vector3s
        tmp = []
        for segment in slice:
            v3_segment = [Vector3(segment[0]), Vector3(segment[1])]
            out_coords.append(v3_segment)
            tmp.append(v3_segment)
        out_slices.append(tmp)
        out_face_indeces.append(face_indeces)

        plane_normal = plane_normal.rotate_z(angle)

    # Print results
    h.print_bp(f"generated {np.size(out_coords)} points")
    # for i in out_face_indeces:
    #     for j in i:
    #         print(j, mesh.vertices[mesh.faces[j]])
    return out_slices, out_face_indeces

def get_normals(mesh: trimesh.base.Trimesh, in_lines: list, in_face_indeces: list):
    if not h.is_structured(in_lines, "(n, 2)"):
        raise ValueError("list not structured correctly")

    out_normals = []

    for i, idx in enumerate(in_face_indeces):
        n1 = mesh.face_normals[idx]

        # print(n1, Vector3(n1), Vector3(n1) == Vector3.zero())

        if i == 0:
            n2 = mesh.face_normals[in_face_indeces[len(in_face_indeces) - 1]]
        else:
            n2 = mesh.face_normals[in_face_indeces[i - 1]]

        # if i == 0:
        #     print(Vector3(n1), Vector3(n2), "/", in_face_indeces[i], in_face_indeces[i + 1])
        normal = ((Vector3(n1) + Vector3(n2)) / 2).normalized()
        out_normals.append(normal)

    return out_normals

def project_to_plane(
    in_slice: list, in_normals: list, plane_offset: float, angle_rad: float
):
    """Sadly, trimesh.points.project_to_plane seems to be faulty
    Therefore, I'm forced to implement it myself T-T"""
    if not h.is_structured(in_slice, "(n, 2)"):
        raise ValueError("list not structured correctly")
    # TODO: Update to use Vector2s
    y_axis = Vector3(0, 1, 0)
    x_axis = Vector3(1, 0, 0)

    out_points = []
    plane_origin = [0, plane_offset, 0]
    plane_normal = y_axis.normalized().to_np_array()

    # ------- OLD --------
    old_points = [line[0].rotate_z(-angle_rad).to_list() for line in in_slice]
    distances = trimesh.points.point_plane_distance(
        old_points, plane_normal, plane_origin
    )
    old_points = [
        Vector3(point - plane_normal * distances[i])
        for i, point in enumerate(old_points)
    ]
    # --------------------
    # # TODO make projections follow normal vector
    points = [line[0].rotate_z(-angle_rad) for line in in_slice]
    normals = [n.rotate_z(-angle_rad).normalized() for n in in_normals]
    lines = [[], []]
    for i, (point, normal) in enumerate(zip(points, normals)):
        sign = np.sign(plane_offset)
        # Prevent the cross product from returning zero
        if x_axis == normal:
            dir_v = Vector3(0, 1, 0)
        else:
            dir_v = sign * abs(h.cross(x_axis, normal).normalized())

        lines[0].insert(i, point.to_list())
        lines[1].insert(i, (point + dir_v * 2 * abs(plane_offset)).to_list())

    intersections, valid = trimesh.intersections.plane_lines(
        plane_origin, plane_normal, lines
    )
    out_points = (Vector3(inter) for i, inter in enumerate(intersections))

    if not all(valid):
        h.print_error("invalid intersections")
        error_indeces = []
        for i in range(len(valid)):
            if not valid[i]:
                error_indeces.append(i)
        # return None

    return out_points

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
    if all(
        extent < bound for extent, bound in zip(extents.to_list(), bounds.to_list())
    ):
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
    slices, face_indeces_list = slice_mesh_axisymmetric(mesh, prefs.STEPS)
    for i in range(len(slices)):
        slice, face_indeces = redefine_segments(slices[i], face_indeces_list[i])
        slice = scale_to_fit(slice, Vector3(300, 300, 400), Vector3(extents))

        for line in slice:
            coords.append(line)
        normals = get_normals(mesh, slice, face_indeces)
        angle_rad = rotation_angle(prefs.STEPS, deg=False) * i
        plane_offset = sorted(extents)[1] / 2

        # try:
        proj = project_to_plane(slice, normals, plane_offset, angle_rad)
        for point in proj:
            XYs.append(point)

        proj = project_to_plane(slice, normals, -plane_offset, angle_rad)
        for point in proj:
            UVs.append(point)

        # except Exception(ValueError):
        #     h.print_error("projection returned an invalid value")
        #     return 1

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

