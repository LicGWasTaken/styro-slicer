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

# x forward, y right, z up

PCD_SIZE = 200000  # The total amount of points in the point cloud
Z_SLICE_COUNT = 100  # The amount of subsections when remeshing along the z axis
ROTATIONAL_SLICE_COUNT = 8  # The amount of subsections when subdividing rotationally
SUB_PCD_PRECISION = (
    1 / 250
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
    tri_mesh = trimesh.load_mesh(file_)
    u.msg(f"loaded mesh at {file_}", "info")

    # Get mesh boundaries
    # trimesh.bounds.oriented_bounds(tri_mesh) also rotates the mesh, which isn't helpful in this scenario
    to_origin, extents = u.axis_oriented_extents(tri_mesh)
    u.msg(f"mesh extents: {tri_mesh.extents}", "info")

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
            sorted_extents = np.sort(extents)

            # Sort the materials by volume
            sorted_sizes = sorted(kwargs["material-sizes"], key=math.prod)

            for size in sorted_sizes:
                valid = True
                for i in range(3):
                    if sorted_extents[i] >= sorted(size)[i]:
                        valid = False

                if valid:
                    selected_material_size = size
                    u.msg(f"selected material size {selected_material_size}", "info")
                    break

            if not valid:
                u.msg("mesh doesn't fit within available sizes", "warning")

    to_origin, extents = u.axis_oriented_extents(tri_mesh)
    tri_mesh = tri_mesh.apply_transform(to_origin)

    # Translate to only have positive vertices
    tri_mesh.vertices += extents / 2
    tri_mesh.vertices = np.round(tri_mesh.vertices, prefs.NUMPY_DECIMALS)
    tri_mesh.vertices = np.where(tri_mesh.vertices == -0.0, 0.0, tri_mesh.vertices)

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
            idx = np.where(extents == sorted_extents[2])[0][0]
            if idx == 0:  # x-axis
                rotation_matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
            elif idx == 1:  # y-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
            else:  # z-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        rotation_matrix = np.append(rotation_matrix, [[0, 0, 0, 1]], axis=0)
        tri_mesh = tri_mesh.apply_transform(rotation_matrix)
        to_origin, extents = u.axis_oriented_extents(tri_mesh)

    # Scale it according to the kwargs
    selected_material_size = None
    if "scale-to-material" in kwargs.keys() and kwargs["scale-to-machine"]:
        scale = kwargs["machine-size"] / extents
        tri_mesh.apply_scale(scale)
        u.msg(f"applied machine scaling: {kwargs["machine-size"]}", "info")
    elif (
        selected_material_size != None
        and "scale-to-material" in kwargs.keys()
        and kwargs["scale-to-material"]
    ):
        scale = selected_material_size / extents
        tri_mesh.apply_scale(scale)
        u.msg("applied material scaling", "info")

    to_origin, extents = u.axis_oriented_extents(tri_mesh)

    # --------------- Slicing ---------------
    u.msg("Computing convex hulls", "process")

    convex_slices = [[] for _ in range(Z_SLICE_COUNT)]
    for i in range(Z_SLICE_COUNT):
        # Slice from below
        plane_origin = [0, 0, (extents[2] / Z_SLICE_COUNT) * i * 0.999]
        tmp = trimesh.intersections.slice_mesh_plane(
            tri_mesh, plane_normal=[0, 0, 1], plane_origin=plane_origin
        )

        # Slice from above
        plane_origin = [0, 0, (extents[2] / Z_SLICE_COUNT) * (i + 1) * 1.001]
        tmp = trimesh.intersections.slice_mesh_plane(
            tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin
        )

        # Remove top and bottom of convex hull
        tmp = tmp.convex_hull
        if i > 0:
            plane_origin = [0, 0, (extents[2] / Z_SLICE_COUNT) * i]
            tmp = trimesh.intersections.slice_mesh_plane(
                tmp, plane_normal=[0, 0, 1], plane_origin=plane_origin
            )
        if i < Z_SLICE_COUNT - 1:
            plane_origin = [0, 0, (extents[2] / Z_SLICE_COUNT) * (i + 1)]
            tmp = trimesh.intersections.slice_mesh_plane(
                tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin
            )

        convex_slices[i] = tmp
    u.msg("sliced mesh", "info")

    # --------------- O3D ---------------
    pcd = o3d.geometry.PointCloud()

    # Scale the number of points with the extents to get a more even distribution
    volumes = []
    sum = 0
    for i, mesh in enumerate(convex_slices):
        mesh_to_origin, mesh_extents = u.axis_oriented_extents(mesh)
        volume = mesh_extents[0] * mesh_extents[1] * mesh_extents[2]
        volumes.append(volume)
        sum += volume

    sub_pcd_sizes = []
    for i, v in enumerate(volumes):
        size = math.ceil(PCD_SIZE * v / sum)

        # Manually increase the value for the top and bottom slice
        # to account for the increase in surface area
        if i < 1 or i >= len(volumes) - 1:
            size *= 5
        sub_pcd_sizes.append(size)
    u.msg(f"calculated pcd sizes", "info")

    for i, mesh in enumerate(convex_slices):
        # Covert the mesh from trimesh to o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(convex_slices[i].vertices)
        # astype(np.int32) avoids a segmentation fault
        o3d_mesh.triangles = o3d.utility.Vector3iVector(
            convex_slices[i].faces.astype(np.int32)
        )

        # Sample a point cloud from the mesh
        o3d_mesh.compute_vertex_normals()
        sub_pcd = o3d_mesh.sample_points_poisson_disk(
            number_of_points=sub_pcd_sizes[i], init_factor=5
        )
        pcd += sub_pcd

        if i + 1 < Z_SLICE_COUNT:
            u.msg(f"finished {i + 1} sub-pointclouds", "info", "\r")
        else:
            u.msg(f"finished {i + 1} sub-pointclouds", "info")

    # --------------- Remesh ---------------
    pcd.estimate_normals()

    # Add kerf to pcd
    if "kerf" in kwargs.keys():
        kerf = kwargs["kerf"]
        for i, p in enumerate(pcd.points):
            pcd.points[i] = p + pcd.normals[i] * kerf
        u.msg("added kerf", "info")

    # Estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    convex_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )

    # Create the triangular mesh with the vertices and faces from open3d
    v = np.asarray(convex_mesh.vertices)
    t = np.asarray(convex_mesh.triangles)
    mesh = trimesh.Trimesh(v, t, vertex_normals=np.asarray(convex_mesh.vertex_normals))
    trimesh.convex.is_convex(mesh)
    # trimesh.repair.fill_holes(mesh)
    mesh.export(prefs.MESH_FOLDER + "convex_mesh.stl")

    # Update mesh extents
    extents += 2 * kerf
    convex_mesh.translate([kerf, kerf, kerf])

    # --------------- Rotational coordinates  ---------------
    u.msg("Computing rotational coordinates", "process")
    convex_mesh.compute_vertex_normals()
    convex_pcd = convex_mesh.sample_points_poisson_disk(
        number_of_points=PCD_SIZE, init_factor=5
    )

    # remove top and bottom capsule to not collide with the machine
    capsule_radius = u.magnitude([extents[0], extents[1]]) / 20
    capsule_height = extents[2] / 50
    convex_pcd_points = []
    for i, p in enumerate(convex_pcd.points):
        if (
            u.vertical_capsule_SDF(
                origin=np.asarray([extents[0] / 2, extents[1] / 2, 0]),
                point=p,
                radius=capsule_radius,
                height=capsule_height,
            )
            > 0
            and u.vertical_capsule_SDF(
                origin=np.asarray([extents[0] / 2, extents[1] / 2, extents[2]]),
                point=p,
                radius=capsule_radius,
                height=capsule_height,
            )
            > 0
        ):
            convex_pcd_points.append(convex_pcd.points[i])
    convex_pcd_points = np.asarray(convex_pcd_points)
    u.msg("removed collision capsules", "info")

    box_x = u.magnitude([extents[0], extents[1]]) / 2
    box_y = min(extents[0], extents[1]) * SUB_PCD_PRECISION
    box_z = extents[2]
    box_extents = np.asarray([box_x, box_y, box_z])

    concave_hulls = []
    for n in range(ROTATIONAL_SLICE_COUNT):
        box_rotation = n * 2 * math.pi / ROTATIONAL_SLICE_COUNT
        to_box_center_vector = np.asarray(
            [math.cos(box_rotation), math.sin(box_rotation), 0]
        )
        to_box_center_vector /= u.magnitude(to_box_center_vector)
        box_origin = extents / 2 + to_box_center_vector * box_x / 2

        # Find points within a slice of the mesh
        contour = []
        for p in convex_pcd_points:
            if np.isclose(
                u.box_SDF(
                    origin=box_origin,
                    point=p,
                    extents=box_extents,
                    rotation_z=box_rotation,
                ),
                0,
            ):
                contour.append(p)
        contour = np.asarray(contour)

        # Rotate them along the z axis
        for i, p in enumerate(contour):
            p -= box_origin - to_box_center_vector * box_x / 2
            contour[i] = u.rotate_z_rad(p, -box_rotation)

        # Calculate their 2d concave hull
        concave_hull_idxs = ch.concave_hull_indexes(
            points=np.asarray(contour[:, [0, 2]]),
            concavity=2.0,
            length_threshold=0.0,
        )
        concave_hull_points = contour[:, [0, 2]][concave_hull_idxs]

        # Sort the hulls counterclockwise
        d = [extents[0], extents[0]]
        min_idx = [0, 0]
        # top = np.asarray([-extents[0] / 4, extents[2] / 2])
        # bottom = np.asarray([-extents[0] / 4, -extents[2] / 2])
        top = np.asarray([0, extents[2] / 2])
        bottom = np.asarray([0, -extents[2] / 2])
        for i, p in enumerate(concave_hull_points):
            # Find points closest to the z axis at the origin
            tmp = u.magnitude(p - bottom)
            if tmp < d[0]:
                d[0] = tmp
                min_idx[0] = i

            tmp = u.magnitude(p - top)
            if tmp < d[1]:
                d[1] = tmp
                min_idx[1] = i

        # Loop between extrema to sort a list from bottom to top
        # Compare distances to get the closest neighbor
        idx = min_idx[0]
        sorted_hull_indices = []
        while True:
            sorted_hull_indices.append(idx)

            if idx == min_idx[1]:
                break

            minimum = np.inf
            minimum_idx = 0
            for i, p in enumerate(concave_hull_points):
                if i == idx:
                    continue

                distance = u.sphere_SDF(
                    origin=concave_hull_points[idx], point=p, radius=1
                )

                if distance < minimum and i not in sorted_hull_indices:
                    minimum = distance
                    minimum_idx = i

            idx = minimum_idx

        sorted_hull = concave_hull_points[sorted_hull_indices]

        # Add the y coordinate back as zeros
        zeros = np.zeros((sorted_hull.shape[0],))
        sorted_hull = np.column_stack((sorted_hull[:, 0], zeros, sorted_hull[:, 1]))

        # Rotate them back to their original position
        for i, p in enumerate(sorted_hull):
            # Comment this line out when not plotting
            # p = u.rotate_z_rad(p, box_rotation)
            p = u.rotate_z_rad(p, math.pi)
            sorted_hull[i] = p + (extents / 2)

        # Scale it if necessary
        # TODO add argument
        # machine_extents = [250, 250, 250]
        # for i, p in enumerate(sorted_hull):
        #     for j in range(3):
        #         p[j] *= machine_extents[j] / extents[j]
        #     sorted_hull[i] = p

        concave_hulls.append(np.asarray(sorted_hull))

        if n + 1 < ROTATIONAL_SLICE_COUNT:
            u.msg(f"finished {n + 1} sections", "info", "\r")
        else:
            u.msg(f"finished {n + 1} sections", "info")

    gcode.to_gcode("output", concave_hulls, 2 * math.pi / ROTATIONAL_SLICE_COUNT)

    arr = []
    for x in concave_hulls:
        for y in x:
            arr.append(y)
    arr = np.asarray(arr)
    u.plot(arr)

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

