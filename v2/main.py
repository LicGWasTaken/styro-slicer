# import alphashape
import argv
import concave_hull as ch
import math
import numpy as np
import open3d as o3d
import prefs as prefs
import scipy as sp
import sys
import time
import trimesh
import utils as u

# tmp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x forward, y right, z up

PCD_SIZE = 50000 # The total amount of points in the point cloud
Z_SLICE_COUNT = 100 # The amount of subsections when remeshing along the z axis
ROTATIONAL_SLICE_COUNT = 4 # The amount of subsections when subdividing rotationally
SUB_PCD_PRECISION = 1 / 10 # Factor to calculate the bounding box width compared to the total extent

def main(file_, **kwargs):
    # Define global variables within main
    global PCD_SIZE
    global Z_SLICE_COUNT
    global ROTATIONAL_SLICE_COUNT
    global SUB_PCD_PRECISION

    u.msg(f"args: {file_}, {kwargs}", "debug")
    u.msg("Running main", "process")

    # Load mesh
    tri_mesh = trimesh.load_mesh(file_)
    u.msg(f"loaded mesh at {file_}", "info")

    # Get mesh boundaries
    to_origin, extents = trimesh.bounds.oriented_bounds(tri_mesh)
    u.msg(f"mesh extents in mm: {tri_mesh.extents}", "info")

    # Select the smallest possible material size
    selected_material_size = None
    if "selected_material_size" in kwargs:
        selected_material_size = kwargs["selected-material-size"]
        pass
    else:
        if "material-sizes" not in kwargs:
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

    # Center mesh to world origin
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

        # Apply the transformation
        tri_mesh = tri_mesh.apply_transform(rotation_matrix)

        # Update extents
        to_origin, extents = trimesh.bounds.oriented_bounds(tri_mesh)
        u.msg("aligned mesh", "info")

    # --------------- Slicing ---------------
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

    # --------------- O3D ---------------
    pcd = o3d.geometry.PointCloud()

    # Scale the number of points with the extents to get a more even distribution
    volumes = []
    sum = 0
    for mesh in convex_slices:
        to_origin, extents = trimesh.bounds.oriented_bounds(tri_mesh)
        volume = extents[0] * extents[1] * extents[2]
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

    # # Plot the points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2])
    # # ax.scatter(pcd_slices[19][:, 0], pcd_slices[19][:, 1], pcd_slices[19][:, 2])
    # plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

    # --------------- Remesh ---------------
    pcd.estimate_normals()

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
    mesh = trimesh.Trimesh(
        v, t, vertex_normals=np.asarray(convex_mesh.vertex_normals)
    )
    trimesh.convex.is_convex(mesh)
    # trimesh.repair.fill_holes(mesh)
    mesh.export(prefs.MESH_FOLDER + "convex_mesh.stl")

    # --------------- Rotational coordinates  ---------------
    convex_mesh.compute_vertex_normals()
    convex_pcd = convex_mesh.sample_points_poisson_disk(
        number_of_points=PCD_SIZE, init_factor=5
    )

    box_x = math.sqrt(extents[0] ** 2 + extents[1] ** 2) / 2
    box_y = min(extents[0], extents[1]) * SUB_PCD_PRECISION
    box_z = extents[2]
    box_extents = np.asarray([box_x, box_y, box_z])

    concave_hulls = []
    for i in range(ROTATIONAL_SLICE_COUNT):
        box_rotation = i * 2 * math.pi / ROTATIONAL_SLICE_COUNT
        to_box_center_vector = np.asarray(
            [math.cos(box_rotation), math.sin(box_rotation), 0]
        )
        box_origin = extents / 2 + to_box_center_vector * box_x / 2

        # Find points within a slice of the mesh
        contour = []
        for p in convex_pcd.points:
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
            p -= box_origin
            contour[i] = u.rotate_z_rad(p, -box_rotation)
            # = p + origin

        # Calculate their 2d concave hull
        concave_hull_idxs = ch.concave_hull_indexes(
            points=np.asarray(contour[:, [0, 2]]),
            concavity=2.0,
            length_threshold=0.0,
        )
        concave_hull_points = contour[:, [0, 2]][concave_hull_idxs]

        # Add the y coordinate back as zeros
        zeros = np.zeros((concave_hull_points.shape[0],))
        concave_hull_points = np.column_stack(
            (concave_hull_points[:, 0], zeros, concave_hull_points[:, 1])
        )

        # Rotate them back to their original position
        for i, p in enumerate(concave_hull_points):
            # p -= origin
            p = u.rotate_z_rad(p, box_rotation)
            concave_hull_points[i] = p + box_origin

        concave_hulls.append(np.asarray(concave_hull_points))

    # arr = []
    # for c in concave_hulls:
    #     for p in c:
    #         arr.append(p)
    # arr = np.asarray(arr)

    # # Plot the points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color="hotpink")
    # plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

    # --------------- Minkowski Sum for kerf ---------------
    circle_quality = 16
    circle = []
    for i in range(circle_quality):
        circle.append(u.rotate_z_rad([0, -0.1], i * 2 * math.pi / circle_quality))

    # for i, concave_hull in enumerate(concave_hulls):
    #     concave_hulls[i] = u.minkowski(concave_hull, circle)
    a = np.asarray([np.asarray([0, 0]), np.asarray([1, 0]), np.asarray([1, 1]),  np.asarray([0, 1])])
    minkowski = u.minkowski(a, circle)

    arr = np.asarray(minkowski)
    # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(arr[:, 0], arr[:, 1], color="hotpink")
    plt.savefig(prefs.MESH_FOLDER + "unnamed.png")
        
    return 0

    # for n, concave_hull in enumerate(concave_hulls):
    #     d = [extents[0], extents[0]]
    #     min_idx = [0, 0]
    #     top = np.asarray([extents[0] / 2, extents[1] / 2, extents[2]])
    #     bottom = np.asarray([extents[0] / 2, extents[1] / 2, 0])
    #     for i, p in enumerate(concave_hull):
    #         # Find points closest to the z axis at the origin
    #         tmp = u.magnitude(p - top)
    #         if tmp < d[0]:
    #             d[0] = tmp
    #             min_idx[0] = i

    #         tmp = u.magnitude(p - bottom)
    #         if tmp < d[1]:
    #             d[1] = tmp
    #             min_idx[1] = i

    #     # Loop between extrema and remove the side of the hull closest to the centerline
    #     avg_dist = [0, 0]
    #     b = False
    #     count = 0
    #     for i in range(len(concave_hull)):
    #         i += min_idx[0]
    #         if i >= len(concave_hull):
    #             i = i - len(concave_hull)

    #         if i == min_idx[1]:
    #             avg_dist[0] /= count
    #             count = 0
    #             b = True
    #             continue

    #         if i == min_idx[0] and b == True:
    #             avg_dist[1] /= count
    #             break

    #         p = concave_hull[i]
    #         if not b:
    #             avg_dist[0] += u.magnitude(p[:2] - (extents / 2)[:2])
    #         else:
    #             avg_dist[1] += u.magnitude(p[:2] - (extents / 2)[:2])

    #         count += 1

    #     new_contour = []
    #     for i in range(len(concave_hull)):
    #         if avg_dist[0] > avg_dist[1]:
    #             i += min_idx[0]
    #         else:
    #             i += min_idx[1]

    #         if i >= len(concave_hull):
    #             i = i - len(concave_hull)

    #         if (
    #             avg_dist[0] > avg_dist[1]
    #             and i == min_idx[1] + 1
    #             or avg_dist[0] <= avg_dist[1]
    #             and i == min_idx[0] + 1
    #         ):
    #             break

    #         new_contour.append(concave_hull[i])
        
    #     print(len(concave_hull), len(new_contour), min_idx)
    #     concave_hulls[n] = new_contour

    # # Plot the points
    # arr = []
    # for c in concave_hulls:
    #     for p in c:
    #         arr.append(p)
    # arr = np.asarray(arr)

    # # Plot the points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color="black")
    # # ax.scatter(pcd_slices[19][:, 0], pcd_slices[19][:, 1], pcd_slices[19][:, 2])
    # plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

    # ---------------  ---------------
    return 0

    convex_mesh.compute_vertex_normals()
    convex_pcd = convex_mesh.sample_points_poisson_disk(
        number_of_points=PCD_SIZE, init_factor=5
    )

    # Subdivide points along the x axis
    Z_SLICE_COUNT = 10
    x_slices = [[] for _ in range(Z_SLICE_COUNT)]
    for p in convex_pcd.points:
        i = math.floor(p[0] / (extents[0] / Z_SLICE_COUNT))

        # points at the end are slightly outside the extents.
        # To avoid errors we add them to the last slice.
        if i >= len(x_slices):
            i = len(x_slices) - 1
        x_slices[i].append(p)

    # Convert to a np array
    for i, s in enumerate(x_slices):
        x_slices[i] = np.asarray(s)

    xs = []
    for i, s in enumerate(x_slices):
        avg_x = (i - 0.5) * extents[0] / Z_SLICE_COUNT
        for p in s:
            xs.append([avg_x, p[1], p[2]])
    xs = np.asarray(xs)

    # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2])
    plt.savefig(prefs.MESH_FOLDER + "unnamed.png")
    return 0

    # --------------- 2D convex hulls ---------------
    # convex_pcd = []
    # for i, s in enumerate(pcd_slices):
    #     hull = s[sp.spatial.ConvexHull(s).vertices]
    #     z = (extents[2] / slice_count) * i
    #     for p in hull:
    #         convex_pcd.append(np.asarray([p[0], p[1], z]))
    # convex_pcd = np.asarray(convex_pcd)

    # # Plot the points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(convex_pcd[:, 0], convex_pcd[:, 1], convex_pcd[:, 2])
    # plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

if __name__ == "__main__":
    # Start timer
    timer = time.perf_counter()

    # Get the passed file and keyword arguments
    try:
        file_, kwargs = argv.get_arguments()
    except TypeError:
        sys.exit()

    # Run the program
    main(file_, **kwargs)

    # Stop timer and print results
    u.msg(f"time elapsed: {round(time.perf_counter() - timer, 3)}s", "debug")

