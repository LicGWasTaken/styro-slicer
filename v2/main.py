# import alphashape
import argv
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

origin_offset_pct = 0.1  # Offset from the edge of the bounding box when slicing

def main(file_, **kwargs):
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
    # extents/2 * (1 + pct) to avoid 0s that might cause problems with divisions later
    # tri_mesh.vertices += (extents / 2) * (1 + origin_offset_pct)
    tri_mesh.vertices += (extents / 2)
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
    slice_count = 100
    convex_slices = [[] for _ in range(slice_count)]
    for i in range(slice_count):
        # Slice from below
        plane_origin = [0, 0, (extents[2] / slice_count) * i * 0.999]
        tmp = trimesh.intersections.slice_mesh_plane(tri_mesh, plane_normal=[0, 0, 1], plane_origin=plane_origin)
        
        # Slice from above
        plane_origin = [0, 0, (extents[2] / slice_count) * (i + 1) * 1.001]
        tmp = trimesh.intersections.slice_mesh_plane(tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin)
        
        # Remove top and bottom of convex hull
        tmp = tmp.convex_hull
        if i > 0:
            plane_origin = [0, 0, (extents[2] / slice_count) * i]
            tmp = trimesh.intersections.slice_mesh_plane(tmp, plane_normal=[0, 0, 1], plane_origin=plane_origin)
        if i < slice_count - 1:
            plane_origin = [0, 0, (extents[2] / slice_count) * (i + 1)]
            tmp = trimesh.intersections.slice_mesh_plane(tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin)

        convex_slices[i] = tmp

    # --------------- O3D ---------------
    pcd = o3d.geometry.PointCloud()
    pcd_size = 50000

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
        size = math.ceil(pcd_size * v / sum)

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
        o3d_mesh.triangles = o3d.utility.Vector3iVector(convex_slices[i].faces.astype(np.int32))

        # Sample a point cloud from the mesh
        o3d_mesh.compute_vertex_normals()
        sub_pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=sub_pcd_sizes[i], init_factor=5)
        pcd += sub_pcd    
    pcd_points = np.asarray(pcd.points)

    # # Subdivide points by height (z)
    # slice_count = 10
    # pcd_slices = [[] for _ in range(slice_count)]
    # for p in pcd_points:
    #     i = math.floor(p[2] / (extents[2] / slice_count))
    #     # points on the top surface are slightly outside the extents. 
    #     # To avoid errors we add them to the last slice.
    #     if i >= len(pcd_slices):
    #         i = len(pcd_slices) - 1
    #     pcd_slices[i].append(p)

    # # Convert to a np array
    # for i, s in enumerate(pcd_slices):
    #     pcd_slices[i] = np.asarray(s)

    # # Plot the points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2])
    # # ax.scatter(pcd_slices[19][:, 0], pcd_slices[19][:, 1], pcd_slices[19][:, 2])
    # plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

    # --------------- Remesh ---------------
    pcd.estimate_normals()

    # # for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
    # #     print(f"alpha={alpha:.3f}")
    # alpha = .5
    # alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # alpha_mesh.compute_vertex_normals()
    # convex_mesh = alpha_mesh

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

    # --------------- 2D convex hulls ---------------
    return 0
    convex_pcd = []
    for i, s in enumerate(pcd_slices):
        hull = s[sp.spatial.ConvexHull(s).vertices]
        z = (extents[2] / slice_count) * i
        for p in hull:
            convex_pcd.append(np.asarray([p[0], p[1], z]))
    convex_pcd = np.asarray(convex_pcd)

    # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(convex_pcd[:, 0], convex_pcd[:, 1], convex_pcd[:, 2])
    plt.savefig(prefs.MESH_FOLDER + "unnamed.png")
    
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

def OLD():
    pass
    # # --------------- Calculate horizontal convex hulls ---------------
    # steps = 100
    # hulls_2D = []
    # hulls_3D = []

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # for i in range(steps):
    #     # Skip the very beginning and end to avoid unclear cuts
    #     factor = extents[2] / (steps - 1)
    #     z = i * factor
    #     if i == 0:
    #         z += extents[2] * origin_offset_pct
    #     elif i == steps - 1:
    #         z -= extents[2] * origin_offset_pct

    #     contour = trimesh.intersections.mesh_plane(
    #         mesh, [0, 0, 1], [0, 0, z], return_faces=False
    #     )

    #     # Convert point-pairs into singular 3D points
    #     try:
    #         points_3D, a = zip(*contour)
    #     # This occurs at the top of round surfaces, we can just skip over those
    #     except ValueError:
    #         continue

    #     # Store the z coordinate and convert the points to 2D
    #     z = points_3D[0][2]
    #     points_2D = np.empty((len(points_3D), 2))
    #     for i, p in enumerate(points_3D):
    #         points_2D[i] = np.array([p[0], p[1]])

    #     # Calulate the hull
    #     hull_2D = sp.spatial.ConvexHull(points_2D)
    #     hull_2D = points_2D[hull_2D.vertices]
    #     hulls_2D.append(hull_2D)

    # # In order to correctly remesh the convex hulls, increase the amount of points within each one.
    # max_distance = extents[2] / steps
    # new_hulls_2D = []
    # for hull in hulls_2D:
    #     new_hull = []
    #     for i, point in enumerate(hull):
    #         new_hull.append(point)

    #         # skip points that are already close enough
    #         if i < len(hull) - 1:
    #             next_point = hull[i + 1]
    #         else:
    #             next_point = hull[0]

    #         vector = next_point - point
    #         distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    #         if distance <= max_distance:
    #             continue

    #         # Append new equidistant points
    #         # We are subtracting 1 to skip the last point, which will be added on the next iteration
    #         step_count = math.ceil(distance / max_distance) - 1
    #         new_distance = round(distance / step_count, 3)

    #         current_point = point
    #         normalized_vector = vector / distance
    #         for n in range(step_count):
    #             new_point = current_point + normalized_vector * new_distance
    #             new_hull.append(new_point)
    #             current_point = new_point

    #     new_hulls_2D.append(new_hull)

    # # Reconvert to 3D ----------- TMP
    # for i, hull_2D in enumerate(new_hulls_2D):
    #     # Skip the very beginning and end to avoid unclear cuts
    #     factor = extents[2] / len(new_hulls_2D) - 1
    #     z = i * factor
    #     if i == 0:
    #         z += extents[2] * origin_offset_pct
    #     elif i == len(new_hulls_2D) - 1:
    #         z -= extents[2] * origin_offset_pct

    #     hull_3D = np.empty((len(hull_2D), 3))
    #     for i, p in enumerate(hull_2D):
    #         hull_3D[i] = np.array([p[0], p[1], z])

    #         # plot the points ----------- TMP
    #         # ax.scatter(p[0], p[1], z, color="black")

    #     hulls_3D.append(hull_3D)

    # # plt.savefig(prefs.MESH_FOLDER + "unnamed.png")

    # # --------------- Remesh ---------------
    # # Generate a new mesh with the convex hull coordinates
    # # Turn the contours into a point cloud
    # points = np.empty((0, 3))
    # for hull in hulls_3D:
    #     points = np.append(points, hull, axis=0)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.estimate_normals()

    # # Estimate radius for rolling ball
    # distances = point_cloud.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist
    # # radius = max_distance

    # convex_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     point_cloud, o3d.utility.DoubleVector([radius, radius * 2])
    # )

    # # convex_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    # #         point_cloud)[0]

    # # Create the triangular mesh with the vertices and faces from open3d
    # v = np.asarray(convex_mesh.vertices)
    # t = np.asarray(convex_mesh.triangles)
    # tri_mesh = trimesh.Trimesh(
    #     v, t, vertex_normals=np.asarray(convex_mesh.vertex_normals)
    # )

    # trimesh.convex.is_convex(tri_mesh)
    # trimesh.repair.fill_holes(tri_mesh)
    # tri_mesh.export(prefs.MESH_FOLDER + "convex_mesh.stl")

    # return 0

