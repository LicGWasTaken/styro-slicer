import trimesh
import math
import numpy as np
import utils as u
import vtk

def main(file_):
    mesh_ = trimesh.load_mesh(file_)

    to_origin, mesh_extents = u.axis_oriented_extents(mesh_)
    mesh_.apply_transform(to_origin)

    # Cache the extents
    global extents_ 
    extents_ = mesh_extents

    out = axysimmetric(mesh_, num_cuts=20, num_points=100*20)
    # out = linear(mesh_)
    return out

def align_mesh(path):
    mesh_ = trimesh.load_mesh(path)
    to_origin, mesh_extents = u.axis_oriented_extents(mesh_)
    mesh = mesh_.apply_transform(to_origin)
    mesh.export("mesh.stl")


def linear(mesh_: trimesh.Trimesh):
    out = []

    mesh = convex_mesh(mesh_)

    return out

def convex_mesh(mesh_: trimesh.Trimesh, num_sub_pcds):
    # Subdivide the mesh into submeshes along the z axis and compute their convex hulls
    convex_sub_meshes = [[] for _ in range(num_sub_pcds)]
    for i in range(num_sub_pcds):
        # Slice from below
        plane_origin = [0, 0, (extents_[2] / num_sub_pcds) * i * 0.999]
        tmp = trimesh.intersections.slice_mesh_plane(
            mesh_, plane_normal=[0, 0, 1], plane_origin=plane_origin
        )

        # Slice from above
        plane_origin = [0, 0, (extents_[2] / num_sub_pcds) * (i + 1) * 1.001]
        tmp = trimesh.intersections.slice_mesh_plane(
            tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin
        )

        # Remove top and bottom of convex hull
        tmp = tmp.convex_hull
        if i > 0:
            plane_origin = [0, 0, (extents_[2] / num_sub_pcds) * i]
            tmp = trimesh.intersections.slice_mesh_plane(
                tmp, plane_normal=[0, 0, 1], plane_origin=plane_origin
            )
        if i < num_sub_pcds - 1:
            plane_origin = [0, 0, (extents_[2] / num_sub_pcds) * (i + 1)]
            tmp = trimesh.intersections.slice_mesh_plane(
                tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin
            )

        convex_sub_meshes[i] = tmp

    # Sample pcds and merge them
    append_filter = vtk.vtkAppendPolyData()

    for sub_mesh in convex_sub_meshes:
        sampler = vtk.vtkPolyDataPointSampler()
        sampler.SetInputData(mesh)
        sampler.SetDistance(1)  # Adjust sampling density
        sampler.Update()
        sub_pcd = sampler.GetOutput()

        append_filter.AddInputData(sub_pcd)

    append_filter.Update()
    pcd = append_filter.GetOutput()
    
        

    # # Scale the number of points with the extents to get a more even distribution
    # volumes = []
    # sum = 0
    # for i, mesh in enumerate(convex_slices):
    #     mesh_to_origin, mesh_extents = u.axis_oriented_extents(mesh)
    #     volume = mesh_extents[0] * mesh_extents[1] * mesh_extents[2]
    #     volumes.append(volume)
    #     sum += volume

    # sub_pcd_sizes = []
    # for i, v in enumerate(volumes):
    #     size = math.ceil(PCD_SIZE * v / sum)

    #     # Manually increase the value for the top and bottom slice
    #     # to account for the increase in surface area
    #     if i < 1 or i >= len(volumes) - 1:
    #         size *= 5
    #     sub_pcd_sizes.append(size)
    # u.msg(f"calculated pcd sizes", "info")

    for i, mesh in enumerate(convex_sub_meshes):
        # Covert the mesh from trimesh to o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(convex_sub_meshes[i].vertices)
        # astype(np.int32) avoids a segmentation fault
        o3d_mesh.triangles = o3d.utility.Vector3iVector(
            convex_sub_meshes[i].faces.astype(np.int32)
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
    _mesh = trimesh.Trimesh(v, t, vertex_normals=np.asarray(convex_mesh.vertex_normals))
    to_origin, _ = u.axis_oriented_extents(_mesh)
    _mesh.apply_transform(to_origin)
    _mesh.export(prefs.MESH_FOLDER + "convex_mesh.stl")

    # Update mesh extents
    extents += 2 * kerf
    convex_mesh.translate([kerf, kerf, kerf])

    pass

def axysimmetric(mesh_: trimesh.Trimesh, num_cuts: int, num_points):
    out = []
    mesh = mesh_

    # Variables for mesh transformation
    rad = 2 * math.pi / num_cuts
    num_points_per_cut = round(num_points / num_cuts)
    add = extents_[2] / num_points_per_cut

    # Variables for raycasting
    start_dist = u.magnitude([extents_[0], extents_[1]]) * 2
    min_dist = 0.5

    Intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    ray_direction = [0, -1, 0]
    ray_directions_perp = [[1, 0, 0], [-1, 0, 0]]
    ray_origin = [0, start_dist, 0]

    for i in range(num_cuts):
        cut = []
        z = -extents_[2] / 2 - add
        for j in range(num_points_per_cut):
            z += add

            hit, _, _ = Intersector.intersects_location([ray_origin], [ray_direction])
            if len(hit) == 0: continue

            # Collision detection (exponential decay)
            bound_top = start_dist
            bound_bottom = 0
            dist = start_dist

            while (bound_top - bound_bottom) / 4 > min_dist:
                ray_origin_perp = [0, dist, z]
                ray_origins_perp = [ray_origin_perp, ray_origin_perp]
                hit = Intersector.intersects_any(ray_origins_perp, ray_directions_perp)
                if np.any(hit):
                    bound_bottom = dist
                else:
                    bound_top = dist
                dist = bound_top - (bound_top -bound_bottom) / 2
            cut.append(np.asarray([0, dist, z]))

        # Rotate the mesh
        rotation_matrix = trimesh.transformations.rotation_matrix(rad, [0, 0, 1])
        mesh.apply_transform(rotation_matrix)
        for j, p in enumerate(cut):
            cut[j] = u.rotate_z_rad(p, rad * i)
   
        for p in cut:
            out.append(p)

    out = np.asarray(out)
    return out