import trimesh
import gcode as gc
import math
import numpy as np
import utils as u
import vtk

def check_cut_validity_vertical_angle(plane_cuts: list, motor_planes, motor_plane_data):
    for i, (cut, ray_direction) in enumerate(plane_cuts):
        intersections = []
        intersections.append(u.plane_vector_intersect(cut[0], ray_direction, motor_planes[0]))
        intersections.append(u.plane_vector_intersect(cut[1], ray_direction, motor_planes[0]))
        intersections.append(u.plane_vector_intersect(cut[0], ray_direction, motor_planes[1]))
        intersections.append(u.plane_vector_intersect(cut[1], ray_direction, motor_planes[1]))

        valid = True
        for intersect in intersections:
            if (
                intersect is None or
                intersect[1] >= motor_plane_data[1] or
                intersect[1] <= motor_plane_data[2] or
                intersect[2] >= motor_plane_data[3] or 
                intersect[2] <= motor_plane_data[4]
            ):
                valid = False
                break
        
        if valid:
            return i
        
    return None
    
def linear(mesh_: trimesh.Trimesh, motor_plane_data: np.ndarray, kerf: float, render_planes: bool):
    """motor_plane_data: array containing the following information:
    [x, positive y, negative y, positive z, negative z]
    where each value is relative to the mesh's origin, x defines position, y and z define size"""
    # TODO make this an offset from the very bottom instead of the mesh's origin, to not depend on mesh size

    out = []
    Intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_)
    motor_planes_origins = [np.asarray([motor_plane_data[0], 0, 0]), np.asarray([-motor_plane_data[0], 0, 0])]
    motor_planes_normals = [np.asarray([-1, 0, 0]), np.asarray([1, 0, 0])]
    motor_planes = [u.plane(motor_planes_origins[0], motor_planes_normals[0]), u.plane(motor_planes_origins[1], motor_planes_normals[1])]

    # Visualize plane boundaries
    if render_planes:
        out = [[[-motor_plane_data[0], motor_plane_data[1], motor_plane_data[3]], 
            [-motor_plane_data[0], motor_plane_data[1], motor_plane_data[4]]],
            [[-motor_plane_data[0], motor_plane_data[1], motor_plane_data[4]], 
            [-motor_plane_data[0], motor_plane_data[2], motor_plane_data[4]]],
            [[-motor_plane_data[0], motor_plane_data[2], motor_plane_data[4]], 
            [-motor_plane_data[0], motor_plane_data[2], motor_plane_data[3]]],
            [[-motor_plane_data[0], motor_plane_data[2], motor_plane_data[3]], 
            [-motor_plane_data[0], motor_plane_data[1], motor_plane_data[3]]], # Left Plane
            [[motor_plane_data[0], motor_plane_data[1], motor_plane_data[3]], 
            [motor_plane_data[0], motor_plane_data[1], motor_plane_data[4]]],
            [[motor_plane_data[0], motor_plane_data[1], motor_plane_data[4]], 
            [motor_plane_data[0], motor_plane_data[2], motor_plane_data[4]]],
            [[motor_plane_data[0], motor_plane_data[2], motor_plane_data[4]], 
            [motor_plane_data[0], motor_plane_data[2], motor_plane_data[3]]],
            [[motor_plane_data[0], motor_plane_data[2], motor_plane_data[3]], 
            [motor_plane_data[0], motor_plane_data[1], motor_plane_data[3]]]] # Right Plane
    else:
        out = []

    # Trimesh's trimesh.facets leaves out a lot of faces, so we had those back in manually
    facets_indices = mesh_.facets.copy()
    facet_faces = np.concatenate(facets_indices)
    all_faces = np.arange(len(mesh_.faces.copy()))
    missing_faces = np.setdiff1d(all_faces, facet_faces)
    for f in missing_faces:
        facets_indices.append(np.array([f]))

    # Add the excluded faces to the normals
    facets_normals = mesh_.facets_normal.copy().tolist()
    face_normals = mesh_.face_normals.copy().tolist()
    for f in missing_faces:
        facets_normals.append(face_normals[f])
    facets_normals = np.array([np.array(sublist) for sublist in facets_normals])

    # Turn the indicies into unique vertices
    facets_vertices = []
    for arr in facets_indices:
        vertex_indices = np.concatenate(mesh_.faces[arr])
        facets_vertices.append(mesh_.vertices[np.unique(vertex_indices)])

    # We calculate the origins with the avarage of the vertices
    # since Trimesh's facets.origins returns a random point on the facet
    facets_origins = np.asarray([verts.mean(axis=0) for verts in facets_vertices])

    valid_cuts = []
    pct_num = round(len(facets_indices) / 100)
    for i, facet_index in enumerate(facets_indices):
        if i % pct_num == 0:
            u.msg(f"Completed {round(i / pct_num)}%", "debug", "\r")

        # Find the longest vertical line on each facet
        facet_origin = facets_origins[i]
        facet_normal = facets_normals[i]
        facet_plane = u.plane(facet_origin, facet_normal)
        facet_vertices = facets_vertices[i]

        cut_plane_normal = np.cross(facet_normal, np.asarray([0, 0, 1]))
        if np.all(cut_plane_normal == 0):
            cut_plane_normal = np.asarray([1, 0, 0])
        cut_plane = u.plane(np.asarray([0, 0, 0]), cut_plane_normal)

        intersection_vector = u.plane_intersect(facet_plane, cut_plane)
        extremes = u.extreme_points_along_vector(facet_vertices, intersection_vector, facet_origin)
        extremes += facet_normal * kerf

        # Detect collisions with perpendicular raycasts
        offset_rad = math.pi / 16
        rad = 0
        ray_origins = [extremes[0], extremes[0], extremes[1], extremes[1]]
        ray_direction = np.cross(intersection_vector, facet_normal)
        extremes_vector = intersection_vector

        # TODO add an extra raycast to make sure that no extremes collide with the mesh
        # If they do, add the "imperfect raycast" and prioritize other cuts in later sections
        valid_plane_cuts = []
        while rad < math.pi:
            ray_directions = [ray_direction, -ray_direction, ray_direction, -ray_direction]
            hit = Intersector.intersects_any(ray_origins, ray_directions)
            if not np.any(hit):
                valid_plane_cuts.append([extremes, ray_direction])
                
            rad += offset_rad
            ray_direction = u.rotate_around_vector(ray_direction, facet_normal, offset_rad)
            extremes_vector = u.rotate_around_vector(extremes_vector, facet_normal, offset_rad)
            extremes = u.extreme_points_along_vector(facet_vertices, extremes_vector, facet_origin)
            extremes += facet_normal * kerf
            ray_origins = [extremes[0], extremes[0], extremes[1], extremes[1]]

        # TODO Add the process from axisymmetric slicing to approximate invalid cuts

        # Make sure the cut is valid within machine boundaries
        rad = 0
        offset_rad = math.pi / 6
        cuts = valid_plane_cuts
        while rad <= math.pi / 2:
            valid_idx = check_cut_validity_vertical_angle(cuts, motor_planes, motor_plane_data)
            if valid_idx is not None:
                valid_cut = cuts[valid_idx]
                valid_cuts.append([valid_cut[0][0], valid_cut[0][1], valid_cut[1], rad])
                break
            else:
                # Try again with rotations around the vertical axis
                cuts = []
                for cut, ray_direction in valid_plane_cuts:
                    new_dir = u.rotate_z_rad(ray_direction, offset_rad)
                    new_cut = np.asarray([u.rotate_z_rad(cut[0], offset_rad), u.rotate_z_rad(cut[1], offset_rad)])
                    cuts.append([new_cut, new_dir])
                rad += offset_rad  

    u.msg(f"Completed 100%", "debug")

    # Valid cuts: 
    # 0,1 --> Two points making up the cut
    # 2 --> Facet normal
    # 3 --> Rotary axis angle rad
    
    # Find paths between cuts that don't intersect the mesh.
    # TODO: eventually improve this with a pathfinding algo like A*
    
    plane_normal = np.asarray([0, 0, 1])
    proj = mesh_.projected(plane_normal).vertices
    ps = []
    for p in proj:
        ps.append(np.asarray([p[0], p[1], 0]))

    for i, p in enumerate(ps):
        out.append([p, p])

    # valid_cuts = [[np.asarray([0, 0, 0]), np.asarray([1, 1, 1]), np.asarray([0, 0, 1]), 0]]
    exit_paths = []
    for i, cut0 in enumerate(valid_cuts):
        # Get the end pos of the last and start pos of the new cut
        pos0 = cut0[1]
        if i == len(valid_cuts) - 1:
            cut1 = valid_cuts[0]
        else:
            cut1 = valid_cuts[i + 1]
        pos1 = cut1[0]

        cut_vector = pos1 - pos0
        # projection_plane_normal = u.normalize(np.cross(cut_vector, np.asarray([0, 0, 1])))
        projection_plane_normal = u.normalize(cut0[2]) # TODO why th does this work!?
        silhouette_2d = mesh_.projected(projection_plane_normal).vertices.view(np.ndarray)
        silhouette_3d = np.hstack((silhouette_2d, np.zeros((silhouette_2d.shape[0], 1))))

        T = trimesh.geometry.align_vectors([0, 0, 1], projection_plane_normal)
        silhouette_3d = trimesh.transform_points(silhouette_3d, T)

        # TODO Make sure this is working properly
        silhouette_3d += projection_plane_normal * np.dot(pos0, projection_plane_normal)

        if i < 20:
            continue
        out = silhouette_3d
        out = np.append(out, pos1.reshape(1, 3), axis=0)
        out = np.append(out, pos0.reshape(1, 3), axis=0)
        return out

        # # TODO this is meant to find the path along the silhouette from pos0 to pos1, but for some reason it always finds pos1 right away
        # visited = []
        # sphere_radius = 0.5
        # max_sphere_radius = 10
        # current_point = pos0
        # silhouette_3d = np.append(silhouette_3d, pos1.reshape(1, 3), axis=0)
        # while not np.array_equal(current_point, pos1):
        #     if len(visited) > 0:
        #         print(len(visited))
        #     visited.append(current_point)
        #     dir_vector = u.normalize(pos1 - current_point)

        #     valid_next = []
        #     sphere_radius_multiplier = 1
        #     while len(valid_next) == 0 and sphere_radius < max_sphere_radius:
        #         exit = False
        #         for p in silhouette_3d:
        #             if np.array_equal(p, pos1):
        #                 exit = True
        #                 break
        #             if any(np.array_equal(p, v) for v in visited):
        #                 continue
        #             if u.sphere_SDF(current_point, p, sphere_radius * sphere_radius_multiplier) <= 0:
        #                 valid_next.append(np.asarray(p))
        #                 continue
        #         if exit:
        #             break
        #         sphere_radius_multiplier += 1
        #     if exit:
        #         break

        #     best_next = None
        #     min_angle = np.inf
        #     for p in valid_next:
        #         dir = u.normalize(p - current_point)
        #         angle = u.vector_angle(dir, dir_vector)
        #         if angle < min_angle:
        #             min_angle = angle
        #             best_next = p

        #     current_point = best_next

        # print(pos0, pos1)
        # print(visited)

    # Shoot a ray from the cut, and follow its reflections until a valid exit path is found
#     exit_paths = []
#     for i, cut0 in enumerate(valid_cuts):
#         # Get the end pos of the last and start pos of the new cut
#         pos0 = cut0[1]
#         dir_0 = np.cross(cut0[1] - cut0[0], cut0[2])
#         ray_directions_0 = [dir_0, -dir_0]
#         if i == len(valid_cuts) - 1:
#             cut1 = valid_cuts[0]
#         else:
#             cut1 = valid_cuts[i + 1]
#         pos1 = cut1[0]
#         dir_1 = np.cross(cut1[1] - cut1[0], cut1[2])
#         ray_directions_1 = [dir_1, -dir_1]
        
#         # Raycast to check for collisions between positions
#         # TODO: cast in a circle with radius k to avoid getting too close to the mesh
#         ray_origins = [pos0]
#         ray_directions = [pos1 - pos0]
#         hit = Intersector.intersects_any(ray_origins, ray_directions)
#         if np.any(hit):
#             u.msg(f"Skipped idx {i}, {i + 1}", "debug")
#             continue
        
#         # Collisions from wire angle
#         valid_rotation = None

#         valid = True
#         ray_origins = [pos1, pos1]
#         hit = Intersector.intersects_any(ray_origins, ray_directions_0)
#         if not np.any(hit):
#             # set valid rotation to pos0's
#             valid_rotation = dir_0
#         else:
#             ray_origins = [pos0, pos0]
#             hit = Intersector.intersects_any(ray_origins, ray_directions_1)
#             if not np.any(hit):
#                 # set valid rotation to pos1's
#                 valid_rotation = dir_1
#             else:
#                 valid = False

#         if not valid:
#             # Check with average rotation
#             avg_dir = u.normalize(dir_0 + dir_1)
#             ray_directions = [avg_dir, -avg_dir]
#             ray_origins = [pos0, pos0, pos1, pos1]
#             hit = Intersector.intersects_any(ray_origins, ray_directions_0)
#             if not np.any(hit):
#                 # set valid rotation to the avg
#                 valid_rotation = avg_dir
#             else:
#                 u.msg(f"Invalid rotation between idx {i}, {i + 1}", "debug")
#                 continue
        
#         # Valid cut with valid rotation --> just pass it to the output
#         exit_paths.append(cut1)

    # Sort the cuts by motor position TODO
    position_zero = [0, 0, 0, 0]

    
    for cut in valid_cuts:
        if cut[3] == 0:
            out.append([cut[0], cut[1]])
    print(len(out))
    return np.asarray(out)

def axysimmetric(mesh_: trimesh.Trimesh, num_cuts: int, num_points: int, kerf: float):
    pcd = []
    coords = []
    mesh = mesh_
    _, extents_ = u.axis_oriented_extents(mesh)

    # Variables for mesh transformation
    rad = 2 * math.pi / num_cuts
    num_points_per_cut = round(num_points / num_cuts)
    add = extents_[2] / num_points_per_cut

    # Variables for raycasting
    start_dist = u.magnitude([extents_[0], extents_[1]]) * 2
    min_dist = kerf
    if min_dist == 0:
        min_dist = 0.01
        # TODO send a warning about this inprecision

    Intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    ray_direction = [0, -1, 0]
    ray_directions_perp = [[1, 0, 0], [-1, 0, 0]]

    for i in range(num_cuts):
        cut = []
        z = -extents_[2] / 2 - add
        for j in range(num_points_per_cut):
            z += add

            ray_origin = [0, start_dist, z]
            hit, _, _ = Intersector.intersects_location([ray_origin], [ray_direction])
            if len(hit) == 0: continue

            # Return the raycast result if no collision
            # The raycast intersects the mesh at two points
            closest_hit = hit[np.argmax(hit[:, 1])]
            
            # Add the kerf
            closest_hit[1] += kerf
            ray_origins_perp = [closest_hit, closest_hit]
            hit = Intersector.intersects_any(ray_origins_perp, ray_directions_perp)
            if not np.any(hit):
                cut.append(np.asarray(closest_hit))
                continue

            # Collision detection (exponential decay)
            bound_top = start_dist
            bound_bottom = 0
            dist = closest_hit[1]

            while (bound_top - bound_bottom) / 4 > min_dist:
                ray_origin_perp = [0, dist, z]
                ray_origins_perp = [ray_origin_perp, ray_origin_perp]
                hit = Intersector.intersects_any(ray_origins_perp, ray_directions_perp)
                if np.any(hit):
                    bound_bottom = dist
                else:
                    bound_top = dist
                dist = bound_top - (bound_top - bound_bottom) / 2

            cut.append(np.asarray([0, dist, z]))
        deg = i * rad * 180 / math.pi
        coords.append([deg, np.asarray(cut)])

        # Rotate the mesh
        rotation_matrix = trimesh.transformations.rotation_matrix(rad, [0, 0, 1])
        mesh.apply_transform(rotation_matrix)
        for j, p in enumerate(cut):
            cut[j] = u.rotate_z_rad(p, -rad * i)
   
        for p in cut:
            pcd.append(p)
    pcd = np.asarray(pcd)
    return pcd, coords