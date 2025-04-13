import trimesh
import gcode as gc
import math
import numpy as np
import utils as u
import vtk

ORIGIN = np.asarray([387, 387, 420 + 75])

def main(file_, name_):
    mesh_ = trimesh.load_mesh(file_)

    to_origin, mesh_extents = u.axis_oriented_extents(mesh_)
    mesh_.apply_transform(to_origin)

    # Cache the extents
    global extents_ 
    extents_ = mesh_extents

    out = linear(mesh_)
    # out, coords = axysimmetric(mesh_, num_cuts=16, num_points=100*2, kerf=0.2)
    # gc.to_gcode_axysimmetric(name_, coords, ORIGIN, np.asarray([400, 600, 1450]), feed=150, slew=800)
    return out

def linear(mesh_: trimesh.Trimesh):
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

    # Visualize normals
    for i, normal in enumerate(facets_normals):
        origin = facets_origins[i]
        out.append(np.asarray([origin + normal, origin + normal * 2]))

    # Find the longest vertical line on each facet
    for i, facet_index in enumerate(facets_indices):
        facet_origin = facets_origins[i]

    return np.asarray(out)

    # Used to store information accessible via facet index
    normals = []   
    planes = []
    vectors = []
    points = []
    for i in range(len(facets_indices)):
        facet_origin = facets_origins[i]
        facet_normal = facets_normals[i]
        facet_plane = u.plane(facet_origin, facet_normal)

        # Find the longest vertical line on each facet
        normal = np.asarray([facet_normal[0], facet_normal[1], 0])
        normal = u.rotate_z_rad(normal, math.pi / 2)
        if u.magnitude(normal) == 0: # Handle vertical normals
            normal = np.asarray([1, 0, 0])
        normal = u.normalize(normal)
        normals.append(normal)
        cut_plane = u.plane(np.asarray([0, 0, 0]), normal)
        planes.append(np.asarray(cut_plane))

        intersection = u.plane_intersect(facet_plane, cut_plane)
        vector = u.normalize(intersection[1] - intersection[0])
        vectors.append(vector)

        min_, max_ = u.extreme_points_along_vector(facets_vertices[i], vector)
        points.append(np.asarray([min_, max_]))

        # TODO: Collision detection

        # out.append(min_)
        # out.append(max_)
        out.append(np.asarray([min_, max_]))

    out = np.asarray(out)
    # print(planes, normals, vectors, points)
    print(out)
    return out

def axysimmetric(mesh_: trimesh.Trimesh, num_cuts: int, num_points: int, kerf: float):
    pcd = []
    coords = []
    mesh = mesh_

    # Variables for mesh transformation
    rad = 2 * math.pi / num_cuts
    num_points_per_cut = round(num_points / num_cuts)
    add = extents_[2] / num_points_per_cut

    # Variables for raycasting
    start_dist = u.magnitude([extents_[0], extents_[1]]) * 2
    min_dist = kerf

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