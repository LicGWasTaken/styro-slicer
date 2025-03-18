import trimesh
import math
import numpy as np
import utils as u

extents_ = None

def main(file_):
    mesh_ = trimesh.load_mesh(file_)
    out = axysimmetric(mesh_, num_cuts=20, num_points=100*20)
    return out

def axysimmetric(mesh_: trimesh.Trimesh, num_cuts: int, num_points):
    out = []

    to_origin, mesh_extents = u.axis_oriented_extents(mesh_)
    mesh = mesh_.apply_transform(to_origin)

    # Cache the extents
    extents_ = mesh_extents
    	
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
    print(out.size, len(cut))
    return out