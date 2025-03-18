import trimesh
import math
import numpy as np
import utils as u

def main(file_):
    in_mesh = trimesh.load_mesh(file_)
    out = axysimmetrical(in_mesh, 20)
    return out

def axysimmetrical(_mesh: trimesh.Trimesh, num_cuts: int):
    to_origin, mesh_extents = u.axis_oriented_extents(_mesh)
    _mesh.apply_transform(to_origin)
    rad = 2 * math.pi / num_cuts
    point_count = 100
    addend = mesh_extents[2] / point_count
    out = []
    for i in range(num_cuts):
        tmp = []
        Intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(_mesh)
        # Temporary, change this to something with extents
        starting_distance = u.magnitude([mesh_extents[0], mesh_extents[1]])
        min_distance = 0.5
        z = -mesh_extents[2] / 2
        for z_ in range(point_count):
            # Cast a ray towards the mesh to find the minimum distance
            ray_origin = [[0, starting_distance, z]]
            ray_direction = [[0, -1, 0]]
            hit, _, _ = Intersector.intersects_location(ray_origin, ray_direction)
            if len(hit) == 0:
                z += addend
                continue
            # hit = np.min(hit) # The ray also intersects the back of the mesh

            top_boundary = starting_distance
            bottom_boundary = 0
            current_distance = starting_distance
            ray_directions = [[1, 0, 0], [-1, 0, 0]]
            while (top_boundary - bottom_boundary) / 4 > min_distance:
                # Shoot a perpendicular ray
                ray_origin = [0, current_distance, z]
                ray_origins = [ray_origin, ray_origin]
                hit = Intersector.intersects_any(ray_origins, ray_directions)
                if np.any(hit):
                    bottom_boundary = current_distance
                else:
                    top_boundary = current_distance
                current_distance = top_boundary - (top_boundary - bottom_boundary) / 2
            tmp.append(np.asarray([0, current_distance, z]))
            z += addend
    
        # Rotate the mesh
        rotation_matrix = trimesh.transformations.rotation_matrix(rad, [0, 0, 1])
        _mesh.apply_transform(rotation_matrix)
        for j, p in enumerate(tmp):
            tmp[j] = u.rotate_z_rad(p, rad * i)
        out.append(tmp)

    tmp = []
    for a in out:
        for b in a:
            tmp.append(b)
    tmp = np.asarray(tmp)

    return tmp