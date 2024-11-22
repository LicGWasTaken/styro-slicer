import argv
from math import prod
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

def main(file_, **kwargs):
    u.msg(f"args: {file_}, {kwargs}", "debug")
    u.msg("Running main", "process")

    # Load mesh
    mesh = trimesh.load_mesh(file_)
    u.msg(f"loaded mesh at {file_}", "info")

    # Get mesh boundaries
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    u.msg(f"mesh extents in mm: {mesh.extents}", "info")

    # Select the smallest possible material size
    selected_material_size = None
    if "selected_material_size" in kwargs:
        selected_material_size = kwargs["selected-material-size"]
        pass
    else:
        if "material-sizes" not in kwargs:
            u.msg("no material-sizes found, skipping process", "warning")
        else:
            sorted_extents = np.sort(extents)

            # Sort the materials by volume
            sorted_sizes = sorted(kwargs["material-sizes"], key=prod)

            for i, size in enumerate(sorted_sizes):
                valid = True
                sorted_sizes[i] = sorted(size)
                for i in range(3):
                    if sorted_extents[i] >= size[i]:
                        valid = False

                if valid:
                    selected_material_size = sorted_sizes[i]
                    u.msg(f"selected material size {selected_material_size}", "info")
                    break
            if not valid:
                u.msg("mesh doesn't fit within available sizes", "warning")

    # Center mesh to world origin
    mesh = mesh.apply_transform(to_origin)

    # Translate to only have positive vertices
    mesh.vertices += (extents / 2)
    mesh.vertices = np.round(mesh.vertices, prefs.NUMPY_DECIMALS)
    mesh.vertices = np.where(mesh.vertices == -0.0, 0.0, mesh.vertices)

    # Align the mesh
    if "align-part" in kwargs and kwargs["align-part"] and selected_material_size != None:
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
            if idx == 0: # x-axis
                rotation_matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
            elif idx == 1: # y-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
            else: # z-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        rotation_matrix = np.append(rotation_matrix, [[0, 0, 0, 1]], axis=0)

        # Apply the transformation
        mesh = mesh.apply_transform(rotation_matrix)

        # Update extents
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        u.msg("aligned mesh", "info")
    
    # Slice the mesh horizontally
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    steps = 15
    for i in range(steps):
        factor = extents[2] / steps
        contour = trimesh.intersections.mesh_plane(
                mesh, [0, 0, 1], [0, 0, i * factor], return_faces=False
            )

        # Convert point-pairs into singular 3D points
        try:
            points_3D, a = zip(*contour)
        except ValueError: # This occurs at the top of round surfaces
            continue

        # Store the z coordinate and convert the points to 2D
        z = points_3D[0][2]
        points_2D = np.empty((len(points_3D), 2))
        for i, p in enumerate(points_3D):
            points_2D[i] = np.array([p[0], p[1]])
            
        # Calulate the hull
        hull_2D = sp.spatial.ConvexHull(points_2D)
        hull_2D = points_2D[hull_2D.vertices]

        # Reconvert to 3D
        hull_3D = np.empty((len(hull_2D), 3))
        for i, p in enumerate(hull_2D):
            hull_3D[i] = np.array([p[0], p[1], z])

        # Plot the results for debugging
        x, y, z = zip(*hull_3D)
        ax.scatter(x, y, z, c='b', marker='o')

    plt.savefig(prefs.MESH_FOLDER + "3d_plot.png", dpi=300)

    return 0

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

