import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_lines(lines, **kwargs):  # color, marker, linestyle
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")

        # Rotate view
        if i == 0:
            ax.view_init(elev=90, azim=0, roll=0)
        elif i == 1:
            ax.view_init(elev=0, azim=0, roll=90)
        elif i == 2:
            ax.view_init(elev=0, azim=90, roll=0)
        else:
            ax.view_init(elev=20.0, azim=-35, roll=0)

        # Plot lines
        for line in lines:
            v0 = line[0]
            v1 = line[1]
            xs = [v0.x, v1.x]
            ys = [v0.y, v1.y]
            zs = [v0.z, v1.z]
            ax.plot(xs, ys, zs, **kwargs)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Remove tick labels
        if i == 0:
            ax.set_zticklabels([])
        elif i == 1:
            ax.set_xticklabels([])
        elif i == 2:
            ax.set_yticklabels([])

        ax.grid(False)
        ax.axis("scaled")

    # Save as png
    plt.savefig("/workspace/obj/plot.png")

