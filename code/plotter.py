import matplotlib
matplotlib.use("Agg")
import numpy as np
import pylab as plt

ax = plt.figure(figsize=(10, 10), dpi=300).add_subplot(projection='3d')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35, roll=0)

plt.savefig("/workspace/obj/plot.png")

def plot_lines(lines, fmt):
    ax = plt.figure().add_subplot(projection='3d')

    for line in lines:
        p0 = line[0]
        p1 = line[1]
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]
        ax.plot(xs, ys, zs, fmt)

    # Set limits 
    ax.set_xlim([-3, +3])
    ax.set_ylim([-3, +3])
    ax.set_zlim([-3, +3])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.grid(False)

    # Rotate view
    ax.view_init(elev=20., azim=-40, roll=0)

    # Save as png
    plt.savefig("/workspace/obj/plot.png")
