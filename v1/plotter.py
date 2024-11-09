import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import helpers as h
from vector import Vector3

def _plot_lines(ax: Axes3D, lines: list, color: str, marker: str):
    if not h.is_structured(lines, "(n, 2)"):
        raise ValueError("list not structured correctly")

    for l in lines:
        v0 = l[0]
        v1 = l[1]
        if isinstance(v0, Vector3) and isinstance(v1, Vector3):
            ax.plot(
                [v0.x, v1.x], [v0.y, v1.y], [v0.z, v1.z], color=color, marker=marker
            )
        else:
            ax.plot(
                [v0[0], v1[0]],
                [v0[1], v1[1]],
                [v0[2], v1[2]],
                color=color,
                marker=marker,
            )

def _plot_points(ax: Axes3D, points: list, color: str, marker: str):
    if not h.is_structured(points, "(n, 3)"):
        raise ValueError("list not structured correctly")

    for p in points:
        ax.plot(p.x, p.y, p.z, color=color, marker=marker)

def plot(**kwargs):
    # Check for proper usage
    file_name = "unnamed"
    lines = None
    points = None

    for key, value in kwargs.items():
        if key == "file_name":
            file_name = value
        elif key == "lines":
            lines = value
            try:
                line_color = kwargs["line_color"]
                line_marker = kwargs["line_marker"]
            except KeyError:
                h.print_error(
                    "plot lines failed: please include line_color and line_marker"
                )
                return 1
        elif key == "points":
            points = value
            try:
                point_color = kwargs["point_color"]
                point_marker = kwargs["point_marker"]
            except KeyError:
                h.print_error(
                    "plot points failed: please include point_color and point_marker"
                )
                return 1

    if isinstance(lines, list) or isinstance(points, list):
        fig = plt.figure()
        for i in range(4):
            if "subplots" in kwargs and kwargs["subplots"]:
                ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            else:
                if i < 1:
                    ax = fig.add_subplot(projection="3d")
                else:
                    break

            # Plot values
            if isinstance(lines, list):
                _plot_lines(ax, lines, line_color, line_marker)
            if isinstance(points, list):
                _plot_points(ax, points, point_color, point_marker)

            # Rotate view
            if i < 1:
                ax.view_init(elev=20.0, azim=-35, roll=0)
            elif i < 2:
                ax.view_init(elev=-90, azim=-90, roll=0)
            elif i < 3:
                ax.view_init(elev=0, azim=-90, roll=0)
            else:
                ax.view_init(elev=0, azim=0, roll=90)

            # Set labels
            if i != 3:
                ax.set_xlabel("X")
            if i != 2:
                ax.set_ylabel("Y")
            if i != 1:
                ax.set_zlabel("Z")

            # Remove tick labels
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_zticklabels([])

            ax.grid(False)
            ax.axis("scaled")

        # Save as png
        plt.savefig("/workspace/plots/" + file_name)
        return 0
    h.print_error("plotter has no valid points or lines")
    return 1

