import matplotlib.pyplot as plt
import numpy as np

def plot_3d_coords(coords, title="3D Coordinates Plot"):
    """
    Plots 3D coordinates.

    Parameters:
        coords (list or np.ndarray): A list or array of 3D coordinates. Each coordinate should be a tuple (x, y, z).
        title (str): Title of the plot.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if coords.shape[1] != 3:
        raise ValueError("Input coordinates must have three columns representing x, y, and z.")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='blue', marker='o', label='Points')

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_aspect('equal', adjustable='box') # by default the scales of the axis are NOT the equal (z-axis is shorter)
    ax.legend()
    plt.show()
