import torch
import matplotlib.pyplot as plt
import numpy as np


def random_rotate_data(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(
            "Input tensor must have the last dimension of size 3 (representing 3D vectors)."
        )

    # Generate a random rotation matrix using axis-angle representation
    angle = torch.rand(1) * 2 * torch.pi  # Random angle in radians
    axis = torch.randn(3)  # Random axis
    axis = axis / axis.norm()  # Normalize axis to unit vector

    # Compute rotation matrix using Rodrigues' rotation formula
    K = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )  # Skew-symmetric matrix for cross product
    I = torch.eye(3)  # Identity matrix
    rotation_matrix = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    # Apply the rotation
    rotated_vector = torch.einsum("ij,...j->...i", rotation_matrix, vector)

    return rotated_vector


def plot_3d_coords(coords: np.ndarray, title="3D Coordinates Plot"):
    """
    Plots 3D coordinates.

    Parameters:
        coords (list or np.ndarray): A list or array of 3D coordinates. Each coordinate should be a tuple (x, y, z).
        title (str): Title of the plot.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if coords.shape[1] != 3:
        raise ValueError(
            "Input coordinates must have three columns representing x, y, and z."
        )

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c="blue", marker="o", label="Points")

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_aspect(
        "equal", adjustable="box"
    )  # by default the scales of the axis are NOT the equal (z-axis is shorter)
    ax.legend()
    plt.show()
