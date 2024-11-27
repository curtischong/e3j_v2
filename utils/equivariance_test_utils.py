import torch


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
