import torch
from e3nn.o3._wigner import wigner_D


def get_random_rotation_matrix_3d() -> torch.Tensor:
    """
    Example:
    # Batch of 3D vectors
    vectors = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])

    # Rotate the batch of vectors
    rotated_vectors = vectors @ R.T
    """
    # Generate a random rotation matrix using axis-angle representation
    angle = torch.rand(1) * 2 * torch.pi  # Random angle in radians
    axis = torch.randn(3)  # Random axis
    axis = axis / axis.norm()  # Normalize axis to unit vector

    # Compute rotation matrix using Rodrigues' rotation formula
    K = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )  # Skew-symmetric matrix for cross product
    I = torch.eye(3)  # Identity matrix
    return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def random_rotate_data(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(
            "Input tensor must have the last dimension of size 3 (representing 3D vectors)."
        )
    rotation_matrix = get_random_rotation_matrix_3d()

    # Apply the rotation
    rotated_vector = torch.einsum("ij,...j->...i", rotation_matrix, vector)

    return rotated_vector


# from e3nn
def D_from_matrix(R: torch.Tensor, l: int, parity: int) -> torch.Tensor:
    r"""Matrix of the representation

    Parameters
    ----------
    R : `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
    """
    d = torch.det(R).sign()
    R = d[..., None, None] * R
    k = (1 - d) / 2
    return D_from_angles(*matrix_to_angles(R), l, parity, k)


def D_from_angles(alpha, beta, gamma, l: int, parity: int, k=None) -> torch.Tensor:
    r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

    (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`
        How many times the parity is applied.

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 2l+1, 2l+1)`

    See Also
    --------
    o3.wigner_D
    Irreps.D_from_angles
    """
    if k is None:
        k = torch.zeros_like(alpha)

    alpha, beta, gamma, k = torch.broadcast_tensors(alpha, beta, gamma, k)
    return wigner_D(l, alpha, beta, gamma) * parity ** k[..., None, None]


def matrix_to_angles(R):
    r"""conversion from matrix to angles

    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert torch.allclose(torch.det(R), R.new_tensor(1))
    x = R @ R.new_tensor([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def xyz_to_angles(xyz):
    r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`

    .. math::

        \vec r = R(\alpha, \beta, 0) \vec e_z


    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    xyz = torch.nn.functional.normalize(
        xyz, p=2, dim=-1
    )  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


def angles_to_matrix(alpha, beta, gamma) -> torch.Tensor:
    r"""conversion from angles to matrix

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around X axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Y axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_z(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Z axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )
