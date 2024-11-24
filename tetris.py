"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""

import torch
import random
import numpy as np

from model import Model
from constants import default_dtype


def tetris() -> tuple[torch.Tensor, torch.Tensor]:
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        # [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]
    pos = torch.tensor(pos, dtype=torch.get_default_dtype())

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = torch.tensor(
        [
            [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
            # [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
            [0, 1, 0, 0, 0, 0, 0],  # square
            [0, 0, 1, 0, 0, 0, 0],  # line
            [0, 0, 0, 1, 0, 0, 0],  # corner
            [0, 0, 0, 0, 1, 0, 0],  # L
            [0, 0, 0, 0, 0, 1, 0],  # T
            [0, 0, 0, 0, 0, 0, 1],  # zigzag
        ],
        dtype=torch.get_default_dtype(),
    )

    return pos, labels


def main() -> None:
    x, y = tetris()
    # train_x, train_y = x[1:], y[1:]  # dont train on both chiral shapes
    train_x, train_y = x, y

    x, y = tetris()
    test_x, test_y = x, y

    model = Model(num_classes=train_y.shape[1])

    print("Built a model:")
    print(model)
    print(list(model.parameters()))

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # == Training ==
    for step in range(300):
        cur_loss = 0
        for i, positions in enumerate(train_x):
            pred = model(positions)
            loss = (pred - train_y[i]).pow(2).sum()
            # print("pred", pred)
            # print("target", train_y[i])

            optim.zero_grad()
            loss.backward()
            optim.step()
            cur_loss += loss.item()
        cur_loss /= len(train_x)

        if step % 10 == 0:
            current_accuracy = 0
            for i, positions in enumerate(test_x):
                pred = model(positions)
                accuracy = (
                    model(positions)
                    .round()
                    .eq(test_y[i])
                    .mean(dtype=default_dtype)
                    .double()
                    .item()
                )
                current_accuracy += accuracy
            current_accuracy /= len(test_x)
            print(
                f"epoch {step:5d} | loss {loss:<10.1f} | {100 * accuracy:5.1f}% accuracy"
            )


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


def equivariance_test() -> None:
    torch.set_default_dtype(torch.float64)

    x, y = tetris()
    # x, y = x[1:], y[1:]  # predict both chiral shapes

    num_equivariance_tests = 10
    for _step in range(num_equivariance_tests):
        model = Model(num_classes=y.shape[1])
        for positions in x:
            out = model(positions)
            out2 = model(random_rotate_data(positions))
            assert torch.allclose(out, out2, atol=1e-6), "model is not equivariant"
    print("the model is equivariant!")


def seed_everything(seed: int):
    # Seed Python's built-in random module
    random.seed(seed)
    # Seed NumPy
    np.random.seed(seed)
    # Seed PyTorch
    torch.manual_seed(seed)
    # If using GPU, seed CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    seed_everything(143)
    main()
    # equivariance_test()
