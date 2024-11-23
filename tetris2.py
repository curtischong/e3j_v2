"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""

import torch

from model2 import Model


def tetris() -> tuple[torch.Tensor, torch.Tensor]:
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
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
            [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
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


# def make_batch(pos):
#     # put in torch_geometric format
#     dataset = [Data(pos=pos, x=torch.ones(4, 1)) for pos in pos]
#     return next(iter(DataLoader(dataset, batch_size=len(dataset))))

# def make_batch(pos):
#     # put in torch_geometric format
#     dataset = []
#     for p in pos:
#         irreps = []
#         for _ in range(len(p)):
#             irreps.append(Irreps.from_id("1x0e", [torch.ones(1)]))
#         dataset.append(Data(pos=p, x=irreps))

#     return next(iter(DataLoader(dataset, batch_size=len(dataset))))


def main() -> None:
    x, y = tetris()
    train_x, train_y = x[1:], y[1:]  # dont train on both chiral shapes

    x, y = tetris()
    test_x, test_y = x, y

    model = Model(num_classes=train_y.shape[1])

    print("Built a model:")
    print(model)
    print(list(model.parameters()))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # == Training ==
    for step in range(300):
        cur_loss = 0
        for positions in train_x:
            pred = model(positions)
            loss = (pred - train_y).pow(2).sum()

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
                    .mean(dtype=torch.float32)
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

    data, labels = tetris()
    data = make_batch(data)
    f = Network()

    pred = f(data)
    loss = (pred - labels).pow(2).sum()
    loss.backward()

    rotated_data, _ = tetris()
    rotated_data = make_batch(rotated_data)
    error = f(rotated_data) - f(data)
    assert error.abs().max() < 1e-10


if __name__ == "__main__":
    main()
