"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""

import os
import torch
import random
import numpy as np
import wandb

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
    # wandb.login()
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="e3simple-tetris",
    #     # Track hyperparameters and run metadata
    #     config={
    #         # "learning_rate": lr,
    #         # "epochs": epochs,
    #     },
    # )
    # os.environ["WANDB_MODE"] = "disabled"

    x, y = tetris()
    # train_x, train_y = x[1:], y[1:]  # dont train on both chiral shapes
    train_x, train_y = x, y

    x, y = tetris()
    test_x, test_y = x, y

    model = Model(num_classes=train_y.shape[1])

    print("Built a model:")
    print(model)
    print(list(model.parameters()))
    print("------------------end of params--------------------")

    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(300):
        optim.zero_grad()
        loss = torch.zeros(1)
        for i, positions in enumerate(train_x):
            pred: torch.Tensor = model(positions)
            loss += (pred - train_y[i]).pow(2).sum()
        loss.backward()
        print(f"loss {loss.item():<10.20f}")

        # cur_loss /= len(train_x)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if step % 10 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}_______", param.grad.tolist())
                else:
                    print(f"{name}_______")

            current_accuracy = 0
            for i, positions in enumerate(test_x):
                pred = model(positions)
                print("raw pred", pred.tolist())
                one_hot = torch.zeros_like(pred)
                predicted_class = torch.argmax(pred, dim=0)
                one_hot[predicted_class] = 1
                print("pred", one_hot.tolist())
                print("targ", test_y[i].tolist())
                accuracy = (
                    model(positions)
                    .argmax(dim=0)
                    .eq(test_y[i].argmax(dim=0))
                    # .mean(dtype=default_dtype)
                    .double()
                    .item()
                )
                current_accuracy += accuracy
            current_accuracy /= len(test_x)
            print(f"epoch {step:5d} | {100 * current_accuracy:5.1f}% accuracy")

    # wandb.finish()


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


def profile() -> None:
    data, labels = tetris()
    # data = data.to(device="cuda")
    # labels = labels.to(device="cuda")

    f = Model(labels.shape[1])
    # f.to(device="cuda")

    optim = torch.optim.Adam(f.parameters(), lr=1e-2)

    called_num = [0]

    def trace_handler(p) -> None:
        os.makedirs("profiles", exist_ok=True)
        # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        p.export_chrome_trace("profiles/test_trace_" + str(called_num[0]) + ".json")
        called_num[0] += 1

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=50, warmup=1, active=1),
        on_trace_ready=trace_handler,
    ) as p:
        for _ in range(52):
            for i, positions in enumerate(data):
                optim.zero_grad()
                pred = f(positions)
                loss = (pred - labels[i]).pow(2).sum()

                loss.backward()
                optim.step()

                p.step()


if __name__ == "__main__":
    seed_everything(143)
    # profile()
    main()
    # equivariance_test()
