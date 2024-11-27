"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""

import os
import torch
import random
import numpy as np

from model import Model
from constants import default_dtype
from utils.model_utils import random_rotate_data, seed_everything


def tetris() -> tuple[torch.Tensor, torch.Tensor]:
    # pos = [
    #     [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
    #     # [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
    #     [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
    #     [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
    #     [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
    #     [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
    #     [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
    #     [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    # ]

    pos = [
        # Line.
        [
            [-1.50, 0.00, 0.00],
            [-0.50, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [1.50, 0.00, 0.00],
        ],
        # Square.
        [
            [-0.50, -0.50, 0.00],
            [-0.50, 0.50, 0.00],
            [0.50, 0.50, 0.00],
            [0.50, -0.50, 0.00],
        ],
        # S/Z-shape.
        [
            [-1.00, 0.50, 0.00],
            [0.00, 0.50, 0.00],
            [0.00, -0.50, 0.00],
            [1.00, -0.50, 0.00],
        ],
        # L/J-shape.
        # [
        #     [-0.75, -0.75, 0.00],
        #     [-0.75, 0.25, 0.00],
        #     [0.25, 0.25, 0.00],
        #     [1.25, 0.25, 0.00],
        # ],
        # T-shape.
        [
            [-1.00, 0.25, 0.00],
            [0.00, 0.25, 0.00],
            [1.00, 0.25, 0.00],
            [0.00, -0.75, 0.00],
        ],
        # Corner.
        [
            [-0.75, 0.25, -0.25],
            [0.25, 0.25, -0.25],
            [0.25, -0.75, -0.25],
            [0.25, 0.25, 0.75],
        ],
        # Right screw.
        [
            [-0.50, 0.25, -0.25],
            [-0.50, 0.25, 0.75],
            [0.50, 0.25, -0.25],
            [0.50, -0.75, -0.25],
        ],
        # Left screw.
        [
            [-0.75, 0.50, -0.25],
            [0.25, -0.50, 0.75],
            [0.25, 0.50, -0.25],
            [0.25, -0.50, -0.25],
        ],
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
                    .double()
                    .item()
                )
                current_accuracy += accuracy
            current_accuracy /= len(test_x)
            print(f"epoch {step:5d} | {100 * current_accuracy:5.1f}% accuracy")
            if current_accuracy == 1.0:
                break

    model_location = "tetris.mp"
    print("saving model to", model_location)
    torch.save(model.state_dict(), model_location)


def profile() -> None:
    data, labels = tetris()
    # data = data.to(device="cuda")
    # labels = labels.to(device="cuda")

    f = Model(labels.shape[1])
    # f.to(device="cuda")

    optim = torch.optim.Adam(f.parameters(), lr=0.05)

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
