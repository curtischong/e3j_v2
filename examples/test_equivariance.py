import torch

from examples.tetris import Model
from examples.tetris_data import tetris
from examples.tetris_simple import SimpleModel, SimpleModel2
from utils.model_utils import plot_3d_coords, random_rotate_data, seed_everything
import pytest


@pytest.mark.skip
def test_tetris_equivariance():
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


def test_tetris_simple_equivariance():
    seed_everything(143)
    # torch.set_default_dtype(torch.float64) # changing this to float64 and making the default dtype 64 doesn't improve the equivariance tolerance
    x, y = tetris()

    # no need to set training mode to false since we have no dropout/batch norm layers. so the output should be the same

    num_equivariance_tests = 5
    for _step in range(num_equivariance_tests):
        model = SimpleModel2(
            num_classes=y.shape[1]
        )  # init a new model so it's weights are random
        for positions in x:
            # plot_3d_coords(
            #     positions.numpy()
            # )  # plot the original data and manually verify it looks legit
            # out = torch.mean(model(positions))
            out = model(positions)

            rotated_pos = random_rotate_data(positions)
            # plot_3d_coords(rotated_pos.numpy())
            print("pos", positions.tolist())
            print("rotated_pos", rotated_pos.tolist())

            # out2 = torch.mean(model(rotated_pos))
            out2 = model(rotated_pos)
            print("out1", out.tolist())
            print("out2", out2.tolist())
            assert torch.allclose(out, out2, atol=1e-7), "model is not equivariant"
    print("the model is equivariant!")
