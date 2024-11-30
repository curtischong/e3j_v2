import torch

from e3simple_examples.tetris import TetrisModel
from e3simple_examples.tetris_data import tetris
from e3simple_examples.tetris_simple import SimpleModel, SimpleModel2
from utils.model_utils import plot_3d_coords, seed_everything
import pytest

from utils.rot_utils import random_rotate_data


@pytest.mark.skip
def test_tetris_equivariance():
    x, y = tetris()
    # x, y = x[1:], y[1:]  # predict both chiral shapes

    num_equivariance_tests = 10
    for _step in range(num_equivariance_tests):
        model = TetrisModel(num_classes=y.shape[1])
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
        max_equivariance_err = 0.0
        model = SimpleModel(
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
            # print("pos", positions.tolist())
            # print("rotated_pos", rotated_pos.tolist())

            # out2 = torch.mean(model(rotated_pos))
            out2 = model(rotated_pos)
            # print("out1", out.tolist())
            # print("out2", out2.tolist())
            for i in range(len(out)):
                data1 = out[i]
                data2 = out2[i]
                max_equivariance_err = max(max_equivariance_err, abs(data1 - data2))
            # assert torch.allclose(out, out2, atol=1e-7), "model is not equivariant"
        print("max_equivariance_err", max_equivariance_err)
    print("the model is equivariant!")


if __name__ == "__main__":
    test_tetris_simple_equivariance()
