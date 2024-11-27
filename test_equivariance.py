import torch

from model import Model
from tetris import tetris
from tetris_simple import SimpleModel
from utils.equivariance_test_utils import random_rotate_data
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
    # torch.set_default_dtype(torch.float64) # changing this to float64 and making the default dtype 64 doesn't improve the equivariance tolerance
    x, y = tetris()
    # x, y = x[1:], y[1:]  # predict both chiral shapes

    num_equivariance_tests = 10
    for _step in range(num_equivariance_tests):
        model = SimpleModel(num_classes=y.shape[1])
        for positions in x:
            out = model(positions)
            out2 = model(random_rotate_data(positions))
            print("out", out.tolist())
            print("out2", out2.tolist())
            assert torch.allclose(out, out2, atol=1e-1), "model is not equivariant"
    print("the model is equivariant!")
