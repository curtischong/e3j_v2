import torch
import e3x
import numpy as np

from o3.radial_basis import triangular_window


def test_triangular_window():
    def assert_matches_e3x(feat: list[int]):
        e3x_res = e3x.nn.functions.window.triangular_window(
            np.array(feat), num=3, limit=2.0
        ).tolist()
        e3simple_res = triangular_window(torch.tensor(feat), num=3, limit=2.0).tolist()
        assert e3simple_res == e3x_res, f"e3x={e3x_res}, e3simple={e3simple_res}"

    assert_matches_e3x([1.0, 2.0, 3.0])
    assert_matches_e3x([1.0, 0.0, 0.0])
    assert_matches_e3x([0.0, 0.0, 0.0])
