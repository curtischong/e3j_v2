import e3nn_jax as e3nn
import jax.numpy as jnp
from e3x.so3.irreps import spherical_harmonics
from utils.constants import ODD_PARITY_IDX, EVEN_PARITY_IDX
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
import torch
import pytest


def test_spherical_harmonics_fn_matches_e3x():
    def assert_matches_e3x(feat):
        jfeat = jnp.array(feat)
        e3x_res = spherical_harmonics(
            jfeat, 1, cartesian_order=True, normalization="racah"
        ).flatten()
        e3simple_res = map_3d_feats_to_spherical_harmonics_repr(
            torch.tensor(feat).unsqueeze(0), max_l=1
        )[0].data_flattened()
        assert jnp.allclose(
            e3x_res, jnp.array(e3simple_res)
        ), f"e3x={e3x_res}, e3simple={e3simple_res}"

    assert_matches_e3x([0.0, 0.0, 1.0])
    assert_matches_e3x([1.2, 2.0, -1.0])
    assert_matches_e3x([-2, -2.0, -1.0])


def test_diff_lengths_but_same_dir_have_same_sh_repr():
    assert jnp.array_equal(
        map_3d_feats_to_spherical_harmonics_repr(torch.tensor([[1.0, 1, 1]]))[
            0
        ].data_flattened(),
        map_3d_feats_to_spherical_harmonics_repr(torch.tensor([[2.0, 2, 2]]))[
            0
        ].data_flattened(),
    ), "two vectors facing the same direction should have the same representation (despite having diff magnitudes)"


def test_spherical_harmonics_fn_matches_e3nn():
    pass
    # feat = [0.0, 0.0, 1.0]
    # print(spherical_harmonics(jnp.array(feat), 1, cartesian_order=False))
    # # vector = e3nn.IrrepsArray("1o", jnp.array(feat))
    # # print(e3nn.spherical_harmonics(2, vector, normalize=False))
    # # print(e3nn.spherical_harmonics(2, vector, normalize=True))
    # # print(e3nn.spherical_harmonics([], vector, normalize=True))

    # print(map_3d_feats_to_spherical_harmonics_repr([feat]).array)
    # def assert_matches_e3nn(feat):
    #     jfeat = jnp.array(feat)
    #     e3x_res = spherical_harmonics(jfeat, 1, cartesian_order=False, normalization='racah')
    #     e3simple_res = jnp.squeeze(map_3d_feats_to_spherical_harmonics_repr(jnp.expand_dims(feat, axis=0)).array[ODD_PARITY_IDX])
    #     assert jnp.allclose(e3x_res, e3simple_res), f"e3x={e3x_res}, e3simple={e3simple_res}"
    # # assert it works with e3nn
    # feat = jnp.array([1.2, 2.0, -1.0])
    # # print(e3nn.spherical_harmonics("1x0e + 1x1o", np.array(feat), normalize=True))
    # print(e3nn.spherical_harmonics("1x0e + 1x1o", feat, normalize=True, normalization="norm").array)
    # e3simple_res = jnp.squeeze(map_3d_feats_to_spherical_harmonics_repr(jnp.expand_dims(feat, axis=0)).array[ODD_PARITY_IDX])
    # print(e3simple_res)
