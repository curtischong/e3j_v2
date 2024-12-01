from o3 import spherical_harmonics
from o3.irrep import Irrep, Irreps

# from tensor_product import tensor_product_v1, tensor_product_v2
import jax.numpy as jnp
import e3nn_jax
import torch
import numpy as np
import pytest

import e3nn.o3
from e3nn.util.test import equivariance_error

from o3.spherical_harmonics import map_3d_feats_to_basis_functions
from utils.dummy_data_utils import create_irreps_with_dummy_data
from utils.model_utils import seed_everything
from utils.rot_utils import D_from_matrix, get_random_rotation_matrix_3d


def flatten_e3nn_tensor(irrep: e3nn_jax.IrrepsArray) -> list[float]:
    return irrep.array
    # all_data = []
    # for chunk in irrep.chunks:
    #     all_data.extend(chunk.flatten().tolist())
    # return all_data


def test_matches_e3nn():
    feat1 = jnp.array([1, 1, 1])
    feat2 = jnp.array([1, 1, 2])

    # first get the e3nn tensor product
    e3nn_irrep1 = e3nn_jax.spherical_harmonics(
        "1x0e + 1x1o + 1x2e + 1x3o", feat1, normalize=True, normalization="norm"
    )
    e3nn_irrep2 = e3nn_jax.spherical_harmonics(
        "1x0e + 1x1o + 1x2e + 1x3o", feat2, normalize=True, normalization="norm"
    )
    print("e3nn irreps:")
    print(e3nn_irrep1)
    print(e3nn_irrep2)
    print("e3nn tensor product:")
    e3nn_tensor_product = e3nn_jax.tensor_product(
        e3nn_irrep1, e3nn_irrep2, irrep_normalization="component"
    )
    print(e3nn_tensor_product)

    # first get the e3simple tensor product
    irreps1 = Irreps(
        [
            Irrep.from_id("0e", torch.tensor(e3nn_irrep1["0e"].chunks[0].tolist()[0])),
            Irrep.from_id("1o", torch.tensor(e3nn_irrep1["1o"].chunks[0].tolist()[0])),
            Irrep.from_id("2e", torch.tensor(e3nn_irrep1["2e"].chunks[0].tolist()[0])),
            Irrep.from_id("3o", torch.tensor(e3nn_irrep1["3o"].chunks[0].tolist()[0])),
        ]
    )
    irreps2 = Irreps(
        [
            Irrep.from_id("0e", torch.tensor(e3nn_irrep2["0e"].chunks[0].tolist()[0])),
            Irrep.from_id("1o", torch.tensor(e3nn_irrep2["1o"].chunks[0].tolist()[0])),
            Irrep.from_id("2e", torch.tensor(e3nn_irrep2["2e"].chunks[0].tolist()[0])),
            Irrep.from_id("3o", torch.tensor(e3nn_irrep2["3o"].chunks[0].tolist()[0])),
        ]
    )

    print("e3simple irreps:")
    print(irreps1)
    print(irreps2)
    print("e3simple tensor product:")

    e3simple_tensor_product = irreps1.tensor_product(irreps2, norm_type="component")

    e3simple_tensor_product_data = np.array(e3simple_tensor_product.data_flattened())
    e3nn_tensor_product_data = np.array(flatten_e3nn_tensor(e3nn_tensor_product))

    print(e3simple_tensor_product)
    assert len(e3simple_tensor_product_data) == len(
        e3nn_tensor_product_data
    ), "the two tensor products should have the same number of coefficients"

    is_tp_equal = np.allclose(
        e3simple_tensor_product_data,
        e3nn_tensor_product_data,
        atol=1e-2,
    )
    if not is_tp_equal:
        for i in range(len(e3simple_tensor_product_data)):
            print(
                f"{i}: e3simple_coeff={e3simple_tensor_product_data[i]}, e3nn_coeff={e3nn_tensor_product_data[i]}"
            )
            if not np.allclose(
                e3simple_tensor_product_data[i], e3nn_tensor_product_data[i], atol=1e-2
            ):
                print(
                    f"{i}: e3simple_coeff={e3simple_tensor_product_data[i]}, e3nn_coeff={e3nn_tensor_product_data[i]}"
                )
                raise ValueError("the two tensor products should be equivalent")
    print("the two tensor products are equivalent!")


def test_matches_e3nn2():
    irrep_ids = [
        ("1x0e + 1x1o", "1x0e + 1x1o + 1x2e"),
        ("1x0e + 1x1o + 1x2e", "1x0e + 1x1o"),
        ("1x0e + 1x1o + 1x2e", "1x0e + 1x1o + 1x2e"),
        ("1x3o", "1x3o"),
        ("1x3e + 1x0e", "1x3e"),
        ("1x2e", "1x3e"),
        ("1x0e", "1x1o + 1x0e"),
    ]
    for irrep1_id, irrep2_id in irrep_ids:
        irrep1 = create_irreps_with_dummy_data(irrep1_id, randomize_data=True)
        irrep2 = create_irreps_with_dummy_data(irrep2_id, randomize_data=True)
        irrep1_e3nn = e3nn_jax.IrrepsArray(
            # IMPORTANT: we need to use irrep1.id() since it sorts the id
            irrep1.id(),
            jnp.array(irrep1.data_flattened()),
        )
        irrep2_e3nn = e3nn_jax.IrrepsArray(
            irrep2.id(), jnp.array(irrep2.data_flattened())
        )
        tp = irrep1.tensor_product(irrep2)
        tp_e3nn = e3nn_jax.tensor_product(irrep1_e3nn, irrep2_e3nn)
        # assert jnp.allclose(jnp.array(tp.data_flattened()), jnp.array(tp_e3nn.array))
        tp_data = tp.data_flattened()
        tp_e3nn_data = tp_e3nn.array

        assert (
            tp_e3nn.irreps == tp.id()
        ), f"irrep ids do not match. tp_e3nn={tp_e3nn.irreps}, tp_e3simple={tp.id()}"
        assert len(tp_data) == len(
            tp_e3nn_data
        ), "the two tensor products should have the same number of coefficients"

        for i in range(len(tp_data)):
            if not jnp.allclose(tp_data[i], tp_e3nn_data[i]):
                print(f"irrep1_id={irrep1_id}, irrep2_id={irrep2_id}")
                print(f"i={i}")
                print(f"tp_data[i]={tp_data[i]}")
                print(f"tp_e3nn_data[i]={tp_e3nn_data[i]}")
                print(tp)
                print(tp_e3nn)
                raise ValueError("the two tensor products should be equivalent")
        print("the two tensor products are equivalent!")


# @pytest.mark.skip
def test_equivariance_err():
    for max_l in range(0, 4):
        max_equivariance_err = 0.0
        print("max_l", max_l)
        feats1 = torch.randn(5, 3)
        all_irreps1 = map_3d_feats_to_basis_functions(
            feats1, num_scalar_feats=1, max_l=max_l
        )
        feats2 = torch.randn(5, 3)
        all_irreps2 = map_3d_feats_to_basis_functions(
            feats2, num_scalar_feats=1, max_l=max_l
        )
        for irreps1, irreps2 in zip(all_irreps1, all_irreps2):
            rot_mat = get_random_rotation_matrix_3d()
            irreps1_rot = irreps1.rotate_with_r3_rot_matrix(rot_mat)
            irreps2_rot = irreps2.rotate_with_r3_rot_matrix(rot_mat)

            tp1 = irreps1.tensor_product(irreps2, norm_type="none")
            tp1_rot = tp1.rotate_with_wigner_d_rot_matrix(rot_mat)
            tp2_rot = irreps1_rot.tensor_product(irreps2_rot, norm_type="none")

            for data1, data2 in zip(tp1_rot.data_flattened(), tp2_rot.data_flattened()):
                # print(data1, data2)
                max_equivariance_err = max(max_equivariance_err, abs(data1 - data2))
            # assert (
            #     abs(data1 - data2) < 1e-2
            # ), f"{irrep_id} max_equivariance_err {max_equivariance_err}"
            # print(f"{irrep_id} max_equivariance_err", max_equivariance_err)
            if max_equivariance_err > 1e-2:
                print(f"max_l={max_l} max_equivariance_err {max_equivariance_err}")
                print("tp1_rot", tp1_rot)
                print("tp2_rot", tp2_rot)
                exit()


if __name__ == "__main__":
    seed_everything(143)
    # test_equivariance_err()
    # test_matches_e3nn()
    test_matches_e3nn2()
