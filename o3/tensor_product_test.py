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
        "1x0e + 1x1o + 1x2e", feat1, normalize=True, normalization="norm"
    )
    e3nn_irrep2 = e3nn_jax.spherical_harmonics(
        "1x0e + 1x1o + 1x2e", feat2, normalize=True, normalization="norm"
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
        ]
    )
    irreps2 = Irreps(
        [
            Irrep.from_id("0e", torch.tensor(e3nn_irrep2["0e"].chunks[0].tolist()[0])),
            Irrep.from_id("1o", torch.tensor(e3nn_irrep2["1o"].chunks[0].tolist()[0])),
            Irrep.from_id("2e", torch.tensor(e3nn_irrep2["2e"].chunks[0].tolist()[0])),
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


# @pytest.mark.skip
def test_equivariance_err():
    NUM_TESTS_PER_IRREP_ID = 10

    for irrep_id in ["3x0e", "1x1o", "2x2e"]:
        max_equivariance_err = 0.0
        for _ in range(NUM_TESTS_PER_IRREP_ID):
            irreps1 = create_irreps_with_dummy_data(irrep_id, randomize_data=True)
            irreps2 = create_irreps_with_dummy_data(irrep_id, randomize_data=True)

            rot_mat = get_random_rotation_matrix_3d()
            irreps1_rot = irreps1.rotate_with_r3_rot_matrix(rot_mat)
            irreps2_rot = irreps2.rotate_with_r3_rot_matrix(rot_mat)

            tp1 = irreps1.tensor_product(irreps2)
            tp1_rot = tp1.rotate_with_wigner_d_rot_matrix(rot_mat)
            tp2_rot = irreps1_rot.tensor_product(irreps2_rot)

            for data1, data2 in zip(tp1_rot.data_flattened(), tp2_rot.data_flattened()):
                max_equivariance_err = max(max_equivariance_err, abs(data1 - data2))
        print(f"{irrep_id} max_equivariance_err", max_equivariance_err)


if __name__ == "__main__":
    # test_equivariance_err()
    test_matches_e3nn()
