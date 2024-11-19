from constants import EVEN_PARITY, ODD_PARITY
from irrep import Irrep, Irreps
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr

# from tensor_product import tensor_product_v1, tensor_product_v2
import jax.numpy as jnp
import e3nn_jax
import torch
import numpy as np


def flatten_e3nn_tensor(irrep: e3nn_jax.IrrepsArray) -> list[float]:
    all_data = []
    for chunk in irrep.chunks:
        all_data.extend(chunk.flatten().tolist())
    return all_data


def test_matches_e3nn():
    feat1 = jnp.array([1, 1, 1])
    feat2 = jnp.array([1, 1, 2])

    # first get the e3nn tensor product
    e3nn_irrep1 = e3nn_jax.spherical_harmonics(
        "1x0e + 1x1o", feat1, normalize=True, normalization="norm"
    )
    e3nn_irrep2 = e3nn_jax.spherical_harmonics(
        "1x0e + 1x1o", feat2, normalize=True, normalization="norm"
    )
    print("e3nn irreps:")
    print(e3nn_irrep1)
    print(e3nn_irrep2)
    print("e3nn tensor product:")
    e3nn_tensor_product = e3nn_jax.tensor_product(
        e3nn_irrep1, e3nn_irrep2, irrep_normalization="none"
    )
    print(e3nn_tensor_product)

    # first get the e3simple tensor product
    irreps1 = Irreps(
        [
            Irrep.from_id("0e", torch.tensor(e3nn_irrep1["0e"].chunks[0].tolist()[0])),
            Irrep.from_id("1o", torch.tensor(e3nn_irrep1["1o"].chunks[0].tolist()[0])),
        ]
    )
    irreps2 = Irreps(
        [
            Irrep.from_id("0e", torch.tensor(e3nn_irrep2["0e"].chunks[0].tolist()[0])),
            Irrep.from_id("1o", torch.tensor(e3nn_irrep2["1o"].chunks[0].tolist()[0])),
        ]
    )

    print("e3simple irreps:")
    print(irreps1)
    print(irreps2)
    print("e3simple tensor product:")

    e3simple_tensor_product = irreps1.tensor_product(irreps2)

    e3simple_tensor_product_data = np.array(e3simple_tensor_product.data_flattened())
    e3nn_tensor_product_data = np.array(flatten_e3nn_tensor(e3nn_tensor_product))

    print(e3simple_tensor_product_data)
    print(e3nn_tensor_product_data)
    assert np.allclose(
        e3simple_tensor_product_data,
        e3nn_tensor_product_data,
        atol=1e-6,
    )
