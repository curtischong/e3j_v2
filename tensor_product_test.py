from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from tensor_product import tensor_product_v1, tensor_product_v2
import jax.numpy as jnp
from e3nn_jax import tensor_product
import e3nn_jax


def test_same_results_across_versions():
    feat1 = jnp.array([1,1,1])
    feat2 = jnp.array([1,1,2])

    irrep1 = map_3d_feats_to_spherical_harmonics_repr(jnp.expand_dims(feat1, axis=0))
    irrep2 = map_3d_feats_to_spherical_harmonics_repr(jnp.expand_dims(feat2, axis=0))
    v1_res = tensor_product_v1(irrep1, irrep2, max_l3=5)
    v2_res = tensor_product_v2(irrep1, irrep2)
    print("v1 res:", v1_res)
    print("v2 res:", v2_res)
    assert jnp.allclose(v1_res, v2_res)

def test_matches_e3nn():
    feat1 = jnp.array([1,1,1])
    feat2 = jnp.array([1,1,2])

    irrep1 = map_3d_feats_to_spherical_harmonics_repr(jnp.expand_dims(feat1, axis=0))
    irrep2 = map_3d_feats_to_spherical_harmonics_repr(jnp.expand_dims(feat2, axis=0))

    print ("e3j irreps:")
    print(irrep1)
    print(irrep2)

    print(tensor_product_v1(irrep1, irrep2).tolist())


    e3nn_irrep1 = e3nn_jax.spherical_harmonics("1x0e + 1x1o", feat1, normalize=True, normalization="norm")
    e3nn_irrep2 = e3nn_jax.spherical_harmonics("1x0e + 1x1o", feat2, normalize=True, normalization="norm")
    print("e3nn irreps:")
    print(e3nn_irrep1)
    print(e3nn_irrep2)
    print("e3nn tensor product:")
    print(tensor_product(e3nn_irrep1, e3nn_irrep2,irrep_normalization="norm"))


    # # Create simple irreps with known coefficients
    # max_l = 1
    # num_feats = 1
    # array_shape = (2, (max_l + 1)**2, num_feats)
    # array1 = jnp.zeros(array_shape)
    # array2 = jnp.zeros(array_shape)

    # # Set some coefficients to non-zero values
    # # For parity index 0 (even parity), l=1, m=0
    # coef_idx_l1_m0 = Irrep.coef_idx(1, 0)
    # array1 = array1.at[0, coef_idx_l1_m0, 0].set(1.0)
    # array2 = array2.at[0, coef_idx_l1_m0, 0].set(1.0)

    # irrep1 = Irrep(array1)
    # irrep2 = Irrep(array2)
    # print("irrep1:", irrep1)

    # # Compute tensor product
    # result = tensor_product_v1(irrep1, irrep2)

    # # Print the result
    # print("Resulting array shape:", result.shape)
    # print("Resulting coefficients:", result)

if __name__ == "__main__":
    test_same_results_across_versions()