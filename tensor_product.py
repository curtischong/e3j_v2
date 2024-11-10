from clebsch_gordan import get_clebsch_gordan
from constants import NUM_PARITY_DIMS
from irrep import Irrep
import jax.numpy as jnp


def tensor_product_v1(irrep1: Irrep, irrep2: Irrep, max_output_l: int) -> jnp.ndarray:
    max_l1 = irrep1.l()
    max_l2 = irrep2.l()

    # after we do the tensor product, there will be num_irrep1_feats * num_irrep2_feats features
    num_irrep1_feats = irrep1.multiplicity()
    num_irrep2_feats = irrep2.multiplicity()
    num_output_feats = num_irrep1_feats * num_irrep2_feats

    assert max_output_l <= max(max_l1, max_l2) + 1 # TODO: verify that the max_output_l can grow at most 1 more than the largest of the input ls

    num_coefficients_per_feat = (max_output_l+1)**2 # l=0 has 1 coefficient, l=1 has 3, l=2 has 5, etc. This formula gives the sum of all these coefficients

    out = jnp.zeros((NUM_PARITY_DIMS, num_coefficients_per_feat, num_output_feats), dtype=jnp.float32)

    for feat1_idx in range(num_irrep1_feats):
        for parity1 in range(0,2):
            for parity2 in range(0,2):
                for feat2_idx in range(num_irrep2_feats):
                    if irrep1.is_feature_zero(parity1, feat1_idx) or irrep2.is_feature_zero(parity2, feat2_idx):
                        continue

                    feat3_idx = feat1_idx * num_irrep2_feats + feat2_idx
                    parity3 = parity1 * parity2

                    # for each of the features in irrep1 and irrep2, calculate the tensor product
                    for l3 in range(max_output_l):
                        start_idx_of_l = l3**2
                        for m3 in range(-l3, l3 + 1):

                            # calculate the repr for the output l
                            for l1 in range(max_l1):
                                for l2 in range(max_l2):
                                    for m1 in range(-l1, l1 + 1):
                                        for m2 in range(-l2, l2 + 1):
                                            v1 = irrep1.get_coefficient(parity1, feat1_idx, l1, m1)
                                            v2 = irrep2.get_coefficient(parity2, feat2_idx, l2, m2)
                                            out = out.at[parity3, start_idx_of_l + m3, feat3_idx].add(get_clebsch_gordan(l1, l2, l3, m1, m2, m3)*v1*v2)
    return out


# how do you do the tensor product if you have n irreps for one input, and m irreps for the other input?
# we do n*m tensor products

# where does e3nn and e3x apply the weights?