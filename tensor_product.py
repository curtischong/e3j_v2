from clebsch_gordan import get_clebsch_gordan
from constants import NUM_PARITY_DIMS, PARITY_IDXS
from parity import parity_idx_to_parity, parity_to_parity_idx
from irrep import Irrep
import jax.numpy as jnp
import math


def tensor_product_v1(irrep1: Irrep, irrep2: Irrep) -> jnp.ndarray:
    max_l1 = irrep1.l()
    max_l2 = irrep2.l()
    num_irrep1_feats = irrep1.multiplicity()
    num_irrep2_feats = irrep2.multiplicity()
    num_output_feats = num_irrep1_feats * num_irrep2_feats

    max_output_l = max_l1 + max_l2
    num_coefficients_per_feat = (max_output_l + 1) ** 2
    out = jnp.zeros((NUM_PARITY_DIMS, num_coefficients_per_feat, num_output_feats), dtype=jnp.float32)

    for feat1_idx in range(num_irrep1_feats):
        for feat2_idx in range(num_irrep2_feats):
            for parity1_idx in PARITY_IDXS:
                for parity2_idx in PARITY_IDXS:
                    if irrep1.is_feature_zero(parity1_idx, feat1_idx) or irrep2.is_feature_zero(parity2_idx, feat2_idx):
                        continue
                    feat3_idx = feat1_idx * num_irrep2_feats + feat2_idx
                    parity3 = parity_idx_to_parity(parity1_idx) * parity_idx_to_parity(parity2_idx)
                    parity3_idx = parity_to_parity_idx(parity3)

                    for l1 in range(max_l1 + 1):
                        for m1 in range(-l1, l1 + 1):
                            v1 = irrep1.get_coefficient(parity1_idx, feat1_idx, l1, m1)
                            if v1 == 0:
                                continue
                            for l2 in range(max_l2 + 1):
                                for m2 in range(-l2, l2 + 1):
                                    v2 = irrep2.get_coefficient(parity2_idx, feat2_idx, l2, m2)
                                    if v2 == 0:
                                        continue
                                    l3_min = abs(l1 - l2)
                                    l3_max_current = l1 + l2
                                    for l3 in range(l3_min, l3_max_current + 1):
                                        if l3 > max_output_l:
                                            continue
                                        m3 = m1 + m2
                                        if m3 < -l3 or m3 > l3:
                                            continue
                                        cg = get_clebsch_gordan(l1, l2, l3, m1, m2, m3)
                                        if cg == 0:
                                            continue
                                        # normalization =  math.sqrt(2 * l3 + 1)
                                        # normalization = math.sqrt(num_coefficients_per_feat)
                                        normalization = 1
                                        # print("CG coefficient:", cg, l3, m3)
                                        coef_idx = Irrep.coef_idx(l3, m3)
                                        out = out.at[parity3_idx, coef_idx, feat3_idx].add(cg * v1 * v2 * normalization)
    return out



# how do you do the tensor product if you have n irreps for one input, and m irreps for the other input?
# we do n*m tensor products

# where does e3nn and e3x apply the weights?