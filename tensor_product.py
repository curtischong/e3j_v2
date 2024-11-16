from typing import Optional
from clebsch_gordan import get_clebsch_gordan
from constants import EVEN_PARITY, EVEN_PARITY_IDX, NUM_PARITY_DIMS, ODD_PARITY, ODD_PARITY_IDX, PARITY_IDXS
from parity import parity_idx_to_parity, parity_to_parity_idx
from irrep import Irrep
import jax
import jax.numpy as jnp
import e3x

def tensor_product_v1(irrep1: jnp.ndarray, irrep2: jnp.ndarray, max_l3: Optional[int]) -> jnp.ndarray:
    max_l1 = Irrep.l(irrep1)
    max_l2 = Irrep.l(irrep2)

    # after we do the tensor product, there will be num_irrep1_feats * num_irrep2_feats features
    num_irrep1_feats = Irrep.num_features(irrep1)
    num_irrep2_feats = Irrep.num_features(irrep2)
    num_output_feats = num_irrep1_feats * num_irrep2_feats

    max_output_l = max_l1 + max_l2
    # if max_l3 is None:
    #     max_output_l = max_l1 + max_l2
    # else:
    #     max_output_l = max_l3

    num_coefficients_per_feat = (max_output_l+1)**2 # l=0 has 1 coefficient, l=1 has 3, l=2 has 5, etc. This formula gives the sum of all these coefficients

    out = jnp.zeros((NUM_PARITY_DIMS, num_coefficients_per_feat, num_output_feats), dtype=jnp.float32)

    for feat1_idx in range(num_irrep1_feats):
        for parity1_idx in PARITY_IDXS:
            for parity2_idx in PARITY_IDXS:
                for feat2_idx in range(num_irrep2_feats):
                    if Irrep.is_feature_zero(irrep1, parity1_idx, feat1_idx) or Irrep.is_feature_zero(irrep2, parity2_idx, feat2_idx):
                        continue

                    feat3_idx = feat1_idx * num_irrep2_feats + feat2_idx
                    parity3 = parity_idx_to_parity(parity1_idx) * parity_idx_to_parity(parity2_idx)
                    parity3_idx = parity_to_parity_idx(parity3)


                    # calculate the repr for the output l
                    for l1 in range(max_l1):
                        for l2 in range(max_l2):

                            # for each of the features in irrep1 and irrep2, calculate the tensor product
                            l3_min = abs(l1 - l2)
                            for l3 in range(l3_min, l1 + l2 + 1):
                                for m1 in range(-l1, l1 + 1):
                                    for m2 in range(-l2, l2 + 1):
                                        m3 = m1 + m2
                                        if m3 < -l3 or m3 > l3:
                                            continue
                                        v1 = Irrep.get_coefficient(irrep1, parity1_idx, feat1_idx, l1, m1)
                                        v2 = Irrep.get_coefficient(irrep2, parity2_idx, feat2_idx, l2, m2)
                                        coef_idx = Irrep.coef_idx(l3, m3)
                                        cg = get_clebsch_gordan(l1, l2, l3, m1, m2, m3)
                                        normalization = 1
                                        out = out.at[parity3_idx, coef_idx, feat3_idx].add(cg*v1*v2*normalization)
    return out

e3x.Config.set_clebsch_gordan_cache("/Users/curtischong/Documents/dev/e3j/cache")

@jax.jit
def tensor_product_v2(irrep1: jnp.ndarray, irrep2: jnp.ndarray) -> jnp.ndarray:
    max_l1 = Irrep.l(irrep1)
    max_l2 = Irrep.l(irrep2)
    
    num_irrep1_feats = Irrep.num_features(irrep1)
    num_irrep2_feats = Irrep.num_features(irrep2)
    num_output_feats = num_irrep1_feats * num_irrep2_feats
    
    # if max_l3 is None:
    #     max_l3 = max_l1 + max_l2
    # else:
    #     max_l3 = min(max_l3, max_l1 + max_l2)
    max_l3 = max_l1 + max_l2
    
    num_coefficients_per_feat = (max_l3 + 1) ** 2
    out = jnp.zeros((NUM_PARITY_DIMS, num_coefficients_per_feat, num_output_feats), dtype=jnp.float32)
    
    # Precompute parity combinations
    parity3_indices = jnp.array([[EVEN_PARITY_IDX, ODD_PARITY_IDX], [ODD_PARITY_IDX, EVEN_PARITY_IDX]])
    
    # Precompute feature indices combinations
    feat1_indices = jnp.arange(num_irrep1_feats)
    feat2_indices = jnp.arange(num_irrep2_feats)
    feat3_indices = jnp.reshape(feat1_indices[:, None] * num_irrep2_feats + feat2_indices[None, :], -1)
    
    # Loop over parity combinations
    for p1 in range(NUM_PARITY_DIMS):
        for p2 in range(NUM_PARITY_DIMS):
            p3 = parity3_indices[p1, p2]
            irrep1_p = irrep1[p1]  # Shape: [(max_l1+1)**2, num_irrep1_feats]
            irrep2_p = irrep2[p2]  # Shape: [(max_l2+1)**2, num_irrep2_feats]
            
            # Loop over l1, l2, l3
            for l1 in range(max_l1 + 1):
                indices1 = Irrep.coef_indices_for_l(l1)
                num_m1 = 2 * l1 + 1
                v1 = irrep1_p[indices1, :]  # Shape: [num_m1, num_irrep1_feats]
                
                for l2 in range(max_l2 + 1):
                    indices2 = Irrep.coef_indices_for_l(l2)
                    num_m2 = 2 * l2 + 1
                    v2 = irrep2_p[indices2, :]  # Shape: [num_m2, num_irrep2_feats]
                    
                    l3_min = abs(l1 - l2)
                    l3_max = min(l1 + l2, max_l3)
                    for l3 in range(l3_min, l3_max + 1):
                        indices3 = Irrep.coef_indices_for_l(l3)
                        num_m3 = 2 * l3 + 1
                        
                        # Get Clebsch-Gordan coefficients
                        cg_matrix = e3x.so3.irreps.clebsch_gordan_for_degrees(l1, l2, l3)  # Shape: [num_m1, num_m2, num_m3]
                        cg_matrix = cg_matrix.reshape((num_m1 * num_m2, num_m3))

                        # Compute outer product of v1 and v2
                        v1v2 = jnp.einsum('im,jn->ijmn', v1, v2)  # Shape: [num_m1, num_irrep1_feats, num_m2, num_irrep2_feats]
                        v1v2 = v1v2.reshape((num_m1 * num_m2, num_irrep1_feats * num_irrep2_feats))
                        
                        # Multiply with Clebsch-Gordan coefficients
                        v3 = cg_matrix.T @ v1v2  # Shape: [num_m3, num_output_feats]
                        
                        # Accumulate into output tensor
                        out = out.at[p3, indices3[:, None], feat3_indices[None, :]].add(v3)
    
    print("done tensor product")
    return out
