from __future__ import annotations
import jax.numpy as jnp
import dataclasses
from jaxtyping import Float, Array
from constants import ODD_PARITY_IDX
import math
# https://e3x.readthedocs.io/stable/overview.html
# this page is pretty informative^


# @dataclasses.dataclass(init=False)
# class Irrep:
#     l: int
#     p: int # TODO: use a bool instead?

#     def __init__(self, l, p):
#         assert l >= 0, "l (the degree of your representation) must be non-negative"
#         assert p in {1, -1}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {p}"
#         self.l = l
#         self.p = p

# do we need to register into jax?
# jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)

IrrepType = Float[Array, "parity=2   coefficients_for_every_l_and_m=(max_l+1)^2   num_feats"]
@dataclasses.dataclass(init=False)
class Irrep():
    # why can't we separate the dimensions for l and m? why are we indexing the coefficients in the coefficients_for_every_l_and_m dimension?
    # it's because each l has different numbers of m. e.g. l=0 has 1 m, l=1 has 3, l=2 has 5, etc.
    # if we used different indices for different l and m, the resulting array would look like a staircase, not a rectangle
    # and jnumpy arrays must have the same size for each index

    @staticmethod
    def coef_idx(l: int, m:int) -> "Irrep":
        start_idx_of_l = l**2 # there are l**2 - 1 coefficients for the lower levels of l. (since l=0 has 1 coefficient, l=1 has 3, l=2 has 5, etc)
        m_offset = m + l # (add l since m starts at -l and goes to l)
        return start_idx_of_l + m_offset

    # calculate l based on the dimensions of the array
    @staticmethod
    def l(array: jnp.ndarray) -> int:
        num_irrep_coefficients = int(array.shape[1])
        max_l = int(math.sqrt(num_irrep_coefficients)) - 1
        return max_l

    # this is the number of times the irrep is repeated
    @staticmethod
    def num_features(array):
        return int(array.shape[2])

    @staticmethod
    def get_coefficient(array: jnp.ndarray, parity_idx:int, ith_feature: int, l: int, m: int) -> float:
        return array[parity_idx, Irrep.coef_idx(l,m), ith_feature]

    @staticmethod
    def get_ith_feature(array, parity_idx: int, ith_feature: int) -> float:
        return array[parity_idx, :, ith_feature]
    
    @staticmethod
    def slice_ith_feature(array, ith_feature: int) -> float:
        return jnp.expand_dims(array[:, :, ith_feature], axis=-1)
    # returns true if there is no feature at the given parity and index i

    @staticmethod
    def is_feature_zero(array: jnp.ndarray, parity_idx: int, ith_feature: int) -> jnp.bool_:
        subset = Irrep.get_ith_feature(array, parity_idx, ith_feature)
        # print("subset:", subset, jnp.all(subset == 0))
        return jnp.all(subset == 0)
    
    @staticmethod
    def coef_indices_for_l(l: int) -> jnp.ndarray:
        return jnp.arange(l**2, (l+1)**2)

    # Note: you only use this for predicting outputs I believe. cause it's kinda sus to just throw out all other coefficients, especially the l=0 coefficient
    # But also I think this is a bad way of predicitng outputs? I think summing across all coefficients is better. I'm only leaving this in here
    # to document the fact that we are NOT using cartesian order, and the e3x docs said you can get the vector representation by using these indices
    @staticmethod
    def get_xyz_vectors(array: IrrepType) -> Float[Array, "num_feats 3"]:
        assert array.shape[0][0] >= 4, f"This irrep doesn't have enough coefficients to get the xyz vectors. it only has {array.shape[0][0]} coefficients"

        # since we are NOT using cartesian order (see https://e3x.readthedocs.io/stable/pitfalls.html), we need to rearrange the array
        y = array[ODD_PARITY_IDX,1,:] # start at index=1 since that is the start of the coefficients of l=1
        z = array[ODD_PARITY_IDX,2,:]
        x = array[ODD_PARITY_IDX,3,:]
        return jnp.stack([x, y, z], axis=1)