from __future__ import annotations
import jax.numpy as jnp
import dataclasses
from jaxtyping import Float, Array
from constants import ODD_PARITY_IDX
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

@dataclasses.dataclass(init=False)
class Irrep():
    array: Float[Array, "parity=2   coefficients_for_every_l_and_m=2*(max_l+1)^2   num_feats"]
    # why can't we separate the dimensions for l and m? why are we indexing the coefficients in the coefficients_for_every_l_and_m dimension?
    # it's because each l has different numbers of m. e.g. l=0 has 1 m, l=1 has 3, l=2 has 5, etc.
    # if we used different indices for different l and m, the resulting array would look like a staircase, not a rectangle
    # and jnumpy arrays must have the same size for each index



    def __init__(self, array: jnp.ndarray):
        assert array.shape[0] %2 == 0, f"index 0 is the parity index. it should be size 2. It's size {array.shape[0]}"
        self.array = array

    # calculate l based on the dimensions of the array
    def l(self):
        num_irrep_coefficients = self.array.shape[0][0]

        # recall that 2l + 1 = the number of coefficients for that irrep
        # so 2l = num_irrep_coefficients - 1
        # l = (num_irrep_coefficients - 1) // 2
        return (num_irrep_coefficients - 1) // 2

    # this is the number of times the irrep is repeated
    def multiplicity(self):
        return self.array.shape[0][0][-1] # the number of features is defined in the the very last index
    
    def get_coefficient(self, parity_idx:int, ith_feature: int, l: int, m: int) -> float:
        start_idx_of_l = l**2 # there are l**2 - 1 coefficients for the lower levels of l. (since l=0 has 1 coefficient, l=1 has 3, l=2 has 5, etc)
        return self.array[parity_idx, start_idx_of_l + m, ith_feature]

    # returns true if there is no feature at the given parity and index i
    def is_feature_zero(self, parity: int, ith_feature: int) -> bool:
        subset = self.array[parity, ith_feature]
        return jnp.all(subset == 0)

    # Note: you only use this for predicting outputs I believe. cause it's kinda sus to just throw out all other coefficients, especially the l=0 coefficient
    # But also I think this is a bad way of predicitng outputs? I think summing across all coefficients is better. I'm only leaving this in here
    # to document the fact that we are NOT using cartesian order, and the e3x docs said you can get the vector representation by using these indices
    def get_xyz_vectors(self) -> Float[Array, "num_feats 3"]:
        assert self.array.shape[0][0] >= 4, f"This irrep doesn't have enough coefficients to get the xyz vectors. it only has {self.array.shape[0][0]} coefficients"

        # since we are NOT using cartesian order (see https://e3x.readthedocs.io/stable/pitfalls.html), we need to rearrange the array
        y = self.array[ODD_PARITY_IDX,1,:] # start at index=1 since that is the start of the coefficients of l=1
        z = self.array[ODD_PARITY_IDX,2,:]
        x = self.array[ODD_PARITY_IDX,3,:]
        return jnp.stack([x, y, z], axis=1)