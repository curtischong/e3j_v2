import jax.numpy as jnp

import sympy as sp

import e3x
from irrep import Irrep
from parity import parity_for_l, parity_to_parity_idx
from spherical_harmonics_playground import _spherical_harmonics
from constants import ODD_PARITY_IDX, EVEN_PARITY, default_dtype
import numpy as np
from jaxtyping import Array, Float
import jax
from e3x.so3._spherical_harmonics_lut import _generate_spherical_harmonics_lookup_table
from e3x.so3.irreps import spherical_harmonics, solid_harmonics
# from jax import jit

# @jit
# def get_num_feats(feats_3d):
#     return feats_3d.shape[0]

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep

# https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
# I am using the spherical harmonics definition from e3x

# This is like scatter sum but for feats?
# Important! spherical harmonics are the angular solutions to the Laplace equation. So we normalize
# the feature before mapping to a representation via spherical harmonics
# This means that two vectors of different lengths but facing the same direction will have the same representation

# _generate_spherical_harmonics_lookup_table(
#       max_degree=2, num_processes=4
# )


# If you look at the spherical harmonics here: http://openmopac.net/manual/real_spherical_harmonics.html, you'll see that each cartesian axis is raised to the same power
def map_3d_feats_to_spherical_harmonics_repr(feats_3d: Float[Array, "num_feats 3"], normalize: bool=False) -> jnp.ndarray:
    num_feats = feats_3d.shape[0]
    max_l = 2
    num_coefficients_per_feat = (max_l+1)**2 # l=0 has 1 coefficient, l=1 has 3. so 4 total coefficients
    arr = jnp.zeros((2, num_coefficients_per_feat, num_feats), dtype=default_dtype)

    for ith_feat, feat in enumerate(feats_3d):
        # we are arranging the feats NOT in cartesian order: https://e3x.readthedocs.io/stable/pitfalls.html

        for l in range(max_l + 1):
            for m in range(-l, l + 1):

                # normalize the feature
                # feat_np = np.array(feat)
                # magnitude = np.linalg.norm(feat_np)
                magnitude = jnp.linalg.norm(feat)
                feat = (feat / magnitude)

                # coefficient = float(_spherical_harmonics(l, m)(*feat.tolist()))
                # coefficient = float(e3x.so3._symbolic._spherical_harmonics(l, m)(*feat))
                coefficient = solid_harmonics(feat, l, cartesian_order=False)[m + l]

                # https://chatgpt.com/share/67306530-4680-800e-b259-fd767593126c
                # be careful to assign the right parity to the coefficients!
                parity_idx = parity_to_parity_idx(parity_for_l(l))
                arr = arr.at[parity_idx, Irrep.coef_idx(l,m), ith_feat].set(coefficient)

    return arr


def map_1d_feats_to_spherical_harmonics_repr(feats_1d: list[jnp.ndarray], parity=EVEN_PARITY) -> Irrep:
    arr = jnp.array(feats_1d, dtype=default_dtype)
    return Irrep(arr, parity) # 1D features are even parity by default (cause scalars are invariant even if you rotate/flip them. e.g. the total energy of a system is invariant to those transformations)

# # returns a function that you can pass x,y,z into to get the spherical harmonic
# def spherical_harmonics(l: int, m: int, x: int, y:int, z:int) -> sp.Poly:
#     # TODO: cache the polynomials?
#     return _spherical_harmonics(l, m)



if __name__ == "__main__":
    distances = [
        [0, 0, 2],
        [0, 0, 4],
    ]
    
    print(map_3d_feats_to_spherical_harmonics_repr(jnp.array(distances)))