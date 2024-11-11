import jax.numpy as jnp

import sympy as sp

import e3x
from irrep import Irrep
from parity import parity_for_l, parity_to_parity_idx
from spherical_harmonics_playground import _spherical_harmonics
from constants import ODD_PARITY_IDX, EVEN_PARITY, default_dtype
import numpy as np

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep

# https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
# I am using the spherical harmonics definition from e3x

# This is like scatter sum but for feats?
# Important! spherical harmonics are the angular solutions to the Laplace equation. So we normalize
# the feature before mapping to a representation via spherical harmonics
# This means that two vectors of different lengths but facing the same direction will have the same representation
def map_3d_feats_to_spherical_harmonics_repr(feats_3d: list[list[float]], normalize: bool=False) -> Irrep:
    num_feats = len(feats_3d)
    assert type(feats_3d[0][0]) in [float, int], f"feats_3d must be a list of lists of floats. type(feats_3d[0][0])={type(feats_3d[0][0])}"
    max_l = 1 # l=1 since we're dealing with 3D features
    num_coefficients_per_feat = (max_l+1)**2 # l=0 has 1 coefficient, l=1 has 3. so 4 total coefficients
    arr = jnp.zeros((2, num_coefficients_per_feat, num_feats), dtype=default_dtype)

    for ith_feat, feat in enumerate(feats_3d):
        # we are arranging the feats NOT in cartesian order: https://e3x.readthedocs.io/stable/pitfalls.html
        # feats = [] # this is a 1D array of all the spherical harmonics features

        for l in range(max_l + 1):
            for m in range(-l, l + 1):

                # normalize the feature
                feat_np = np.array(feat)
                magnitude = np.linalg.norm(feat_np)
                feat = (feat_np / magnitude).tolist()

                coefficient = float(_spherical_harmonics(l, m)(*feat))
                # coefficient = float(e3x.so3._symbolic._spherical_harmonics(l, m)(*feat))

                # https://chatgpt.com/share/67306530-4680-800e-b259-fd767593126c
                # be careful to assign the right parity to the coefficients!
                parity_idx = parity_to_parity_idx(parity_for_l(l))
                arr = arr.at[parity_idx, Irrep.coef_idx(l,m), ith_feat].set(coefficient)

                # feats.append(_spherical_harmonics(l, m)(*feat))
        # arr = arr.at[num_coefficients_per_feat].set(feats)

    return Irrep(arr) # 3D features are odd parity by default (cause in real life, if you invert a magnetic field to the opposite direction, you'd want the tensor to switch direction as well)


# what is a good way to store the features?
# I think we should just store the features as a jnp array
def map_1d_feats_to_spherical_harmonics_repr(feats_1d: list[jnp.ndarray]) -> Irrep:
    arr = jnp.array(feats_1d, dtype=default_dtype)
    return Irrep(arr, EVEN_PARITY) # 1D features are even parity by default (cause scalars are invariant even if you rotate/flip them. e.g. the total energy of a system is invariant to those transformations)

# # returns a function that you can pass x,y,z into to get the spherical harmonic
# def spherical_harmonics(l: int, m: int, x: int, y:int, z:int) -> sp.Poly:
#     # TODO: cache the polynomials?
#     return _spherical_harmonics(l, m)



if __name__ == "__main__":
    distances = [
        [0, 0, 2],
        [0, 0, 4],
    ]
    
    print(map_3d_feats_to_spherical_harmonics_repr(distances))