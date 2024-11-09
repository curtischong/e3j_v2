import jax.numpy as jnp

import sympy as sp

from irrep import IrrepsArray
from spherical_harmonic_playground import _spherical_harmonics

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep

# https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
# I am using the spherical harmonics definition from e3x

# This is like scatter sum but for feats?
def map_feats_to_spherical_harmonic(largest_l: int, features: jnp.ndarray, normalize: bool=False) -> IrrepsArray:
    irreps_l = list(range(1, largest_l + 1))
    coefficients = []
    # jnp.zeros((num_car, num_sph), dtype=np.float64),
    for l in irreps_l:
        feats = []
        for m in range(-l, l + 1):
            feats.append(spherical_harmonics(l, m)(features))
        coefficients.append(jnp.array(feats))
    return jnp.array(coefficients)

# returns a function that you can pass x,y,z into to get the spherical harmonic
def spherical_harmonics(l: int, m: int) -> sp.Poly:
    # TODO: cache the polynomials?
    return _spherical_harmonics(l, m)

def tensor_product(irrep1: IrrepsArray, irrep2: IrrepsArray, output_l: int) -> jnp.ndarray:
    l1 = irrep1.irreps.l
    l2 = irrep2.irreps.l



if __name__ == "__main__":
    map_feats_to_spherical_harmonic()