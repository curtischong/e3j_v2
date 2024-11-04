import jax.numpy as jnp

import sympy as sp

from spherical_harmonic_playground import _spherical_harmonics

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep

# https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
# I am using the spherical harmonics definition from e3x
def map_feat_to_spherical_harmonic(largest_l: int, features: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    irreps_l = list(range(1, largest_l + 1))
    return jnp.array([1])

# returns a function that you can pass x,y,z into to get the spherical harmonic
def spherical_harmonics(l: int, m: int) -> jnp.ndarray:
    # TODO: cache the polynomials?
    return _spherical_harmonics(l, m)

def tensor_product(irrep1: jnp.ndarray, irrep2: jnp.ndarray) -> jnp.ndarray:
    pass
