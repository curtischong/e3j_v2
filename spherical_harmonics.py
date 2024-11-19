import jax.numpy as jnp

from parity import parity_for_l, parity_to_parity_idx
from constants import default_dtype
from jaxtyping import Array, Float
from e3x.so3.irreps import solid_harmonics
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
                # Note: parity_for_l matches e3nn here. you cannot assign a parity to an l that doesn't make physical sense (e.g. l=0 cannot have an odd parity)
                parity_idx = parity_to_parity_idx(parity_for_l(l))
                arr = arr.at[parity_idx, IrrepDef.coef_idx(l,m), ith_feat].set(coefficient)

    return arr


def to_cartesian_order_idx(l: int, m: int):
    # to learn more about what Cartesian order is, see https://e3x.readthedocs.io/stable/pitfalls.html
    # TLDR: it's the order in which we index the coefficients of a spherical harmonic function
    abs_m = abs(m)
    num_m_in_l = 2*l + 1

    if m == 0:
        return num_m_in_l - 1
    elif m < 0:
        return num_m_in_l - 1 - 2*abs_m + 1
    else:
        return num_m_in_l - 1 - 2*abs_m


if __name__ == "__main__":
    distances = [
        [0, 0, 2],
        [0, 0, 4],
    ]
    
    print(map_3d_feats_to_spherical_harmonics_repr(jnp.array(distances)))