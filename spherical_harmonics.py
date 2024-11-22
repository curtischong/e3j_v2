import jax.numpy as jnp

from constants import EVEN_PARITY, ODD_PARITY, default_dtype
from e3x.so3.irreps import solid_harmonics
import torch

from irrep import Irrep, Irreps
from spherical_harmonics_playground import _spherical_harmonics
from utils.spherical_harmonics_utils import parity_for_l, to_cartesian_order_idx
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
def map_3d_feats_to_spherical_harmonics_repr(
    feats_3d: torch.tensor, max_l: int = 2
) -> list[Irreps]:
    irreps_out = []
    for feat in feats_3d:
        # ensure we're using cartesian order: https://e3x.readthedocs.io/stable/pitfalls.html

        irreps = []
        for l in range(max_l + 1):
            coefficients = torch.zeros(2 * l + 1)
            for m in range(-l, l + 1):
                # normalize the feature
                magnitude = torch.linalg.norm(feat)
                feat = feat / magnitude

                coefficient = float(
                    _spherical_harmonics(l, m)(
                        feat[0].item(), feat[1].item(), feat[2].item()
                    )
                )  # assuming feat is [x,y,z]
                # coefficient = float(e3x.so3._symbolic._spherical_harmonics(l, m)(*feat))
                # coefficient = solid_harmonics(feat, l, cartesian_order=False)[m + l]

                # https://chatgpt.com/share/67306530-4680-800e-b259-fd767593126c
                # be careful to assign the right parity to the coefficients!
                # Note: parity_for_l matches e3nn here. you cannot assign a parity to an l that doesn't make physical sense (e.g. l=0 cannot have an odd parity)
                coefficient_idx = to_cartesian_order_idx(l, m)
                coefficients[coefficient_idx] = coefficient
            irreps.append(Irrep(l, parity_for_l(l), coefficients))
        irreps_out.append(Irreps(irreps))

    return irreps_out


if __name__ == "__main__":
    distances = [
        [0, 0, 2],
        [0, 0, 4],
    ]

    print(map_3d_feats_to_spherical_harmonics_repr(jnp.array(distances)))
