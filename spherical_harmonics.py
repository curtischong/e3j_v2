import jax.numpy as jnp

import torch

from irrep import Irrep, Irreps
from spherical_harmonics_playground import _spherical_harmonics
from utils.spherical_harmonics_utils import parity_for_l, to_cartesian_order_idx


def map_3d_feats_to_spherical_harmonics_repr(
    feats_3d: torch.tensor, max_l: int = 2
) -> list[Irreps]:
    # maps the 3D feature to the specified spherical harmonic

    # https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
    # I am using the spherical harmonics definition from e3x
    # cartesian form of the real spherical harmonics http://openmopac.net/manual/real_spherical_harmonics.html
    irreps_out = []
    for feat in feats_3d:
        # ensure we're using cartesian order: https://e3x.readthedocs.io/stable/pitfalls.html

        irreps = []
        for l in range(max_l + 1):
            coefficients = torch.zeros(2 * l + 1)
            for m in range(-l, l + 1):
                # IMPORTANT! normalize the feature (so it is a point on the surface of a unit sphere). This is cause spherical harmonics are the ANGULAR solutions to the Laplace equation.
                # This means that two vectors of different lengths but facing the same direction will have the same representation
                magnitude = torch.linalg.norm(feat)
                feat = feat / magnitude

                coefficient = float(
                    _spherical_harmonics(l, m)(
                        feat[0].item(), feat[1].item(), feat[2].item()
                    )
                )  # assuming feat is [x,y,z]

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
