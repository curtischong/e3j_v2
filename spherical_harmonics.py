import jax.numpy as jnp
import sympy as sp

import torch

from constants import EVEN_PARITY
from irrep import Irrep, Irreps
from utils.spherical_harmonics_utils import parity_for_l, to_cartesian_order_idx


def map_3d_feats_to_basis_functions(
    feats_3d: torch.Tensor, num_scalar_feats: int, max_l: int = 2
) -> list[Irreps]:
    sh_irreps = map_3d_feats_to_spherical_harmonics_repr(feats_3d, max_l)
    # return feats_3d
    # now we need to use a radial function to create the 0e representation

    all_out_irreps: list[Irreps] = []
    for irreps in sh_irreps:
        tensor_norm = torch.norm(irreps.irreps[0].data)
        new_scalars = triangular_window(x=tensor_norm, num=num_scalar_feats, limit=2.0)
        # the question I have is: a radial basis will basically give it more 0x irreps (since we are using a linear combination of hte basis functions to represent the radial part)
        # for tetris implimentations, it shouldn't matter since the neighbors are just 1 away
        new_irreps: list[Irrep] = irreps.irreps[1:]

        for scalar in new_scalars:
            new_irreps.append(Irrep(l=0, parity=EVEN_PARITY, data=scalar.unsqueeze(0)))
        all_out_irreps.append(Irreps(new_irreps))

    return all_out_irreps


def map_3d_feats_to_spherical_harmonics_repr(
    feats_3d: torch.Tensor, max_l: int = 2
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
                # copy e3x, where they avoid dividing by 0 by dividing by 1
                feat = torch.where(magnitude == 0, feat, feat / magnitude)

                coefficient = float(
                    _spherical_harmonics(l, m)(
                        feat[0].item(), feat[1].item(), feat[2].item()
                    )
                )

                # https://chatgpt.com/share/67306530-4680-800e-b259-fd767593126c
                # be careful to assign the right parity to the coefficients!
                # Note: parity_for_l matches e3nn here. you cannot assign a parity to an l that doesn't make physical sense (e.g. l=0 cannot have an odd parity)
                coefficient_idx = to_cartesian_order_idx(l, m)
                coefficients[coefficient_idx] = coefficient
            irreps.append(Irrep(l, parity_for_l(l), coefficients))
        irreps_out.append(Irreps(irreps))

    return irreps_out


# this code is from e3x
# they use sympy (so we an simplify the math into polynomials that take in x,y,z and outputs a number for the spherical harmonic coefficient)
# wait no. the bigger question is why are they not taking in x,y,z as a param?
# you can find usages of _spherical_harmonics by searching for: from ._symbolic import _spherical_harmonics
# or just go to usages on the definition. the linter is broken since it says it's not used since it has a _ in the file, but it's not directly used in that file


# here is the math: https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
def _spherical_harmonics(l: int, m: int) -> sp.Poly:
    """Real Cartesian spherical harmonics.

    Computes a symbolic expression for the spherical harmonics of degree l and
    order m (as polynomial) with sympy. Note: The spherical harmonics computed
    here use Racah's normalization (also known as Schmidt's semi-normalization):
                ∫ Ylm(r)·Yl'm'(r) dΩ = 4π/(2l+1)·δ(l,l')·δ(m,m')
    (the integral runs over the unit sphere Ω and δ is the delta function).

    Args:
      l: Degree of the spherical harmonic.
      m: Order of the spherical harmonic.

    Returns:
      A sympy.Poly object with a symbolic expression for the spherical harmonic
      of degree l and order m.
    """

    def B(m: int, x: sp.Symbol, y: sp.Symbol) -> sp.Symbol:
        a = sp.S(0)
        for k in range(m + 1):
            a += sp.binomial(m, k) * x**k * y ** (m - k) * sp.cos((m - k) * sp.pi / 2)
        return a

    def A(m: int, x: sp.Symbol, y: sp.Symbol) -> sp.Symbol:
        b = sp.S(0)
        for k in range(m + 1):
            b += sp.binomial(m, k) * x**k * y ** (m - k) * sp.sin((m - k) * sp.pi / 2)
        return b

    def pi(l: int, m: int, x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> sp.Symbol:
        pi = sp.S(0)
        r2 = x**2 + y**2 + z**2
        for k in range((l - m) // 2 + 1):
            pi += (
                (-1) ** k
                * sp.S(2) ** (-l)
                * sp.binomial(l, k)
                * sp.binomial(2 * l - 2 * k, l)
                * sp.factorial(l - 2 * k)
                / sp.factorial(l - 2 * k - m)
                * z ** (l - 2 * k - m)
                * r2
                ** k  # this is kinda sus see https://github.com/google-research/e3x/issues/22
            )
        return sp.sqrt(sp.factorial(l - m) / sp.factorial(l + m)) * pi

    x, y, z = sp.symbols("x y z")
    if m < 0:
        ylm = sp.sqrt(2) * pi(l, -m, x, y, z) * A(-m, x, y)
    elif m > 0:
        ylm = sp.sqrt(2) * pi(l, m, x, y, z) * B(m, x, y)
    else:
        ylm = pi(l, m, x, y, z)

    return sp.Poly(sp.simplify(ylm), x, y, z)


if __name__ == "__main__":
    # test how _spherical_harmonics works

    print(_spherical_harmonics(0, 0))
    print(_spherical_harmonics(1, 0))
    print(_spherical_harmonics(1, 0)(0, 0, 0.2))

    print(_spherical_harmonics(3, 2))
    print(
        _spherical_harmonics(3, 3)
    )  # what does domain EX mean? it just means that it's an expression (if the polynomial's coefficients are integers, it's zz - this happens when the expression is a constant like 1 or a monomial like z - since z = 1*z)

    print(_spherical_harmonics(2, 2).terms())

    # test my logic to map 3D features to spherical harmonics
    distances = [
        [0, 0, 2],
        [0, 0, 4],
    ]

    print(map_3d_feats_to_spherical_harmonics_repr(torch.tensor(distances)))
