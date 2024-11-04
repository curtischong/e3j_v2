import sympy as sp

# this is how e3x does it
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
          * r2**k # this is kinda sus see https://github.com/google-research/e3x/issues/22
      )
    return sp.sqrt(sp.factorial(l - m) / sp.factorial(l + m)) * pi

  x, y, z = sp.symbols('x y z')
  if m < 0:
    ylm = sp.sqrt(2) * pi(l, -m, x, y, z) * A(-m, x, y)
  elif m > 0:
    ylm = sp.sqrt(2) * pi(l, m, x, y, z) * B(m, x, y)
  else:
    ylm = pi(l, m, x, y, z)

  return sp.Poly(sp.simplify(ylm), x, y, z)

if __name__ == "__main__":
  print(_spherical_harmonics(0, 0))
  print(_spherical_harmonics(1, 0))
  print(_spherical_harmonics(1, 0)(1,0,1))

  print(_spherical_harmonics(3, 2))
  print(_spherical_harmonics(3, 3)) # what does domain EX mean? it just means that it's an expression (if the polynomial's coefficients are integers, it's zz - this happens when the expression is a constant like 1 or a monomial like z - since z = 1*z)

  print(_spherical_harmonics(2, 2).terms())