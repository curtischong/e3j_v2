from constants import EVEN_PARITY, ODD_PARITY


def parity_for_l(l: int) -> int:
    if l % 2 == 0:
        return EVEN_PARITY
    else:
        return ODD_PARITY
    # PERF: reduce this to one line
    # return (l % 2)*(-2) + 1

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

