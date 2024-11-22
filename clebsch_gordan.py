from functools import lru_cache
import e3nn_jax

from utils import to_cartesian_order_idx


def get_clebsch_gordan(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
    m1_idx = to_cartesian_order_idx(l1, m1)
    m2_idx = to_cartesian_order_idx(l2, m2)
    m3_idx = to_cartesian_order_idx(l3, m3)
    return _get_clebsch_gordan(l1, l2, l3)[m1_idx, m2_idx, m3_idx]

@lru_cache(maxsize=None)
def _get_clebsch_gordan(l1: int, l2: int, l_out: int) -> str:
    return e3nn_jax.clebsch_gordan(l1, l2, l_out)

if __name__ == "__main__":
    print(e3nn_jax.clebsch_gordan(1, 1, 2))
    print(e3nn_jax.clebsch_gordan(1, 1, 2).shape) # this has shape (3,3,5)