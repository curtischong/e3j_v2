from functools import lru_cache
import e3nn_jax as e3nn

@lru_cache(maxsize=None)
def get_clebsch_gordan(l1: int, l2: int, l_out: int) -> str:
    cg = e3nn.clebsch_gordan(l1, l2, l_out)
    return cg
