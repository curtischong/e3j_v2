import jax.numpy as jnp
import dataclasses
# https://e3x.readthedocs.io/stable/overview.html
# this page is pretty informative^


@dataclasses.dataclass(init=False)
class Irrep:
    l: int
    p: int # TODO: use a bool instead?

    def __init__(self, l, p):
        assert l >= 0, "l (the degree of your representation) must be non-negative"
        assert p in {1, -1}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {p}"
        self.l = l
        self.p = p

# do we need to register into jax?
# jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)

@dataclasses.dataclass()
class IrrepsArray():
    irreps: Irrep
    array: jnp.ndarray