import dataclasses


@dataclasses.dataclass(init=False)
class Irrep:
    l: int
    p: int

    def __init__(self, l, p):
        assert l >= 0, "l (the degree of your representation) must be non-negative"
        assert p in {1, -1}, "p (the parity of your representation) must be 1 or -1"
        self.l = l
        self.p = p

# jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)

@dataclasses.dataclass(init=False)
class MulIrrep:
    mul: int
    irrep: Irrep
    def __init__(self, mul, irrep):
        assert mul >= 0, "mul must be non-negative"
        self.mul = mul
        self.irrep = irrep



# I want to use the builder pattern to build irreps
@dataclasses.dataclass(init=False)
class Irreps():
    irreps: list[MulIrrep]

    # one problem with using a string to define irreps is
    # that you don't label the features
    # you also are not saying: "these feautres have dimension x". and when you
    # pass it into the model, it all just mixes togehter
    # it's not very safe when you're reading the diff outputs to caluclate loss for exmaple
    def __init__(self, irreps: str):
        pass
