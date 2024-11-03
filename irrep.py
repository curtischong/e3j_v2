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

@dataclasses.dataclass(init=False)
class MulIrrep:
    multiplicity: int
    irrep: Irrep
    def __init__(self, multiplicity, irrep):
        assert multiplicity >= 0, "multiplicity (the number of times you want to repeat your representation) must be non-negative"
        self.multiplicity = multiplicity
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

    # I think in general: we want the layer-wise irreps to be defined via a string
    # but the input/output irreps need to be defined by a better abstraction so we explicitly
    # mention the input features
    def __init__(self, irreps_str: str):
        self.irreps = []

        individual_irreps = [raw_irrep.strip() for raw_irrep in irreps_str.split("+")]
        for raw_irrep in individual_irreps:
            parts = raw_irrep.split("x")

            if len(parts) > 1:
                multiplicity = int(parts[0])
                irrep_kind = parts[1]
            else:
                multiplicity = 1
                irrep_kind = raw_irrep

            l = int(irrep_kind[:-1])

            is_even = irrep_kind[-1] == "e"
            parity = 1 if is_even else -1
            irrep = Irrep(l, parity)

            self.irreps.append(MulIrrep(multiplicity, irrep))
            
# print(Irreps("2x0e + 1o").irreps)
assert(Irreps("2x0e + 1o").irreps == [MulIrrep(2, Irrep(0, 1)), MulIrrep(1, Irrep(1, -1))])