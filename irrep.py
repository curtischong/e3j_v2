from __future__ import annotations
from collections import defaultdict
import torch
import dataclasses
import re

@dataclasses.dataclass(init=False)
class IrrepDef:
    l: int
    parity: int

    def __init__(self, l, parity):
        assert l >= 0, "l (the degree of your representation) must be non-negative"
        assert parity in {1, -1}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {p}"
        self.l = l
        self.parity = parity

    def __repr__(self):
        if self.parity == 1:
            parity_str = "e"
        elif self.parity == -1:
            parity_str = "o"
        return f"{self.l}{parity_str}"
    
    def tensor_product(self, other: IrrepDef) -> list[IrrepDef]:
        out_parity = self.parity * other.parity
        min_l = abs(self.l - other.l)
        max_l = self.l + other.l

        res_irreps = []
        for l in range(min_l, max_l + 1):
            res_irreps.append(IrrepDef(l, out_parity))
        return res_irreps

    def id(self):
        return self.__repr__()

class Irreps:
    # irreps: dict[str, list[IrrepDef]]
    irreps: list[IrrepDef]

    def __init__(self, irrep_defs_str: str) -> None:
        self.irreps = []
        irreps_defs = irrep_defs_str.split("+")
        irreps_defs = [irrep_def.strip() for irrep_def in irreps_defs]
        irrep_pattern = r"^(\d+)x+(\d+)([eo])$"
        for irrep_def in irreps_defs:

            # create irrepDefs from the string
            match = re.match(irrep_pattern, irrep_def)
            if not bool(match):
                raise ValueError(f"irrep_def {irrep_def} is not valid. it need to look something like: 1x1o + 1x2e + 1x3o")
            num_irreps, l_str, parity_str = match.groups()
            parity = -1 if parity_str == "o" else 1
            for _ in range(int(num_irreps)):
                self.irreps.append(IrrepDef(int(l_str), parity))

    @staticmethod
    def from_list(irreps_list: list[IrrepDef]):
        return Irreps("+".join(["1x" + irrep.id() for irrep in irreps_list]))

    def __repr__(self) -> str:
        irreps_count_of_same_l_and_parity = defaultdict(int)
        max_l = 0
        for irrep in self.irreps:
            max_l = max(max_l, irrep.l)
            irreps_count_of_same_l_and_parity[irrep.id()] += 1

        # order the representations by l and parity (so it is easier to read)
        consolidated_repr = []
        for i in range(0, max_l + 1):
            for parity in [1, -1]:
                irrep_def_id = IrrepDef(i, parity).id()
                if irrep_def_id in irreps_count_of_same_l_and_parity:
                    num_irreps_of_id = irreps_count_of_same_l_and_parity[irrep_def_id]
                    consolidated_repr.append(f"{num_irreps_of_id}x{irrep_def_id}")
        return " + ".join(consolidated_repr)

    def id(self):
        return self.__repr__()

    def tensor_product(self, other: Irreps):
        new_irreps = []
        for irrep1 in self.irreps:
            for irrep2 in other.irreps:
                new_irreps.extend(irrep1.tensor_product(irrep2))
        return Irreps.from_list(new_irreps)

class Irrep:
    irrepDef: IrrepDef
    data: torch.tensor

    def __init__(self, irrepDefStr: str, data):
        self.irrepDef = IrrepDef(irrepDefStr)
        self.data = data

if __name__ == "__main__":
    print(Irreps("2x1e + 1x2e + 1x3o"))
    print(Irreps.from_arr())
    print(Irreps("1x0e+1x1o+1x2e").tensor_product(Irreps("1x0e+1x1o+1x2e")))