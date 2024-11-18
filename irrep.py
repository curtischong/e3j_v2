from __future__ import annotations
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

    def id(self):
        return str(self)

class Irreps:
    irreps: map[str, list[IrrepDef]]

    def __init__(self, irrep_defs_str) -> None:
        self.irreps = map()
        irreps_defs = irrep_defs_str.split("+")
        irreps_defs = [irrep_def.strip() for irrep_def in irreps_defs]
        for irrep_def in irreps_defs:
            assert self._is_valid_irrep_def(irrep_def), f"irrep_def {irrep_def} is not valid. it need to look something like: 1x1o + 1x2e + 1x3o"
            # TODO: is there data here???

    def _is_valid_irrep_def(self, irrep_def):
        pattern = r"^\d+x\d+[eo]$"
        return bool(re.match(pattern, irrep_def))

    @staticmethod
    def from_arr(irreps_list: list[IrrepDef]):
        return Irreps("+".join([str(irrep) for irrep in irreps_list]))

    def __repr__(self) -> str:
        irreps = []
        for key, irrep_list in self.irreps.items():
            num_irreps_of_id = len(irrep_list)
            irreps.append(f"{num_irreps_of_id}x{key}")


class Irrep:
    irrepDef: IrrepDef
    data: torch.tensor

    def __init__(self, irrepDefStr: str, data):
        self.irrepDef = IrrepDef(irrepDefStr)
        self.data = data

if __name__ == "__main__":
    print(Irreps("2x1e + 1x2e + 1x3o"))
    print(Irreps.from_arr())