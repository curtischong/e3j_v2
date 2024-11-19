from __future__ import annotations
from collections import defaultdict
import torch
import dataclasses
import re

from clebsch_gordan import get_clebsch_gordan
from constants import EVEN_PARITY, ODD_PARITY
from spherical_harmonics import to_cartesian_order_idx


class Irreps:
    irreps: list[Irrep] # this is always sorted from smallest l to largest l and odd parity to even parity

    def __init__(self, irreps_list: list[Irrep]):
        assert irreps_list, "irreps_list must not be empty"
        self.irreps = sorted(irreps_list, key=lambda irrep: (irrep.l, irrep.parity))

    @staticmethod
    def from_id(id: str, data: list[torch.Tensor]) -> Irreps:
        irreps_defs = id.split("+")
        irreps_defs = [irrep_def.strip() for irrep_def in irreps_defs]
        irreps_pattern = r"^(\d+)x+(\d+)([eo])$"

        data_idx = 0 # advance to the next data when we create the next Irrep object
        irreps = []
        for irrep_def in irreps_defs:
            # create irreps from the string
            match = re.match(irreps_pattern, irrep_def)
            if not bool(match):
                raise ValueError(f"irrep_def {irrep_def} is not valid. it need to look something like: 1x1o + 1x2e + 1x3o")
            num_irreps, l_str, parity_str = match.groups()

            l = int(l_str)
            parity = ODD_PARITY if parity_str == "o" else EVEN_PARITY

            for _ in range(int(num_irreps)):
                if data_idx >= len(data):
                    raise ValueError(f"not enough data for the irrep {l}x{parity_str}. you need {l} data tensors")
                irreps.append(Irrep(l, parity, data[data_idx]))
                data_idx += 1

        assert len(irreps) == len(data), f"the number of irreps ({len(irreps)}) must match the number of data tensors ({len(data)})"
        return Irreps(irreps)


    def __repr__(self) -> str:
        consolidated_data = []
        for irrep in self.irreps:
            consolidated_data.extend(irrep.data.tolist())
        return f"{self.id()}: {str(consolidated_data)}"


    def id(self):
        irrep_ids_with_cnt = []
        current_id = self.irreps[0].id()
        irrep_of_same_id_cnt = 1

        for irrep in self.irreps[1:]:
            if irrep.id() == current_id:
                irrep_of_same_id_cnt += 1
            else:
                irrep_ids_with_cnt.append(f"{irrep_of_same_id_cnt}x{current_id}")
                current_id = irrep.id()
                irrep_of_same_id_cnt = 1

        # Append the last group
        irrep_ids_with_cnt.append(f"{irrep_of_same_id_cnt}x{current_id}")

        return "+".join(irrep_ids_with_cnt)


    def tensor_product(self, other: Irreps):
        new_irreps = []
        for irrep1 in self.irreps:
            for irrep2 in other.irreps:
                new_irreps.extend(irrep1.tensor_product(irrep2))
        return Irreps(new_irreps)

    def data(self):
        return [irrep.data for irrep in self.irreps]

@dataclasses.dataclass(init=False)
class Irrep():
    l: int
    parity: int
    data: torch.Tensor

    def __init__(self, l: int, parity: int, data: torch.Tensor):
        assert l >= 0, "l (the degree of your representation) must be non-negative"
        assert parity in {EVEN_PARITY, ODD_PARITY}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {parity}"
        assert data.numel() == 2*l + 1, f"Expected {2*l + 1} coefficients for l={l}, parity={parity}. Got {data.numel()} coefficients instead"
        assert data.dim() == 1, f"data array passed to irrep is {data.dim}-dimensional. Please make sure it's 1D instead"
        self.l = l
        self.parity = parity
        self.data = data

    @staticmethod
    def from_id(irrep_id: str, data: torch.Tensor):
        irrep_pattern = r"^(\d+)([eo])$"
        match = re.match(irrep_pattern, irrep_id)
        if not bool(match):
            raise ValueError(f"irrep_id {irrep_id} is not valid. it need to look something like: 1o or 7e. (this is the order l followed by the parity (e or o)")

        l_str, parity_str = match.groups()
        parity = -1 if parity_str == "o" else 1

        return Irrep(int(l_str), parity, data)

    def get_coefficient(self, m:int) -> float:
        return self.data[to_cartesian_order_idx(self.l, m)]

    def tensor_product(self, irrep2: Irrep) -> list[Irrep]:
        irrep1 = self

        l1 = irrep1.l
        l2 = irrep2.l
        l_min = abs(l1 - l2)
        l_max = l1 + l2

        parity_out = irrep1.parity * irrep2.parity

        res_irreps = []

        # the tensor product of these two irreps will generate (max_l+1 - min_l) new irreps. where each new irrep has l=l_out and parity=parity_out
        for l_out in range(l_min, l_max + 1):
            # calculate all 2l+1 coefficients for this irrep here
            coefficients = [0]*(2*l_out+1)
            for m3 in range(-l_out, l_out + 1):

                # here we are doing the summation to get the coefficient for this m3 (see assets/tensor_product.png for the formula)
                coefficient = 0
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        cg = get_clebsch_gordan(l1, l2, l_out, m1, m2, m3)
                        v1 = irrep1.get_coefficient(m1)
                        v2 = irrep2.get_coefficient(m2)
                        normalization = 1
                        coefficient += cg*v1*v2*normalization

                m3_idx = to_cartesian_order_idx(l_out, m3) # put the coefficient in the right index since we're following Cartesian order convention https://e3x.readthedocs.io/stable/pitfalls.html
                coefficients[m3_idx] = coefficient

                coefficients = torch.tensor(coefficients)
            res_irreps.append(Irrep(l_out, parity_out, coefficients))
        return res_irreps

    
    def __repr__(self) -> str:
        id = self.id()
        return f"{id}: {self.data.tolist()}"

    def id(self):
        if self.parity == 1:
            parity_str = "e"
        elif self.parity == -1:
            parity_str = "o"
        return f"{self.l}{parity_str}"
