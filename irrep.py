from __future__ import annotations
from collections import defaultdict
import torch
import dataclasses
import re

from clebsch_gordan import get_clebsch_gordan
from constants import EVEN_PARITY, ODD_PARITY
from e3nn.o3._wigner import _so3_clebsch_gordan
from e3nn_jax import clebsch_gordan
import e3nn_jax

from spherical_harmonics import to_cartesian_order_idx


class Irreps:
    irreps: list[Irrep] # this is always sorted from smallest l to largest l and odd parity to even parity

    def __init__(self, irreps_list: list[Irrep]):
        self.irreps = irreps_list


    def __repr__(self) -> str:
        irreps_of_same_l_and_parity = defaultdict(list[Irrep])
        max_l = 0
        for irrep in self.irreps:
            max_l = max(max_l, irrep.l)
            irreps_of_same_l_and_parity[irrep.id()].append(irrep)

        # order the representations by l and parity (so it is easier to read)
        consolidated_repr = []
        consolidated_data = []
        for i in range(0, max_l + 1):
            for parity in [ODD_PARITY, EVEN_PARITY]:
                irrep_id = Irrep(i, parity, None).id()
                if irrep_id not in irreps_of_same_l_and_parity:
                    continue

                num_irreps_of_id = len(irreps_of_same_l_and_parity[irrep_id])
                consolidated_repr.append(f"{num_irreps_of_id}x{irrep_id}")

                for irrep in irreps_of_same_l_and_parity[irrep_id]:
                    if irrep.data is None:
                        continue
                    consolidated_data.extend(irrep.data.tolist())
        consolidated_ids = "+".join(consolidated_repr)
        return f"{consolidated_ids}: {str(consolidated_data)}"


    def id(self):
        return self.__repr__()

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
    
    data: torch.tensor | None

    def __init__(self, l: int, parity: int, data: torch.tensor | None):
        assert l >= 0, "l (the degree of your representation) must be non-negative"
        assert parity in {EVEN_PARITY, ODD_PARITY}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {parity}"
        self.l = l
        self.parity = parity
        self.data = data

    @staticmethod
    def from_id(irrep_id: str, data: torch.tensor | None):
        irrep_pattern = r"^(\d+)([eo])$"
        match = re.match(irrep_pattern, irrep_id)
        if not bool(match):
            raise ValueError(f"irrep_id {irrep_id} is not valid. it need to look something like: 1o or 7e. (this is the order l followed by the parity (e or o)")
        l_str, parity_str = match.groups()
        parity = -1 if parity_str == "o" else 1
        return Irrep(int(l_str), parity, data)

    def get_coefficient(self, m:int) -> float:
        if self.data is None:
            raise ValueError(f"data is None when trying to get m={m} coefficient for {self}")
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
            if irrep1.data is None or irrep2.data is None:
                coefficients = None
            else:
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
        if self.data is None:
            return f"{id}"
        return f"{id}: {self.data.tolist()}"

    def id(self):
        if self.parity == 1:
            parity_str = "e"
        elif self.parity == -1:
            parity_str = "o"
        return f"{self.l}{parity_str}"
