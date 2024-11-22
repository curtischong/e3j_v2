import torch

from irrep import Irreps


def dummy_data(l: int) -> torch.Tensor:
    # creates dummy data for a specific l value
    arr = torch.zeros(2 * l + 1)
    arr[0] = 1  # so the array is normalized to length 1
    return arr


def create_irreps_with_dummy_data(id: str) -> Irreps:
    data_out = []
    for _irreps_def, num_irreps, l, _parity in Irreps.parse_id(id):
        for _ in range(num_irreps):
            data_out.append(dummy_data(l))
    return Irreps.from_id(id, data_out)
