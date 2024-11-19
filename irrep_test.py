from constants import EVEN_PARITY, ODD_PARITY
from irrep import Irrep, Irreps
import torch

# the purpose of this file is to ensure that the tensor product produces the correct irreps
# read the last section of this file to understand how tensor products work


def dummy_data(l: int) -> torch.Tensor:
    # creates dummy data for a specific l value
    arr = torch.zeros(2 * l + 1)
    arr[0] = 1  # so the array is normalized to length 1
    return arr


def dummy_data_for_multiple_irreps(l_list: list[int]) -> torch.Tensor:
    return [dummy_data(l) for l in l_list]


def test_irrep_id():
    assert Irrep(l=1, parity=EVEN_PARITY, data=dummy_data(1)).id() == "1e"
    assert Irrep(l=1, parity=ODD_PARITY, data=dummy_data(1)).id() == "1o"
    assert Irrep(l=2, parity=EVEN_PARITY, data=dummy_data(2)).id() == "2e"
    assert Irrep(l=2, parity=ODD_PARITY, data=dummy_data(2)).id() == "2o"

    assert Irrep.from_id("1e", dummy_data(1)).id() == "1e"
    assert Irrep.from_id("1o", dummy_data(1)).id() == "1o"
    assert Irrep.from_id("20e", dummy_data(20)).id() == "20e"
    assert Irrep.from_id("20o", dummy_data(20)).id() == "20o"


def test_irreps_id():
    assert (
        Irreps.from_id(
            "1x1e + 1x1o + 2x2e+2x2o",
            dummy_data_for_multiple_irreps([1, 1, 2, 2, 2, 2]),
        ).id()
        == "1x1o+1x1e+2x2o+2x2e"
    )
    assert (
        Irreps.from_id("1x1e + 1x1e", dummy_data_for_multiple_irreps([1, 1])).id()
        == "2x1e"
    )  # test consolidating irreps


# def test_irrep_tensor_product():
#     assert IrrepDef(1, 1).tensor_product(IrrepDef(1, 1)) == [IrrepDef(2, 1)]


def test_irrep_tensor_product():
    # To do a tensor product, we basically use the distributive property on the left side:
    # e.g. in algebra, (a + b)(c + d) = ac + ad + bc + bd
    # doing a tensor product is the same idea. we multiply each irrep on the LHS (analogous to a + b) with each irrep on the RHS (analogous to c + d) much like what we do when using the distributive property

    # let X be the tensor product operation (aka direct product)
    # let + be the direct sum operation

    # 1x0e+1x1o tp 1x0e+1x1o = (1x0e X 1x0e) + (1x0e X 1x1o) + (1x1o X 1x0e) + (1x1o X 1x1o)
    # (1x0e X 1x1o) produces: 1x0e
    # (1x0e X 1x1o) produces: 1x1o
    # (1x1o X 1x0e) produces: 1x1o
    # (1x1o X 1x1o) produces: 1x0e + 1x1e + 1x2e

    # if we take the direct sum of the produced irreps, we get: 2x0e + 2x1o + 1x1e + 1x2e
    assert (
        Irreps.from_id("1x0e+1x1o", dummy_data_for_multiple_irreps([0, 1]))
        .tensor_product(
            Irreps.from_id("1x0e+1x1o", dummy_data_for_multiple_irreps([0, 1]))
        )
        .id()
        == "2x0e+2x1o+1x1e+1x2e"
    )

    # this next tensor product test is here just to ensure my code works
    assert (
        Irreps.from_id("1x0e+1x1o+1x2e", dummy_data_for_multiple_irreps([0, 1, 2]))
        .tensor_product(
            Irreps.from_id("1x0e+1x1o+1x2e", dummy_data_for_multiple_irreps([0, 1, 2]))
        )
        .id()
        == "3x0e+4x1o+2x1e+2x2o+4x2e+2x3o+1x3e+1x4e"
    )
