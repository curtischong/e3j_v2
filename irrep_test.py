import pytest
from irrep import Irrep, Irreps


def test_irrep_id():
    assert Irrep(l=1, parity=1, data=None).id() == "1e"
    assert Irrep(l=1, parity=-1, data=None).id() == "1o"
    assert Irrep(l=2, parity=1, data=None).id() == "2e"
    assert Irrep(l=2, parity=-1, data=None).id() == "2o"

    assert Irrep.from_id("1e", None).id() == "1e"
    assert Irrep.from_id("1o", None).id() == "1o"
    assert Irrep.from_id("20e", None).id() == "20e"
    assert Irrep.from_id("20o", None).id() == "20o"


def test_irreps_id():
    assert Irreps.from_id("1x1e + 1x1o + 2x2e+2x2o").id() == "1x1o+1x1e+2x2o+2x2e"
    # assert Irreps("1x1e + 1x1e").id() == "2x1e"

# def test_irrep_tensor_product():
#     assert IrrepDef(1, 1).tensor_product(IrrepDef(1, 1)) == [IrrepDef(2, 1)]

def test_irrep_tensor_product():
    # 1x0e+1x1o tensor product 1x0e+1x1o
    # 1x0e +
    # 1x1o +
    # 1x1o +
    # 1x0e + 1x1e + 1x2e
    assert Irreps("1x0e+1x1o").tensor_product(Irreps("1x0e+1x1o")).id() == "2x0e + 2x1o + 1x1e + 1x2e"
    assert Irreps("1x0e+1x1o+1x2e").tensor_product(Irreps("1x0e+1x1o+1x2e")).id() == "3x0e + 4x1o + 2x1e + 2x2o + 4x2e + 2x3o + 1x3e + 1x4e"