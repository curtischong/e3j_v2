import pytest
from irrep import IrrepDef, Irreps


def test_irrep_id():
    assert IrrepDef(1, 1).id() == "1e"
    assert IrrepDef(1, -1).id() == "1o"
    assert IrrepDef(2, 1).id() == "2e"
    assert IrrepDef(2, -1).id() == "2o"

def test_irreps_id():
    assert Irreps("1x1e + 1x1o + 2x2e+2x2o").id() == "1x1e + 1x1o + 2x2e + 2x2o"
    # assert Irreps("1x1e + 1x1e").id() == "2x1e"