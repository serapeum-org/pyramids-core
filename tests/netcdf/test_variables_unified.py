"""Variable read + mutate live on the single `nc.variables` accessor.

Pins P2-5 of the engine-accessor migration: the separate `varops` engine is folded into
`nc.variables`, which now carries `add`/`remove`/`rename`/`set` alongside its read surface.
`nc.varops` stays as a deprecating alias for the window (the real mutation path is exercised
by `tests/netcdf/selection/test_reduce_extended_how.py::...swapping_its_raster...`).
"""

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf.engines.variables import _Variables

NC_PATH = Path(__file__).parents[1] / "data" / "netcdf" / "cf__5v__1d4-4d1__y-asc.nc"


@pytest.fixture
def nc() -> NetCDF:
    return NetCDF.read_file(str(NC_PATH))


def test_variables_exposes_the_mutation_methods(nc: NetCDF):
    for name in ("add", "remove", "rename", "set"):
        assert callable(getattr(nc.variables, name)), name


def test_varops_alias_warns_and_returns_the_internal_engine(nc: NetCDF):
    with pytest.warns(DeprecationWarning, match=r"`nc.varops`.*`nc.variables`"):
        alias = nc.varops
    assert isinstance(alias, _Variables)
    assert alias._ds is nc


def test_item_assignment_is_still_refused(nc: NetCDF):
    name = nc.variable_names[0]
    with pytest.raises(TypeError):
        nc.variables[name] = object()
