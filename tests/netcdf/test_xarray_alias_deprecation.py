"""The xarray-style aliases on NetCDF warn and forward to their canonical members.

Pins P2-7 of the engine-accessor migration: ``attrs``/``data_vars``/``dims``/``sizes`` are
xarray-vocabulary aliases with near-zero usage and heavy name-collision, so they are
deprecated (removed in 0.90.0). Each must emit a ``DeprecationWarning`` naming the
canonical member and still return the same value.
"""

import warnings
from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

NC_PATH = Path(__file__).parents[1] / "data" / "netcdf" / "cf__5v__1d4-4d1__y-asc.nc"


@pytest.fixture
def nc() -> NetCDF:
    return NetCDF.read_file(str(NC_PATH))


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("attrs", "global_attributes"),
        ("data_vars", "variables"),
        ("dims", "dimension_sizes"),
        ("sizes", "dimension_sizes"),
    ],
)
def test_alias_warns_naming_the_canonical_member(nc: NetCDF, alias: str, canonical: str):
    with pytest.warns(DeprecationWarning, match=rf"`{alias}`.*`{canonical}`"):
        getattr(nc, alias)


def test_attrs_returns_the_global_attributes(nc: NetCDF):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert nc.attrs == nc.global_attributes


@pytest.mark.parametrize("alias", ["dims", "sizes"])
def test_dim_aliases_return_the_dimension_sizes(nc: NetCDF, alias: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert getattr(nc, alias) == nc.dimension_sizes


def test_data_vars_lists_the_same_variables(nc: NetCDF):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert list(nc.data_vars) == list(nc.variables)
