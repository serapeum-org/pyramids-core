"""NC-5: ``NetCDF.get_all_metadata`` -> ``refresh_metadata``.

The new name says what differs from the cached ``metadata`` property (an uncached re-read),
and it now also stores the fresh result as the cached value. The old name keeps working as a
``DeprecationWarning``-emitting alias.
"""

from __future__ import annotations

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_SAMPLE = "cf__7v__1d3-2d3-3d1__y-asc.nc"


def test_refresh_metadata_refreshes_the_cache(sample):
    nc = NetCDF.read_file(sample(_SAMPLE))
    fresh = nc.refresh_metadata()
    # refresh_metadata stores its result as the cached value, so .metadata returns it.
    assert nc.metadata is fresh


def test_get_all_metadata_warns_and_matches(sample):
    nc = NetCDF.read_file(sample(_SAMPLE))
    with pytest.warns(DeprecationWarning, match="refresh_metadata"):
        legacy = nc.get_all_metadata()
    assert legacy.variables.keys() == nc.refresh_metadata().variables.keys()
