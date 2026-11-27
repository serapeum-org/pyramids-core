"""The internal, non-deprecated dispatchers route per type and never self-warn (H2).

`_read_array` / `_band_scale` / `_band_offset` / `_crop` / `_to_crs` / `_zonal_stats` let first-party
code read polymorphically without tripping the flat-facade `DeprecationWarning`: a plain `Dataset` goes
through its engine, a `NetCDF` through its own override. These pin both halves and prove the canonical
public ops no longer emit a deprecation from pyramids' own internals.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf import GeoReference as NCGeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0.0, 5.0), cell_size=1.0, epsg=4326)
NC_GEO = NCGeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)


def _dataset() -> Dataset:
    """A small plain single-band `Dataset`."""
    return Dataset.from_array(np.arange(25, dtype="float32").reshape(5, 5), geo_ref=GEO)


def _variable() -> NetCDF:
    """A one-variable `NetCDF` view."""
    return NetCDF.from_array(
        np.full((4, 4), 2.0, "float32"), geo_ref=NC_GEO, variable_name="t"
    ).get_variable("t")


class TestInternalReadDispatch:
    """`_read_array` reads through the right path for each receiver type without warning."""

    def test_dataset_read_matches_the_io_engine_and_does_not_warn(self):
        """On a plain `Dataset`, `_read_array` equals `io.read_array` and raises no deprecation.

        Test scenario:
            A plain dataset's internal read must go through the io engine and stay silent under
            `-W error::DeprecationWarning`, so a user calling only the canonical API is never warned.
        """
        ds = _dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            internal = np.asarray(ds._read_array())
        assert np.array_equal(internal, np.asarray(ds.io.read_array()))

    def test_netcdf_read_matches_its_override(self):
        """On a `NetCDF`, `_read_array` routes to the variable/dimension-aware override.

        Test scenario:
            The NetCDF path must not be forced through `io.read_array`; `_read_array` returns exactly
            what the public `NetCDF.read_array` override returns.
        """
        var = _variable()
        assert np.array_equal(
            np.asarray(var._read_array()), np.asarray(var.read_array())
        )


class TestInternalScaleOffsetDispatch:
    """`_band_scale` / `_band_offset` read the right getter for each receiver type."""

    def test_dataset_scale_offset_match_the_bands_engine_silently(self):
        """A plain dataset's internal scale/offset come from the bands engine with no warning.

        Test scenario:
            Reading scale/offset internally must not warn under `-W error::DeprecationWarning`.
        """
        ds = _dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert ds._band_scale() == ds.bands.scale
            assert ds._band_offset() == ds.bands.offset

    def test_netcdf_scale_offset_use_the_cf_packing_getter(self):
        """A NetCDF's internal scale/offset come from its CF-packing override, not the bands engine.

        Test scenario:
            `_band_scale`/`_band_offset` on a NetCDF equal its public `scale`/`offset` getters.
        """
        var = _variable()
        assert var._band_scale() == var.scale
        assert var._band_offset() == var.offset


class TestInternalSpatialDispatch:
    """`_crop` / `_to_crs` / `_zonal_stats` route to the NetCDF overrides on a NetCDF receiver."""

    def test_netcdf_crop_and_to_crs_route_to_the_override(self):
        """`container._crop` / `container._to_crs` behave as the NetCDF overrides, not the Dataset path.

        Test scenario:
            On a root container the override returns a `NetCDF` (per-variable), which the bypassing
            `spatial` path would not; `_crop`/`_to_crs` must match the override's result type.
        """
        path = os.path.join(tempfile.mkdtemp(), "c.nc")
        NetCDF.from_array(
            np.full((4, 4), 10.0, "float32"), geo_ref=NC_GEO, variable_name="t"
        ).to_file(path)
        cropped = NetCDF.read_file(path)._crop(bbox=(0.0, 0.0, 2.0, 2.0))
        direct = NetCDF.read_file(path).crop(bbox=(0.0, 0.0, 2.0, 2.0))
        assert type(cropped) is type(direct)
        reprojected = NetCDF.read_file(path)._to_crs(3857)
        assert type(reprojected) is type(NetCDF.read_file(path).to_crs(3857))


class TestCanonicalOpsDoNotSelfWarn:
    """The public accessor ops do not emit a first-party `DeprecationWarning` from internals (H2)."""

    def test_to_dataframe_is_silent_under_warning_errors(self):
        """`ds.io.to_dataframe()` returns without raising under `-W error::DeprecationWarning`.

        Test scenario:
            The canonical accessor reads pixels internally; after H2 that read is non-deprecated, so
            the call completes rather than raising a deprecation from pyramids' own code.
        """
        ds = _dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            frame = ds.io.to_dataframe()
        assert len(frame) == 25
