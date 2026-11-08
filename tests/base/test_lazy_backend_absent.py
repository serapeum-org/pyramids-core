"""Thin branded-error suite: the public lazy ops fail cleanly on a bare core (no compute backend).

The lazy/dask operation bodies live in the ``pyramids-lazy`` compute backend; core dispatches to it
through ``require_compute_backend``. This suite is the core-side contract that, when **no** backend is
installed, every public lazy entry point raises the branded
:class:`~pyramids.base._errors.OptionalPackageDoesNotExist` (not an ``AttributeError`` /
``ModuleNotFoundError``). The comprehensive behaviour tests (that the ops actually compute correctly)
live with the backend in the ``pyramids-lazy`` repo and run against it there and in core's integration
CI; here we only pin the absent-backend behaviour, which needs no backend at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyramids.base._compute_backend as _cb
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection


@pytest.fixture
def no_backend(monkeypatch):
    """Force ``require_compute_backend`` to find nothing: empty registry + absent first-party plugin."""
    monkeypatch.setattr(
        _cb, "extra_missing", lambda name: True
    )  # pyramids_lazy "not installed"
    _cb.clear_compute_backend_cache()
    monkeypatch.setattr(_cb, "_discover", dict)  # no entry-point backends
    _cb.clear_compute_backend_cache()
    yield
    _cb.clear_compute_backend_cache()


def test_require_compute_backend_raises_branded(no_backend):
    """The seam itself raises the branded error with a pip hint."""
    with pytest.raises(OptionalPackageDoesNotExist):
        _cb.require_compute_backend()


def test_dataset_to_zarr_branded(no_backend, tmp_path):
    """Dataset.to_zarr raises the branded error rather than a raw import error."""
    ds = Dataset.from_array(
        np.ones((2, 2), "float32"),
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    )
    with pytest.raises(OptionalPackageDoesNotExist):
        ds.to_zarr(str(tmp_path / "x.zarr"))


def test_dataset_from_zarr_branded(no_backend, tmp_path):
    """Dataset.from_zarr raises the branded error."""
    with pytest.raises(OptionalPackageDoesNotExist):
        Dataset.from_zarr(str(tmp_path / "x.zarr"))


def test_netcdf_lazy_entry_points_branded(no_backend):
    """NetCDF.open_mfdataset / to_kerchunk / combine_kerchunk raise the branded error."""
    from pyramids.netcdf import NetCDF

    with pytest.raises(OptionalPackageDoesNotExist):
        NetCDF.open_mfdataset(["a.nc"], "v")
    with pytest.raises(OptionalPackageDoesNotExist):
        NetCDF.combine_kerchunk(["a.nc"], "out.json")


def test_feature_dask_read_branded(no_backend, tmp_path):
    """FeatureCollection.read_file(backend='dask') raises the branded error."""
    with pytest.raises(OptionalPackageDoesNotExist):
        FeatureCollection.read_file(str(tmp_path / "x.geojson"), backend="dask")


def test_public_lazy_classes_branded(no_backend):
    """The public lazy class names raise the branded error when no backend provides them."""
    import pyramids.feature as feature_pkg
    import pyramids.netcdf as netcdf_pkg

    with pytest.raises(OptionalPackageDoesNotExist):
        _ = netcdf_pkg.LazyNetCDF
    with pytest.raises(OptionalPackageDoesNotExist):
        _ = feature_pkg.LazyFeatureCollection


def test_public_lazy_fc_without_dask_geopandas_is_branded(monkeypatch):
    """Without dask-geopandas the public name raises the BRANDED error, not a plain ImportError.

    Test scenario:
        This is the shape of the ``py311``-``py315`` matrix legs: the backend is installed but
        ``dask-geopandas`` (the ``[parquet]`` extra) is not. ``OptionalPackageDoesNotExist`` is an
        ``ImportError`` subclass, so a plain ``ImportError`` here is invisible to every caller that
        matches the branded type.
    """
    import pyramids.feature as feature_pkg

    monkeypatch.setattr(feature_pkg, "_HAS_DASK_GEOPANDAS", False)
    with pytest.raises(OptionalPackageDoesNotExist):
        _ = feature_pkg.LazyFeatureCollection


def test_netcdf_to_kerchunk_branded(no_backend, tmp_path):
    """NetCDF.to_kerchunk raises the branded error (covers the to_kerchunk op)."""
    from pyramids.netcdf import GeoReference as NCGeoRef
    from pyramids.netcdf import NetCDF

    nc = NetCDF.from_array(
        np.zeros((1, 2, 2), dtype="float32"),
        geo_ref=NCGeoRef(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        variable_name="v",
    )
    with pytest.raises(OptionalPackageDoesNotExist):
        nc.to_kerchunk(str(tmp_path / "ref.json"))


def test_netcdf_chunk_branded(no_backend, tmp_path):
    """NetCDF.chunk raises the branded error (covers the build_lazy_cube op)."""
    from pyramids.netcdf import GeoReference as NCGeoRef
    from pyramids.netcdf import NetCDF

    path = str(tmp_path / "c.nc")
    nc = NetCDF.from_array(
        np.zeros((1, 2, 2), dtype="float32"),
        geo_ref=NCGeoRef(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        variable_name="v",
    )
    nc.to_file(path)
    nc_read = NetCDF.read_file(path)
    with pytest.raises(OptionalPackageDoesNotExist):
        nc_read.chunk()


def test_netcdf_to_dask_dataframe_branded(no_backend):
    """NetCDF.to_dask_dataframe raises the branded error (covers the to_dask_dataframe op)."""
    from pyramids.netcdf import GeoReference as NCGeoRef
    from pyramids.netcdf import NetCDF

    nc = NetCDF.from_array(
        np.zeros((1, 2, 2), dtype="float32"),
        geo_ref=NCGeoRef(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        variable_name="v",
    )
    with pytest.raises(OptionalPackageDoesNotExist):
        nc.to_dask_dataframe()


def test_collection_zarr_roundtrip_branded(no_backend, tmp_path):
    """DatasetCollection.from_zarr raises the branded error (covers read_collection_from_zarr).

    The write side (write_collection_to_zarr, via DatasetCollection.to_zarr) dispatches through the
    same ``require_compute_backend`` gate — which this fixture makes raise before any backend work —
    so the read-side check is representative; the write path's compute behaviour is exercised in the
    backend's own repo where a collection can be built cheaply.
    """
    from pyramids.dataset.collection import DatasetCollection

    with pytest.raises(OptionalPackageDoesNotExist):
        DatasetCollection.from_zarr(str(tmp_path / "c.zarr"))


def test_feature_read_parquet_dask_branded(no_backend, tmp_path):
    """FeatureCollection.read_parquet(backend='dask') raises the branded error.

    Covers the read_vector_parquet_lazy op, distinct from read_vector_lazy (read_file).
    """
    with pytest.raises(OptionalPackageDoesNotExist):
        FeatureCollection.read_parquet(str(tmp_path / "x.parquet"), backend="dask")


@pytest.mark.skipif(
    _cb.extra_missing("pyramids_lazy") is False,
    reason="a backend IS installed; this check is for the genuinely-bare environment",
)
def test_genuinely_bare_core_is_branded_without_monkeypatch():
    """With no backend actually installed, the public lazy names raise branded — no fixture.

    The other tests simulate absence with the ``no_backend`` fixture; this one takes no fixture and
    asserts the real contract, so core's CI ``bare`` environment (no pyramids_lazy, no dask) proves
    it for real rather than only under a monkeypatch.
    """
    import pyramids.feature as feature_pkg
    import pyramids.netcdf as netcdf_pkg

    _cb.clear_compute_backend_cache()
    with pytest.raises(OptionalPackageDoesNotExist):
        _ = netcdf_pkg.LazyNetCDF
    with pytest.raises(OptionalPackageDoesNotExist):
        _ = feature_pkg.LazyFeatureCollection
