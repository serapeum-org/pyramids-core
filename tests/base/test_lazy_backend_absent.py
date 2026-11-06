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
