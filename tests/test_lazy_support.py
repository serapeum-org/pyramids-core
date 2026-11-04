"""Guard the lazy-extraction seam (plan Phase 0, tasks #2 and #3).

Two invariants that a behaviour test never notices:

1. ``pyramids.lazy_support`` re-exports the *identical* core objects (never a reimplementation) --
   a copied ``CachingFileManager`` / ``default_lock`` would silently break the process-global
   handle/lock sharing dask-worker reads depend on.
2. Every lazy submodule still imports cleanly *without* the ``[lazy]`` extra -- the import deferral
   in core (task #2) must not hide a module that is itself broken at import time.
"""

from __future__ import annotations

import importlib

import pytest

import pyramids.lazy_support as ls

#: facade name -> the home module it must be re-exported from (identically).
_ORIGINS = {
    "AXIS_NAMES": "pyramids.base._axes",
    "OptionalPackageDoesNotExist": "pyramids.base._errors",
    "OverviewTargetError": "pyramids.base._errors",
    "CachingFileManager": "pyramids.base._file_manager",
    "gdal_mdarray_open": "pyramids.base._file_manager",
    "DummyLock": "pyramids.base._locks",
    "default_lock": "pyramids.base._locks",
    "_is_identity_packing": "pyramids.base._utils",
    "apply_unpack": "pyramids.base._utils",
    "extra_hint": "pyramids.base._utils",
    "import_dask": "pyramids.base._utils",
    "import_dask_geopandas": "pyramids.base._utils",
    "import_h5py": "pyramids.base._utils",
    "import_xarray": "pyramids.base._utils",
    "import_zarr": "pyramids.base._utils",
    "lazy_extra_hint": "pyramids.base._utils",
    "require_optional": "pyramids.base._utils",
    "crs_from_user_input": "pyramids.base.crs",
    "crs_spec": "pyramids.base.crs",
    "epsg_from_user_input": "pyramids.base.crs",
    "sr_from_epsg": "pyramids.base.crs",
    "sr_from_wkt": "pyramids.base.crs",
    "GeoReference": "pyramids.base.georeference",
    "ArrayLike": "pyramids.base.protocols",
    "as_numpy": "pyramids.base.protocols",
    "is_lazy": "pyramids.base.protocols",
    "cloud_config_from_env": "pyramids.base.remote",
    "to_fsspec_url": "pyramids.base.remote",
    "GeoTransform": "pyramids.dataset.transform",
    "axis_flips": "pyramids.netcdf._mdim",
    "no_data_value_of": "pyramids.netcdf._mdim",
    "open_mdarray": "pyramids.netcdf._mdim",
    "_dtype_to_str": "pyramids.netcdf.utils",
}

#: Lazy submodules whose top-level import defers dask, so they must import even on a bare core
#: install. ``feature._lazy_collection`` is excluded: it hard-imports ``dask_geopandas`` at module
#: top (needs the ``[parquet]`` extra), so it is not import-clean without that extra.
_LAZY_SUBMODULES = [
    "pyramids.netcdf._lazy",
    "pyramids.netcdf._lazy_cube",
    "pyramids.netcdf._mfdataset",
    "pyramids.netcdf._kerchunk_facade",
    "pyramids.netcdf._kerchunk_builder",
    "pyramids.dataset.ops._zarr",
    "pyramids.dataset.ops._geobox_zarr",
]


def test_api_version_is_a_positive_int():
    """The contract version is present and usable for a ``>=`` pin by the plugin."""
    assert isinstance(ls.LAZY_SUPPORT_API_VERSION, int)
    assert ls.LAZY_SUPPORT_API_VERSION >= 1


def test_every_all_name_is_present():
    """Everything the facade promises in ``__all__`` actually resolves on the module."""
    missing = [name for name in ls.__all__ if not hasattr(ls, name)]
    assert not missing, f"lazy_support.__all__ names missing: {missing}"


def test_origins_cover_all_reexports():
    """The identity map covers every re-exported name (so none drifts in uncovered)."""
    reexported = {name for name in ls.__all__ if name != "LAZY_SUPPORT_API_VERSION"}
    assert set(_ORIGINS) == reexported


@pytest.mark.parametrize("name", sorted(_ORIGINS))
def test_reexport_is_the_same_object(name: str):
    """Each facade symbol IS the home-module object -- re-exported, never reimplemented."""
    home = importlib.import_module(_ORIGINS[name])
    assert getattr(ls, name) is getattr(home, name), (
        f"lazy_support.{name} is not the same object as {_ORIGINS[name]}.{name}; "
        "the facade must re-export, never copy"
    )


@pytest.mark.parametrize("module_path", _LAZY_SUBMODULES)
def test_lazy_submodule_imports_without_the_extra(module_path: str):
    """Each lazy submodule imports cleanly -- core's import deferral must not hide a broken module."""
    assert importlib.import_module(module_path) is not None
