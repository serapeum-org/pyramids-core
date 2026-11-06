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
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

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
    "RasterMeta": "pyramids.base._raster_meta",
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
    "ZARR_SCHEMA_VERSION": "pyramids.dataset.ops._geobox_zarr",
    "detect_data_var": "pyramids.dataset.ops._geobox_zarr",
    "finalize_zarr_metadata": "pyramids.dataset.ops._geobox_zarr",
    "geobox_crs": "pyramids.dataset.ops._geobox_zarr",
    "normalize_compressors": "pyramids.dataset.ops._geobox_zarr",
    "read_geobox": "pyramids.dataset.ops._geobox_zarr",
    "_resolve_store": "pyramids.dataset.ops._zarr",
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


#: Heavy optional deps the lazy layer owns. Blocked in a fresh interpreter so the import guards below
#: prove behaviour on a bare `pyramids-core` install, not merely in this dev env where they exist.
_BLOCKED_DEPS = [
    "dask",
    "distributed",
    "zarr",
    "kerchunk",
    "h5py",
    "fsspec",
    "s3fs",
    "dask_geopandas",
    "pyarrow",
    "xarray",
]


def _run_with_blocked_deps(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter with the heavy lazy deps forced unimportable.

    Setting ``sys.modules[name] = None`` makes a later ``import name`` raise ``ImportError``, so any
    top-level import of a blocked dep fails loudly instead of silently succeeding because the dev env
    happens to have it installed.
    """
    prelude = "import sys\n" + "".join(
        f"sys.modules[{dep!r}] = None\n" for dep in _BLOCKED_DEPS
    )
    script = prelude + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )


@pytest.mark.parametrize("module_path", _LAZY_SUBMODULES)
def test_lazy_submodule_imports_without_the_extra(module_path: str):
    """Each lazy submodule imports in a fresh interpreter with dask/zarr/kerchunk/... forced absent.

    Blocking the heavy deps is what makes this a real guard: a plain import in this dev env would pass
    even if a submodule grew a top-level `import dask`, because dask is installed here.
    """
    result = _run_with_blocked_deps(
        f"import importlib; importlib.import_module({module_path!r})"
    )
    assert result.returncode == 0, (
        f"{module_path} failed to import with lazy extras blocked:\n{result.stderr}"
    )


def test_core_packages_import_without_the_extra_and_do_not_pull_lazy_cube():
    """Core packages import with the lazy stack absent, and `import pyramids.netcdf` must not pull `_lazy_cube`.

    The second half is the extraction-readiness contract (plan #2): once `_lazy_cube` moves to
    pyramids-lazy, a bare `import pyramids.netcdf` must still succeed, so the package must resolve
    `LazyNetCDF` lazily (PEP-562) rather than importing it at module top.
    """
    result = _run_with_blocked_deps(
        """
        import importlib, sys
        for pkg in ("pyramids", "pyramids.dataset", "pyramids.netcdf", "pyramids.feature"):
            importlib.import_module(pkg)
        assert "pyramids.netcdf._lazy_cube" not in sys.modules, (
            "importing pyramids.netcdf eagerly pulled the lazy _lazy_cube module"
        )
        """
    )
    assert result.returncode == 0, result.stderr


def test_no_public_name_leaks_beyond_all():
    """The module exposes no public name that ``__all__`` does not list (re-export hygiene)."""
    public = {
        name
        for name in vars(ls)
        if not name.startswith("_") and not inspect.ismodule(getattr(ls, name))
    }
    extra = public - set(ls.__all__)
    assert not extra, f"public names missing from __all__: {sorted(extra)}"


def test_core_never_imports_lazy_support():
    """Core code must never import ``pyramids.lazy_support`` -- only an external backend does (plan 0.3-A).

    A hard rule of the facade: keeping core off it is what keeps the facade out of every core import
    cycle. This scans the installed ``pyramids`` source tree (excluding the facade itself) for any
    reference to ``lazy_support``.
    """
    pkg_root = Path(ls.__file__).parent
    offenders = [
        path.relative_to(pkg_root).as_posix()
        for path in pkg_root.rglob("*.py")
        if path.name != "lazy_support.py"
        and "lazy_support" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"core modules reference lazy_support (must not): {offenders}"
