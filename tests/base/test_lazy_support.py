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
    "import_pyarrow": "pyramids.base._utils",
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
    "COUNTING_REDUCERS": "pyramids.base._reductions",
    "REDUCERS": "pyramids.base._reductions",
    "_Applied": "pyramids.netcdf.engines._along_dim",
    "_CumSum": "pyramids.netcdf.engines._along_dim",
    "_diffed_array": "pyramids.netcdf.engines._along_dim",
    "_Extremum": "pyramids.netcdf.engines._along_dim",
    "_Interpolate": "pyramids.netcdf.engines._along_dim",
    "_InterpTo": "pyramids.netcdf.engines._along_dim",
    "_Pad": "pyramids.netcdf.engines._along_dim",
    "_Rank": "pyramids.netcdf.engines._along_dim",
    "_read_no_data": "pyramids.netcdf.engines._along_dim",
    "_reduced_array": "pyramids.netcdf.engines._along_dim",
    "_rolled_array": "pyramids.netcdf.engines._along_dim",
    "_Shift": "pyramids.netcdf.engines._along_dim",
    "_variable_from_applied": "pyramids.netcdf.engines._along_dim",
    "_is_in_memory": "pyramids.netcdf.engines.interop",
    "_is_container": "pyramids.netcdf.engines.interop",
    "_frame_values": "pyramids.netcdf.engines.interop",
    "_frame_axes": "pyramids.netcdf.engines.interop",
    "_frame_variables": "pyramids.netcdf.engines.interop",
    "_public_spatial_names": "pyramids.netcdf.engines.interop",
    "_assert_band_dimension": "pyramids.netcdf.engines.selection",
    "_BOUNDARIES": "pyramids.netcdf.engines.selection",
    "_check_fill_value": "pyramids.netcdf.engines.selection",
    "_check_how": "pyramids.netcdf.engines.selection",
    "_check_limit": "pyramids.netcdf.engines.selection",
    "_check_min_periods": "pyramids.netcdf.engines.selection",
    "_check_order": "pyramids.netcdf.engines.selection",
    "_check_periods": "pyramids.netcdf.engines.selection",
    "_check_quantile": "pyramids.netcdf.engines.selection",
    "_check_window": "pyramids.netcdf.engines.selection",
    "_coarsen_windows": "pyramids.netcdf.engines.selection",
    "_DIFF_LABELS": "pyramids.netcdf.engines.selection",
    "_interp_targets": "pyramids.netcdf.engines.selection",
    "_INTERP_MIN_POINTS": "pyramids.netcdf.engines.selection",
    "_pad_before_after": "pyramids.netcdf.engines.selection",
    "_refuse_spatial_band_op": "pyramids.netcdf.engines.selection",
    "_resolve_interp_kind": "pyramids.netcdf.engines.selection",
    "_resolve_one_dim": "pyramids.netcdf.engines.selection",
    "_resolve_positional_indices": "pyramids.netcdf.engines.selection",
    "_transpose_order": "pyramids.netcdf.engines.selection",
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

#: The lazy op *bodies* moved to pyramids-lazy; what remains in core are the small shared helpers
#: still used by eager paths. These must import even on a bare core install (no dask): ``_lazy`` keeps
#: only ``mask_no_data`` (the post-read mask), ``_zarr`` keeps only ``_resolve_store``, and
#: ``_geobox_zarr`` is the geobox helper set the eager STAC/collection paths share.
_LAZY_SUBMODULES = [
    "pyramids.netcdf._lazy",
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
        # The lazy cube bodies live in the pyramids-lazy backend now, so asserting the old in-core
        # `pyramids.netcdf._lazy_cube` is absent would be vacuous (it no longer exists). The live
        # contract is that importing core must not eagerly pull the backend distribution at all.
        pulled = [m for m in sys.modules if m == "pyramids_lazy" or m.startswith("pyramids_lazy.")]
        assert not pulled, (
            f"importing core eagerly pulled the lazy backend: {pulled}"
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
    cycle. This parses the installed ``pyramids`` source tree (excluding the facade itself) and
    flags any real ``import`` of ``pyramids.lazy_support`` — AST-based, not a substring scan, so a
    mention in a comment or docstring does not false-positive and an aliased/``from`` import cannot
    slip past.
    """
    import ast

    pkg_root = Path(ls.__file__).parent

    def _imports_lazy_support(source: str) -> bool:
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - a parse failure is caught by other tooling
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "pyramids.lazy_support" for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "pyramids.lazy_support":
                    return True
                # `from pyramids import lazy_support`
                if node.module == "pyramids" and any(
                    a.name == "lazy_support" for a in node.names
                ):
                    return True
        return False

    offenders = [
        path.relative_to(pkg_root).as_posix()
        for path in pkg_root.rglob("*.py")
        if path.name != "lazy_support.py"
        and _imports_lazy_support(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"core modules reference lazy_support (must not): {offenders}"
