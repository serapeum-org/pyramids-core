"""Optional-dependency probes, skipif decorators, and marker mapping.

One source of truth for every `[project.optional-dependencies]` group.

Two consumers:

* Inline `@requires_<extra>` decorators — kept for test files that
  already guard individual methods with `pytest.mark.skipif(...)`.
* `pytest_collection_modifyitems` in :mod:`tests.conftest` — reads
  :data:`EXTRA_MARKERS` to auto-apply a skip when a test is tagged with
  `@pytest.mark.<extra>` but the extra is not installed.

Marker names use underscores (valid Python identifiers required by
`pytest.mark.<name>`), while the PyPI extra names use hyphens. The
mapping below spells both out explicitly.
"""

from __future__ import annotations

import importlib.util

import pytest


def _has(*module_names: str) -> bool:
    """Return True iff every named top-level module is importable.

    Uses :func:`importlib.util.find_spec` so the probe never triggers a
    heavy import — cheap to run during collection. A dotted name whose
    parent package is missing raises :class:`ModuleNotFoundError`
    (rather than returning `None`), so the guard catches that and
    reports the module as absent.
    """
    found = True
    for name in module_names:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            found = False
            break
    return found


HAS_DASK = _has("dask")
HAS_DASK_ARRAY = _has("dask.array")
HAS_DASK_GEOPANDAS = _has("dask_geopandas")
HAS_ZARR = _has("zarr")
HAS_INTEROP = _has("xarray")
HAS_KERCHUNK = _has("kerchunk")
HAS_H5PY = _has("h5py")
HAS_PYARROW = _has("pyarrow")
HAS_CLEOPATRA = _has("cleopatra")
HAS_PYSTAC_CLIENT = _has("pystac_client")


# Per-extra installed? flag. Values are True when EVERY underlying
# module of that extra is importable — so `netcdf_lazy` requires
# kerchunk AND h5py AND the lazy stack, all of which now ship in
# `pip install pyramids-core[lazy]`.
#
# Why `netcdf_lazy` is kept distinct from `lazy` rather than collapsed into it: it gates on
# EXTRA modules on top of the base extra's headline dependency (kerchunk + h5py). Those ship
# inside the [lazy] extra, but an install can still carry the base stack (zarr/dask) without
# them. A test re-marked plain `lazy` would then FAIL to skip when kerchunk/h5py is the
# missing piece — so the finer-grained marker is retained to keep the skip precise. The
# matching `parquet_lazy` marker went to pyramids-lazy with the lazy vector tests.
_HAS_VIZ = HAS_CLEOPATRA
_HAS_LAZY = HAS_DASK_ARRAY and HAS_ZARR
_HAS_INTEROP = HAS_INTEROP
_HAS_NETCDF_LAZY = _HAS_LAZY and HAS_KERCHUNK and HAS_H5PY
_HAS_PARQUET = HAS_PYARROW
_HAS_STAC = HAS_PYSTAC_CLIENT


requires_plot = pytest.mark.skipif(
    not _HAS_VIZ, reason="pyramids-core[viz] not installed"
)
requires_lazy = pytest.mark.skipif(
    not _HAS_LAZY, reason="pyramids-core[lazy] not installed"
)
requires_interop = pytest.mark.skipif(
    not _HAS_INTEROP, reason="the optional interop dependency is not installed"
)
requires_netcdf_lazy = pytest.mark.skipif(
    not _HAS_NETCDF_LAZY, reason="kerchunk + h5py (pyramids-core[lazy]) not installed"
)
requires_parquet = pytest.mark.skipif(
    not _HAS_PARQUET, reason="pyramids-core[parquet] not installed"
)
requires_stac = pytest.mark.skipif(
    not _HAS_STAC, reason="pyramids-core[stac] not installed"
)


# Precise dask.array-only gate for tests that exercise the array backend
# without needing zarr (e.g. the ArrayLike protocol tests). Distinct from
# `requires_lazy`, which also requires zarr.
requires_dask_array = pytest.mark.skipif(
    not HAS_DASK_ARRAY, reason="dask.array not installed"
)
# dask.dataframe (dask-expr) imports pyarrow, which ships in the `parquet` extra, not `lazy`;
# gate on both so `to_dask_dataframe` tests skip in a lazy-only env rather than erroring on import.
requires_dask_dataframe = pytest.mark.skipif(
    not (HAS_DASK_ARRAY and HAS_PYARROW),
    reason="dask.dataframe needs dask + pyarrow (pyramids-core[parquet])",
)


# Marker-name → skipif-decorator mapping consumed by the collection hook
# in `tests/conftest.py`. A test annotated with `@pytest.mark.<name>`
# auto-gains the matching skip when its extra is not installed.
EXTRA_MARKERS: dict[str, pytest.MarkDecorator] = {
    # `plot` is the canonical gate for the [viz] extra; no separate
    # `viz` marker is registered to avoid duplicate surface.
    "plot": requires_plot,
    "lazy": requires_lazy,
    "interop": requires_interop,
    "netcdf_lazy": requires_netcdf_lazy,
    "parquet": requires_parquet,
    "stac": requires_stac,
}
