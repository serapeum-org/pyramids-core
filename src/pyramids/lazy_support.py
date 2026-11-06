"""Stable re-export boundary between pyramids-core and the lazy/dask layer.

This module is the **one** contract an out-of-tree lazy backend (``pyramids-lazy``, once the lazy
layer is extracted -- see the lazy-plugin extraction plan, issue #1455) imports from
pyramids-core. It re-exports -- never reimplements -- the exact core primitives the lazy modules
depend on, so those primitives can be refactored inside ``pyramids.base`` / ``pyramids.netcdf`` /
``pyramids.dataset`` without breaking the plugin, as long as this surface and
:data:`LAZY_SUPPORT_API_VERSION` stay stable.

Rules (enforced by ``tests/test_lazy_support.py``):

* **Re-export, never copy.** Every name here MUST be the identical object from its home module. A
  plugin that reimplemented, e.g., :class:`CachingFileManager` or :func:`default_lock` would break
  the process-global file-handle / lock sharing that dask-worker reads rely on.
* **Core never imports this module.** Core code imports the home modules directly; only an external
  lazy backend (and the identity test) imports ``pyramids.lazy_support``. This keeps the facade out
  of every core import cycle.
* **Additive and versioned.** Bump :data:`LAZY_SUPPORT_API_VERSION` only on a *breaking* change (a
  name removed, or a signature/semantics changed); additions never need a bump. A lazy backend pins
  ``pyramids.lazy_support.LAZY_SUPPORT_API_VERSION >= N`` at its own import and fails loudly
  ("upgrade pyramids-core") rather than on a later missing attribute.
* **Private primitives are re-exported on purpose.** A few names here are underscore-prefixed core
  primitives (``_is_identity_packing``, ``_dtype_to_str``) the lazy layer genuinely needs; they are
  part of this contract despite the leading underscore.
* **Patch deferred forwarders at their home module.** Core re-imports these primitives inside the
  method bodies that use them (plan #2), so a test double belongs on the *home* module (e.g.
  ``pyramids.dataset.ops._zarr._resolve_store``), not on the consumer module, which no longer holds
  the name at module scope.

Examples:
    - A lazy backend checks the contract version, then uses a re-exported primitive -- which is the
      *same* object as its canonical home (the facade re-exports, never copies):
        ```python
        >>> import pyramids.lazy_support as ls
        >>> ls.LAZY_SUPPORT_API_VERSION >= 1
        True
        >>> from pyramids.base._utils import apply_unpack
        >>> ls.apply_unpack is apply_unpack
        True

        ```
"""

from pyramids.base._axes import AXIS_NAMES
from pyramids.base._errors import OptionalPackageDoesNotExist, OverviewTargetError
from pyramids.base._file_manager import CachingFileManager, gdal_mdarray_open
from pyramids.base._locks import DummyLock, default_lock
from pyramids.base._raster_meta import RasterMeta
from pyramids.base._utils import (
    _is_identity_packing,
    apply_unpack,
    extra_hint,
    import_dask,
    import_dask_geopandas,
    import_h5py,
    import_pyarrow,
    import_xarray,
    import_zarr,
    lazy_extra_hint,
    require_optional,
)
from pyramids.base.crs import (
    crs_from_user_input,
    crs_spec,
    epsg_from_user_input,
    sr_from_epsg,
    sr_from_wkt,
)
from pyramids.base._reductions import COUNTING_REDUCERS, REDUCERS
from pyramids.base.georeference import GeoReference
from pyramids.base.protocols import ArrayLike, as_numpy, is_lazy
from pyramids.base.remote import cloud_config_from_env, to_fsspec_url
# The lazy NetCDF cube (LazyNetCDF, moved to pyramids-lazy) delegates its along-dimension cube
# operations (rolling / diff / cumsum / shift / rank / interpolate / pad / reduce / extremum) to
# these shared core engine primitives, which the EAGER NetCDF surface uses too -- so they stay in
# core and are re-exported here, per the facade's "private primitives re-exported on purpose" rule.
from pyramids.netcdf.engines._along_dim import (
    _Applied,
    _CumSum,
    _diffed_array,
    _Extremum,
    _Interpolate,
    _InterpTo,
    _Pad,
    _Rank,
    _read_no_data,
    _reduced_array,
    _rolled_array,
    _Shift,
    _variable_from_applied,
)
from pyramids.netcdf.engines.interop import (
    _frame_axes,
    _frame_values,
    _is_container,
    _is_in_memory,
    _public_spatial_names,
)
from pyramids.netcdf.engines.selection import (
    _assert_band_dimension,
    _BOUNDARIES,
    _check_fill_value,
    _check_how,
    _check_limit,
    _check_min_periods,
    _check_order,
    _check_periods,
    _check_quantile,
    _check_window,
    _coarsen_windows,
    _DIFF_LABELS,
    _interp_targets,
    _INTERP_MIN_POINTS,
    _pad_before_after,
    _refuse_spatial_band_op,
    _resolve_interp_kind,
    _resolve_one_dim,
    _resolve_positional_indices,
    _transpose_order,
)
from pyramids.dataset.ops._geobox_zarr import (
    ZARR_SCHEMA_VERSION,
    detect_data_var,
    finalize_zarr_metadata,
    geobox_crs,
    normalize_compressors,
    read_geobox,
)
from pyramids.dataset.ops._zarr import _resolve_store
from pyramids.dataset.transform import GeoTransform
from pyramids.netcdf._mdim import axis_flips, no_data_value_of, open_mdarray
from pyramids.netcdf.utils import _dtype_to_str

#: Contract version of this re-export surface. Bump ONLY on a breaking change; additions do not
#: require a bump. A lazy backend requires ``LAZY_SUPPORT_API_VERSION >= N`` and errors clearly when
#: core is older, instead of failing later on a missing attribute.
LAZY_SUPPORT_API_VERSION = 1

__all__ = [
    "LAZY_SUPPORT_API_VERSION",
    "AXIS_NAMES",
    "ArrayLike",
    "CachingFileManager",
    "DummyLock",
    "GeoReference",
    "GeoTransform",
    "OptionalPackageDoesNotExist",
    "OverviewTargetError",
    "RasterMeta",
    "ZARR_SCHEMA_VERSION",
    "COUNTING_REDUCERS",
    "REDUCERS",
    "_Applied",
    "_BOUNDARIES",
    "_CumSum",
    "_DIFF_LABELS",
    "_Extremum",
    "_INTERP_MIN_POINTS",
    "_InterpTo",
    "_Interpolate",
    "_Pad",
    "_Rank",
    "_Shift",
    "_assert_band_dimension",
    "_check_fill_value",
    "_check_how",
    "_check_limit",
    "_check_min_periods",
    "_check_order",
    "_check_periods",
    "_check_quantile",
    "_check_window",
    "_coarsen_windows",
    "_diffed_array",
    "_dtype_to_str",
    "_frame_axes",
    "_frame_values",
    "_interp_targets",
    "_is_container",
    "_is_identity_packing",
    "_is_in_memory",
    "_pad_before_after",
    "_public_spatial_names",
    "_read_no_data",
    "_reduced_array",
    "_refuse_spatial_band_op",
    "_resolve_interp_kind",
    "_resolve_one_dim",
    "_resolve_positional_indices",
    "_resolve_store",
    "_rolled_array",
    "_transpose_order",
    "_variable_from_applied",
    "apply_unpack",
    "as_numpy",
    "axis_flips",
    "cloud_config_from_env",
    "crs_from_user_input",
    "crs_spec",
    "default_lock",
    "detect_data_var",
    "epsg_from_user_input",
    "extra_hint",
    "finalize_zarr_metadata",
    "gdal_mdarray_open",
    "geobox_crs",
    "import_dask",
    "import_dask_geopandas",
    "import_h5py",
    "import_pyarrow",
    "import_xarray",
    "import_zarr",
    "is_lazy",
    "lazy_extra_hint",
    "no_data_value_of",
    "normalize_compressors",
    "open_mdarray",
    "read_geobox",
    "require_optional",
    "sr_from_epsg",
    "sr_from_wkt",
    "to_fsspec_url",
]
