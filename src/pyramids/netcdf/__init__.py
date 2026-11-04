"""NetCDF subpackage for pyramids."""

from __future__ import annotations

import warnings

from pyramids.netcdf.array_options import (
    CFAttributes,
    Encoding,
    ExtraDimensions,
    GeoReference,
)
from pyramids.netcdf.engines.selection import CumulativeAccessor
from pyramids.netcdf.labeled import LabeledArray, LabeledDataset
from pyramids.netcdf.metadata import from_json, get_metadata, to_dict, to_json
from pyramids.netcdf.models import (
    CFInfo,
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
    VariableInfo,
)
from pyramids.netcdf.netcdf import Container, NetCDF, Variable
from pyramids.netcdf.plot_options import (
    CoordinateSpec,
    FacetSpec,
    Selectors,
)
from pyramids.netcdf.ugrid import MeshDataset

__all__ = [
    "CFAttributes",
    "CFInfo",
    "Container",
    "CoordinateSpec",
    "CumulativeAccessor",
    "DimensionInfo",
    "Encoding",
    "ExtraDimensions",
    "FacetSpec",
    "from_json",
    "GeoReference",
    "get_metadata",
    "GroupInfo",
    "LabeledArray",
    "LabeledDataset",
    "LazyNetCDF",
    "NetCDF",
    "NetCDFMetadata",
    "Selectors",
    "StructuralInfo",
    "to_dict",
    "to_json",
    "MeshDataset",
    "Variable",
    "VariableInfo",
]


def __getattr__(name: str):
    """Resolve lazily-exposed `pyramids.netcdf` names (`LazyNetCDF`, legacy `UgridDataset`).

    `LazyNetCDF` is resolved here rather than imported at module top so `import pyramids.netcdf`
    does not eagerly pull the lazy `_lazy_cube` submodule (extraction-readiness, plan #2).

    `UgridDataset` was renamed to :class:`~pyramids.netcdf.ugrid.MeshDataset`; this top-level
    re-export kept the old name importable for one release so `from pyramids.netcdf import
    UgridDataset` does not break.

    Args:
        name: The attribute requested from this package.

    Returns:
        :class:`~pyramids.netcdf.ugrid.MeshDataset` for the old ``"UgridDataset"`` name.

    Raises:
        AttributeError: Any other unknown name.
    """

    if name == "LazyNetCDF":
        # Resolved lazily so importing `pyramids.netcdf` does not pull the lazy `_lazy_cube`
        # module (extraction-readiness, plan #2); the class itself imports without the [lazy] extra.
        # Mirrors feature/__init__'s PEP-562 deferral of LazyFeatureCollection.
        from pyramids.netcdf._lazy_cube import LazyNetCDF

        return LazyNetCDF
    if name == "UgridDataset":
        warnings.warn(
            "UgridDataset is renamed to MeshDataset; import MeshDataset instead. "
            "UgridDataset will be removed in a future release (see the migration guide).",
            DeprecationWarning,
            stacklevel=2,
        )
        return MeshDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
