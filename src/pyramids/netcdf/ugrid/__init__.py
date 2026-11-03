"""UGRID (Unstructured Grid) NetCDF support for pyramids.

This subpackage provides classes for reading, writing, and operating
on UGRID-compliant NetCDF files containing unstructured mesh data.

Beyond the core data model (:class:`MeshDataset`, :class:`Mesh2d`, :class:`Mesh1d`) it
carries the mesh operation modules: :mod:`~pyramids.netcdf.ugrid.graph` (topology and graph
operations), :mod:`~pyramids.netcdf.ugrid.fill` (spatial gap filling),
:mod:`~pyramids.netcdf.ugrid.vector_ops` (mesh↔vector), :mod:`~pyramids.netcdf.ugrid.regrid`
(sampling and regridding), :mod:`~pyramids.netcdf.ugrid.construct` (building a mesh from a
raster or polygons), :mod:`~pyramids.netcdf.ugrid.interop_xarray` (xarray / zarr hand-off),
:mod:`~pyramids.netcdf.ugrid.analysis` (point extraction, random sampling, contouring) and
:mod:`~pyramids.netcdf.ugrid.mesh1d` (1-D networks).
Each is reachable through a :class:`MeshDataset` facade; the modules are exported too for
callers holding a bare :class:`Mesh2d`.
"""

from __future__ import annotations

from pyramids.netcdf.ugrid import (
    analysis,
    construct,
    fill,
    graph,
    interop_xarray,
    mesh1d,
    regrid,
    vector_ops,
)
from pyramids.netcdf.ugrid._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    MESH_DRIVER_API_VERSION,
    NETCDF_SUFFIXES,
    RESERVED_NAMES,
    MeshDriver,
    NoMeshTopologyError,
    UnsupportedMeshFormatError,
    mesh_drivers,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import MeshDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.mesh1d import Mesh1d
from pyramids.netcdf.ugrid.models import (
    MeshMetadata,
    MeshTopologyInfo,
    MeshVariable,
    VolumeTopology,
)
from pyramids.netcdf.ugrid.regrid import Regridder
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

__all__ = [
    "CAPABILITIES",
    "ENTRY_POINT_GROUP",
    "MESH_DRIVER_API_VERSION",
    "NETCDF_SUFFIXES",
    "RESERVED_NAMES",
    "Connectivity",
    "MeshDriver",
    "Mesh2d",
    "MeshSpatialIndex",
    "MeshTopologyInfo",
    "MeshVariable",
    "Mesh1d",
    "NoMeshTopologyError",
    "Regridder",
    "MeshDataset",
    "MeshMetadata",
    "UnsupportedMeshFormatError",
    "VolumeTopology",
    "analysis",
    "construct",
    "fill",
    "graph",
    "interop_xarray",
    "mesh_drivers",
    "mesh1d",
    "regrid",
    "vector_ops",
]


#: Old public names kept importable for one release, mapped to their replacements.
#: `Network1d`/`UgridDataset`/`UgridMetadata` were renamed when the unstructured-mesh
#: data model dropped the UGRID-convention and hydrodynamic wording from its type names
#: (the convention is an I/O format, not the identity of the object).
_DEPRECATED_NAMES = {
    "Network1d": ("Mesh1d", Mesh1d),
    "UgridDataset": ("MeshDataset", MeshDataset),
    "UgridMetadata": ("MeshMetadata", MeshMetadata),
}


def __getattr__(name: str):
    """Resolve a renamed public name, warning once, or raise `AttributeError`.

    Args:
        name: The attribute requested from this package.

    Returns:
        The replacement object for a deprecated name.

    Raises:
        AttributeError: `name` is neither a current nor a deprecated public name.
    """
    import warnings

    if name == "network":
        # Return the real `network.py` shim, not the `mesh1d` module: the shim both re-exports
        # `Mesh1d` and resolves the old `Network1d` name through its own `__getattr__`, so
        # `ugrid.network.Network1d` works regardless of whether the shim was imported first.
        # Returning `mesh1d` here would leave `ugrid.network.Network1d` raising AttributeError.
        import importlib

        warnings.warn(
            "pyramids.netcdf.ugrid.network is renamed to pyramids.netcdf.ugrid.mesh1d; "
            "import from mesh1d instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module("pyramids.netcdf.ugrid.network")
    entry = _DEPRECATED_NAMES.get(name)
    if entry is not None:
        new_name, obj = entry
        warnings.warn(
            f"{name} is renamed to {new_name}; import {new_name} instead "
            f"({name} will be removed in a future release).",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
