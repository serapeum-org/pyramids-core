"""Deprecated shim: the 1D mesh topology moved to `pyramids.netcdf.ugrid.mesh1d`.

`Network1d` was renamed to :class:`~pyramids.netcdf.ugrid.mesh1d.Mesh1d` -- it is the 1D
unstructured mesh topology (nodes + edges), peer to :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`,
not a hydrodynamic-specific type. Import from `mesh1d` instead; this module re-exports the
new name under the old one for one release and warns on use.
"""

from __future__ import annotations

import warnings

from pyramids.netcdf.ugrid.mesh1d import Mesh1d

__all__ = ["Mesh1d"]


def __getattr__(name: str):
    """Return `Mesh1d` for the old `Network1d` name, warning once.

    Args:
        name: The attribute requested from this shim module.

    Returns:
        :class:`~pyramids.netcdf.ugrid.mesh1d.Mesh1d` for ``"Network1d"``.

    Raises:
        AttributeError: Any other name.
    """
    if name == "Network1d":
        warnings.warn(
            "pyramids.netcdf.ugrid.network.Network1d is renamed to "
            "pyramids.netcdf.ugrid.mesh1d.Mesh1d; import Mesh1d from mesh1d instead. "
            "It will be removed in a future release (see the migration guide).",
            DeprecationWarning,
            stacklevel=2,
        )
        return Mesh1d
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
