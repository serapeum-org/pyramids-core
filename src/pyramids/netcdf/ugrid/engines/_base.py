"""Base class shared by every :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` engine.

Defines :class:`_MeshEngine`, the back-reference holder every public mesh engine subclasses.
It mirrors the raster ``pyramids.dataset.engines._base._Engine`` pattern (``__slots__`` single
slot, a ``__reduce__`` placeholder so a directly-pickled engine does not pickle through the
dataset), but binds to :class:`MeshDataset` rather than ``RasterBase`` because the mesh model
is a separate hierarchy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyramids.netcdf.ugrid.dataset import MeshDataset

logger = logging.getLogger("pyramids.netcdf.ugrid.engines")


class _MeshPlaceholder:
    """Stand-in returned by :meth:`_MeshEngine.__reduce__` for a directly-pickled engine.

    A ``MeshDataset`` recreates fresh engines on demand (the accessors are properties), so an
    engine is never part of the dataset's own pickle. If user code pickles an engine directly,
    this placeholder is the unpickle target rather than a circular pickle through ``_ds``.
    """


def _recreate_mesh_placeholder() -> _MeshPlaceholder:
    return _MeshPlaceholder()


class _MeshEngine:
    """Base class for every ``MeshDataset`` engine.

    Holds a reference to the owning :class:`MeshDataset` in a single slot. The core accessors
    are ``MeshDataset`` properties that pass ``self`` (a strong reference held only for the
    duration of the call, with no stored ``mesh -> engine`` cycle), so a temporary such as
    ``MeshDataset.read_file(p).terrain.slope()`` stays alive for the call.
    """

    __slots__ = ("_ds",)

    _ds: MeshDataset

    def __init__(self, ds: MeshDataset) -> None:
        self._ds = ds

    def __reduce__(self) -> tuple[Any, tuple]:
        return (_recreate_mesh_placeholder, ())
