"""Terrain engine.

Owns the DEM-derivative family -- ``slope``, ``aspect``, ``hillshade`` -- on a
Dataset. Accessed as ``ds.terrain``; the Dataset exposes same-named facade
methods so ``ds.slope(...)`` and ``ds.terrain.slope(...)`` are equivalent.

These derivatives technically fit the ``analysis`` family, but they form a
cohesive DEM sub-domain, so grouping them under their own accessor keeps
``analysis`` from swelling and makes them discoverable as ``ds.terrain.<TAB>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.ops._focal import aspect, hillshade, slope

if TYPE_CHECKING:
    import dask.array as da

    from pyramids.dataset.dataset import (  # noqa: F401  (forward ref in _Engine["Dataset"])
        Dataset,
    )


class _Terrain(_Engine["Dataset"]):
    """DEM-derivative operations on a Dataset.

    Owns the real implementations of ``slope``, ``aspect`` and ``hillshade``.
    ``Dataset`` exposes a same-named facade for each, so ``ds.slope(...)`` and
    ``ds.terrain.slope(...)`` are equivalent.
    """

    def slope(
        self, *, chunks: Any = None, band: int = 0, units: str = "degrees"
    ) -> np.ndarray | da.Array:
        """Surface slope of a DEM in degrees (default) or radians.

        Args:
            chunks: Lazy-path chunk spec; ``None`` runs eagerly.
            band: Zero-based band index.
            units: ``"degrees"`` (default) or ``"radians"``.

        Returns:
            numpy.ndarray or dask.array.Array: Per-cell slope magnitude; no-data
            cells and their immediate neighbours carry the band's sentinel.
        """
        return slope(self._ds, chunks=chunks, band=band, units=units)

    def aspect(self, *, chunks: Any = None, band: int = 0) -> np.ndarray | da.Array:
        """Surface aspect (down-slope compass direction) of a DEM.

        Args:
            chunks: Lazy-path chunk spec; ``None`` runs eagerly.
            band: Zero-based band index.

        Returns:
            numpy.ndarray or dask.array.Array: Per-cell aspect; no-data cells and
            their immediate neighbours carry the band's sentinel.
        """
        return aspect(self._ds, chunks=chunks, band=band)

    def hillshade(
        self,
        *,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        chunks: Any = None,
        band: int = 0,
    ) -> np.ndarray | da.Array:
        """Hillshade (shaded relief) of a DEM for a light source.

        Args:
            azimuth: Light-source compass azimuth in degrees (default ``315``).
            altitude: Light-source altitude above the horizon in degrees (default ``45``).
            chunks: Lazy-path chunk spec; ``None`` runs eagerly.
            band: Zero-based band index.

        Returns:
            numpy.ndarray or dask.array.Array: Per-cell shaded-relief intensity.
        """
        return hillshade(
            self._ds, azimuth=azimuth, altitude=altitude, chunks=chunks, band=band
        )
