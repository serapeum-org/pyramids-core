"""Terrain engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.terrain``).

Owns the DEM-derivative family -- slope, aspect, hillshade -- on an unstructured mesh. Mirrors
the raster ``Dataset.terrain`` engine; the per-face plane-gradient maths live in
:mod:`pyramids.netcdf.ugrid.terrain` and are shared by both call forms.
"""

from __future__ import annotations

import numpy as np

from pyramids.netcdf.ugrid import terrain as _terrain
from pyramids.netcdf.ugrid.engines._base import _MeshEngine


class _MeshTerrain(_MeshEngine):
    """DEM derivatives on an unstructured mesh: slope, aspect, hillshade (``mesh.terrain``)."""

    __slots__ = ()

    def slope(
        self, units: str = "degrees", variable_name: str | None = None
    ) -> np.ndarray:
        """Per-face slope of a mesh DEM, in degrees (default) or radians (issue #1374).

        An unstructured bed-level mesh is a DEM. The raster version reads a 3x3 pixel window,
        which has no mesh equivalent; this fits a plane through each face's own node elevations
        and reads the slope off that -- exact for a triangle, least-squares for a polygon.

        Args:
            units: ``"degrees"`` (default) or ``"radians"``.
            variable_name: The elevation variable (default: the only one). A face-located one
                is averaged onto the nodes first.

        Returns:
            numpy.ndarray: Per-face slope of length ``n_face``; ``NaN`` on a degenerate face,
            and on any face touching a ``nodata`` elevation -- a void has no defined gradient,
            which is the contract the raster kernels state too.

        Raises:
            ValueError: ``units`` is invalid, the variable is edge-located, or it is not one
                value per element.

        See Also:
            aspect: The direction the slope faces.
            hillshade: Shaded relief from the same gradient.
        """
        return _terrain.slope(
            self._ds._mesh,
            self._ds._elevation_nodes("slope", variable_name),
            units=units,
        )

    def aspect(self, variable_name: str | None = None) -> np.ndarray:
        """Per-face aspect of a mesh DEM, in degrees clockwise from north (issue #1374).

        Names the direction the slope **faces** (downhill), matching the raster convention: a
        surface rising to the east faces west, 270 degrees.

        Args:
            variable_name: The elevation variable (default: the only one).

        Returns:
            numpy.ndarray: Per-face aspect in ``[0, 360)``; ``NaN`` on a degenerate face, and
            on any face touching a ``nodata`` elevation.

        Raises:
            ValueError: The variable is edge-located, or is not one value per element.

        See Also:
            slope: The magnitude of the same gradient.
        """
        return _terrain.aspect(
            self._ds._mesh,
            self._ds._elevation_nodes("aspect", variable_name),
        )

    def hillshade(
        self,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        variable_name: str | None = None,
    ) -> np.ndarray:
        """Per-face shaded relief of a mesh DEM, clipped to ``0..255`` (issue #1374).

        Args:
            azimuth: Sun azimuth in degrees (0 = north, 90 = east). Defaults to 315 (NW).
            altitude: Sun altitude above the horizon in degrees. Defaults to 45.
            variable_name: The elevation variable (default: the only one).

        Returns:
            numpy.ndarray: Per-face intensity in ``[0, 255]``; ``NaN`` on a degenerate face,
            and on any face touching a ``nodata`` elevation. A flat mesh saturates at
            ``sin(altitude) * 255``, as the raster version does.

        Raises:
            ValueError: The variable is edge-located, or is not one value per element.

        See Also:
            slope: The slope magnitude the shading is derived from.
        """
        return _terrain.hillshade(
            self._ds._mesh,
            self._ds._elevation_nodes("hillshade", variable_name),
            azimuth=azimuth,
            altitude=altitude,
        )
