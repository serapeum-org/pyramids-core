"""Slope and aspect trigonometry over a gradient field (D7).

The angle math that turns a surface gradient ``(dz_dx, dz_dy)`` into a slope or an
aspect is identical for a raster DEM and a mesh DEM -- only the way the gradient is
produced differs (centred differences over a box for a raster, a per-face plane fit for
a mesh). That geometry stays with the caller; this module owns only the trig, so the
two hierarchies share one definition of what "slope in degrees" and "aspect clockwise
from north" mean, and one place that rejects an unknown ``units``.

``pyramids.base.geodesy`` answers how far apart two coordinates are on the ellipsoid;
this module answers the angle of a local height gradient -- a different question, which
is why it is its own module rather than folded into ``geodesy``.

Pure numpy, so a caller may pass an eager array or a dask block alike.
"""

from __future__ import annotations

import numpy as np

_ANGLE_UNITS = ("degrees", "radians")


def validate_angle_units(units: str, *, operation: str = "slope") -> None:
    """Reject a ``units`` that is neither ``"degrees"`` nor ``"radians"``.

    Args:
        units: The spelling the caller passed.
        operation: The public method name, so the message blames the caller's call.

    Raises:
        ValueError: ``units`` is not one of the two accepted spellings. The check is
            exact and case-sensitive, so ``"DEGREES"`` and ``"radian"`` are refused --
            otherwise a typo silently returns radians, a 57x error (D7).
    """
    if units not in _ANGLE_UNITS:
        raise ValueError(
            f"{operation}: units must be 'degrees' or 'radians', got {units!r}."
        )


def slope_from_gradient(
    dz_dx: np.ndarray, dz_dy: np.ndarray, *, units: str = "degrees"
) -> np.ndarray:
    """Slope magnitude ``arctan(hypot(dz_dx, dz_dy))`` in ``units``.

    ``dz_dy`` is the row-wise (southward-increasing) derivative, matching
    :func:`numpy.gradient` over a north-up raster; a mesh caller negates its northward
    derivative to match. ``hypot`` is sign-blind, so the orientation does not change the
    slope -- it matters only for :func:`aspect_from_gradient`.

    Args:
        dz_dx: Gradient in the column (easting) direction.
        dz_dy: Gradient in the row (southward) direction.
        units: ``"degrees"`` (default) or ``"radians"``.

    Returns:
        numpy.ndarray: Per-cell slope, ``float64``, in the requested unit.

    Raises:
        ValueError: ``units`` is neither ``"degrees"`` nor ``"radians"``.
    """
    validate_angle_units(units, operation="slope")
    radians = np.arctan(np.hypot(dz_dx, dz_dy))
    return np.degrees(radians) if units == "degrees" else radians


def aspect_from_gradient(dz_dx: np.ndarray, dz_dy: np.ndarray) -> np.ndarray:
    """Aspect in ``[0, 360)`` degrees clockwise from north -- the downhill direction.

    ``dz_dy`` is the row-wise derivative (see :func:`slope_from_gradient`); the sign is
    load-bearing here. A perfectly flat cell answers ``270.0`` as an artefact of
    ``arctan2`` on signed zeros -- both hierarchies already did, so treat it as an
    artefact, not a direction.

    Args:
        dz_dx: Gradient in the column (easting) direction.
        dz_dy: Gradient in the row (southward) direction.

    Returns:
        numpy.ndarray: Per-cell aspect in ``[0, 360)`` degrees, clockwise from north.
    """
    angle = np.degrees(np.arctan2(dz_dy, -dz_dx))
    return np.mod(450.0 - angle, 360.0)
