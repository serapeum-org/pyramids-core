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

    Examples:
        - An accepted spelling passes silently:
            ```python
            >>> validate_angle_units("degrees")

            ```
        - A miscased or unknown unit is refused:
            ```python
            >>> validate_angle_units("DEGREES")
            Traceback (most recent call last):
                ...
            ValueError: slope: units must be 'degrees' or 'radians', got 'DEGREES'.

            ```
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

    Examples:
        - A unit east-west gradient is a 45-degree slope:
            ```python
            >>> import numpy as np
            >>> slope_from_gradient(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
            array([ 0., 45.])

            ```
        - The same in radians:
            ```python
            >>> import numpy as np
            >>> np.round(slope_from_gradient(np.array([1.0]), np.array([0.0]), units="radians"), 6)
            array([0.785398])

            ```
    """
    validate_angle_units(units, operation="slope")
    radians = np.arctan(np.hypot(dz_dx, dz_dy))
    result = np.degrees(radians) if units == "degrees" else radians
    return np.asarray(result, dtype=np.float64)


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

    Examples:
        - A surface rising to the east faces west, 270 degrees:
            ```python
            >>> import numpy as np
            >>> aspect_from_gradient(np.array([1.0]), np.array([0.0]))
            array([270.])

            ```
        - A surface rising to the north faces south, 180 degrees:
            ```python
            >>> import numpy as np
            >>> aspect_from_gradient(np.array([0.0]), np.array([-1.0]))
            array([180.])

            ```
    """
    angle = np.degrees(np.arctan2(dz_dy, -dz_dx))
    return np.asarray(np.mod(450.0 - angle, 360.0), dtype=np.float64)
