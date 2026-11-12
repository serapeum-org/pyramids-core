"""Named curve-fit models for :meth:`MeshDataset.curvefit` (issue #1542).

:meth:`~pyramids.netcdf.ugrid.dataset.MeshDataset.curvefit` fits an arbitrary ``f(x, *params)``
model per element along time. That callable has no shell form, so the CLI cannot pass one — the
same reason :meth:`focal_apply` is surfaced through named reducers. This module is the mesh
counterpart for fitting: a registry of the common non-polynomial models, each a plain function
whose positional parameters after ``x`` are the ones fitted (so
:func:`scipy.optimize.curve_fit` can read their count by introspection, no ``p0`` required).

``x`` is the time coordinate as seconds elapsed from the first stamp, so it starts at ``0`` — the
models are chosen to be well behaved there (no ``log(x)`` / ``x ** b`` domain trap at the origin).
Polynomial trends, including a straight line, are the closed-form fast path of
:meth:`~pyramids.netcdf.ugrid.dataset.MeshDataset.polyfit`; ``linear`` is kept here only so the
registry is self-contained for the CLI selector.

Because ``x`` is in **seconds**, a realistic multi-day or multi-month axis reaches ``1e5``–``1e7``.
``curve_fit``'s default initial guess is ``1.0`` per parameter, so the non-linear models
(``exponential`` / ``logistic`` / ``sine``) evaluate e.g. ``exp(1 * x)`` and overflow on the first
iteration, and a non-converging element yields ``NaN`` parameters — the fit can silently come back
all-``NaN``. On such an axis pass a scaled initial guess (a small rate for ``exponential``, a
frequency near ``2 * pi / period`` for ``sine``), or reach for ``polyfit`` / ``linear`` for a trend.

Example:
    >>> from pyramids.netcdf.ugrid.curve_models import CURVE_MODELS
    >>> sorted(CURVE_MODELS)
    ['exponential', 'gaussian', 'linear', 'logistic', 'sine']
    >>> round(float(CURVE_MODELS["linear"](2.0, 3.0, 1.0)), 3)  # a*x + b
    7.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def linear(x: Any, a: float, b: float) -> Any:
    """A straight line ``a * x + b`` (the closed-form case lives in ``polyfit``)."""
    return a * x + b


def exponential(x: Any, a: float, b: float) -> Any:
    """Exponential growth or decay ``a * exp(b * x)``."""
    return a * np.exp(b * x)


def logistic(x: Any, capacity: float, rate: float, midpoint: float) -> Any:
    """A logistic / sigmoid saturation ``L / (1 + exp(-k * (x - x0)))``."""
    return capacity / (1.0 + np.exp(-rate * (x - midpoint)))


def gaussian(x: Any, amplitude: float, mean: float, sigma: float) -> Any:
    """A Gaussian peak ``a * exp(-(x - mu) ** 2 / (2 * sigma ** 2))``."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2.0 * sigma**2))


def sine(x: Any, amplitude: float, frequency: float, phase: float, offset: float) -> Any:
    """A sinusoid ``a * sin(b * x + c) + d`` for a periodic signal."""
    return amplitude * np.sin(frequency * x + phase) + offset


CURVE_MODELS: dict[str, Callable[..., Any]] = {
    "linear": linear,
    "exponential": exponential,
    "logistic": logistic,
    "gaussian": gaussian,
    "sine": sine,
}
"""Mapping of model name to its ``f(x, *params)`` callable, for the CLI ``--model`` selector."""
