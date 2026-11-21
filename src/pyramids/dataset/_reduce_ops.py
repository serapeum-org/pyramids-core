"""Validation and op tables for DatasetCollection time-axis reductions.

The lazy reduction *builders* (the :mod:`dask.array` dispatch and the mode / trend numpy kernels) live
in the pyramids-lazy backend (``pyramids_lazy._reduce``); this module keeps only the op tables and the
pure-Python validation :meth:`DatasetCollection.reduce_time` uses to check an op / level before
dispatching to the backend. The tables ``_SUPPORTED_OPS`` / ``_NAN_TABLE`` are re-exported to the
backend through :mod:`pyramids.lazy_support`.
"""

from __future__ import annotations

from pyramids.base._reductions import CORE_REDUCER_NAMES

_SUPPORTED_OPS = CORE_REDUCER_NAMES
_NAN_TABLE = {op: f"nan{op}" for op in _SUPPORTED_OPS}

#: Holistic ops :meth:`DatasetCollection.reduce_time` adds on top of the six
#: associative ones; none is a simple :mod:`dask.array` reduction.
_HOLISTIC_OPS = ("median", "quantile", "percentile", "mode", "trend", "sens_slope")

#: Every op :meth:`DatasetCollection.reduce_time` accepts.
TIME_REDUCE_OPS = _SUPPORTED_OPS + _HOLISTIC_OPS

#: Ops that require a level ``q`` (and reject it for every other op).
_Q_OPS = ("quantile", "percentile")

#: Per-pixel slope ops that need the window's numeric timestamps.
_TREND_OPS = ("trend", "sens_slope")


def validate_time_op(op: str, q: float | None) -> None:
    """Validate a :meth:`DatasetCollection.reduce_time` ``op`` / ``q`` pair.

    Args:
        op: The reduction name; must be one of :data:`TIME_REDUCE_OPS`.
        q: The level for ``"percentile"`` (``0 < q <= 100``) or ``"quantile"``
            (``0 < q <= 1``). Required for those two ops, and refused (must be
            ``None``) for every other op.

    Raises:
        ValueError: ``op`` is unsupported; ``q`` is missing for a
            percentile/quantile op or out of range; or ``q`` is given for an
            op that takes none.
    """
    if op not in TIME_REDUCE_OPS:
        raise ValueError(f"op must be one of {TIME_REDUCE_OPS}, got {op!r}.")
    if op in _Q_OPS:
        high = 100.0 if op == "percentile" else 1.0
        if q is None:
            raise ValueError(f"op={op!r} requires a level q in (0, {high:g}].")
        if not 0 < q <= high:
            raise ValueError(
                f"q must satisfy 0 < q <= {high:g} for op={op!r}, got {q!r}."
            )
    elif q is not None:
        raise ValueError(
            f"q is only valid for 'percentile' / 'quantile', not op={op!r}."
        )
