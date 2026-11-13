"""Core lazy helper for NetCDF reads: the no-data mask applied after a lazy read.

The lazy MDArray readers (``build_lazy_array`` and the per-chunk machinery) now live in the
``pyramids-lazy`` compute backend; :meth:`NetCDF.read_array` with ``chunks=`` dispatches to it through
``require_compute_backend``. What stays here is :func:`mask_no_data`, the small dask-array masking step
core applies to the returned lazy array after unpacking — the lazy counterpart of the eager
``read_array(masked=True)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids.base._utils import extra_hint, import_dask

_DASK_MISSING_MESSAGE = extra_hint(
    "dask is required for lazy NetCDF reads.",
    "lazy",
)


def mask_no_data(arr: Any, fill: Any) -> Any:
    """Return a lazy dask masked array with `arr`'s no-data cells masked.

    The lazy counterpart of the eager `read_array(masked=True)`: cells equal to the (physical)
    `fill` — or NaN cells when `fill` is NaN — are masked, so computing the result yields a
    :class:`numpy.ma.MaskedArray`. `fill=None` (a variable with no declared no-data) yields an
    all-unmasked masked array, matching the eager path. `fill` is the sentinel **in read units**
    (physical, post-unpack); applied after `apply_unpack`, so it compares equal to the unpacked
    fill cells exactly, the same way the eager path's `_read_no_data` does.

    Args:
        arr: A :class:`dask.array.Array` of (already unpacked) values.
        fill: The physical no-data sentinel, or `None`.

    Returns:
        dask.array.Array: A dask masked array over `arr`.
    """
    import_dask(_DASK_MISSING_MESSAGE)
    import dask.array as da

    if fill is None:
        result = da.ma.masked_array(arr)
    elif np.isnan(fill):
        result = da.ma.masked_invalid(arr)
    else:
        result = da.ma.masked_equal(arr, fill)
    return result
