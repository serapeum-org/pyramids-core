"""Shared COG overview enumeration.

`validate` and `inspect` both walk a band's overview pyramid and read the same
four facts per level (size, block size, decimation). This is the single source
of that walk; each caller wraps the yielded tuples into its own shape (a plain
dict for the validation report, an :class:`OverviewLevel` for the inspector).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def enumerate_overviews(
    band: Any, full_width: int
) -> Iterator[tuple[int, int, int, tuple[int, int], int]]:
    """Yield one record per overview level of `band`.

    Args:
        band: A GDAL raster band (`gdal.Band`) to walk the overviews of.
        full_width: The full-resolution raster width in pixels, used as the
            numerator of each level's decimation factor.

    Yields:
        tuple[int, int, int, tuple[int, int], int]: `(index, width, height,
            (block_x, block_y), decimation)` for each overview level, where
            `decimation` is `round(full_width / level_width)` (0 when the
            level reports a zero width).
    """
    for index in range(band.GetOverviewCount()):
        overview = band.GetOverview(index)
        block_x, block_y = overview.GetBlockSize()
        decimation = round(full_width / overview.XSize) if overview.XSize else 0
        yield index, overview.XSize, overview.YSize, (block_x, block_y), decimation
