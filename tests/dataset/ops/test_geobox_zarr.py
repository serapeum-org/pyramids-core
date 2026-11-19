"""Unit tests for :mod:`pyramids.dataset.ops._geobox_zarr` (pixel-centre maths).

The real-zarr GeoZarr tests (write_geobox / read_geobox / detect_data_var / finalize_zarr_metadata)
moved to the pyramids-lazy plugin — they need the zarr stack. Only the backend-free pixel-centre
coordinate maths stays here.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset.ops._geobox_zarr import pixel_centre_coords

pytestmark = pytest.mark.core

_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


class TestPixelCentreCoords:
    """Tests for pixel_centre_coords."""

    def test_centre_offsets(self):
        """Coordinates are at pixel centres, not edges.

        Test scenario:
            For origin (0, 4), dx=1, dy=-1, a 4×5 grid: x starts at 0.5 and steps
            by 1; y starts at 3.5 and steps by -1 (north-up).
        """
        x, y = pixel_centre_coords(_GT, rows=4, cols=5)
        np.testing.assert_allclose(x, [0.5, 1.5, 2.5, 3.5, 4.5], err_msg=f"x={x}")
        np.testing.assert_allclose(y, [3.5, 2.5, 1.5, 0.5], err_msg=f"y={y}")

    def test_lengths_match_rows_cols(self):
        """x has length cols and y has length rows.

        Test scenario:
            A 3-row, 7-col request yields len(x)==7 and len(y)==3.
        """
        x, y = pixel_centre_coords(_GT, rows=3, cols=7)
        assert (len(x), len(y)) == (7, 3), f"lengths wrong: x={len(x)}, y={len(y)}"

    def test_single_row_and_col(self):
        """A 1×1 grid yields one centre coordinate on each axis.

        Test scenario:
            rows=1, cols=1 returns x=[0.5] and y=[3.5] for the sample transform.
        """
        x, y = pixel_centre_coords(_GT, rows=1, cols=1)
        np.testing.assert_allclose(x, [0.5], err_msg=f"x={x}")
        np.testing.assert_allclose(y, [3.5], err_msg=f"y={y}")
