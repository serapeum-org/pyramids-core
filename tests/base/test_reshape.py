"""Tests for the band-axis reshape helpers `reshape_as_image` / `reshape_as_raster`."""

import numpy as np
import pytest

from pyramids import reshape_as_image, reshape_as_raster


class TestReshape:
    """Band-first <-> band-last conversions, preserving data and masks."""

    def test_image_shape(self):
        """`(bands, rows, cols)` becomes `(rows, cols, bands)`."""
        assert reshape_as_image(np.zeros((3, 4, 5))).shape == (4, 5, 3)

    def test_raster_shape(self):
        """`(rows, cols, bands)` becomes `(bands, rows, cols)`."""
        assert reshape_as_raster(np.zeros((4, 5, 3))).shape == (3, 4, 5)

    def test_round_trip(self):
        """image(raster(x)) and raster(image(x)) return the original data."""
        arr = np.arange(60).reshape(3, 4, 5)
        assert np.array_equal(reshape_as_raster(reshape_as_image(arr)), arr)

    def test_masked_preserved(self):
        """A masked array stays a masked array and its mask is transposed."""
        arr = np.ma.masked_equal(np.arange(60).reshape(3, 4, 5), 0)
        out = reshape_as_image(arr)
        assert isinstance(out, np.ma.MaskedArray)
        assert out.mask.shape == (4, 5, 3)

    def test_image_ndim_guard(self):
        """A non-3-D array to reshape_as_image raises ValueError."""
        with pytest.raises(ValueError, match="3-D"):
            reshape_as_image(np.zeros((4, 5)))

    def test_raster_ndim_guard(self):
        """A non-3-D array to reshape_as_raster raises ValueError."""
        with pytest.raises(ValueError, match="3-D"):
            reshape_as_raster(np.zeros((4, 5)))
