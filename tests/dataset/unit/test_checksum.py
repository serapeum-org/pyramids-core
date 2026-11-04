"""Tests for per-band `Dataset.checksum`."""

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

REF = GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326)


class TestChecksum:
    """`Dataset.checksum` returns GDAL per-band checksums."""

    def test_list_over_bands(self):
        """With no band argument a list of per-band checksums is returned."""
        ds = Dataset.from_array(np.ones((2, 4, 5), np.float32), geo_ref=REF)
        sums = ds.checksum()
        assert isinstance(sums, list)
        assert len(sums) == 2

    def test_single_band_int(self):
        """A 0-based band index returns a single int checksum."""
        ds = Dataset.from_array(np.arange(20.0).reshape(1, 4, 5), geo_ref=REF)
        assert isinstance(ds.checksum(band=0), int)

    def test_identical_data_same_checksum(self):
        """Two datasets with identical data share a band checksum."""
        a = Dataset.from_array(np.arange(20.0).reshape(1, 4, 5), geo_ref=REF)
        b = Dataset.from_array(np.arange(20.0).reshape(1, 4, 5), geo_ref=REF)
        assert a.checksum(band=0) == b.checksum(band=0)

    def test_different_data_differs(self):
        """Different data (practically always) yields a different checksum."""
        a = Dataset.from_array(np.zeros((1, 4, 5), np.float32), geo_ref=REF)
        b = Dataset.from_array(np.ones((1, 4, 5), np.float32), geo_ref=REF)
        assert a.checksum(band=0) != b.checksum(band=0)

    def test_out_of_range_band_raises(self):
        """An out-of-range band index raises IndexError (via `_iloc`)."""
        ds = Dataset.from_array(np.ones((1, 4, 5), np.float32), geo_ref=REF)
        with pytest.raises(IndexError):
            ds.checksum(band=5)
