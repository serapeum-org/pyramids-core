"""Tests for the in-memory raster handle `pyramids.MemoryFile`."""

import numpy as np
import pytest
from osgeo import gdal

from pyramids import MemoryFile
from pyramids._io import read_vsi_bytes
from pyramids.dataset import Dataset, GeoReference

REF = GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326)


def _raster_bytes():
    """Encoded GeoTIFF bytes of a small 1-band raster."""
    return Dataset.from_array(np.arange(12.0).reshape(1, 3, 4), geo_ref=REF).to_bytes()


class TestMemoryFile:
    """Stage raster bytes in memory, open them as a Dataset, read them back, clean up."""

    def test_stage_bytes_and_open(self):
        """Bytes passed to the constructor open as a Dataset with the right shape."""
        with MemoryFile(_raster_bytes()) as mf:
            ds = mf.open()
            assert ds.shape == (1, 3, 4)
            assert ds.epsg == 4326

    def test_name_is_vsimem_path(self):
        """`name` is a real /vsimem/ path."""
        with MemoryFile(_raster_bytes()) as mf:
            assert mf.name.startswith("/vsimem/")

    def test_write_then_open(self):
        """An empty handle can be filled via write() and then opened."""
        with MemoryFile() as mf:
            mf.write(_raster_bytes())
            assert mf.open().shape == (1, 3, 4)

    def test_read_round_trip(self):
        """read() returns bytes that reopen to the same array."""
        data = _raster_bytes()
        with MemoryFile(data) as mf:
            back = mf.read()
        reopened = MemoryFile(back).open()
        assert np.array_equal(
            reopened.read_array(), Dataset.from_bytes(data).read_array()
        )

    def test_close_unlinks_buffer(self):
        """After close() the backing /vsimem/ buffer is gone and the handle is unusable."""
        mf = MemoryFile(_raster_bytes())
        path = mf.name
        mf.close()
        with pytest.raises((FileNotFoundError, OSError)):
            read_vsi_bytes(path)
        with pytest.raises(ValueError):
            mf.name
        with pytest.raises(ValueError):
            mf.open()

    def test_double_close_is_safe(self):
        """Closing twice does not raise."""
        mf = MemoryFile(_raster_bytes())
        mf.close()
        mf.close()

    def test_context_manager_cleans_up(self):
        """Leaving the with-block unlinks the buffer."""
        with MemoryFile(_raster_bytes()) as mf:
            path = mf.name
        with pytest.raises((FileNotFoundError, OSError)):
            read_vsi_bytes(path)

    def test_bad_data_type_raises(self):
        """Non-bytes data to the constructor raises TypeError."""
        with pytest.raises(TypeError):
            MemoryFile("not bytes")

    def test_write_bad_type_raises(self):
        """Non-bytes data to write() raises TypeError."""
        with MemoryFile() as mf:
            with pytest.raises(TypeError):
                mf.write("not bytes")

    def test_read_before_write_raises(self):
        """read() on an empty handle (nothing written) raises FileNotFoundError/OSError."""
        with MemoryFile() as mf:
            with pytest.raises((FileNotFoundError, OSError)):
                mf.read()
