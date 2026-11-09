"""Deprecation alias from the dataset naming cleanup: ``to_raster`` -> ``to_file``.

``to_raster`` was a redundant second name for :meth:`Dataset.to_file`. It keeps working but emits a
``DeprecationWarning`` that points at ``to_file``.
"""

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

_GEO = dict(top_left_corner=(0, 2), cell_size=1.0, epsg=4326)


def _value_raster() -> Dataset:
    """A 2x2 value raster."""
    values = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
    return Dataset.from_array(values, geo_ref=GeoReference(**_GEO))


class TestToRasterRenamedToToFile:
    """``to_raster`` is the deprecated alias; ``to_file`` is canonical."""

    def test_dataset_to_raster_warns_and_writes(self, tmp_path):
        """``Dataset.to_raster`` still writes the file but warns."""
        out = tmp_path / "out.tif"
        with pytest.warns(DeprecationWarning, match="to_file"):
            _value_raster().to_raster(str(out))
        assert out.exists(), "to_raster must still write the raster"

    def test_io_engine_to_raster_warns(self, tmp_path):
        """The engine-level ``IO.to_raster`` warns too."""
        out = tmp_path / "out2.tif"
        ds = _value_raster()  # keep a strong ref; the io engine holds only a weakref
        with pytest.warns(DeprecationWarning, match="to_file"):
            ds.io.to_raster(str(out))
        assert out.exists(), "IO.to_raster must still write the raster"
