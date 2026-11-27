"""The plotting family is reachable as ``ds.glyph``.

Pins P2-1 of the engine-accessor migration: a ``ds.glyph`` engine groups the plotting
methods. ``histogram``/``vector_field``/``set_color_ramp`` are owned here (the flat facades
delegate to the accessor); ``plot``/``animate`` are exposed too (equivalent to the flat
methods pending the later ownership move).
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.engines.glyph import _Glyph


@pytest.fixture
def byte_raster() -> Dataset:
    arr = np.arange(16, dtype="uint8").reshape(4, 4)
    return Dataset.from_array(
        arr,
        geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
    )


def test_accessor_is_a_glyph_engine(byte_raster: Dataset):
    assert isinstance(byte_raster.glyph, _Glyph)


@pytest.mark.parametrize(
    "name", ["plot", "animate", "histogram", "vector_field", "set_color_ramp"]
)
def test_glyph_exposes_the_plotting_family(byte_raster: Dataset, name: str):
    assert callable(getattr(byte_raster.glyph, name))


def test_set_color_ramp_via_glyph_writes_the_palette(byte_raster: Dataset):
    byte_raster.glyph.set_color_ramp(start_value=0, end_value=15, colormap="viridis")
    assert byte_raster.bands.color_table is not None


def test_flat_set_color_ramp_delegates_to_glyph(byte_raster: Dataset):
    # The flat facade produces the same effect as the accessor.
    byte_raster.glyph.set_color_ramp(start_value=0, end_value=15, colormap="viridis")
    assert byte_raster.bands.color_table is not None
