"""The `ds.terrain` accessor owns slope/aspect/hillshade; the flat facades agree.

Pins P2-2 of the engine-accessor migration: the DEM-derivative family moves onto a
dedicated `ds.terrain` engine, and the flat `ds.slope`/`ds.aspect`/`ds.hillshade`
methods delegate to it (so both call forms return identical results).
"""

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.engines.terrain import _Terrain


@pytest.fixture
def dem() -> Dataset:
    """A small tilted-plane DEM with a non-trivial slope."""
    rows, cols = np.mgrid[0:8, 0:8]
    heights = (rows * 3.0 + cols * 1.5).astype(np.float32)
    return Dataset.from_array(
        heights,
        geo_ref=GeoReference(top_left_corner=(0.0, 8.0), cell_size=1.0, epsg=32636),
    )


def test_accessor_is_a_terrain_engine(dem: Dataset):
    assert isinstance(dem.terrain, _Terrain)


def test_slope_accessor_matches_flat_facade(dem: Dataset):
    np.testing.assert_array_equal(
        np.asarray(dem.slope()), np.asarray(dem.terrain.slope())
    )


def test_aspect_accessor_matches_flat_facade(dem: Dataset):
    np.testing.assert_array_equal(
        np.asarray(dem.aspect()), np.asarray(dem.terrain.aspect())
    )


def test_hillshade_accessor_matches_flat_facade(dem: Dataset):
    np.testing.assert_array_equal(
        np.asarray(dem.hillshade(azimuth=300.0, altitude=40.0)),
        np.asarray(dem.terrain.hillshade(azimuth=300.0, altitude=40.0)),
    )


def test_slope_units_forwarded_through_accessor(dem: Dataset):
    deg = float(np.nanmax(np.asarray(dem.terrain.slope(units="degrees"))))
    rad = float(np.nanmax(np.asarray(dem.terrain.slope(units="radians"))))
    assert deg > rad > 0.0
