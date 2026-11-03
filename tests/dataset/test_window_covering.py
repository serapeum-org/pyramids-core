"""Tests for `Window.covering` (window covering a geometry's bounding box)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pyramids.dataset.window import Window

# Unit-cell grid, origin (0, 4), north-up.
GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


class TestWindowCovering:
    """`covering` extracts a bbox from several geometry forms and delegates to from_bounds."""

    def test_tuple_bbox_matches_from_bounds(self):
        """A 4-tuple bbox gives the same window as from_bounds."""
        bbox = (1.0, 1.0, 3.0, 3.0)
        assert Window.covering(bbox, GT) == Window.from_bounds(bbox, GT)

    def test_shapely_geometry(self):
        """A shapely geometry uses its .bounds."""
        geom = box(1.0, 1.0, 3.0, 3.0)
        assert Window.covering(geom, GT) == Window.from_bounds((1.0, 1.0, 3.0, 3.0), GT)

    def test_geodataframe_total_bounds(self):
        """A GeoDataFrame uses total_bounds across all its geometries."""
        gdf = gpd.GeoDataFrame(
            geometry=[box(1.0, 1.0, 2.0, 2.0), box(2.0, 2.0, 3.0, 3.0)]
        )
        assert Window.covering(gdf, GT) == Window.from_bounds((1.0, 1.0, 3.0, 3.0), GT)

    def test_geoseries_total_bounds(self):
        """A GeoSeries also routes through total_bounds."""
        gs = gpd.GeoSeries([box(1.0, 1.0, 2.0, 2.0), box(2.0, 2.0, 3.0, 3.0)])
        assert Window.covering(gs, GT) == Window.from_bounds((1.0, 1.0, 3.0, 3.0), GT)

    def test_numpy_array_bbox(self):
        """A numpy array bbox falls through to the plain-sequence branch."""
        bbox = np.array([1.0, 1.0, 3.0, 3.0])
        assert Window.covering(bbox, GT) == Window.from_bounds((1.0, 1.0, 3.0, 3.0), GT)

    def test_fully_covers(self):
        """A non-aligned bbox is fully contained by the covering window's bounds."""
        win = Window.covering((1.3, 1.7, 2.1, 2.4), GT)
        min_x, min_y, max_x, max_y = win.to_bounds(GT)
        assert min_x <= 1.3 and min_y <= 1.7
        assert max_x >= 2.1 and max_y >= 2.4

    def test_read_args(self):
        """The covering window reads as (col_off, row_off, cols, rows)."""
        assert Window.covering((1.0, 1.0, 3.0, 3.0), GT).to_read_args() == (1, 1, 2, 2)

    def test_inverted_bbox_raises(self):
        """An inverted bounding box raises ValueError (inherited from from_bounds)."""
        with pytest.raises(ValueError):
            Window.covering((3.0, 3.0, 1.0, 1.0), GT)

    def test_empty_geodataframe_raises_clear_error(self):
        """An empty GeoDataFrame (NaN total_bounds) raises a clear 'empty' ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Window.covering(gpd.GeoDataFrame(geometry=[]), GT)

    def test_empty_geometry_raises_clear_error(self):
        """An empty shapely geometry (NaN bounds) raises a clear 'empty' ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Window.covering(Polygon(), GT)
