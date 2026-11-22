"""The data extent is the tight box around a band's valid cells.

`Dataset.total_bounds` answers "where is the grid"; `data_extent` answers "where
is the data in it". The two agree only when the band has no gaps -- the moment a
sentinel marks a cell, the data extent shrinks inside the grid. These pin that
shrink, that an all-sentinel band has no extent at all, that the mask is read
from the band's stored values (so a CF-packed band is measured like any other),
and that the streamed row-strip reduction stitches a correct box across strips.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _raster(values: np.ndarray, no_data_value: float | None = None) -> Dataset:
    """A raster carrying `values`, top-left at (0, rows), one-unit square cells.

    Args:
        values: The band contents, 2-D for one band or `(bands, rows, cols)`.
        no_data_value: The sentinel to declare, or `None` for a band with no gaps.

    Returns:
        Dataset: An in-memory raster whose north edge sits at `y = rows`, so a
            row index maps to a whole `y` coordinate.
    """
    rows = values.shape[-2]
    return Dataset.from_array(
        values,
        no_data_value=no_data_value,
        geo_ref=GeoReference(top_left_corner=(0.0, float(rows)), cell_size=1.0, epsg=4326),
    )


class TestTheExtentShrinksToTheData:
    """A sub-region of data gives a box smaller than the whole grid."""

    def test_a_central_block_gives_a_tight_box(self):
        """Data in a 2x2 block has an extent of exactly that block.

        Test scenario:
            A 4x4 grid holds data only in rows 1-2, columns 1-2; every other cell
            is the sentinel. The extent is the map box of that block, not the grid.
        """
        values = np.full((4, 4), -9999.0, dtype="float32")
        values[1:3, 1:3] = 5.0
        ds = _raster(values, no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(), np.array([1.0, 1.0, 3.0, 3.0]))

    def test_the_extent_sits_inside_total_bounds(self):
        """A gappy band's extent is contained by, and smaller than, the grid box."""
        values = np.full((4, 4), -9999.0, dtype="float32")
        values[1:3, 1:3] = 5.0
        ds = _raster(values, no_data_value=-9999.0)
        extent, grid = ds.data_extent(), ds.total_bounds
        assert extent[0] >= grid[0] and extent[1] >= grid[1]
        assert extent[2] <= grid[2] and extent[3] <= grid[3]
        assert not np.array_equal(extent, grid)


class TestTheWholeGridCases:
    """The extremes: no gaps at all, and nothing but gaps."""

    def test_a_band_with_no_sentinel_has_the_grid_as_its_extent(self):
        """With no sentinel declared there are no gaps, so the extent is the grid."""
        ds = _raster(np.ones((4, 4), dtype="float32"))
        assert np.array_equal(ds.data_extent(), ds.total_bounds)

    def test_a_full_band_equals_total_bounds(self):
        """A sentinel that marks nothing leaves the extent equal to the grid box."""
        ds = _raster(np.ones((4, 4), dtype="float32"), no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(), ds.total_bounds)

    def test_an_all_sentinel_band_has_no_extent(self):
        """Every cell is the sentinel, so there is no data and the result is None."""
        ds = _raster(np.full((4, 4), -9999.0, dtype="float32"), no_data_value=-9999.0)
        assert ds.data_extent() is None

    def test_a_no_sentinel_float_band_still_shrinks_around_nan(self):
        """Without a sentinel, NaN cells are still gaps, so a NaN border shrinks the box."""
        values = np.ones((4, 4), dtype="float32")
        values[0, :] = np.nan
        values[:, 0] = np.nan
        ds = _raster(values, no_data_value=None)
        assert np.array_equal(ds.data_extent(), np.array([1.0, 0.0, 4.0, 3.0]))
        assert not np.array_equal(ds.data_extent(), ds.total_bounds)

    def test_nan_with_a_concrete_sentinel_is_counted_as_data(self):
        """With a concrete sentinel declared, a NaN cell is data (not a gap), so it widens the box."""
        values = np.full((4, 4), -9999.0, dtype="float32")
        values[0, 0] = 5.0
        values[3, 3] = np.nan
        ds = _raster(values, no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(), np.array([0.0, 0.0, 4.0, 4.0]))


class TestTheMaskIsReadFromStoredValues:
    """A CF-packed band is measured on its stored counts, not the physical values."""

    def test_a_packed_bands_gap_is_found_in_the_stored_sentinel(self):
        """Only the one stored, unpacked data cell contributes to the extent.

        Test scenario:
            A 3x3 int16 band stores the sentinel everywhere but its centre, with a
            `scale_factor` of 0.01. A physical read shows every gap as `-99.99`,
            which does not equal the `-9999` sentinel -- so measuring the physical
            values would count every cell as data and return the whole grid. The
            extent must instead be the single centre cell, proving it reads the
            stored counts.
        """
        stored = np.full((3, 3), -9999, dtype="int16")
        stored[1, 1] = 500
        ds = _raster(stored, no_data_value=-9999)
        ds.scale = [0.01]
        assert np.array_equal(ds.data_extent(), np.array([1.0, 1.0, 2.0, 2.0]))


class TestTheStreamedReductionStitchesStrips:
    """The box is correct when valid cells fall in different row strips."""

    def test_valid_cells_in_separate_strips_are_merged(self):
        """Data in row 10 and row 500 of a 600-row band gives one box spanning both.

        Test scenario:
            `stream_reduce` reads 256-row strips, so row 10 and row 500 arrive in
            different strips. The per-strip boxes must be reduced to a single box
            covering both rows and both columns, proving the cross-strip merge.
        """
        values = np.full((600, 4), -9999.0, dtype="float32")
        values[10, 2] = 1.0
        values[500, 1] = 2.0
        ds = _raster(values, no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(), np.array([1.0, 99.0, 3.0, 590.0]))


class TestBandSelectionAndTheFacade:
    """Each band has its own extent, and the facade matches the engine."""

    def test_each_band_reports_its_own_data(self):
        """Two bands with data in opposite corners report different extents."""
        values = np.full((2, 4, 4), -9999.0, dtype="float32")
        values[0, 0, 0] = 1.0
        values[1, 3, 3] = 2.0
        ds = _raster(values, no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(band=0), np.array([0.0, 3.0, 1.0, 4.0]))
        assert np.array_equal(ds.data_extent(band=1), np.array([3.0, 0.0, 4.0, 1.0]))

    def test_the_facade_delegates_to_the_engine(self):
        """`ds.data_extent()` returns what `ds.analysis.data_extent()` returns."""
        values = np.full((4, 4), -9999.0, dtype="float32")
        values[1:3, 1:3] = 5.0
        ds = _raster(values, no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(), ds.analysis.data_extent())

    def test_a_band_index_the_dataset_lacks_is_refused(self):
        """An out-of-range band raises `ValueError`, naming neither tuple nor index."""
        ds = _raster(np.ones((4, 4), dtype="float32"), no_data_value=-9999.0)
        with pytest.raises(ValueError, match="out of range"):
            ds.data_extent(band=3)


class TestOrientationHandling:
    """The pixel-window projection holds on south-up, non-square and rotated grids."""

    SOUTH_UP = GeoReference(geo=(100.0, 2.0, 0.0, 50.0, 0.0, 3.0), epsg=3857)
    ROTATED = GeoReference(geo=(0.0, 1.0, 0.2, 0.0, 0.1, -1.0), epsg=3857)

    def test_a_gap_free_south_up_non_square_band_equals_total_bounds(self):
        """A gap-free band on a south-up, non-square grid has the whole-grid box."""
        ds = Dataset.from_array(np.ones((4, 5), dtype="float32"), geo_ref=self.SOUTH_UP)
        assert np.array_equal(ds.data_extent(), ds.total_bounds)
        assert np.array_equal(ds.data_extent(), np.array([100.0, 50.0, 110.0, 62.0]))

    def test_a_gappy_south_up_non_square_band_has_the_shrunk_box(self):
        """A rows-1-2, cols-1-3 block projects to its tight box on the south-up grid."""
        values = np.full((4, 5), -9999.0, dtype="float32")
        values[1:3, 1:4] = 1.0
        ds = Dataset.from_array(values, geo_ref=self.SOUTH_UP, no_data_value=-9999.0)
        assert np.array_equal(ds.data_extent(), np.array([102.0, 53.0, 108.0, 59.0]))

    def test_a_gap_free_rotated_band_equals_total_bounds(self):
        """A gap-free band on a rotated grid still reduces to the whole-grid box."""
        ds = Dataset.from_array(np.ones((4, 4), dtype="float32"), geo_ref=self.ROTATED)
        assert np.array_equal(ds.data_extent(), ds.total_bounds)

    def test_a_rotated_grid_projects_the_valid_window_with_offsets(self):
        """A single valid cell on a rotated grid projects its window with col/row offsets."""
        values = np.full((4, 4), -9999.0, dtype="float32")
        values[1, 2] = 5.0
        ds = Dataset.from_array(values, geo_ref=self.ROTATED, no_data_value=-9999.0)
        np.testing.assert_allclose(ds.data_extent(), [2.2, -1.8, 3.4, -0.7])
