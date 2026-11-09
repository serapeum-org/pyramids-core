"""Raster-category zonal statistics, the ``raw_values`` column, and the ``overlay`` deprecation (#1496).

`zonal_stats` accepts a categorical raster as its zones (each distinct pixel value is a zone) and returns a
`GeoDataFrame` with one row per class, matching the polygon path. The deprecated `overlay` delegates to it.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.collection import DatasetCollection

_GEO = dict(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326)


def _values() -> Dataset:
    """A 3x3 float value raster."""
    arr = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]], dtype="float32"
    )
    return Dataset.from_array(arr, geo_ref=GeoReference(**_GEO))


def _classes() -> Dataset:
    """A 3x3 int class raster aligned with :func:`_values` (classes 1, 2, 3)."""
    arr = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype="int32")
    return Dataset.from_array(arr, geo_ref=GeoReference(**_GEO))


class TestRasterZones:
    """`zonal_stats` with a categorical raster as its zones."""

    def test_returns_geodataframe_indexed_by_class(self):
        """One row per distinct class, indexed by the class value, with geometry + CRS.

        Test scenario:
            A 3x3 value raster grouped by a 3-class raster returns a GeoDataFrame indexed
            by `[1, 2, 3]` (`index.name == "class"`), each row carrying a real geometry in
            the raster's CRS.
        """
        result = _values().zonal_stats(_classes(), stats=("mean", "count"))
        assert isinstance(result, gpd.GeoDataFrame), "must return a GeoDataFrame"
        assert list(result.index) == [1, 2, 3], (
            f"unexpected classes: {list(result.index)}"
        )
        assert result.index.name == "class", "index should be named 'class'"
        assert result.geometry.notna().all(), "each class must carry a geometry"
        assert result.crs is not None and result.crs.to_epsg() == 4326

    def test_stats_match_manual_reduction(self):
        """Per-class mean/count/min equal a hand-computed reduction.

        Test scenario:
            Class 1 = {10, 20, 40}, class 2 = {30, 50, 60}, class 3 = {70, 80, 90}.
        """
        result = _values().zonal_stats(_classes(), stats=("mean", "count", "min"))
        assert result.loc[1, "mean"] == pytest.approx(70 / 3), "class 1 mean"
        assert result.loc[2, "count"] == 3.0, "class 2 count"
        assert result.loc[3, "min"] == 70.0, "class 3 min"

    def test_unaligned_class_raster_raises(self):
        """A class raster on a different grid raises AlignmentError.

        Test scenario:
            A 2x2 class raster is not aligned with the 3x3 value raster.
        """
        other = Dataset.from_array(
            np.array([[1, 2], [2, 1]], dtype="int32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )
        with pytest.raises(AlignmentError):
            _values().zonal_stats(other, stats=("mean",))

    def test_cf_packed_values_are_physical(self):
        """Statistics use physical values on a CF-packed band.

        Test scenario:
            A packed int16 band with scale 0.01 over one class reports the physical mean.
        """
        packed = Dataset.from_array(
            np.array([[100, 200], [300, 400]], dtype="int16"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )
        packed.scale = [0.01]
        classes = Dataset.from_array(
            np.array([[1, 1], [1, 1]], dtype="int32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )
        result = packed.zonal_stats(classes, stats=("mean",))
        assert result.loc[1, "mean"] == pytest.approx(2.5), "physical mean of 1,2,3,4"


class TestRawValues:
    """The opt-in `raw_values` column holding each zone's extracted values."""

    def test_raw_values_off_by_default(self):
        """Without `raw_values=True` there is no `raw_values` column."""
        result = _values().zonal_stats(_classes(), stats=("mean",))
        assert "raw_values" not in result.columns

    def test_raw_values_are_row_major_ndarrays(self):
        """Each zone's `raw_values` is an ndarray of its values in row-major order.

        Test scenario:
            Class 1 covers (r0c0, r0c1, r1c0) = 10, 20, 40 in that row-major order.
        """
        result = _values().zonal_stats(_classes(), stats=("count",), raw_values=True)
        class_one = result.loc[1, "raw_values"]
        assert isinstance(class_one, np.ndarray), "raw_values cell must be an ndarray"
        np.testing.assert_array_equal(
            class_one, np.array([10.0, 20.0, 40.0]), err_msg="row-major order broken"
        )

    def test_exclude_value_drops_from_raw_and_stats(self):
        """`exclude_value` removes matching cells from both stats and raw values.

        Test scenario:
            Excluding 20.0 drops it from class 1, leaving {10, 40}.
        """
        result = _values().zonal_stats(
            _classes(), stats=("count",), exclude_value=20.0, raw_values=True
        )
        np.testing.assert_array_equal(
            result.loc[1, "raw_values"], np.array([10.0, 40.0])
        )
        assert result.loc[1, "count"] == 2.0, "excluded value must not count"


class TestOverlayDeprecation:
    """`overlay` is a deprecated alias returning the old `dict[class, list]`."""

    def test_dataset_overlay_warns_and_matches_zonal_stats(self):
        """`Dataset.overlay` warns and equals the dict rebuilt from `zonal_stats`."""
        with pytest.warns(DeprecationWarning, match="zonal_stats"):
            legacy = _values().overlay(_classes())
        gdf = _values().zonal_stats(_classes(), stats=("count",), raw_values=True)
        rebuilt = {
            int(key): [float(value) for value in vals]
            for key, vals in zip(gdf.index, gdf["raw_values"])
            if len(vals)
        }
        normalised = {
            int(key): [float(v) for v in vals] for key, vals in legacy.items()
        }
        assert normalised == rebuilt, (
            "overlay dict diverged from zonal_stats raw_values"
        )

    def test_analysis_engine_overlay_warns(self):
        """The engine-level `Analysis.overlay` warns too."""
        ds = _values()
        with pytest.warns(DeprecationWarning, match="zonal_stats"):
            ds.analysis.overlay(_classes())

    def test_collection_overlay_warns_and_merges(self):
        """`DatasetCollection.overlay` warns and merges per-class values across datasets."""
        cube = DatasetCollection.from_dataset(_values(), time_length=1)
        with pytest.warns(DeprecationWarning, match="zonal_stats"):
            merged = cube.overlay(_classes())
        assert sorted(int(key) for key in merged) == [1, 2, 3]
