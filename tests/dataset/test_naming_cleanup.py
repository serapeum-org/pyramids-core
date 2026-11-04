"""Renames from the naming-cleanup refactor: new names work, old names deprecate."""

import numpy as np
import pandas as pd
import pytest

from pyramids.dataset import Dataset, GeoReference

REF = GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326)


def _ds() -> Dataset:
    """A tiny 1-band raster covering x in [0, 3], y in [-2, 0]."""
    return Dataset.from_array(np.arange(6.0).reshape(1, 2, 3), geo_ref=REF)


class TestFillDomainRename:
    """`fill` is renamed to `fill_domain` (it writes the data cells, not the gaps)."""

    def test_fill_domain_returns_dataset(self):
        """`fill_domain` writes the domain cells and returns a Dataset."""
        out = _ds().fill_domain(5)
        assert isinstance(out, Dataset)

    def test_fill_is_deprecated_alias(self):
        """`fill` still works but warns and points at `fill_domain`."""
        with pytest.warns(DeprecationWarning, match="fill_domain"):
            _ds().fill(5)


class TestNearestCellRename:
    """`map_to_array_coordinates` -> `nearest_cell`; `array_to_map_coordinates` removed."""

    def test_nearest_cell_present(self):
        """The new name exists on Dataset."""
        assert hasattr(Dataset, "nearest_cell")

    def test_array_to_map_coordinates_removed(self):
        """The duplicate of `xy` is gone (hard removed, no alias)."""
        assert not hasattr(Dataset, "array_to_map_coordinates")

    def test_nearest_cell_resolves_a_point(self):
        """`nearest_cell` maps an (x, y) table to the nearest array index."""
        located = _ds().nearest_cell(pd.DataFrame({"x": [0.5], "y": [-0.5]}))
        assert np.asarray(located).tolist() == [[0, 0]]

    def test_map_to_array_coordinates_is_deprecated_alias(self):
        """`map_to_array_coordinates` still works but warns and points at `nearest_cell`."""
        ds = _ds()
        with pytest.warns(DeprecationWarning, match="nearest_cell"):
            ds.map_to_array_coordinates(pd.DataFrame({"x": [0.5], "y": [-0.5]}))


class TestMetadataRename:
    """The `meta_data` family collapses to one-word `metadata` names, old names deprecate."""

    @pytest.mark.parametrize(
        ("new_name", "old_name"),
        [
            ("metadata", "meta_data"),
            ("metadata_domains", "meta_data_domains"),
            ("get_metadata", "get_meta_data"),
            ("set_metadata", "set_meta_data"),
            ("band_metadata", "band_meta_data"),
        ],
    )
    def test_new_name_present_and_old_name_still_there_as_alias(
        self, new_name, old_name
    ):
        """Both names resolve on Dataset (new canonical + deprecated alias kept)."""
        assert hasattr(Dataset, new_name)
        assert hasattr(Dataset, old_name)

    def test_metadata_reads_the_default_domain(self):
        """`metadata` returns the default-domain mapping."""
        assert _ds().metadata == {}

    def test_get_metadata_matches_metadata_on_a_base_dataset(self):
        """On a base Dataset the default domain of `get_metadata` equals `metadata`."""
        ds = _ds()
        assert ds.get_metadata() == ds.metadata

    def test_band_metadata_is_one_mapping_per_band(self):
        """`band_metadata` returns one mapping per band."""
        assert _ds().band_metadata == [{}]

    def test_meta_data_getter_is_deprecated(self):
        """The old `meta_data` attribute warns and points at `metadata`."""
        with pytest.warns(DeprecationWarning, match="metadata"):
            _ = _ds().meta_data

    def test_get_meta_data_is_deprecated(self):
        """The old `get_meta_data` method warns and points at `get_metadata`."""
        with pytest.warns(DeprecationWarning, match="get_metadata"):
            _ds().get_meta_data()

    def test_band_meta_data_is_deprecated(self):
        """The old `band_meta_data` attribute warns and points at `band_metadata`."""
        with pytest.warns(DeprecationWarning, match="band_metadata"):
            _ = _ds().band_meta_data


class TestBboxCollapsedIntoTotalBounds:
    """`bbox` is deprecated in favour of the canonical `total_bounds` (same box)."""

    def test_total_bounds_is_the_canonical_array(self):
        """`total_bounds` returns the box `[minx, miny, maxx, maxy]` as an ndarray."""
        tb = _ds().total_bounds
        assert isinstance(tb, np.ndarray)
        assert tb.tolist() == [0.0, -2.0, 3.0, 0.0]

    def test_bbox_is_deprecated_but_returns_the_same_box(self):
        """`bbox` still works (as a list), warns, and equals `total_bounds`."""
        ds = _ds()
        with pytest.warns(DeprecationWarning, match="total_bounds"):
            box = ds.bbox
        assert isinstance(box, list)
        assert box == ds.total_bounds.tolist()
