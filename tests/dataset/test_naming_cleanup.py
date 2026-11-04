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
