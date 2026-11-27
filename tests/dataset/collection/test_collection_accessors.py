"""DatasetCollection exposes five accessor engines.

Pins P2-8 of the engine-accessor migration: a collection gets its own stack-aware engines
``dc.spatial`` / ``dc.io`` / ``dc.glyph`` / ``dc.analysis`` / ``dc.series`` (and no others).
``dc.series`` carries the whole time axis (reduction + selection).
"""

import numpy as np
import pytest

from pyramids.dataset.collection import (
    DatasetCollection,
    _ColAnalysis,
    _ColGlyph,
    _ColIO,
    _ColSeries,
    _ColSpatial,
)


@pytest.fixture
def cube(three_files) -> DatasetCollection:
    return DatasetCollection.from_files(three_files)


def test_five_accessors_have_the_right_engine_types(cube: DatasetCollection):
    assert isinstance(cube.spatial, _ColSpatial)
    assert isinstance(cube.io, _ColIO)
    assert isinstance(cube.glyph, _ColGlyph)
    assert isinstance(cube.analysis, _ColAnalysis)
    assert isinstance(cube.series, _ColSeries)


def test_only_the_five_engines_are_exposed(cube: DatasetCollection):
    for absent in ("bands", "cog", "georef", "cell", "vector", "terrain"):
        assert not hasattr(cube, absent), absent


@pytest.mark.parametrize(
    "engine, methods",
    [
        ("spatial", ("align", "crop", "to_crs")),
        ("io", ("to_file", "to_zarr", "to_cog_stack", "to_netcdf", "close")),
        ("glyph", ("plot", "animate")),
        ("analysis", ("apply", "merge")),
        (
            "series",
            ("mean", "std", "sum", "min", "max", "var", "first", "last", "groupby"),
        ),
    ],
)
def test_engine_methods_are_callable(cube, engine, methods):
    acc = getattr(cube, engine)
    for m in methods:
        assert callable(getattr(acc, m)), f"{engine}.{m}"


def test_series_first_matches_the_flat_method(cube: DatasetCollection):
    np.testing.assert_array_equal(
        np.asarray(cube.series.first().io.read_array()),
        np.asarray(cube.first().io.read_array()),
    )


def test_series_head_matches_the_flat_method(cube: DatasetCollection):
    np.testing.assert_array_equal(
        np.asarray(cube.series.head(2)), np.asarray(cube.head(2))
    )
