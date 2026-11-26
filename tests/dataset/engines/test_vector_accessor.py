"""The raster-to-vector engine is ``ds.vector``; ``ds.vectorize`` is a deprecated alias.

Pins P2-3 of the engine-accessor migration: the ``vectorize`` engine is renamed to the
noun ``vector``. ``ds.vector`` is the canonical accessor, the flat facades delegate to it,
and ``ds.vectorize`` keeps working for the deprecation window while emitting a warning.
"""

import warnings

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.engines import _Vector


@pytest.fixture
def labelled() -> Dataset:
    """A tiny integer raster with two labelled blobs for polygonizing."""
    arr = np.array(
        [[1, 1, 0, 2], [1, 0, 0, 2], [0, 0, 2, 2], [0, 0, 0, 0]], dtype=np.int32
    )
    return Dataset.from_array(
        arr,
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        no_data_value=0,
    )


def test_vector_accessor_is_a_vector_engine(labelled: Dataset):
    assert isinstance(labelled.vector, _Vector)


def test_flat_to_polygons_matches_accessor(labelled: Dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        flat = labelled.to_polygons()
    accessor = labelled.vector.to_polygons()
    assert len(flat) == len(accessor)


def test_vectorize_alias_warns_and_returns_the_vector_engine(labelled: Dataset):
    with pytest.warns(DeprecationWarning, match="ds.vectorize.*use.*ds.vector"):
        alias = labelled.vectorize
    assert isinstance(alias, _Vector) and alias._ds is labelled
