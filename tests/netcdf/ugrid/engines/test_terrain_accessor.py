"""The ``mesh.terrain`` accessor owns slope/aspect/hillshade; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the DEM-derivative family moves onto a
dedicated ``mesh.terrain`` engine, mirroring ``Dataset.terrain``, and the flat
``mesh.slope`` / ``mesh.aspect`` / ``mesh.hillshade`` methods become deprecating facades that
delegate to it, so both call forms return identical results.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.terrain import _MeshTerrain

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _dem(node_z) -> MeshDataset:
    """A two-triangle mesh DEM carrying ``node_z`` as the elevation."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"z": np.asarray(node_z, dtype="float64")},
        data_locations={"z": "node"},
    )


def test_accessor_is_a_terrain_engine():
    """``mesh.terrain`` is a fresh ``_MeshTerrain`` bound to the dataset."""
    mesh = _dem(_NODE_X)
    assert isinstance(mesh.terrain, _MeshTerrain)


def test_slope_accessor_matches_flat_facade():
    """``mesh.terrain.slope()`` equals the deprecated ``mesh.slope()``."""
    mesh = _dem(_NODE_X)
    with pytest.warns(DeprecationWarning, match="mesh.slope"):
        flat = np.asarray(mesh.slope())
    np.testing.assert_array_equal(np.asarray(mesh.terrain.slope()), flat)


def test_aspect_accessor_matches_flat_facade():
    """``mesh.terrain.aspect()`` equals the deprecated ``mesh.aspect()``."""
    mesh = _dem(_NODE_X)
    with pytest.warns(DeprecationWarning, match="mesh.aspect"):
        flat = np.asarray(mesh.aspect())
    np.testing.assert_array_equal(np.asarray(mesh.terrain.aspect()), flat)


def test_hillshade_accessor_matches_flat_facade():
    """``mesh.terrain.hillshade()`` equals the deprecated ``mesh.hillshade()``."""
    mesh = _dem(_NODE_X)
    with pytest.warns(DeprecationWarning, match="mesh.hillshade"):
        flat = np.asarray(mesh.hillshade(azimuth=300.0, altitude=40.0))
    np.testing.assert_array_equal(
        np.asarray(mesh.terrain.hillshade(azimuth=300.0, altitude=40.0)), flat
    )


def test_slope_units_forwarded_through_accessor():
    """The ``units`` argument reaches the engine: degrees exceed radians on a tilted plane."""
    mesh = _dem(_NODE_X)
    deg = float(np.nanmax(np.asarray(mesh.terrain.slope(units="degrees"))))
    rad = float(np.nanmax(np.asarray(mesh.terrain.slope(units="radians"))))
    assert deg > rad > 0.0


def test_accessor_form_does_not_warn():
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    mesh = _dem(_NODE_X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mesh.terrain.slope()
        mesh.terrain.aspect()
        mesh.terrain.hillshade()
