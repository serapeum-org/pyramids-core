"""The ``mesh.vector`` accessor owns the vector hand-off family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the vectorisation family moves onto a
dedicated ``mesh.vector`` engine, mirroring the raster ``Dataset`` vector surface, and the flat
``mesh.to_geodataframe`` / ``mesh.to_feature_collection`` / ``mesh.contour`` / ``mesh.cluster`` /
``mesh.voronoi`` / ``mesh.voronoi_polygons`` / ``mesh.polygonize`` methods become deprecating
facades that delegate to it, so both call forms return identical results.
"""

import warnings

import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.vector import _MeshVector

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh() -> MeshDataset:
    """A two-triangle mesh carrying a face-located ``v`` variable."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([1.0, 2.0])},
        data_locations={"v": "face"},
    )


def test_accessor_is_a_vector_engine():
    """``mesh.vector`` is a fresh ``_MeshVector`` bound to the dataset."""
    mesh = _mesh()
    assert isinstance(mesh.vector, _MeshVector)


def test_to_geodataframe_accessor_matches_flat_facade():
    """``mesh.vector.to_geodataframe()`` equals the deprecated ``mesh.to_geodataframe()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_geodataframe"):
        flat = mesh.to_geodataframe("v")
    assert_geodataframe_equal(mesh.vector.to_geodataframe("v"), flat)


def test_to_feature_collection_accessor_matches_flat_facade():
    """``mesh.vector.to_feature_collection()`` equals the deprecated flat method."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_feature_collection"):
        flat = mesh.to_feature_collection("v")
    accessor = mesh.vector.to_feature_collection("v")
    assert isinstance(accessor, FeatureCollection)
    assert len(accessor) == len(flat)
    assert list(accessor.geometry) == list(flat.geometry)
    assert accessor.epsg == flat.epsg


def test_contour_accessor_matches_flat_facade():
    """``mesh.vector.contour()`` equals the deprecated ``mesh.contour()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.contour"):
        flat = mesh.contour(levels=2, variable_name="v")
    assert_geodataframe_equal(mesh.vector.contour(levels=2, variable_name="v"), flat)


def test_cluster_accessor_matches_flat_facade():
    """``mesh.vector.cluster()`` equals the deprecated ``mesh.cluster()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.cluster"):
        flat = mesh.cluster(0.5, 1.5)
    accessor = mesh.vector.cluster(0.5, 1.5)
    np.testing.assert_array_equal(
        np.asarray(accessor["v"].data), np.asarray(flat["v"].data)
    )


def test_voronoi_accessor_matches_flat_facade():
    """``mesh.vector.voronoi()`` equals the deprecated ``mesh.voronoi()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.voronoi"):
        flat = mesh.voronoi()
    accessor = mesh.vector.voronoi()
    np.testing.assert_array_equal(accessor.points, flat.points)
    np.testing.assert_array_equal(accessor.vertices, flat.vertices)


def test_voronoi_polygons_accessor_matches_flat_facade():
    """``mesh.vector.voronoi_polygons()`` equals the deprecated ``mesh.voronoi_polygons()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.voronoi_polygons"):
        flat = mesh.voronoi_polygons()
    accessor = mesh.vector.voronoi_polygons()
    assert isinstance(accessor, FeatureCollection)
    assert len(accessor) == len(flat)
    assert list(accessor.geometry) == list(flat.geometry)
    assert accessor.epsg == flat.epsg


def test_polygonize_accessor_matches_flat_facade():
    """``mesh.vector.polygonize()`` equals the deprecated ``mesh.polygonize()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.polygonize"):
        flat = mesh.polygonize("v")
    assert_geodataframe_equal(mesh.vector.polygonize("v"), flat)


def test_accessor_form_does_not_warn():
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    mesh = _mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mesh.vector.to_geodataframe("v")
        mesh.vector.to_feature_collection("v")
        mesh.vector.contour(levels=2, variable_name="v")
        mesh.vector.cluster(0.5, 1.5)
        mesh.vector.voronoi()
        mesh.vector.voronoi_polygons()
        mesh.vector.polygonize("v")
