"""The ``mesh.topology`` accessor owns the unstructured-grid ops; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the mesh-native topology family -- the
operations with no raster analogue -- moves onto a dedicated ``mesh.topology`` engine, and the flat
``mesh.circumcenters`` / ``mesh.exterior_edges`` / ``mesh.exterior_faces`` / ``mesh.bounding_polygon`` /
``mesh.connected_components`` / ``mesh.binary_dilation`` / ``mesh.binary_erosion`` /
``mesh.sparse_adjacency`` / ``mesh.active_mask`` methods become deprecating facades that delegate to
it, so both call forms return identical results.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.topology import _MeshTopology

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh() -> MeshDataset:
    """Two adjacent triangles carrying a face-located ``v`` variable."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([1.0, 2.0])},
        data_locations={"v": "face"},
    )


def _mask_mesh(values) -> MeshDataset:
    """The two-triangle mesh carrying a boolean face mask ``m``."""
    return _mesh().variables.with_variable("m", np.asarray(values), location="face")


def test_accessor_is_a_topology_engine():
    """``mesh.topology`` is a fresh ``_MeshTopology`` bound to the dataset."""
    mesh = _mesh()
    assert isinstance(mesh.topology, _MeshTopology)


def test_circumcenters_accessor_matches_flat_facade():
    """``mesh.topology.circumcenters()`` equals the deprecated ``mesh.circumcenters()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.circumcenters"):
        flat = mesh.circumcenters()
    np.testing.assert_array_equal(mesh.topology.circumcenters(), flat)


def test_exterior_edges_accessor_matches_flat_facade():
    """``mesh.topology.exterior_edges()`` equals the deprecated ``mesh.exterior_edges()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.exterior_edges"):
        flat = mesh.exterior_edges()
    np.testing.assert_array_equal(mesh.topology.exterior_edges(), flat)


def test_exterior_faces_accessor_matches_flat_facade():
    """``mesh.topology.exterior_faces()`` equals the deprecated ``mesh.exterior_faces()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.exterior_faces"):
        flat = mesh.exterior_faces()
    np.testing.assert_array_equal(mesh.topology.exterior_faces(), flat)


def test_bounding_polygon_accessor_matches_flat_facade():
    """``mesh.topology.bounding_polygon()`` equals the deprecated ``mesh.bounding_polygon()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.bounding_polygon"):
        flat = mesh.bounding_polygon()
    assert mesh.topology.bounding_polygon().equals(flat)


def test_connected_components_accessor_matches_flat_facade():
    """``mesh.topology.connected_components()`` equals the deprecated flat method."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.connected_components"):
        flat = mesh.connected_components()
    np.testing.assert_array_equal(mesh.topology.connected_components(), flat)


def test_binary_dilation_accessor_matches_flat_facade():
    """``mesh.topology.binary_dilation()`` equals the deprecated ``mesh.binary_dilation()``."""
    mesh = _mask_mesh([True, False])
    with pytest.warns(DeprecationWarning, match="mesh.binary_dilation"):
        flat = mesh.binary_dilation("m")
    accessor = mesh.topology.binary_dilation("m")
    np.testing.assert_array_equal(
        np.asarray(accessor["m"].data), np.asarray(flat["m"].data)
    )


def test_binary_erosion_accessor_matches_flat_facade():
    """``mesh.topology.binary_erosion()`` equals the deprecated ``mesh.binary_erosion()``."""
    mesh = _mask_mesh([True, True])
    with pytest.warns(DeprecationWarning, match="mesh.binary_erosion"):
        flat = mesh.binary_erosion("m")
    accessor = mesh.topology.binary_erosion("m")
    np.testing.assert_array_equal(
        np.asarray(accessor["m"].data), np.asarray(flat["m"].data)
    )


def test_sparse_adjacency_accessor_matches_flat_facade():
    """``mesh.topology.sparse_adjacency()`` equals the deprecated ``mesh.sparse_adjacency()``."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.sparse_adjacency"):
        flat = mesh.sparse_adjacency("face")
    np.testing.assert_array_equal(
        mesh.topology.sparse_adjacency("face").toarray(), flat.toarray()
    )


def test_active_mask_accessor_matches_flat_facade():
    """``mesh.topology.active_mask()`` equals the deprecated ``mesh.active_mask()``."""
    mesh = _mesh()
    mesh["v"].active = np.array([True, False])
    with pytest.warns(DeprecationWarning, match="mesh.active_mask"):
        flat = mesh.active_mask("v")
    np.testing.assert_array_equal(mesh.topology.active_mask("v"), flat)


def test_accessor_form_does_not_warn():
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    mesh = _mesh()
    mesh["v"].active = np.array([True, False])
    mask_mesh = _mask_mesh([True, False])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mesh.topology.circumcenters()
        mesh.topology.exterior_edges()
        mesh.topology.exterior_faces()
        mesh.topology.bounding_polygon()
        mesh.topology.connected_components()
        mesh.topology.sparse_adjacency("face")
        mesh.topology.active_mask("v")
        mask_mesh.topology.binary_dilation("m")
        mask_mesh.topology.binary_erosion("m")
