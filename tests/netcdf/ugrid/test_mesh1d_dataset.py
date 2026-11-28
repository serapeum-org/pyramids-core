"""A MeshDataset can wrap a 1D topology (Mesh1d) and carry its node/edge data over time.

Pins issue #1652: until now MeshDataset accepted only a Mesh2d (`mesh: Mesh2d`), so a 1D mesh
(Mesh1d: nodes + edges, no faces) had no data container -- a driver reading a 1D results file
could build the geometry but had nowhere to attach the values, which were parsed then dropped.
This widens MeshDataset to accept a Mesh1d and carry node-/edge-located data variables, static
and over a time axis, the same way the 2D path already does.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import Mesh1d, MeshDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0])
_NODE_Y = np.array([0.0, 0.0, 0.0])
_EDGES = np.array([[0, 1], [1, 2]])


def _ds() -> MeshDataset:
    """A 3-node / 2-edge 1D chain carrying a static node variable ``s1``."""
    return MeshDataset.from_mesh1d(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        edge_node_connectivity=_EDGES,
        data={"s1": np.array([1.0, 2.0, 3.0])},
        data_locations={"s1": "node"},
        epsg=4326,
    )


class TestMeshDatasetOnMesh1d:
    """MeshDataset built on a Mesh1d topology carries 1D data."""

    def test_the_topology_is_a_mesh1d(self):
        """``.mesh`` is the Mesh1d it was built from, not a Mesh2d."""
        ds = _ds()
        assert isinstance(ds.mesh, Mesh1d), f"expected Mesh1d, got {type(ds.mesh)}"

    def test_a_node_variable_round_trips(self):
        """A node-located variable keeps its values, name and location."""
        ds = _ds()
        assert ds.variable_names == ["s1"], ds.variable_names
        np.testing.assert_array_equal(np.asarray(ds["s1"].data), [1.0, 2.0, 3.0])
        assert ds["s1"].location == "node", ds["s1"].location

    def test_an_edge_variable_round_trips(self):
        """An edge-located variable (one value per edge) is accepted."""
        ds = MeshDataset.from_mesh1d(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            edge_node_connectivity=_EDGES,
            data={"q1": np.array([10.0, 20.0])},
            data_locations={"q1": "edge"},
            epsg=4326,
        )
        np.testing.assert_array_equal(np.asarray(ds["q1"].data), [10.0, 20.0])
        assert ds["q1"].location == "edge"

    def test_the_1d_topology_counts_are_exposed(self):
        """``n_node`` / ``n_edge`` read off the Mesh1d."""
        ds = _ds()
        assert ds.n_node == 3, ds.n_node
        assert ds.n_edge == 2, ds.n_edge

    def test_a_temporal_node_variable_keeps_its_time_axis(self):
        """A ``(n_time, n_node)`` variable is temporal, as on the 2D path."""
        ds = MeshDataset.from_mesh1d(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            edge_node_connectivity=_EDGES,
            data={"s1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            data_locations={"s1": "node"},
            epsg=4326,
        )
        assert ds["s1"].data.shape == (2, 3), ds["s1"].data.shape


class TestFaceOnlyOpsRefuseA1dMesh:
    """Operations defined over faces raise a clear error on a 1D mesh, not AttributeError."""

    def test_rasterising_to_a_grid_is_refused(self):
        """``io.to_dataset`` (rasterise) has no 1D meaning -> TypeError naming the reason."""
        ds = _ds()
        with pytest.raises(TypeError, match="not available on a 1D mesh"):
            ds.io.to_dataset("s1", cell_size=0.5)

    def test_n_face_is_refused(self):
        """A 1D mesh has no faces; asking is an error, not a silent wrong answer."""
        ds = _ds()
        with pytest.raises((TypeError, AttributeError)):
            _ = ds.n_face
