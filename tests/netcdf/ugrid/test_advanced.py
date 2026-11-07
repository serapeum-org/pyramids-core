"""Unit tests for UGRID advanced operations.

Covers UGRID-18 (CRS/reprojection), UGRID-19 (time dimension),
UGRID-20 (edge/node data support).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import MeshDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


class TestCrsHandling:
    """Tests for CRS handling and reprojection (UGRID-18)."""

    def test_epsg_from_create(self):
        """Test EPSG is set when creating from arrays.

        Test scenario:
            EPSG 32631 should be recoverable from the dataset.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        assert ds.epsg == 32631, f"Expected EPSG 32631, got {ds.epsg}"

    def test_to_crs_reprojection(self):
        """Test reprojecting from UTM to WGS84.

        Test scenario:
            Reproject from EPSG:32631 (UTM 31N) to EPSG:4326 (WGS84).
            Coordinates should change to lon/lat range.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert reprojected.epsg == 4326, f"Expected EPSG 4326, got {reprojected.epsg}"
        assert reprojected.mesh.node_x[0] < 180, (
            f"Expected longitude < 180, got {reprojected.mesh.node_x[0]}"
        )
        assert reprojected.mesh.node_y[0] < 90, (
            f"Expected latitude < 90, got {reprojected.mesh.node_y[0]}"
        )

    def test_to_crs_preserves_data(self):
        """Test that reprojection preserves data values.

        Test scenario:
            Data values should be identical after coordinate reprojection.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"temp": np.array([25.0])},
            data_locations={"temp": "face"},
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert reprojected["temp"].data[0] == pytest.approx(25.0), (
            f"Data should be preserved, got {reprojected['temp'].data[0]}"
        )

    def test_to_crs_preserves_topology(self):
        """Test that reprojection preserves mesh topology.

        Test scenario:
            n_node, n_face should be unchanged after reprojection.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert reprojected.n_node == ds.n_node, "n_node should be preserved"
        assert reprojected.n_face == ds.n_face, "n_face should be preserved"

    def test_to_crs_no_source_crs_raises(self):
        """Test that to_crs raises if source CRS is unknown.

        Test scenario:
            Dataset without CRS should raise ValueError.
        """

        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        ds = MeshDataset(
            mesh=mesh,
            data_variables={},
            global_attributes={},
        )
        with pytest.raises(ValueError, match="source CRS is unknown"):
            ds.to_crs(4326)


class TestTimeDimension:
    """Tests for time dimension support (UGRID-19)."""

    @pytest.fixture
    def temporal_dataset(self):
        """Create a MeshDataset with temporal data.

        Returns:
            MeshDataset with 3 time steps, 2 faces.
        """
        data_2d = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        return MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 3, 2]]),
            data={"water_level": data_2d},
            data_locations={"water_level": "face"},
        )

    def test_time_values_default(self, temporal_dataset):
        """Test time_values returns range indices by default.

        Test scenario:
            Without explicit time coordinates, should return [0, 1, 2].
        """
        tv = temporal_dataset.time_values
        assert tv == [0, 1, 2], f"Expected [0, 1, 2], got {tv}"

    def test_sel_time(self, temporal_dataset):
        """Test selecting a single time step.

        Test scenario:
            sel_time(1) should return data from time step 1.
        """
        ds_t1 = temporal_dataset.sel_time(1)
        var = ds_t1["water_level"]
        assert not var.has_time, "Selected time step should not have time dimension"
        np.testing.assert_array_equal(var.data, [3.0, 4.0])

    def test_sel_time_non_temporal_raises(self):
        """sel_time on a mesh with no temporal variable raises, like isel.

        Test scenario:
            A static (non-temporal) mesh has no time axis to index; sel_time must refuse it
            with a clear ValueError rather than silently returning the mesh unchanged.
        """
        static = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"depth": np.array([2.0, 4.0])},
            data_locations={"depth": "face"},
        )
        with pytest.raises(ValueError, match="time dimension"):
            static.sel_time(0)

    def test_sel_time_strips_time_metadata(self, temporal_dataset):
        """A collapsed step drops the stale time coordinate, units and calendar.

        Test scenario:
            The time axis is gone after sel_time, so its coordinate values / CF units /
            calendar describe nothing and must not ride along (they would mislead the
            on-disk coordinate writer).
        """
        var = temporal_dataset["water_level"]
        var.attributes = {
            **var.attributes,
            "time_values": [0.0, 1.0, 2.0],
            "time_units": "days since 2001-01-01",
            "calendar": "standard",
        }
        collapsed = temporal_dataset.sel_time(1)["water_level"]
        assert "time_values" not in collapsed.attributes, "stale time_values carried over"
        assert "time_units" not in collapsed.attributes, "stale time_units carried over"
        assert "calendar" not in collapsed.attributes, "stale calendar carried over"

    def test_sel_time_trailing_time_axis(self):
        """sel_time collapses the real time axis, not axis 0, for trailing-time data.

        Test scenario:
            A variable dimensioned ``(face, time)`` keeps time last. sel_time(1) must return
            every face at step 1 (the column ``data[:, 1]``), not row 1 (``data[1]``).
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
            face_x=np.array([0.5, 0.5]),
            face_y=np.array([0.3, 0.6]),
        )
        var = MeshVariable(
            name="wl",
            location="face",
            mesh_name="mesh2d",
            shape=(2, 3),
            dimensions=("mesh2d_nFaces", "time"),
            _data=np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
        )
        collapsed = MeshDataset(mesh, {"wl": var}, {}).sel_time(1)["wl"]
        assert collapsed.shape == (2,), f"expected one value per face, got {collapsed.shape}"
        np.testing.assert_array_equal(collapsed.data, [11.0, 21.0])

    def test_sel_time_keeps_static_variables(self):
        """sel_time collapses temporal variables and carries static ones through unchanged.

        Test scenario:
            A mesh with one temporal (3 steps) and one static variable: the temporal one
            becomes a single static step, the static one is returned as-is.
        """
        mixed = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={
                "wl": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "bed": np.array([-9.0, -8.0]),
            },
            data_locations={"wl": "face", "bed": "face"},
        )
        picked = mixed.sel_time(1)
        assert not picked["wl"].has_time, "temporal variable should be collapsed"
        np.testing.assert_array_equal(picked["wl"].data, [3.0, 4.0])
        np.testing.assert_array_equal(picked["bed"].data, [-9.0, -8.0])

    def test_sel_time_trailing_time_without_data_raises(self):
        """A trailing-time variable with no loaded data raises instead of guessing.

        Test scenario:
            The non-leading branch must materialise the array to take the step; with neither
            eager data nor a loader there is nothing to select, so it raises ValueError.
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
            face_x=np.array([0.5, 0.5]),
            face_y=np.array([0.3, 0.6]),
        )
        var = MeshVariable(
            name="wl", location="face", mesh_name="mesh2d", shape=(2, 3),
            dimensions=("mesh2d_nFaces", "time"),
        )
        with pytest.raises(ValueError, match="no loaded data"):
            MeshDataset(mesh, {"wl": var}, {}).sel_time(1)

    def test_sel_time_mismatched_time_lengths_raises(self):
        """sel_time refuses temporal variables that disagree on their time length.

        Test scenario:
            Two temporal variables with 3 and 2 steps cannot share one index set, so sel_time
            raises rather than taking a step that is in range for one and out of range for the
            other.
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
            face_x=np.array([0.5, 0.5]),
            face_y=np.array([0.3, 0.6]),
        )
        long_var = MeshVariable(
            name="a", location="face", mesh_name="mesh2d", shape=(3, 2),
            dimensions=("time", "mesh2d_nFaces"), _data=np.zeros((3, 2)),
        )
        short_var = MeshVariable(
            name="b", location="face", mesh_name="mesh2d", shape=(2, 2),
            dimensions=("time", "mesh2d_nFaces"), _data=np.zeros((2, 2)),
        )
        ds = MeshDataset(mesh, {"a": long_var, "b": short_var}, {})
        with pytest.raises(ValueError, match="time length"):
            ds.sel_time(0)

    def test_sel_time_range(self, temporal_dataset):
        """Test selecting a time range.

        Test scenario:
            sel_time_range(0, 2) should return first 2 time steps.
        """
        ds_range = temporal_dataset.sel_time_range(0, 2)
        var = ds_range["water_level"]
        assert var.has_time, "Should still have time dimension"
        assert var.n_time_steps == 2, f"Expected 2 time steps, got {var.n_time_steps}"

    def test_no_time_variables(self):
        """Test time_values returns None when no temporal data.

        Test scenario:
            Dataset with only 1D data should have time_values=None.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"elev": np.array([5.0])},
            data_locations={"elev": "face"},
        )
        assert ds.time_values is None, f"Expected None, got {ds.time_values}"


class TestEdgeNodeData:
    """Tests for edge and node data support (UGRID-20)."""

    def test_node_data_access(self, ugrid_convention_nc_path):
        """Test accessing node-centered data.

        Test scenario:
            mesh2d_node_z should be accessible and have node location.
        """
        ds = MeshDataset.read_file(ugrid_convention_nc_path)
        var = ds["mesh2d_node_z"]
        assert var.location == "node", f"Expected 'node', got '{var.location}'"
        assert var.n_elements == ds.n_node, (
            f"Node data elements should match n_node: {var.n_elements} vs {ds.n_node}"
        )

    def test_edge_data_access(self, ugrid_convention_nc_path):
        """Test accessing edge-centered data.

        Test scenario:
            mesh2d_edge_type should be accessible and have edge location.
        """
        ds = MeshDataset.read_file(ugrid_convention_nc_path)
        var = ds["mesh2d_edge_type"]
        assert var.location == "edge", f"Expected 'edge', got '{var.location}'"
        assert var.n_elements == ds.n_edge, (
            f"Edge data elements should match n_edge: {var.n_elements} vs {ds.n_edge}"
        )

    def test_node_data_to_geodataframe(self):
        """Test converting node data to GeoDataFrame with Points.

        Test scenario:
            Node data on a 3-node mesh should produce 3 Point geometries.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"altitude": np.array([0.0, 5.0, 10.0])},
            data_locations={"altitude": "node"},
        )
        gdf = ds.to_geodataframe("altitude", location="node")
        assert len(gdf) == 3, f"Expected 3 rows, got {len(gdf)}"
        assert gdf.geometry.iloc[0].geom_type == "Point", "Expected Point geometry"
        assert "altitude" in gdf.columns, "Should have altitude column"

    def test_to_dataset_node_data(self):
        """Test to_dataset with node-centered data.

        Test scenario:
            Should interpolate node data to a regular grid.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"altitude": np.array([0.0, 5.0, 10.0])},
            data_locations={"altitude": "node"},
        )
        raster = ds.to_dataset("altitude", cell_size=0.25)
        assert raster.rows > 0, f"Expected positive rows, got {raster.rows}"
        assert raster.columns > 0, f"Expected positive cols, got {raster.columns}"
