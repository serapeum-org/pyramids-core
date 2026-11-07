"""Unit tests for UGRID write, GeoDataFrame interop, and from_arrays.

Covers UGRID-15 (write), UGRID-16 (GeoDataFrame), UGRID-17 (create from arrays).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid.dataset import MeshDataset
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable

pytestmark = pytest.mark.core


class TestWriteUgrid:
    """Tests for MeshDataset.to_file() (UGRID-15)."""

    @staticmethod
    def _temporal_ramp():
        """A two-face mesh carrying a 3-step linear ramp (slope 10 / 20 per face)."""
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 1.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 2, 3]]),
            data={"v": np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])},
            data_locations={"v": "face"},
        )
        ds["v"].attributes = {**ds["v"].attributes, "time_values": [0.0, 1.0, 2.0]}
        return ds

    def test_to_file_round_trips_a_polyfit_coefficient_dim(self, tmp_path):
        """A polyfit result writes and reads back its non-time `coefficient` axis.

        Test scenario:
            `polyfit` replaces the time axis with a `coefficient` axis, giving a
            `(deg + 1, n_face)` variable whose leading axis is neither time nor the element
            axis. The writer assembled only a time-or-nothing + element dimension list, so it
            built a rank-1 array for a rank-2 variable and GDAL raised
            `RuntimeError: Illegal numpy array rank 2`. The result must now write and read
            back with the coefficient axis and values intact.
        """
        fit = self._temporal_ramp().polyfit(1)
        assert fit["v"].dimensions == ("coefficient", "face")
        out = tmp_path / "polyfit.nc"

        fit.to_file(out)

        back = MeshDataset.read_file(out)
        read = back["v"]
        assert np.asarray(read.data).shape == (2, 2), (
            "the coefficient axis must survive"
        )
        assert read.dimensions[0] == "coefficient", f"got {read.dimensions}"
        np.testing.assert_allclose(np.asarray(read.data), np.asarray(fit["v"].data))

    def test_to_file_round_trips_a_curvefit_parameter_dim(self, tmp_path):
        """The sibling case: `curvefit`'s non-time `parameter` axis also writes."""
        fit = self._temporal_ramp().curvefit(lambda x, a, b: a * x + b)
        assert fit["v"].dimensions == ("parameter", "face")
        out = tmp_path / "curvefit.nc"

        fit.to_file(out)

        read = MeshDataset.read_file(out)["v"]
        assert np.asarray(read.data).shape == (2, 2)
        assert read.dimensions[0] == "parameter", f"got {read.dimensions}"

    def test_write_and_read_round_trip(self, tmp_path):
        """Test write then read produces consistent data.

        Test scenario:
            Create a simple mesh, write to file, read back,
            verify topology and data match.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"elevation": np.array([5.0])},
            data_locations={"elevation": "face"},
            epsg=4326,
        )
        out_path = tmp_path / "test_ugrid.nc"
        ds.to_file(out_path)
        assert out_path.exists(), "Output file should exist"

        ds2 = MeshDataset.read_file(out_path)
        assert ds2.n_node == 3, f"Expected 3 nodes, got {ds2.n_node}"
        assert ds2.n_face == 1, f"Expected 1 face, got {ds2.n_face}"
        assert "elevation" in ds2.data_variable_names, (
            f"Expected 'elevation', got {ds2.data_variable_names}"
        )
        np.testing.assert_array_almost_equal(
            ds2["elevation"].data,
            [5.0],
            err_msg="Elevation data should survive round-trip",
        )
        np.testing.assert_array_almost_equal(
            ds2.mesh.node_x,
            [0.0, 1.0, 0.5],
            err_msg="Node x-coordinates should survive round-trip",
        )

    def test_write_mixed_mesh(self, tmp_path, mixed_mesh):
        """Test writing a mixed mesh (quad + triangles).

        Test scenario:
            Write and read back a mixed mesh to verify connectivity
            round-trip with fill values.
        """
        data_vars = {
            "temp": MeshVariable(
                name="temp",
                location="face",
                mesh_name="mesh2d",
                shape=(3,),
                _data=np.array([1.0, 2.0, 3.0]),
                units="K",
            ),
        }
        topo = MeshTopologyInfo(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="mesh2d_node_x",
            node_y_var="mesh2d_node_y",
            face_node_var="mesh2d_face_nodes",
            data_variables={"temp": "face"},
        )
        ds = MeshDataset(
            mesh=mixed_mesh,
            data_variables=data_vars,
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            topology_info=topo,
        )
        out_path = tmp_path / "mixed.nc"
        ds.to_file(out_path)

        ds2 = MeshDataset.read_file(out_path)
        assert ds2.n_face == 3, f"Expected 3 faces, got {ds2.n_face}"
        assert ds2.n_node == 6, f"Expected 6 nodes, got {ds2.n_node}"
        fnc = ds2.mesh.face_node_connectivity
        assert fnc.nodes_per_element()[0] == 4, "First face (quad) should have 4 nodes"
        assert fnc.nodes_per_element()[1] == 3, "Second face (tri) should have 3 nodes"

    def test_ugrid_convention_nc_round_trip(self, ugrid_convention_nc_path, tmp_path):
        """Test read-write-read round trip on real data.

        Test scenario:
            Read UGRID convention NC, write to new file, read back,
            verify node/face counts match.
        """
        ds1 = MeshDataset.read_file(ugrid_convention_nc_path)
        out_path = tmp_path / "ws_roundtrip.nc"
        ds1.to_file(out_path)

        ds2 = MeshDataset.read_file(out_path)
        assert ds2.n_node == ds1.n_node, (
            f"Node count mismatch: {ds2.n_node} vs {ds1.n_node}"
        )
        assert ds2.n_face == ds1.n_face, (
            f"Face count mismatch: {ds2.n_face} vs {ds1.n_face}"
        )
        np.testing.assert_array_almost_equal(
            ds2["mesh2d_node_z"].data[:10],
            ds1["mesh2d_node_z"].data[:10],
            err_msg="Node z data should survive round-trip",
        )
        np.testing.assert_array_almost_equal(
            ds2.mesh.node_x[:10],
            ds1.mesh.node_x[:10],
            err_msg="Node x-coordinates should survive round-trip",
        )


class TestToGeoDataFrame:
    """Tests for MeshDataset.to_geodataframe() (UGRID-16)."""

    def test_face_geodataframe(self):
        """Test converting faces to GeoDataFrame.

        Test scenario:
            Each face should become a Polygon row.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2, 3]]),
            data={"temp": np.array([25.0])},
            data_locations={"temp": "face"},
        )
        gdf = ds.to_geodataframe("temp", location="face")
        assert len(gdf) == 1, f"Expected 1 row, got {len(gdf)}"
        assert gdf.geometry.iloc[0].geom_type == "Polygon", (
            f"Expected Polygon, got {gdf.geometry.iloc[0].geom_type}"
        )
        assert "temp" in gdf.columns, f"Expected 'temp' column, got {list(gdf.columns)}"

    def test_node_geodataframe(self):
        """Test converting nodes to GeoDataFrame.

        Test scenario:
            Each node should become a Point row.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        gdf = ds.to_geodataframe(location="node")
        assert len(gdf) == 3, f"Expected 3 rows, got {len(gdf)}"
        assert gdf.geometry.iloc[0].geom_type == "Point", (
            f"Expected Point, got {gdf.geometry.iloc[0].geom_type}"
        )

    def test_no_variable(self):
        """Test GeoDataFrame without data variable.

        Test scenario:
            Should produce a GeoDataFrame with geometry only.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        gdf = ds.to_geodataframe(location="face")
        assert len(gdf) == 1, f"Expected 1 row, got {len(gdf)}"
        assert "geometry" in gdf.columns, "Should have geometry column"

    def test_invalid_location_raises(self):
        """Test to_geodataframe with invalid location.

        Test scenario:
            location='invalid' should raise ValueError.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        with pytest.raises(ValueError, match="Unknown location"):
            ds.to_geodataframe(location="invalid")


class TestCreateFromArrays:
    """Tests for MeshDataset.from_arrays() (UGRID-17)."""

    def test_basic_creation(self):
        """Test creating a MeshDataset from raw arrays.

        Test scenario:
            Create a single triangle mesh.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        assert ds.n_node == 3, f"Expected 3 nodes, got {ds.n_node}"
        assert ds.n_face == 1, f"Expected 1 face, got {ds.n_face}"

    def test_with_data(self):
        """Test creating with data variables.

        Test scenario:
            Create mesh with temperature data on faces.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2, 3]]),
            data={"temperature": np.array([20.0])},
            data_locations={"temperature": "face"},
        )
        assert "temperature" in ds.data_variable_names, (
            "Should have temperature variable"
        )
        assert ds["temperature"].data[0] == pytest.approx(20.0), (
            "Temperature should be 20.0"
        )

    def test_mixed_mesh(self):
        """Test creating a mixed mesh from arrays.

        Test scenario:
            Create mesh with tri and quad faces using -1 fill.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array(
                [
                    [0, 1, 4, 3],
                    [1, 2, 5, -1],
                ]
            ),
        )
        assert ds.n_face == 2, f"Expected 2 faces, got {ds.n_face}"
        assert ds.n_node == 6, f"Expected 6 nodes, got {ds.n_node}"

    def test_epsg_set(self):
        """Test that EPSG is correctly set.

        Test scenario:
            Create with EPSG 32631 and verify.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        assert ds.epsg == 32631, f"Expected EPSG 32631, got {ds.epsg}"


class TestEdgeGeoDataFrame:
    """Tests for to_geodataframe with edge location (M9)."""

    def test_edge_geodataframe(self):
        """Test converting edges to GeoDataFrame with LineStrings.

        Test scenario:
            Build edge connectivity, then convert edges to LineStrings.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        ds.mesh.build_edge_connectivity()
        gdf = ds.to_geodataframe(location="edge")
        assert len(gdf) == 3, f"Expected 3 edge rows, got {len(gdf)}"
        assert gdf.geometry.iloc[0].geom_type == "LineString", (
            f"Expected LineString, got {gdf.geometry.iloc[0].geom_type}"
        )

    def test_edge_no_connectivity_raises(self):
        """Test to_geodataframe raises when edge connectivity missing.

        Test scenario:
            Without edge connectivity, should raise ValueError.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        with pytest.raises(ValueError, match="Edge connectivity"):
            ds.to_geodataframe(location="edge")


class TestToFeatureCollection:
    """Tests for to_feature_collection (M10)."""

    def test_feature_collection(self):
        """Test converting mesh to FeatureCollection.

        Test scenario:
            Should wrap GeoDataFrame in a FeatureCollection.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"elev": np.array([5.0])},
            data_locations={"elev": "face"},
        )
        fc = ds.to_feature_collection("elev", location="face")
        assert isinstance(fc, FeatureCollection), (
            f"Expected FeatureCollection, got {type(fc)}"
        )


class TestTemporalWrite:
    """Tests for temporal data write round-trip (M11)."""

    def test_temporal_write_round_trip(self, tmp_path):
        """Test writing and reading temporal data.

        Test scenario:
            Create dataset with 2D temporal data, write, read back,
            verify time dimension and values are preserved.
        """
        temporal_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 3, 2]]),
            data={"wl": temporal_data},
            data_locations={"wl": "face"},
        )
        out_path = tmp_path / "temporal.nc"
        ds.to_file(out_path)

        ds2 = MeshDataset.read_file(out_path)
        assert "wl" in ds2.data_variable_names, "Should have wl variable"
        var = ds2["wl"]
        assert var.has_time, "Should have time dimension"
        assert var.n_time_steps == 2, f"Expected 2 time steps, got {var.n_time_steps}"
        np.testing.assert_array_almost_equal(
            var.data,
            temporal_data,
            err_msg="Temporal data should survive round-trip",
        )


class TestVoronoiPolygons:
    """`MeshDataset.voronoi_polygons` — the file-serializable Voronoi output (#1479)."""

    @staticmethod
    def _square_mesh():
        """A four-node two-triangle mesh in EPSG:4326."""
        return MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 1.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 2, 3]]),
            epsg=4326,
        )

    def test_returns_polygon_cells_in_the_mesh_crs(self):
        """One polygon per node, carrying the mesh CRS (not a bare scipy object).

        Test scenario:
            `voronoi()` returns a `scipy.spatial.Voronoi` with no geospatial form;
            `voronoi_polygons()` returns a `FeatureCollection` of the dual cells so the result
            has a CRS and a vector geometry.
        """
        cells = self._square_mesh().voronoi_polygons()

        assert isinstance(cells, FeatureCollection), f"got {type(cells).__name__}"
        assert len(cells) == 4, "one cell per node"
        assert set(cells.geometry.geom_type) == {"Polygon"}, "cells must be polygons"
        assert cells.epsg == 4326, "the mesh CRS must carry onto the cells"

    def test_cells_are_writable_to_a_file(self, tmp_path):
        """The whole point of the method: the output can be serialized (#1479).

        Test scenario:
            The raw `voronoi()` scipy object cannot be written to a file; the polygon
            `FeatureCollection` can, which is what unblocks a CLI verb.
        """
        out = tmp_path / "voronoi.gpkg"

        self._square_mesh().voronoi_polygons().to_file(str(out))

        assert out.exists() and out.stat().st_size > 0, "the cells must write to a file"

    def test_the_raw_voronoi_primitive_is_unchanged(self):
        """`voronoi()` still returns the low-level scipy tessellation (non-breaking)."""
        from scipy.spatial import Voronoi

        assert isinstance(self._square_mesh().voronoi(), Voronoi)


class TestExtraDimension:
    """`io._extra_dimension` — reuse vs. size-suffix for non-time axes (#1479)."""

    @staticmethod
    def _root_group(tmp_path):
        """A fresh GDAL multidimensional root group to create dimensions in."""
        from osgeo import gdal

        ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(
            str(tmp_path / "dims.nc")
        )
        return ds.GetRootGroup()

    def test_same_name_and_size_is_reused(self, tmp_path):
        """A second request for the same name+size returns the same dimension object.

        Test scenario:
            Two variables that share an extra axis (e.g. both polyfit degree-1 results, each a
            `coefficient` axis of length 2) must share one GDAL dimension, not create two.
        """
        from pyramids.netcdf.ugrid.io import _extra_dimension

        rg = self._root_group(tmp_path)
        dims: dict = {}
        first = _extra_dimension(rg, dims, "coefficient", 2)
        second = _extra_dimension(rg, dims, "coefficient", 2)
        assert second is first, "same name+size must reuse the dimension"

    def test_same_name_different_size_is_suffixed(self, tmp_path):
        """A clash on name with a different size gets a unique size-suffixed name.

        Test scenario:
            A GDAL group cannot hold two dimensions of the same name. A degree-1 fit
            (`coefficient`, size 2) and a degree-2 fit (size 3) in one dataset must not
            collide; the second becomes `coefficient2`.
        """
        from pyramids.netcdf.ugrid.io import _extra_dimension

        rg = self._root_group(tmp_path)
        dims: dict = {}
        first = _extra_dimension(rg, dims, "coefficient", 2)
        second = _extra_dimension(rg, dims, "coefficient", 3)
        assert second is not first, "a different size must not reuse the dimension"
        assert second.GetName() == "coefficient2", f"got {second.GetName()}"

    def test_two_fitted_variables_share_the_coefficient_dim(self, tmp_path):
        """Through the real writer: two temporal vars fit at one degree share the axis.

        Test scenario:
            `polyfit(1)` on a mesh with two temporal variables gives both a `coefficient`
            axis of length 2; writing must reuse the one dimension (the `_extra_dimension`
            same-size path) and round-trip both.
        """
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 1.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 2, 3]]),
            data={
                "a": np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]]),
                "b": np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 5.0]]),
            },
            data_locations={"a": "face", "b": "face"},
        )
        for name in ("a", "b"):
            ds[name].attributes = {
                **ds[name].attributes,
                "time_values": [0.0, 1.0, 2.0],
            }
        out = tmp_path / "twofit.nc"

        ds.polyfit(1).to_file(out)

        back = MeshDataset.read_file(out)
        assert back["a"].dimensions[0] == "coefficient"
        assert back["b"].dimensions[0] == "coefficient"
        assert np.asarray(back["a"].data).shape == (2, 2)
