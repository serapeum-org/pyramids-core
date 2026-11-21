"""D4: the mesh is the single CRS owner; dataset exports never disagree with it."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf.ugrid import MeshDataset


def _square_mesh(epsg: int | None = None) -> MeshDataset:
    """A three-face mesh, optionally carrying a declared CRS."""
    return MeshDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5]]),
        data={"depth": np.array([1.5, 2.5, 3.5])},
        epsg=epsg,
    )


class TestCrsSingleOwner:
    """Every derivation keeps the dataset CRS and the mesh CRS in agreement."""

    def test_from_arrays_sets_the_mesh_crs(self):
        """``from_arrays(epsg=)`` reconciles the mesh, so contour is not CRS-less."""
        ds = _square_mesh(epsg=4326)
        assert ds.epsg == 4326
        assert ds.mesh.crs is not None and ds.mesh.crs.to_epsg() == 4326
        assert ds.contour(3, "depth").crs is not None
        assert ds.contour(3, "depth").crs.to_epsg() == 4326

    def test_two_public_exports_agree(self):
        """``to_geodataframe`` and ``contour`` report the same CRS, not 4326 vs None."""
        ds = _square_mesh(epsg=4326)
        gdf_epsg = ds.to_geodataframe().crs.to_epsg()
        contour_epsg = ds.contour(3, "depth").crs.to_epsg()
        assert gdf_epsg == contour_epsg == 4326

    def test_to_crs_keeps_the_mesh_crs(self):
        """Reprojection carries the CRS onto the new mesh, not just the dataset."""
        reprojected = _square_mesh(epsg=4326).to_crs(3857)
        assert reprojected.mesh.crs is not None
        assert reprojected.contour(3, "depth").crs.to_epsg() == 3857

    def test_clip_keeps_the_mesh_crs(self):
        """A spatial subset no longer returns a CRS-less mesh."""
        clipped = _square_mesh(epsg=4326).clip(box(0.0, 0.0, 1.0, 1.0))
        assert clipped.contour(3, "depth").crs.to_epsg() == 4326

    def test_set_crs_refreshes_topology_info(self):
        """``set_crs`` updates the fourth slot, so topo_info does not go stale."""
        from pyproj import CRS

        ds = _square_mesh(epsg=4326)
        ds.set_crs(epsg=3857)
        assert CRS.from_wkt(ds._topology_info.crs_wkt).to_epsg() == 3857

    def test_malformed_wkt_is_tolerated_not_raised(self):
        """A bare mesh adopting an unparseable WKT stays CRS-less, never raises."""
        from pyramids.netcdf.ugrid.mesh import Connectivity, Mesh2d

        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        assert mesh.crs is None
        mesh.set_crs_wkt("NOT A WKT")
        assert mesh.crs is None
        assert mesh.crs_wkt is None
        mesh.set_crs_wkt(None)  # also a no-op, never raises
        assert mesh.crs is None

    def test_fill_is_monotonic_never_clobbers(self):
        """A mesh that already has a CRS is not overwritten by a different WKT."""
        from pyproj import CRS

        ds = _square_mesh(epsg=4326)
        shared = ds.mesh
        assert shared.crs.to_epsg() == 4326
        # Build a second dataset from the same, already-CRS'd mesh with a different WKT.
        MeshDataset(
            mesh=shared,
            data_variables=ds._data_variables,
            global_attributes=ds._global_attributes,
            topology_info=ds._topology_info,
            crs_wkt=CRS.from_epsg(3857).to_wkt(),
        )
        assert shared.crs.to_epsg() == 4326  # first write wins; no clobber
