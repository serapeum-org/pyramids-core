"""D4: the dataset is the single CRS owner; every export resolves from it, so none disagree."""

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
    """Every geometry export reads the dataset CRS, so they all agree -- even across derivations."""

    def test_from_arrays_drives_contour_from_the_dataset_crs(self):
        """``from_arrays(epsg=)`` gives the dataset a CRS, so contour is not CRS-less."""
        ds = _square_mesh(epsg=4326)
        assert ds.epsg == 4326
        assert ds.vector.contour(3, "depth").crs is not None
        assert ds.vector.contour(3, "depth").crs.to_epsg() == 4326

    def test_two_public_exports_agree(self):
        """``to_geodataframe`` and ``contour`` report the same CRS, not 4326 vs None."""
        ds = _square_mesh(epsg=4326)
        gdf_epsg = ds.vector.to_geodataframe().crs.to_epsg()
        contour_epsg = ds.vector.contour(3, "depth").crs.to_epsg()
        assert gdf_epsg == contour_epsg == 4326

    def test_to_crs_carries_the_crs_to_contour(self):
        """Reprojection's new CRS reaches the geometry exports."""
        reprojected = _square_mesh(epsg=4326).spatial.to_crs(3857)
        assert reprojected.epsg == 3857
        assert reprojected.vector.contour(3, "depth").crs.to_epsg() == 3857

    def test_clip_keeps_the_crs_on_contour(self):
        """A spatial subset's contour is not CRS-less."""
        clipped = _square_mesh(epsg=4326).clip(box(0.0, 0.0, 1.0, 1.0))
        assert clipped.vector.contour(3, "depth").crs.to_epsg() == 4326

    def test_set_crs_refreshes_topology_info(self):
        """``set_crs`` updates the fourth slot, so topo_info does not go stale."""
        from pyproj import CRS

        ds = _square_mesh(epsg=4326)
        ds.spatial.set_crs(epsg=3857)
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

    def test_construction_does_not_mutate_the_caller_mesh(self):
        """Review M2: building a dataset must not write a CRS onto the passed-in mesh.

        The dataset owns its CRS via ``_crs_wkt``; an earlier design filled ``mesh.crs`` in
        ``__init__``, mutating a caller-held object. A bare mesh stays CRS-less after
        construction, and two datasets built from the same bare mesh with different CRSes are
        independent (neither raises, neither clobbers the other).
        """
        from pyproj import CRS

        ds = _square_mesh(epsg=4326)
        bare = ds.mesh
        assert bare.crs is None, (
            "from_arrays builds a CRS-less mesh; __init__ must not fill it"
        )
        d1 = MeshDataset(
            mesh=bare,
            data_variables=ds._data_variables,
            global_attributes={},
            topology_info=None,
            crs_wkt=CRS.from_epsg(4326).to_wkt(),
        )
        assert bare.crs is None, "the caller's mesh was mutated"
        d2 = MeshDataset(
            mesh=bare,
            data_variables=ds._data_variables,
            global_attributes={},
            topology_info=None,
            crs_wkt=CRS.from_epsg(3857).to_wkt(),
        )
        assert d1.epsg == 4326
        assert d2.epsg == 3857

    def test_exports_stay_consistent_when_the_mesh_carries_another_crs(self):
        """Review M1: exports resolve from the dataset, so a stale mesh CRS cannot diverge.

        Even when the mesh happens to carry a different CRS than the dataset's ``crs_wkt`` --
        including an authority-less (no-EPSG) one that defeated the old EPSG-based guard -- every
        geometry export reports the dataset's CRS, never the mesh's.
        """
        from pyproj import CRS

        laea = CRS.from_proj4(
            "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        ds = _square_mesh(epsg=4326)
        ds.mesh.crs = laea  # a different, no-EPSG CRS sitting on the mesh
        gdf_epsg = ds.vector.to_geodataframe().crs.to_epsg()
        contour_epsg = ds.vector.contour(3, "depth").crs.to_epsg()
        assert gdf_epsg == contour_epsg == ds.epsg == 4326
