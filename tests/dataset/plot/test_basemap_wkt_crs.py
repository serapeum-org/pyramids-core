"""Render-time basemap on a WKT-only CRS (no EPSG code) — issue #1577.

After #1572 a projected-but-no-EPSG dataset's returned glyph carries its WKT CRS. This covers
the companion fix: ``plot(basemap=True)`` must dispatch the tile fetch with that WKT (resolved
through ``crs_spec``) instead of raising "must have a CRS (epsg) to use basemap". A genuinely
CRS-less dataset still raises. ``add_basemap`` is patched throughout — these assert the dispatch
CRS, not a real tile fetch.
"""

import geopandas as gpd
import numpy as np
import pyproj
import pytest
from shapely.geometry import Point

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")

from unittest.mock import patch  # noqa: E402

from pyramids.base._errors import CRSError  # noqa: E402
from pyramids.dataset import Dataset  # noqa: E402
from pyramids.dataset._plot_helpers import RenderRequest, mesh_render  # noqa: E402
from pyramids.feature import FeatureCollection  # noqa: E402
from pyramids.netcdf import GeoReference  # noqa: E402
from pyramids.netcdf.ugrid import MeshDataset  # noqa: E402
from pyramids.netcdf.ugrid.plot import plot_mesh_outline  # noqa: E402

pytestmark = pytest.mark.plot

_WKT = pyproj.CRS.from_proj4(
    "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
).to_wkt()


def _wkt_raster():
    """A single-band raster whose CRS is a custom LAEA projection (``epsg`` is None)."""
    ds = Dataset.from_array(
        np.arange(12.0).reshape(3, 4),
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )
    ds.crs = _WKT
    return ds


def _wkt_mesh():
    """A two-face mesh whose CRS is a custom LAEA projection (``epsg`` is None)."""
    ds = MeshDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": np.array([1.0, 2.0])},
        data_locations={"d": "face"},
        epsg=28992,
    )
    ds.set_crs(crs=_WKT)
    return ds


class TestBasemapWktCrs:
    """`plot(basemap=True)` tiles a WKT-only CRS instead of rejecting it (#1577)."""

    def test_raster_plot_basemap_dispatches_wkt(self):
        """`Dataset.plot(basemap=True)` on a WKT-only raster tiles with the WKT."""
        ds = _wkt_raster()
        assert ds.epsg is None, "precondition: the custom CRS has no EPSG code"
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            ds.plot(band=0, basemap=True)
        assert mock_add.call_args.kwargs["crs"] == ds.crs, (
            f"basemap tile fetch must use the dataset WKT, got {mock_add.call_args.kwargs.get('crs')!r}"
        )

    def test_validate_rejects_basemap_without_any_crs(self):
        """A truthy basemap with neither EPSG nor WKT still raises (no spurious tiling)."""
        request = RenderRequest(
            arr=np.zeros((4, 4)), basemap=True, basemap_epsg=None, basemap_crs_wkt=None
        )
        with pytest.raises(ValueError, match="to use basemap"):
            request.validate()

    def test_feature_plot_basemap_no_crs_raises(self):
        """A CRS-less FeatureCollection with `basemap=True` still raises (part (b) of #1577)."""
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs=None)
        fc = FeatureCollection(gdf)
        assert fc.epsg is None, "precondition: the FC has no EPSG code"
        assert fc.crs is None, "precondition: the FC has no WKT either"
        with pytest.raises(CRSError, match="to use basemap"):
            fc.plot(basemap=True)

    def test_mesh_render_basemap_no_crs_raises(self):
        """`mesh_render` with no EPSG and no WKT still raises (the data-path guard holds)."""
        ds = _wkt_mesh()
        with pytest.raises(ValueError, match="to use basemap"):
            mesh_render(
                mesh=ds._mesh,
                data=np.array([1.0, 2.0]),
                location="face",
                basemap=True,
                basemap_epsg=None,
                basemap_crs_wkt=None,
            )

    def test_plot_mesh_outline_basemap_no_crs_raises(self):
        """`plot_mesh_outline` with no EPSG and no WKT still raises (the outline guard holds)."""
        ds = _wkt_mesh()
        with pytest.raises(ValueError, match="to use basemap"):
            plot_mesh_outline(
                ds._mesh, basemap=True, basemap_epsg=None, basemap_crs_wkt=None
            )

    def test_feature_plot_basemap_dispatches_wkt(self):
        """`FeatureCollection.plot(basemap=True)` on a WKT-only FC tiles with the WKT."""
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs=_WKT)
        fc = FeatureCollection(gdf)
        assert fc.epsg is None, "precondition: the custom CRS has no EPSG code"
        with patch("pyramids.feature._plot.add_basemap") as mock_add:
            fc.plot(basemap=True)
        assert mock_add.call_args.kwargs["crs"] == fc.crs.to_wkt(), (
            f"basemap tile fetch must use the FC WKT, got {mock_add.call_args.kwargs.get('crs')!r}"
        )

    def test_mesh_plot_basemap_dispatches_wkt(self):
        """`MeshDataset.plot(basemap=True)` on a WKT-only mesh tiles with the WKT."""
        ds = _wkt_mesh()
        assert ds.epsg is None, "precondition: the custom CRS has no EPSG code"
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            ds.glyph.plot("d", basemap=True)
        assert mock_add.call_args.kwargs["crs"] == ds.crs_wkt, (
            f"basemap tile fetch must use the mesh WKT, got {mock_add.call_args.kwargs.get('crs')!r}"
        )

    def test_mesh_outline_basemap_dispatches_wkt(self):
        """`MeshDataset.plot_outline(basemap=True)` on a WKT-only mesh tiles with the WKT."""
        ds = _wkt_mesh()
        assert ds.epsg is None, "precondition: the custom CRS has no EPSG code"
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            ds.glyph.plot_outline(basemap=True)
        assert mock_add.call_args.kwargs["crs"] == ds.crs_wkt, (
            f"basemap tile fetch must use the mesh WKT, got {mock_add.call_args.kwargs.get('crs')!r}"
        )
