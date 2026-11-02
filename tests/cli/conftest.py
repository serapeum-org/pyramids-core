"""Shared fixtures for the CLI test suite (expansion command groups).

Kept distinct in name from the per-module fixtures in ``test_cli_commands.py``
so both can coexist; these back the new ``vector`` / ``nc`` / ``collection`` /
expansion tests.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection


@pytest.fixture
def cli_raster(tmp_path) -> str:
    """An 8x8 float32 ramp GeoTIFF on EPSG:4326 with nodata -9999.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "cli_src.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def cli_points(tmp_path) -> str:
    """A 5-point GeoJSON layer on EPSG:4326 with a numeric column.

    Returns:
        str: Path to the written vector.
    """
    gdf = gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(0, 0), Point(4, 0), Point(0, 4), Point(4, 4), Point(2, 2)],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "cli_pts.geojson")
    FeatureCollection(gdf).to_file(path)
    return path


@pytest.fixture
def cli_crsless_vector(tmp_path) -> str:
    """A Shapefile point layer with no CRS set.

    A shapefile (no ``.prj``) genuinely reads back with ``crs=None`` — unlike
    GeoJSON, which GDAL always reports as CRS84/EPSG:4326.

    Returns:
        str: Path to the written vector.
    """
    gdf = gpd.GeoDataFrame({"value": [1.0]}, geometry=[Point(0, 0)])
    path = str(tmp_path / "cli_nocrs.shp")
    FeatureCollection(gdf).to_file(path, driver="ESRI Shapefile")
    return path


@pytest.fixture
def cli_stack(tmp_path) -> str:
    """Three co-registered 4x4 rasters t0/t1/t2 (values 0, 1, 2).

    Returns:
        str: A glob matching the three rasters.
    """
    for index in range(3):
        Dataset.from_array(
            np.full((4, 4), float(index), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
        ).to_file(str(tmp_path / f"t{index}.tif"))
    return str(tmp_path / "t*.tif")
