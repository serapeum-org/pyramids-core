"""Zonal statistics over a :class:`Dataset`, grouped by polygon zones or raster categories.

The zones come from **either** a polygon :class:`~pyramids.feature.FeatureCollection` **or** a categorical
raster :class:`~pyramids.dataset.Dataset` (each distinct pixel value is a zone). Both paths reduce the base band
over an integer label grid with the shared :func:`pyramids.base._reductions.reduce_by_label`, and both return a
:class:`geopandas.GeoDataFrame` — one row per zone, carrying the zone geometry plus one column per statistic.

Single-pass design:

* Polygon zones: rasterize every polygon into an integer label grid (xvec's `method="rasterize"` pattern), then
  reduce per label.
* Raster zones: the class raster *is* the label grid; `numpy.unique` maps the distinct class values to 0-based
  labels, and the class footprints are vectorised (`to_polygons` dissolved by value) for the geometry.

Scope kept narrow:

* Supported stats: `mean`, `sum`, `min`, `max`, `count`, `std`, `var`.
* One band at a time (caller selects via `band=`).
* Raster/raster and raster/vector must share a grid/CRS — caller aligns or reprojects first.
* `raw_values=True` adds an object column holding each zone's extracted values (an `ndarray`, in row-major
  order). It is an in-memory convenience and does not export to file formats; it is off by default.

Area-weighted zonal statistics (a cell scaled by its area fraction inside the zone) are planned as a
`method="fractional"` follow-up — see `planning/dask/zonal_stats/`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from pyramids.base._domain import is_no_data
from pyramids.base._reductions import reduce_by_label
from pyramids.base._utils import apply_unpack
from pyramids.base.crs import sr_from_epsg, sr_from_wkt

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature import FeatureCollection


def _rasterize_labels(ds: Dataset, fc: FeatureCollection) -> np.typing.NDArray:
    """Rasterize `fc` into an integer label array shaped like `ds`.

    Label values are 0-based feature-index integers. Pixels not
    covered by any polygon get -1.

    H4: the FeatureCollection's CRS is attached to the OGR layer so
    GDAL's `RasterizeLayer` reprojects coordinates when the vector
    and raster CRSes disagree. A mismatch that `pyproj` considers
    incompatible raises :class:`ValueError` early rather than silently
    producing a mis-aligned label grid.
    """
    fc_crs = getattr(fc, "crs", None)
    ds_epsg = int(ds.epsg) if ds.epsg else None
    if fc_crs is not None and ds_epsg is not None:
        fc_epsg = fc_crs.to_epsg()
        if fc_epsg is not None and fc_epsg != ds_epsg:
            raise ValueError(
                f"zonal_stats: FeatureCollection CRS (EPSG:{fc_epsg}) does "
                f"not match Dataset CRS (EPSG:{ds_epsg}). Reproject the "
                "FeatureCollection via fc.to_crs(ds.epsg) first."
            )

    if fc_crs is not None:
        srs = sr_from_wkt(fc_crs.to_wkt())
    elif ds_epsg is not None:
        srs = sr_from_epsg(ds_epsg)
    else:
        srs = osr.SpatialReference()
    mem_driver = ogr.GetDriverByName("MEM")
    ds_vec = mem_driver.CreateDataSource("zonal_mem")
    layer = ds_vec.CreateLayer("features", srs=srs, geom_type=ogr.wkbPolygon)
    id_field = ogr.FieldDefn("pid", ogr.OFTInteger)
    layer.CreateField(id_field)
    for idx, geom in enumerate(fc.geometry):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("pid", int(idx))
        ogr_geom = ogr.CreateGeometryFromWkb(geom.wkb)
        feat.SetGeometry(ogr_geom)
        layer.CreateFeature(feat)
        feat = None

    mem_drv = gdal.GetDriverByName("MEM")
    label_ds = mem_drv.Create("", ds.columns, ds.rows, 1, gdal.GDT_Int32)
    label_ds.SetGeoTransform(ds.geotransform)
    label_ds.SetProjection(ds.raster.GetProjection())
    label_ds.GetRasterBand(1).Fill(-1)
    gdal.RasterizeLayer(
        label_ds,
        [1],
        layer,
        options=["ATTRIBUTE=pid", "ALL_TOUCHED=FALSE"],
    )
    labels = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None
    ds_vec = None
    return labels


def _prep_band(
    ds: Dataset, band: int, exclude_value: float | None
) -> np.typing.NDArray:
    """Return the band's physical values with invalid cells blanked to `NaN`.

    The band is read once in stored units; the no-data cells are found there, where the sentinel lives, and the
    values are then unpacked to physical units with `ds._effective_packing(band)`. No-data cells and any cell equal
    to `exclude_value` (in physical units) are set to `NaN` so `reduce_by_label` ignores them and `raw_values`
    drops them.

    Args:
        ds: The source dataset.
        band: Zero-based band index.
        exclude_value: A physical value to drop, or `None`.

    Returns:
        np.ndarray: `float64`, shaped like the band, with invalid cells as `NaN`.
    """
    stored = np.asarray(ds.read_array(band=band, unpack=False))
    raster = np.asarray(
        apply_unpack(stored, *ds._effective_packing(band)), dtype=np.float64
    )
    no_data_list = ds.no_data_value
    no_data = no_data_list[band] if no_data_list else None
    valid = np.ones(raster.shape, dtype=bool)
    if no_data is not None:
        # `is_no_data`, not `==`: a NaN sentinel never equals itself.
        valid &= ~is_no_data(stored, no_data)
    if exclude_value is not None:
        valid &= raster != float(exclude_value)
    return np.where(valid, raster, np.nan)


def _vector_zones(
    ds: Dataset, fc: FeatureCollection
) -> tuple[np.typing.NDArray, int, Any, Any, Any]:
    """Build the label grid, zone count, index, geometry and CRS for polygon zones."""
    labels = _rasterize_labels(ds, fc)
    index = fc.index
    geoms = fc.geometry.values
    crs = getattr(fc, "crs", None)
    return labels, len(fc), index, geoms, crs


def _raster_zones(
    ds: Dataset, classes_ds: Dataset
) -> tuple[np.typing.NDArray, int, Any, Any, Any]:
    """Build the label grid, zone count, index, geometry and CRS for raster-category zones.

    The class raster must be aligned with `ds` (same grid). Each distinct class value (excluding the class
    raster's no-data) becomes a zone; the class footprints are vectorised and dissolved per value for the
    geometry.
    """
    class_arr = np.asarray(classes_ds.read_array(band=0, unpack=False))
    cls_no_data_list = classes_ds.no_data_value
    cls_no_data = cls_no_data_list[0] if cls_no_data_list else None
    class_valid = np.ones(class_arr.shape, dtype=bool)
    if cls_no_data is not None:
        class_valid &= ~is_no_data(class_arr, cls_no_data)

    classes = np.unique(class_arr[class_valid])
    labels = np.full(class_arr.shape, -1, dtype=np.intp)
    labels[class_valid] = np.searchsorted(classes, class_arr[class_valid])

    polygons = classes_ds.to_polygons()
    value_col = polygons.columns[0]
    dissolved = polygons.dissolve(by=value_col)
    geom_map = {
        float(value): geom for value, geom in zip(dissolved.index, dissolved.geometry)
    }
    geoms = [geom_map.get(float(class_value)) for class_value in classes]
    index = pd.Index(classes, name="class")
    crs = ds.crs
    return labels, len(classes), index, geoms, crs


def _extract_raw(
    raster: np.typing.NDArray, labels: np.typing.NDArray, n_zones: int
) -> list[np.typing.NDArray]:
    """Per zone, the base band's valid values in row-major order (NaN cells dropped)."""
    flat_labels = labels.ravel()
    flat_values = raster.ravel()
    raw: list[np.typing.NDArray] = []
    for label in range(n_zones):
        zone_values = flat_values[flat_labels == label]
        raw.append(zone_values[~np.isnan(zone_values)])
    return raw


def zonal_stats(
    ds: Dataset,
    zones: FeatureCollection | Dataset,
    *,
    stats: Sequence[str] = ("mean",),
    method: str = "rasterize",
    band: int = 0,
    exclude_value: float | None = None,
    raw_values: bool = False,
) -> gpd.GeoDataFrame:
    """Group `ds`'s values by polygon or raster-category zones and reduce them per zone.

    The statistics are taken over physical values, as `read_array` returns them: a CF-packed band
    (`scale_factor` / `add_offset`) is unpacked. The band's no-data cells, and any cell equal to `exclude_value`,
    are left out of every statistic.

    Args:
        ds: The source :class:`~pyramids.dataset.Dataset`.
        zones: Either a polygon :class:`~pyramids.feature.FeatureCollection` (one zone per feature) or a
            categorical raster :class:`~pyramids.dataset.Dataset` aligned with `ds` (one zone per distinct pixel
            value). A `FeatureCollection`'s CRS must match `ds`; a class raster must share `ds`'s grid.
        stats: Statistics per zone. One or more of `"mean"`, `"sum"`, `"min"`, `"max"`, `"std"`, `"var"`,
            `"count"`.
        method: `"rasterize"` is the only supported value for polygon zones today (raster zones ignore it). An
            area-weighted `"fractional"` method is planned.
        band: Zero-based band index on `ds`. Default 0.
        exclude_value: A physical value to drop from the inputs before reducing, or `None`.
        raw_values: When `True`, add a `raw_values` object column holding each zone's extracted values as an
            `ndarray` (row-major order, NaN/no-data dropped). Off by default; the column is in-memory only and
            does not export to file formats.

    Returns:
        geopandas.GeoDataFrame: One row per zone, carrying the zone `geometry`, one `float64` column per stat
        (`NaN` for a zone with no valid cell, `count` is `0.0` there), and optionally `raw_values`. Indexed by
        `fc.index` for polygon zones, or by the class value (`index.name="class"`) for raster zones.

    Raises:
        ValueError: An unknown `stat` name or an unknown `method`.
        AlignmentError: A class raster not aligned with `ds`.

    Examples:
        - Mean of a constant raster over one polygon equals the raster value:
            ```python
            >>> import geopandas as gpd
            >>> import numpy as np
            >>> from shapely.geometry import box
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.dataset.ops._zonal import zonal_stats
            >>> from pyramids.feature import FeatureCollection
            >>> arr = np.full((4, 4), 5.0, dtype=np.float32)
            >>> ds = Dataset.from_array(
            ...     arr,
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
            ... )
            >>> fc = FeatureCollection(gpd.GeoDataFrame(
            ...     {"zone": ["a"]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:4326",
            ... ))
            >>> out = zonal_stats(ds, fc, stats=("mean",))
            >>> float(out["mean"].iloc[0])
            5.0

            ```
        - Group a value raster by an aligned class raster; one row per class:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.dataset.ops._zonal import zonal_stats
            >>> geo = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
            >>> values = Dataset.from_array(
            ...     np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"), geo_ref=geo,
            ... )
            >>> classes = Dataset.from_array(
            ...     np.array([[1, 1], [2, 2]], dtype="int32"), geo_ref=geo,
            ... )
            >>> out = zonal_stats(values, classes, stats=("mean", "count"))
            >>> out[["mean", "count"]].to_dict("index")
            {1: {'mean': 15.0, 'count': 2.0}, 2: {'mean': 35.0, 'count': 2.0}}

            ```
    """
    if method != "rasterize":
        raise ValueError(f"method must be 'rasterize', got {method!r}")

    # Local import breaks the dataset <-> dataset.ops._zonal cycle (dataset.py imports this module).
    from pyramids.dataset import Dataset as _Dataset

    raster = _prep_band(ds, band, exclude_value)
    if isinstance(zones, _Dataset):
        if not ds.spatial._check_alignment(zones):
            from pyramids.base._errors import AlignmentError

            raise AlignmentError(
                "The class Dataset is not aligned with the current raster, please use the "
                "method 'align' to align both rasters."
            )
        labels, n_zones, index, geoms, crs = _raster_zones(ds, zones)
    else:
        labels, n_zones, index, geoms, crs = _vector_zones(ds, zones)

    columns = reduce_by_label(raster, labels, n_zones, list(stats), unassigned=-1)
    data = {stat: columns[stat] for stat in stats}
    if raw_values:
        data["raw_values"] = _extract_raw(raster, labels, n_zones)
    return gpd.GeoDataFrame(data, geometry=list(geoms), index=index, crs=crs)


__all__ = ["zonal_stats"]
