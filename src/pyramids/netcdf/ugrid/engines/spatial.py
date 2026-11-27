"""Spatial engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.spatial``).

Owns the spatial-subset / reprojection / resampling family -- ``crop`` (polygon or bbox),
``subset_by_bounds`` (axis-aligned envelope), ``to_crs`` (reproject node coordinates),
``set_crs`` (declare a CRS without warping) and ``resample`` (bin the time axis by a calendar
frequency). Mirrors the raster ``Dataset.spatial`` engine; the per-operation maths live in
:mod:`pyramids.netcdf.ugrid.spatial` (the subsetters) and :mod:`pyramids.netcdf.ugrid.mesh`,
with the CRS helpers in :mod:`pyramids.base.crs`, so none of them imports ``ugrid.dataset`` and
the import graph stays acyclic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

from pyramids.base._errors import CRSError
from pyramids.base.crs import (
    crs_from_user_input,
    crs_spec,
    epsg_from_user_input,
    sr_from_epsg,
    sr_from_wkt,
)
from pyramids.netcdf.ugrid.engines._base import _MeshEngine
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.spatial import resolve_crs, subset_by_bounds

if TYPE_CHECKING:
    from pyramids.netcdf.ugrid.dataset import MeshDataset


class _MeshSpatial(_MeshEngine):
    """Spatial ops for a mesh: crop, subset_by_bounds, to_crs, set_crs, resample (``mesh.spatial``)."""

    __slots__ = ()

    def set_crs(self, crs: str | None = None, epsg: int | None = None) -> None:
        """Assign a CRS to a mesh that has none — no reprojection.

        Exactly one of *crs* or *epsg* must be given. The CRS is written
        in place to the dataset **and** the underlying :class:`Mesh2d`,
        so :attr:`epsg`, :attr:`crs_wkt`, and :attr:`crs` all reflect
        the new value immediately.

        This is the complement of :meth:`to_crs`, which *reprojects*
        coordinates. ``set_crs`` only declares the coordinate system the
        existing coordinates are already in.

        Args:
            crs: WKT string describing the CRS.
            epsg: EPSG code (e.g. ``4326``).

        Raises:
            ValueError: If neither *crs* nor *epsg* is provided, or if
                both are provided.
        """
        wkt, crs_obj = resolve_crs(self._ds._mesh, crs=crs, epsg=epsg)
        self._ds._crs_wkt = wkt
        self._ds._cached_crs = crs_obj
        # Keep the fourth slot from going stale (D4): topo_info.crs_wkt is only read back
        # on the read path, but leaving it behind after set_crs is a loaded gun.
        if self._ds._topology_info is not None:
            self._ds._topology_info = replace(self._ds._topology_info, crs_wkt=wkt)

    def crop(
        self,
        mask: Any = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: int | None = None,
    ) -> MeshDataset:
        """Crop the mesh to a polygon mask or a bbox — the unstructured-mesh analogue of crop.

        The mesh equivalent of :meth:`pyramids.dataset.Dataset.crop` / :meth:`NetCDF.crop`. Rather
        than warping a raster, it selects the **faces** that fall inside the region (renumbering the
        node connectivity for the resulting sub-mesh) and keeps the data on the surviving elements.
        Delegates to :meth:`clip` for a polygon and :meth:`subset_by_bounds` for a bbox; this method
        exists so the spatial-subset call is named ``crop`` across the raster and mesh classes alike.

        Args:
            mask (Any):
                Polygon mask — a shapely geometry, ``GeoDataFrame``, or ``FeatureCollection``.
                Mutually exclusive with ``bbox``.
            touch (bool):
                Applies only to a polygon ``mask``: if ``True`` (default), keep faces that touch the
                mask boundary; if ``False``, keep only faces fully inside it. Ignored for ``bbox``,
                which always selects faces by its axis-aligned envelope (``subset_by_bounds``).
                Defaults to True.
            bbox (tuple or list of 4 floats, keyword-only):
                ``(west, south, east, north)`` in the mesh CRS, or in ``epsg`` when supplied. Accepts
                a tuple or a list. Selects faces by the axis-aligned envelope; not affected by
                ``touch``. Mutually exclusive with ``mask``.
            epsg (int, keyword-only):
                CRS of ``bbox``. When it differs from the mesh CRS the box is reprojected to the mesh
                CRS and subset by its envelope; when it equals the mesh CRS it is a no-op. Defaults to
                the mesh CRS.

        Returns:
            MeshDataset: A new sub-mesh — faces inside the region, connectivity renumbered, and data
                variables subset to the surviving elements.

        Raises:
            ValueError: If both ``mask`` and ``bbox`` are supplied, if ``bbox`` is not a 4-tuple, or
                if ``epsg`` is given for a ``bbox`` but the mesh has no CRS to reproject into.
            TypeError: If neither ``mask`` nor ``bbox`` is supplied.

        Examples:
            - Crop a mesh to a polygon (faces intersecting it survive):
                ```python
                >>> from shapely.geometry import Polygon
                >>> from pyramids.netcdf import MeshDataset
                >>> ug = MeshDataset.read_file("mesh.nc")                        # doctest: +SKIP
                >>> sub = ug.spatial.crop(Polygon([(-1, -1), (0, -1), (0, 1), (-1, 1)]))  # doctest: +SKIP
                >>> sub.n_face <= ug.n_face                                       # doctest: +SKIP
                True

                ```
            - Crop to a bounding box in the mesh's own CRS:
                ```python
                >>> sub = ug.spatial.crop(bbox=(-1.0, -1.0, 0.0, 1.0))            # doctest: +SKIP

                ```

        See Also:
            clip: Polygon-mask subsetting that ``crop`` delegates to when ``mask`` is given.
            subset_by_bounds: Bounding-box subsetting that ``crop`` delegates to when ``bbox`` is
                given.
            pyramids.dataset.Dataset.crop: The raster equivalent on a gridded dataset.
        """
        if bbox is not None:
            if mask is not None:
                raise ValueError("crop accepts either `mask` or `bbox`, not both")
            if len(bbox) != 4:
                raise ValueError(
                    "bbox must be a 4-tuple of (west, south, east, north), "
                    f"got {len(bbox)} value(s)"
                )
            west, south, east, north = bbox
            if epsg is not None and (
                self._ds.epsg is None or int(epsg) != int(self._ds.epsg)
            ):
                if self._ds.epsg is None:
                    raise ValueError(
                        f"cannot reproject a bbox given in EPSG:{int(epsg)} into a mesh that has no "
                        "CRS; drop epsg to treat the bbox as native coordinates"
                    )
                # Reproject the bbox to the mesh CRS and subset by its envelope, so the bbox path
                # selects faces with the same rule (subset_by_bounds) regardless of the source CRS.
                west, south, east, north = (
                    gpd.GeoSeries([box(west, south, east, north)], crs=epsg)
                    .to_crs(self._ds.epsg)
                    .total_bounds
                )
            # Call the engine sibling, not the deprecated flat facade, so a bbox crop does
            # not emit the `subset_by_bounds` deprecation warning from inside `crop`.
            result = self.subset_by_bounds(west, south, east, north)
        elif mask is not None:
            result = self._ds.clip(mask, touch=touch)
        else:
            raise TypeError(
                "crop requires a `mask` (polygon) or a `bbox` (west, south, east, north) tuple"
            )
        return result

    def subset_by_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> MeshDataset:
        """Subset mesh to faces within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            New MeshDataset with subset mesh and data.
        """
        mesh, data_variables = subset_by_bounds(self._ds, xmin, ymin, xmax, ymax)
        return self._ds._wrap_subset(mesh, data_variables)

    def to_crs(self, to_epsg: int | str | Any) -> MeshDataset:
        """Reproject all node coordinates to a new CRS.

        Uses pyproj.Transformer to reproject node coordinates.
        Face/edge center coordinates are recomputed after reprojection.
        Data values are preserved — only coordinates change.

        Args:
            to_epsg: The target CRS, in any form `pyproj.CRS.from_user_input` accepts — an
                EPSG reference number (`3857`), an authority string (`"EPSG:3857"`,
                `"ESRI:54030"` for Robinson), a bare numeric string (`"3857"`), a PROJ4 or
                WKT string, or a `pyproj.CRS`. This matches `Dataset.to_crs` and
                `Spatial.to_crs`; a mesh is no longer the one layer that takes a bare code
                only. A CRS without an EPSG code — most globe and custom projections — is now
                a valid target.

        Returns:
            New MeshDataset with reprojected coordinates.

        Raises:
            ValueError: The mesh declares neither an EPSG code nor a WKT, so there is no
                source CRS to reproject from.
            pyramids.base._errors.CRSError: The target CRS cannot be interpreted.

        Examples:
            - Reproject to an authority string, which used to be rejected (#1404):
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ... )
                >>> mesh.spatial.set_crs(epsg=4326)
                >>> mesh.spatial.to_crs("EPSG:3857").epsg
                3857
                >>> mesh.spatial.to_crs(3857).epsg
                3857

                ```
            - Reproject to a CRS that has no EPSG code (Robinson); the result keeps its WKT
                and reports `epsg` as `None`:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ... )
                >>> mesh.spatial.set_crs(epsg=4326)
                >>> robinson = mesh.spatial.to_crs("ESRI:54030")
                >>> robinson.epsg is None
                True
                >>> "Robinson" in robinson.crs.name
                True

                ```
        """
        # The mesh may describe its CRS by WKT alone, with no EPSG code to
        # resolve; `crs_spec` returns whichever of the two is usable, so a
        # projected mesh stops being refused for a CRS it plainly has.
        source_crs = crs_spec(self._ds.epsg, self._ds.crs_wkt)
        if source_crs is None:
            raise ValueError(
                "Cannot reproject: source CRS is unknown. "
                "Set CRS before calling to_crs()."
            )

        # Both sides go through `crs_from_user_input` as-is, so a mesh in a CRS whose
        # code only GDAL's PROJ database carries still reprojects (issue #943), and the
        # target accepts the same range of inputs the source does. Prefixing `EPSG:` onto
        # the target was the bug (#1404): it built the nonsense `EPSG:EPSG:3857` /
        # `EPSG:+proj=robin` / `EPSG:PROJCS[...]` for anything other than a bare integer,
        # the exact hazard the source comment above warns about.
        target_crs = crs_from_user_input(to_epsg)
        transformer = Transformer.from_crs(
            crs_from_user_input(source_crs),
            target_crs,
            always_xy=True,
        )
        new_node_x, new_node_y = transformer.transform(
            self._ds._mesh.node_x,
            self._ds._mesh.node_y,
        )

        new_face_x = None
        new_face_y = None
        if self._ds._mesh.has_face_coords:
            new_face_x, new_face_y = transformer.transform(
                self._ds._mesh.face_x,
                self._ds._mesh.face_y,
            )

        new_edge_x = None
        new_edge_y = None
        if self._ds._mesh.has_edge_coords:
            new_edge_x, new_edge_y = transformer.transform(
                self._ds._mesh.edge_x,
                self._ds._mesh.edge_y,
            )

        new_mesh = Mesh2d(
            node_x=new_node_x,
            node_y=new_node_y,
            face_node_connectivity=self._ds._mesh.face_node_connectivity,
            edge_node_connectivity=self._ds._mesh.edge_node_connectivity,
            face_edge_connectivity=self._ds._mesh.face_edge_connectivity,
            face_face_connectivity=self._ds._mesh.face_face_connectivity,
            edge_face_connectivity=self._ds._mesh.edge_face_connectivity,
            face_x=new_face_x,
            face_y=new_face_y,
            edge_x=new_edge_x,
            edge_y=new_edge_y,
        )

        # The stored WKT comes from the resolved target. When the target carries an EPSG
        # code, `sr_from_epsg` gives GDAL's authority WKT -- byte-for-byte what this method
        # produced before #1404 for every coded target (keeping the child datum / spheroid /
        # primem AUTHORITY nodes a WKT2->WKT1 round-trip drops, and, for a code only GDAL's
        # PROJ database carries (#943), the sole place that code survives at all). The code
        # is recovered with `epsg_from_user_input`, not `target_crs.to_epsg()`: pyproj's own
        # `to_epsg()` returns None for those GDAL-only codes, and `epsg_from_user_input`
        # verifies the candidate against the parsed definition (`IsSame`), so a near-match or
        # a relabelled authority is refused rather than silently stored. A genuinely
        # code-less CRS (Robinson, a PROJ4 string) raises `CRSError` there; its WKT comes
        # from the already-resolved `target_crs`, not a second resolution of the input.
        try:
            target_epsg = epsg_from_user_input(to_epsg)
        except CRSError:
            srs = sr_from_wkt(target_crs.to_wkt())
        else:
            srs = sr_from_epsg(target_epsg)
        new_crs_wkt = srs.ExportToWkt()

        new_topo_info = None
        if self._ds._topology_info is not None:
            new_topo_info = replace(self._ds._topology_info, crs_wkt=new_crs_wkt)

        # `type(self._ds)` and `file_name`, as every other derivation does: hard-coding
        # `MeshDataset(` silently downgraded a subclass, and dropping `file_name` made the
        # attribute unreliable rather than absent.
        result = type(self._ds)(
            mesh=new_mesh,
            data_variables=self._ds._data_variables,
            global_attributes=self._ds._global_attributes,
            topology_info=new_topo_info,
            crs_wkt=new_crs_wkt,
            file_name=self._ds._file_name,
            volume_topology=self._ds._volume_topology,
        )
        return result

    def resample(
        self,
        freq: str,
        how: str = "mean",
        *,
        skipna: bool = True,
        q: float | None = None,
    ) -> MeshDataset:
        """Bin the time axis by a calendar frequency and reduce each bin.

        Sugar over ``reduce(how, groupby=freq)``: the mesh counterpart of the raster
        ``resample``. The time coordinates must be date-like, since the bins come from a pandas
        offset alias (``"1MS"``, ``"YS"``, ``"7D"``, …). The time dimension is kept, one step
        per non-empty bin, stamped with **pandas' own label for that bin** — which is the bin's
        start for a start-anchored alias and its end for an end-anchored one. Over steps in
        January, February and March 2020, ``"MS"`` labels the bins 01-01 / 02-01 / 03-01 while
        ``"ME"`` labels them 01-31 / 02-29 / 03-31, and ``"W"`` labels each week by its Sunday.

        Args:
            freq: A pandas offset alias naming the bin width.
            how: The statistic each bin is reduced with. Defaults to ``"mean"``.
            skipna: Skip the declared no-data value and NaN. Defaults to True.
            q: The quantile, required when ``how="quantile"``.

        Returns:
            MeshDataset: A new dataset with one time step per bin.

        Raises:
            ValueError: No variable has a time dimension, the coordinates are not date-like,
                or ``how``/``q`` is invalid.
        """
        return self._ds.reduce(how, skipna=skipna, q=q, groupby=freq)
