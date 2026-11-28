"""Analysis engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.analysis``).

Owns the per-element analysis family -- the cell-wise transforms (``apply`` / ``astype`` /
``round`` / ``clip_values`` / ``fillna`` / ``where`` / ``isin`` / ``isnull`` / ``notnull``), the
value comparisons (``equals`` / ``identical``), the focal (neighbourhood) aggregations
(``focal_mean`` / ``focal_std`` / ``focal_apply``), the morphology / coverage helpers (``sieve`` /
``proximity`` / ``footprint``), the spatial fillers (``fill_na_spatial`` / ``laplace_interpolate``),
the sampling / extraction family (``sample`` / ``sample_line`` / ``sample_random`` / ``extract`` /
``zonal_stats``), the domain descriptors (``get_histogram`` / ``domain_area`` /
``count_domain_cells`` / ``masked_values``), the topology clip (``clip``) and the whole-axis time
reductions (``max`` / ``min`` / ``mean`` / ``std`` / ``sum`` / ``var``). It mirrors the raster
``Dataset`` analysis surface; the flat ``mesh.<name>`` methods remain as deprecating facades that
delegate here.

Most methods carry their body here, pulling what they need from the mesh-local kernels
(:mod:`~pyramids.netcdf.ugrid.analysis`, :mod:`~pyramids.netcdf.ugrid.focal`,
:mod:`~pyramids.netcdf.ugrid.fill`, :mod:`~pyramids.netcdf.ugrid.graph`,
:mod:`~pyramids.netcdf.ugrid.vector_ops`, :mod:`~pyramids.netcdf.ugrid.spatial`) and the shared
:mod:`pyramids.base._array_ops` / :mod:`pyramids.base._reductions`, none of which imports
``ugrid.dataset``. A few (``equals`` / ``identical`` / ``masked_values``) lean on names defined in
``ugrid.dataset`` -- the ``MeshDataset`` class itself for the runtime ``isinstance`` check, and the
module-level ``_apply_active_mask`` helper -- so they cannot move their body here without a cycle;
they keep their body on :class:`MeshDataset` as a ``_<name>_impl`` method and this engine delegates
to it. The six reductions forward to the already-migrated ``mesh.selection.reduce`` engine, so they
do not re-emit the selection family's deprecation warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely

from pyramids.base import _array_ops
from pyramids.base._reductions import reduce_by_label
from pyramids.netcdf.ugrid import analysis as _analysis
from pyramids.netcdf.ugrid import fill as _fill
from pyramids.netcdf.ugrid import focal as _focal
from pyramids.netcdf.ugrid import graph as _graph
from pyramids.netcdf.ugrid import vector_ops as _vector_ops
from pyramids.netcdf.ugrid._facades import _KEEP, _representable
from pyramids.netcdf.ugrid.engines._base import _MeshEngine
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex, clip_mesh

if TYPE_CHECKING:
    from pyramids.netcdf.ugrid.dataset import MeshDataset


class _MeshAnalysis(_MeshEngine):
    """Per-element analysis ops for a mesh (``mesh.analysis``)."""

    __slots__ = ()

    def apply(self, func: Any, variable_name: str | None = None) -> Any:
        """Apply ``func`` to every element of a variable (issue #1375).

        The mesh counterpart of the raster ``apply``. ``func`` receives the whole array, so a
        numpy ufunc runs vectorised rather than once per element. The shape must survive, since
        the result still has to line up with the mesh elements it describes.

        Args:
            func: A callable taking and returning an array of the same shape.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset carrying the transformed values.

        Raises:
            ValueError: ``func`` changed the array shape.

        See Also:
            astype: Cast values to another dtype.
            round: Round values to a number of decimal places.
        """
        var = self._ds._facade_variable("apply", variable_name)
        values = np.asarray(var.data)
        domain = _array_ops.notnull(values, var.nodata)
        result, declared = _array_ops.apply_values(
            values, func, domain=domain, sentinel=var.nodata
        )
        return self._ds._rewrap(var, result, nodata=declared)

    def astype(self, dtype: Any, variable_name: str | None = None) -> Any:
        """Cast variable values to a different dtype.

        Args:
            dtype: Target numpy dtype.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with the new dtype.

        See Also:
            round: Round floating-point values before an integer cast.

        """
        var = self._ds._facade_variable("astype", variable_name)
        values = np.asarray(var.data)
        domain = _array_ops.notnull(values, var.nodata)
        result, declared = _array_ops.astype_values(
            values, dtype, domain=domain, sentinel=var.nodata, operation="astype"
        )
        return self._ds._rewrap(var, result, nodata=declared)

    def clip(self, mask: Any, touch: bool = True) -> MeshDataset:
        """Clip the mesh to a polygon mask.

        Selects faces that intersect (touch=True) or are fully
        contained within (touch=False) the mask polygon.

        Args:
            mask: Polygon mask (GeoDataFrame, FeatureCollection,
                or Shapely geometry).
            touch: If True, include faces touching the boundary.

        Returns:
            New MeshDataset with clipped mesh and data.
        """
        mesh, data_variables = clip_mesh(self._ds, mask, touch=touch)
        return self._ds._wrap_subset(mesh, data_variables)

    def clip_values(
        self, lower: Any, upper: Any, variable_name: str | None = None
    ) -> Any:
        """Clamp variable values to ``[lower, upper]``.

        Named ``clip_values`` to avoid collision with the spatial
        :meth:`~pyramids.netcdf.ugrid.dataset.MeshDataset.clip` method.

        Args:
            lower: Minimum allowed value.
            upper: Maximum allowed value.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with clamped values.

        See Also:
            round: Round values to a number of decimal places.

        """
        var = self._ds._facade_variable("clip_values", variable_name)
        values = np.asarray(var.data)
        domain = _array_ops.notnull(values, var.nodata)
        result, declared = _array_ops.clip_values(
            values,
            lower,
            upper,
            domain=domain,
            sentinel=var.nodata,
            operation="clip_values",
        )
        return self._ds._rewrap(var, result, nodata=declared)

    def round(self, decimals: int = 0, variable_name: str | None = None) -> Any:
        """Round variable values to *decimals* decimal places.

        Args:
            decimals: Number of decimal places.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with rounded values.

        See Also:
            clip_values: Clamp values to a range.

        """
        var = self._ds._facade_variable("round", variable_name)
        values = np.asarray(var.data)
        domain = _array_ops.notnull(values, var.nodata)
        result, declared = _array_ops.round_values(
            values, decimals, domain=domain, sentinel=var.nodata, operation="round"
        )
        return self._ds._rewrap(var, result, nodata=declared)

    def fillna(self, value: Any, variable_name: str | None = None) -> Any:
        """Replace missing values with *value*.

        Args:
            value: The fill value.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with missing values replaced.

        See Also:
            isnull: Boolean mask of missing positions.
            fill_na_spatial: Fill gaps from the nearest valid spatial
                element.

        """
        var = self._ds._facade_variable("fillna", variable_name)
        values = np.asarray(var.data)
        domain = _array_ops.notnull(values, var.nodata)
        result, declared = _array_ops.fillna(
            values, value, domain=domain, sentinel=var.nodata
        )
        return self._ds._rewrap(var, result, nodata=declared)

    def fill_na_spatial(
        self, variable_name: str | None = None, *, max_distance: float | None = None
    ) -> Any:
        """Fill each gap from the nearest valid element in space.

        Where :meth:`interpolate_na` fills along time, this fills across the mesh: a gap takes
        the value of the closest element that has one.

        Args:
            variable_name: The variable to fill, or ``None`` for the only one.
            max_distance: Leave a gap unfilled when no valid element is this close. ``None``
                (the default) is unbounded.

        Returns:
            MeshDataset: A new dataset with the gaps filled.
        """
        var = self._ds._facade_variable("fill_na_spatial", variable_name)
        filled = _fill.fill_na_spatial(
            self._ds._mesh,
            np.asarray(var.data),
            var.location,
            max_distance=max_distance,
        )
        return self._ds._rewrap(var, filled)

    def where(
        self,
        cond: Any,
        other: Any = np.nan,
        variable_name: str | None = None,
    ) -> Any:
        """Keep the elements ``cond`` selects and mask the rest (issue #1375).

        The mesh counterpart of the raster ``where``. A ``nodata`` / NaN element is never
        selected — there is no value there to keep — so it goes to ``other`` whatever ``cond``
        says, leaving the result with one masking convention rather than two.

        The per-timestep active mask (:attr:`~pyramids.netcdf.ugrid.models.MeshVariable.active`)
        is **not** applied here. Keeping "dry" distinct from "missing" is the whole point of
        that field (issue #1331), so activity-aware masking is something a caller composes
        explicitly: ``ds.where(ds.active_mask(name) & (values > 0))``.

        Args:
            cond: A boolean array broadcastable to the variable's values.
            other: What an unselected element holds. Defaults to NaN.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with the unselected elements masked.

        See Also:
            clip_values: Clamp values to a range instead of masking outside it.
            masked_values: Apply the active mask to the values, on demand.
        """
        var = self._ds._facade_variable("where", variable_name)
        result = _array_ops.where_values(np.asarray(var.data), cond, other, var.nodata)
        # With a non-NaN `other`, every no-data element is overwritten by it, so the result no
        # longer uses the source's sentinel to mark anything and declaring it would claim a
        # marker that is now an ordinary value. With the default NaN fill the gaps stay
        # missing, so the sentinel still describes them.
        overwrites_gaps = other is not None and not (
            isinstance(other, float) and np.isnan(other)
        )
        return self._ds._rewrap(var, result, nodata=None if overwrites_gaps else _KEEP)

    def isin(self, test_values: Any, variable_name: str | None = None) -> Any:
        """Boolean mask where variable values are in *test_values*.

        Args:
            test_values: Values to test membership against.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with boolean data.
        """
        var = self._ds._facade_variable("isin", variable_name)
        values = np.asarray(var.data)
        domain = _array_ops.notnull(values, var.nodata)
        result = _array_ops.isin_values(values, test_values, domain=domain)
        # A boolean membership mask is a different quantity from the data; it declares no gap.
        return self._ds._rewrap(var, result, nodata=None)

    def isnull(self, variable_name: str | None = None) -> Any:
        """Boolean mask of missing positions.

        Args:
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with boolean data.

        See Also:
            notnull: Inverse mask (non-missing positions).
            fillna: Replace missing values with a constant.

        """
        var = self._ds._facade_variable("isnull", variable_name)
        result = _array_ops.isnull(np.asarray(var.data), var.nodata)
        return self._ds._rewrap(var, result)

    def notnull(self, variable_name: str | None = None) -> Any:
        """Boolean mask of non-missing positions.

        Args:
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with boolean data.

        See Also:
            isnull: Inverse mask (missing positions).

        """
        var = self._ds._facade_variable("notnull", variable_name)
        result = _array_ops.notnull(np.asarray(var.data), var.nodata)
        return self._ds._rewrap(var, result)

    def equals(self, other: object) -> bool:
        """Value equality — same mesh, same variable values at non-missing positions.

        Compares node coordinates, face connectivity, CRS (by EPSG),
        variable names, and per-variable values (ignoring positions
        where either side is nodata/NaN).

        Args:
            other: The object to compare against.  Returns ``False``
                immediately if *other* is not a ``MeshDataset``.

        Returns:
            ``True`` when the two datasets are value-equal.

        See Also:
            identical: Strict comparison that also checks dtype,
                nodata sentinels, CRS WKT, and global attributes.
        """
        return self._ds._equals_impl(other)

    def identical(self, other: object) -> bool:
        """Strict equality — same dtype, same values, same nodata, same attributes.

        Everything must match exactly, including NaN positions, dtypes,
        nodata sentinels, CRS WKT, and global attributes.

        Args:
            other: The object to compare against.  Returns ``False``
                immediately if *other* is not a ``MeshDataset``.

        Returns:
            ``True`` when the two datasets are bitwise-identical.

        See Also:
            equals: Looser comparison that ignores nodata positions,
                dtype differences, and attribute mismatches.
        """
        return self._ds._identical_impl(other)

    def focal_mean(
        self,
        rings: int = 1,
        variable_name: str | None = None,
        reach: Any = None,
    ) -> np.ndarray:
        """The mean of each face's k-ring neighbourhood, centre included (issue #1376).

        The mesh counterpart of the raster ``focal_mean``. The raster version aggregates over a
        ``(2*radius+1)`` **pixel window**, which has no meaning here — faces have neither a
        row/column address nor a fixed neighbour count. This aggregates over the faces within
        ``rings`` hops of the face adjacency instead, which is why the parameter is ``rings``
        and not ``radius``.

        Boundary faces aggregate over whatever neighbours they have: no padding, no fabricated
        values. ``nodata`` faces are excluded, and so are inactive ones when the variable
        carries an active mask — a dry neighbour should not drag a wet face's mean down.

        Args:
            rings: Adjacency hops the neighbourhood spans. Defaults to 1.
            variable_name: Variable to act on (default: the only one).
            reach: A neighbourhood matrix to reuse across calls, built with
                ``pyramids.netcdf.ugrid.focal.neighbourhood(ds.mesh, rings)``. Without it each
                call builds its own, so a sweep over many variables or steps rebuilds the same
                sparse power every time. Leave ``rings`` at its default when passing it.

        Returns:
            numpy.ndarray: Per-face mean of length ``n_face``; ``NaN`` where the neighbourhood
            holds no usable value.

        Raises:
            ValueError: ``rings`` is less than 1, or the variable is not one value per face.

        See Also:
            focal_std: Spread over the same neighbourhood.
            focal_apply: An arbitrary aggregation over the same neighbourhood.
        """
        var, values, active = self._ds._focal_inputs("focal_mean", variable_name)
        return _focal.focal_mean(
            self._ds._mesh,
            values,
            rings=rings,
            nodata=var.nodata,
            active=active,
            reach=reach,
        )

    def focal_std(
        self,
        rings: int = 1,
        variable_name: str | None = None,
        reach: Any = None,
    ) -> np.ndarray:
        """The standard deviation of each face's k-ring neighbourhood (issue #1376).

        The spread counterpart of :meth:`focal_mean`, with the same neighbourhood, exclusion
        and boundary conventions.

        Args:
            rings: Adjacency hops the neighbourhood spans. Defaults to 1.
            variable_name: Variable to act on (default: the only one).
            reach: A neighbourhood matrix to reuse across calls, built with
                ``pyramids.netcdf.ugrid.focal.neighbourhood(ds.mesh, rings)``. Without it each
                call builds its own, so a sweep over many variables or steps rebuilds the same
                sparse power every time. Leave ``rings`` at its default when passing it.

        Returns:
            numpy.ndarray: Per-face population standard deviation; ``NaN`` where the
            neighbourhood holds no usable value.

        Raises:
            ValueError: ``rings`` is less than 1, or the variable is not one value per face.

        See Also:
            focal_mean: The mean over the same neighbourhood.
        """
        var, values, active = self._ds._focal_inputs("focal_std", variable_name)
        return _focal.focal_std(
            self._ds._mesh,
            values,
            rings=rings,
            nodata=var.nodata,
            active=active,
            reach=reach,
        )

    def focal_apply(
        self,
        func: Any,
        rings: int = 1,
        variable_name: str | None = None,
        reach: Any = None,
    ) -> np.ndarray:
        """Aggregate each face's k-ring neighbourhood with ``func`` (issue #1376).

        ``func`` receives the 1-D array of usable values in one neighbourhood and returns a
        scalar, so ``np.ptp``, ``np.median`` or a lambda all work. Unlike :meth:`focal_mean`
        and :meth:`focal_std` this cannot be expressed as a sparse product, so it visits each
        face in turn — noticeably slower on a large mesh, which is the price of an arbitrary
        aggregation.

        Args:
            func: Callable taking a 1-D array and returning a scalar.
            rings: Adjacency hops the neighbourhood spans. Defaults to 1.
            variable_name: Variable to act on (default: the only one).
            reach: A neighbourhood matrix to reuse across calls, built with
                ``pyramids.netcdf.ugrid.focal.neighbourhood(ds.mesh, rings)``. Without it each
                call builds its own, so a sweep over many variables or steps rebuilds the same
                sparse power every time. Leave ``rings`` at its default when passing it.

        Returns:
            numpy.ndarray: Per-face aggregate; ``NaN`` where the neighbourhood holds no usable
            value, so ``func`` is never handed an empty array.

        Raises:
            ValueError: ``rings`` is less than 1, or the variable is not one value per face.

        See Also:
            focal_mean: The vectorised mean, when that is all you need.
        """
        var, values, active = self._ds._focal_inputs("focal_apply", variable_name)
        return _focal.focal_apply(
            self._ds._mesh,
            values,
            func,
            rings=rings,
            nodata=var.nodata,
            active=active,
            reach=reach,
        )

    def sieve(
        self,
        threshold: int,
        variable_name: str | None = None,
    ) -> Any:
        """Remove small connected regions below *threshold* faces.

        Faces in a connected component smaller than *threshold* are
        blanked to NaN.  Operates on the face adjacency graph.

        Args:
            threshold: Minimum component size (face count) to keep.
            variable_name: Variable to sieve (default: the only one).

        Returns:
            MeshDataset: A new dataset with small-region faces
                replaced by NaN.

        Raises:
            ValueError: If the variable has more than one time step;
                select a single step first.

        See Also:
            proximity: Compute graph distances from target faces.

        """
        var = self._ds._facade_variable("sieve", variable_name)
        data = np.asarray(var.data)
        if data.ndim > 1:
            raise ValueError(
                "sieve requires a single time step; select a time "
                "step first to reduce temporal data to 1-D."
            )
        result = _graph.sieve(
            self._ds._mesh,
            data,
            threshold,
        )
        return self._ds._rewrap(var, result)

    def proximity(self, target_faces: Any) -> Any:
        """Graph distance from every face to the nearest target face.

        Uses BFS over the face adjacency graph.

        Args:
            target_faces: 1-D array of face indices that are targets
                (distance zero).

        Returns:
            numpy array of per-face integer distances.  ``-1`` for
            faces in disconnected components with no target.

        See Also:
            sieve: Remove small connected regions by area threshold.
        """
        return _graph.proximity(
            self._ds._mesh,
            np.atleast_1d(np.asarray(target_faces, dtype=np.intp)),
        )

    def footprint(self, variable_name: str | None = None) -> Any:
        """The dissolved outline of the faces carrying valid values (#1379).

        The mesh counterpart of the raster ``footprint`` — "the real coverage of the values".
        Distinct from :meth:`bounding_polygon`, which dissolves **every** face and so describes
        the mesh *domain* rather than where data exists, and from :meth:`polygonize`, which
        returns one polygon per distinct value.

        "Valid" means not ``nodata`` and not NaN. It deliberately does **not** consult the
        per-timestep active mask: a dry element has a real value and is still covered by the
        data, so excluding it would conflate "no data here" with "dry here", which is the
        distinction issue #1331 exists to preserve. For the wet extent, mask first
        (:meth:`masked_values` or :meth:`where`) and take the footprint of that.

        Args:
            variable_name: Variable to act on (default: the only one).

        Returns:
            shapely.Polygon | shapely.MultiPolygon: The dissolved coverage, empty when no face
            carries a valid value. Carries no CRS — attach one if you need it.

        Raises:
            ValueError: The variable is not one value per face (select a step first).

        See Also:
            bounding_polygon: The whole mesh outline, data or not.
            polygonize: One polygon per distinct value.
        """
        var = self._ds._facade_variable("footprint", variable_name)
        return _graph.valid_footprint(
            self._ds._mesh,
            np.asarray(var.data),
            var.nodata,
        )

    def zonal_stats(
        self,
        zones: Any,
        *,
        variable_name: str,
        stats: tuple[str, ...] = ("mean",),
        weighted: bool = True,
        time_index: int = 0,
    ) -> pd.DataFrame:
        """Aggregate a face variable within polygons, weighted by face area.

        The mesh counterpart of :func:`pyramids.dataset.ops._zonal.zonal_stats`. Each face
        is assigned to the zone whose polygon contains its centroid (the mesh analogue of a
        raster pixel's cell centre), then the shared
        :func:`pyramids.base._reductions.reduce_by_label` reduces the face values per zone.
        With ``weighted`` (the default) the **averaging** statistics (``mean`` / ``std`` /
        ``var``) are area-weighted, so a zone's mean is area-exact — a capability the raster
        path does not yet have. ``sum`` is always the plain Σ value (never an area integral),
        and ``count`` / ``min`` / ``max`` are weight-invariant.

        Args:
            zones: Polygons, as a :class:`~pyramids.feature.FeatureCollection` (or any
                object exposing ``geometry``, ``index`` and ``len``). CRS must match the
                mesh; reproject first if not.
            variable_name: Name of a **face-located** data variable.
            stats: Statistics per zone, any of ``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
                ``"std"``, ``"var"``, ``"count"``. Defaults to ``("mean",)``.
            weighted: Area-weight the averaging statistics (``mean`` / ``std`` / ``var``).
                Defaults to True. ``sum`` / ``count`` / ``min`` / ``max`` are unaffected.
            time_index: Step for a temporal variable. Defaults to 0.

        Returns:
            pandas.DataFrame: Indexed like ``zones``; one column per statistic. A zone
            containing no face centroid is NaN (``count`` ``0.0``).

        Raises:
            ValueError: The variable is not face-located, or the zones' CRS disagrees with
                the mesh CRS.
        """
        var, arr = self._ds._element_values(variable_name, time_index=time_index)
        if var.location != "face":
            raise ValueError(
                "mesh zonal_stats needs a face-located variable; "
                f"{variable_name!r} is on {var.location!r}."
            )
        zone_crs = getattr(zones, "crs", None)
        if zone_crs is not None and self._ds.epsg is not None:
            zone_epsg = zone_crs.to_epsg()
            if zone_epsg is not None and zone_epsg != self._ds.epsg:
                raise ValueError(
                    f"zonal_stats: zones CRS (EPSG:{zone_epsg}) does not match mesh CRS "
                    f"(EPSG:{self._ds.epsg}). Reproject the zones to the mesh CRS first."
                )
        geometries = list(zones.geometry)
        cx, cy = self._ds._mesh.face_centroids
        centroids = shapely.points(cx, cy)
        tree = shapely.STRtree(geometries)
        # Face centroid within a zone polygon -> that zone. A centroid in more than one
        # (overlapping zones) keeps the first, mirroring a raster pixel landing in one cell.
        centroid_idx, zone_idx = tree.query(centroids, predicate="within")
        labels = np.full(self._ds._mesh.n_face, -1, dtype=np.intp)
        for face_i, zone_i in zip(centroid_idx, zone_idx):
            if labels[face_i] == -1:
                labels[face_i] = zone_i
        n_zones = len(geometries)
        columns: dict[str, np.typing.NDArray] = {}
        # Area weighting applies only to the averaging statistics. `sum` stays the plain
        # Σ value (a weighted sum would be an area integral — a different quantity, and a
        # silent surprise); `count` / `min` / `max` are weight-invariant anyway.
        averaging = [s for s in stats if s in ("mean", "std", "var")]
        plain = [s for s in stats if s not in ("mean", "std", "var")]
        if weighted and averaging:
            columns.update(
                reduce_by_label(
                    arr, labels, n_zones, averaging, weights=self._ds._mesh.face_areas
                )
            )
            if plain:
                columns.update(reduce_by_label(arr, labels, n_zones, plain))
        else:
            columns.update(reduce_by_label(arr, labels, n_zones, list(stats)))
        return pd.DataFrame({stat: columns[stat] for stat in stats}, index=zones.index)

    def sample(
        self,
        variable_name: str,
        x: float | np.ndarray,
        y: float | np.ndarray,
        *,
        method: str = "contains",
        time_index: int = 0,
    ) -> np.typing.NDArray:
        """Sample a mesh variable at point locations.

        The mesh counterpart of :meth:`pyramids.dataset.Dataset.sample`. Where the raster
        version inverts an affine geotransform to find the pixel under each point, a mesh
        resolves the point to an element through :class:`MeshSpatialIndex`: a face-located
        variable by which face contains the point (``method="contains"``) or the nearest
        face centroid (``method="nearest"``); a node-located variable by nearest node.

        Args:
            variable_name: Name of the data variable.
            x: Query x-coordinate(s), in the mesh CRS.
            y: Query y-coordinate(s), in the mesh CRS.
            method: ``"contains"`` (point-in-face, face variables only) or ``"nearest"``.
                Defaults to ``"contains"``.
            time_index: Step for a temporal variable. Defaults to 0.

        Returns:
            numpy.ndarray: One value per query point; NaN where a ``"contains"`` query
            falls outside every face.

        Raises:
            ValueError: An unknown ``method``, ``"contains"`` on a non-face variable, or a
                variable located on edges (unsupported).
        """
        var, arr = self._ds._element_values(variable_name, time_index=time_index)
        xs = np.atleast_1d(np.asarray(x, dtype="float64"))
        ys = np.atleast_1d(np.asarray(y, dtype="float64"))
        index = MeshSpatialIndex(self._ds._mesh)
        if var.location == "face":
            if method == "contains":
                idx = index.locate_faces(xs, ys)
            elif method == "nearest":
                idx = np.atleast_1d(index.locate_nearest_face(xs, ys)).ravel()
            else:
                raise ValueError(
                    f"sample method must be 'contains' or 'nearest', got {method!r}"
                )
        elif var.location == "node":
            # A node is a point, so "contains" has no meaning here — only "nearest"
            # applies. Reject "contains" rather than silently doing a nearest lookup, as
            # the docstring promises.
            if method == "contains":
                raise ValueError(
                    "sample method='contains' is only valid for a face-located variable; "
                    f"{variable_name!r} is on 'node' — use method='nearest'."
                )
            if method != "nearest":
                raise ValueError(
                    f"sample method must be 'contains' or 'nearest', got {method!r}"
                )
            idx = np.atleast_1d(index.locate_nearest_node(xs, ys)).ravel()
        else:
            raise ValueError(
                f"sample supports 'face' and 'node' variables; {variable_name!r} is on "
                f"{var.location!r}."
            )
        out = np.full(len(xs), np.nan, dtype="float64")
        inside = idx >= 0
        out[inside] = arr[idx[inside]]
        return out

    def sample_line(self, line: Any, variable_name: str | None = None) -> Any:
        """Sample a variable along a line, ordered by distance along it.

        One sample per *crossing*, at the midpoint of each part of the line-face intersection.
        A line that leaves a face and re-enters it is sampled once per entry, so
        ``face_indices`` can repeat -- see :func:`pyramids.netcdf.ugrid.vector_ops.sample_line`.

        Args:
            line: A shapely ``LineString``, or a sequence of coordinates.
            variable_name: The variable to sample, or ``None`` for the only one.

        Returns:
            tuple: ``(distances, values, face_indices)``, one entry per crossing in
            along-line order. ``face_indices`` is not unique when the line re-enters a face.

        Raises:
            ValueError: ``variable_name`` is ``None`` while the dataset holds no variable or
                more than one, or the variable's last axis does not match ``n_face``.
        """
        var = self._ds._facade_variable("sample_line", variable_name)
        return _vector_ops.sample_line(
            self._ds._mesh,
            np.asarray(var.data),
            line,
        )

    def sample_random(
        self,
        n: int,
        variable_name: str | None = None,
        seed: int | None = None,
    ) -> tuple:
        """Draw *n* spatially random samples from the mesh domain.

        Args:
            n: Number of samples.
            variable_name: Variable to sample (default: the only one).
            seed: RNG seed for reproducibility.

        Returns:
            ``(x, y, values)`` — arrays of length *n*.

        See Also:
            extract: Sample at specific coordinates rather than random
                ones.

        """
        var = self._ds._facade_variable("sample_random", variable_name)
        result = _analysis.sample_random(
            self._ds._mesh,
            np.asarray(var.data),
            n,
            seed,
        )
        return result

    def extract(
        self,
        x: Any,
        y: Any,
        variable_name: str | None = None,
    ) -> Any:
        """Sample variable values at arbitrary (x, y) coordinates.

        Uses an STRtree spatial index to locate the containing face
        for each query point.

        Args:
            x: Query x-coordinates (scalar or array).
            y: Query y-coordinates (scalar or array).
            variable_name: Variable to extract (default: the only one).

        Returns:
            numpy array of extracted values.  Points outside the mesh
            get ``NaN``.

        See Also:
            sample_random: Draw spatially random samples instead of
                querying specific coordinates.
            sample: Interpolated point sampling (existing method on
                MeshDataset).
        """
        var = self._ds._facade_variable("extract", variable_name)
        result = _analysis.extract_points(
            self._ds._mesh,
            np.asarray(var.data),
            np.atleast_1d(np.asarray(x, dtype=float)),
            np.atleast_1d(np.asarray(y, dtype=float)),
            var.location,
        )
        return result

    def get_histogram(
        self,
        bins: int = 256,
        value_range: tuple[float, float] | None = None,
        variable_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The variable's value distribution as data (issue #1375).

        The mesh counterpart of the raster ``get_histogram``. :meth:`plot_histogram` could
        already *draw* the distribution but gave no way to obtain it; both mask no-data through
        the same helper, so these counts describe exactly the samples those bars show.

        Args:
            bins: Number of equal-width bins. Defaults to 256.
            value_range: ``(low, high)`` to bin within, or ``None`` to span the data.
            variable_name: Variable to act on (default: the only one).

        Returns:
            tuple: ``(counts, edges)`` -- per-bin counts and the ``bins + 1`` edges, in the
            variable's own units.

        Raises:
            ValueError: Nothing survives the no-data masking, so there is no distribution.

        See Also:
            plot_histogram: The same distribution, drawn.
            summary: Per-variable min / max / mean / std over the whole mesh.
        """
        var = self._ds._facade_variable("get_histogram", variable_name)
        return _analysis.histogram(np.asarray(var.data), var.nodata, bins, value_range)

    def domain_area(self) -> float:
        """Total area of the mesh domain.

        Returns:
            float: Sum of all face areas in the mesh's coordinate
                units squared.

        See Also:
            count_domain_cells: Count of active faces rather than
                their total area.
        """
        return float(self._ds._mesh.face_areas.sum())

    def count_domain_cells(self, variable_name: str | None = None) -> int:
        """Number of active (non-masked) mesh faces.

        When the variable carries no active mask, all faces are counted.

        Args:
            variable_name: Variable whose active mask to consult.
                ``None`` counts all faces.

        Returns:
            int: Active face count.

        See Also:
            domain_area: Total area rather than face count.

        """
        result = int(self._ds._mesh.n_face)
        if variable_name is not None:
            mask = self._ds.topology.active_mask(variable_name)
            if mask is not None:
                if mask.ndim > 1:
                    mask = mask[-1]
                result = int(np.count_nonzero(mask))
        return result

    def masked_values(self, variable_name: str) -> np.typing.NDArray:
        """The variable's values with both ``nodata`` **and** the active mask blanked to NaN.

        The **opt-in** counterpart of a bare ``.data`` read (issue #1331): dry elements (where
        :meth:`active_mask` is ``False``) become NaN on top of the usual ``nodata`` blanking,
        so a caller that wants "treat dry as missing" gets it explicitly — the values are
        never masked by default. Keeps the variable's full shape (every time step and layer),
        unlike the single-step :meth:`sample` / :meth:`weighted` path. When the variable has no
        mask the result is just the ``nodata``-blanked values.

        Args:
            variable_name: Name of the data variable.

        Returns:
            np.ndarray: The float64 values, dry and missing elements alike as NaN.

        Raises:
            KeyError: ``variable_name`` is not a data variable of this dataset.
            ValueError: The variable has no loaded data.

        Examples:
            - Dry elements become NaN, so a mean over the wet cells is straightforward:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> ds = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([2.0, 4.0])},
                ... )
                >>> ds["depth"].active = np.array([True, False])
                >>> ds.analysis.masked_values("depth").tolist()
                [2.0, nan]
                >>> float(np.nanmean(ds.analysis.masked_values("depth")))
                2.0

                ```
        """
        return self._ds._masked_values_impl(variable_name)

    def laplace_interpolate(
        self,
        variable_name: str | None = None,
        *,
        use_weights: bool = True,
        direct_solve: bool = False,
    ) -> Any:
        """Fill the interior gaps by solving Laplace's equation over the mesh.

        The valid values are the boundary condition and each gap becomes the smooth solution
        between them — a better hole-filler than nearest when the field varies smoothly.

        Args:
            variable_name: The variable to fill, or ``None`` for the only one.
            use_weights: Weight the graph Laplacian by inverse distance. Defaults to True.
            direct_solve: Solve directly rather than iteratively. Defaults to False.

        Returns:
            MeshDataset: A new dataset with the interior gaps filled.
        """
        var = self._ds._facade_variable("laplace_interpolate", variable_name)
        filled = _fill.laplace_interpolate(
            self._ds._mesh,
            np.asarray(var.data),
            var.location,
            use_weights=use_weights,
            direct_solve=direct_solve,
        )
        return self._ds._rewrap(var, filled)

    def max(self, *, skipna: bool = True) -> MeshDataset:
        """Maximum over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            MeshDataset: A new dataset; each temporal variable collapsed to its time maximum.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds.selection.reduce("max", skipna=skipna)

    def min(self, *, skipna: bool = True) -> MeshDataset:
        """Minimum over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            MeshDataset: A new dataset; each temporal variable collapsed to its time minimum.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds.selection.reduce("min", skipna=skipna)

    def mean(self, *, skipna: bool = True) -> MeshDataset:
        """Mean over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            MeshDataset: A new dataset; each temporal variable collapsed to its time mean.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds.selection.reduce("mean", skipna=skipna)

    def std(self, *, skipna: bool = True) -> MeshDataset:
        """Std-dev over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            MeshDataset: A new dataset; each temporal variable collapsed to its time std.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds.selection.reduce("std", skipna=skipna)

    def sum(self, *, skipna: bool = True) -> MeshDataset:
        """Sum over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            MeshDataset: A new dataset; each temporal variable collapsed to its time sum.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds.selection.reduce("sum", skipna=skipna)

    def var(self, *, skipna: bool = True) -> MeshDataset:
        """Variance over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            MeshDataset: A new dataset; each temporal variable collapsed to its time variance.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds.selection.reduce("var", skipna=skipna)

    def to_location(self, location: str, variable_name: str | None = None) -> Any:
        """Remap a variable onto another facet of the mesh.

        Averages the values of the contributing elements, so a face field asked for on the
        nodes becomes each node's mean of the faces around it.

        Args:
            location: The facet to remap onto — ``"node"``, ``"edge"`` or ``"face"``.
            variable_name: The variable to remap, or ``None`` for the only one.

        Returns:
            MeshDataset: A new dataset whose variable sits on ``location``.

        Raises:
            ValueError: An unknown ``location``, or a remap the kernel does not offer.
        """
        var = self._ds._facade_variable("to_location", variable_name)
        remap = {"node": _graph.to_node, "edge": _graph.to_edge, "face": _graph.to_face}
        if location not in remap:
            raise ValueError(
                f"to_location: location must be one of {sorted(remap)}, got {location!r}."
            )
        if var.location == location:
            # Already there. The averaging kernels only accept the two *other* facets as a
            # source, so without this an identity remap failed with "source must be 'node' or
            # 'edge'" -- a confusing answer to a request that is trivially already satisfied.
            return self._ds._rebuild(dict(self._ds._data_variables))
        values = remap[location](
            self._ds._mesh,
            np.asarray(var.data),
            source=var.location,
        )
        moved = var.with_data(np.asarray(values))
        moved.location = location
        # An element with no contributor (an orphan node, say) averages to NaN, so the remapped
        # variable has to declare NaN as its no-data value -- the same reasoning the regrid facade
        # already applies. Otherwise `to_file` writes those NaNs under the old `_FillValue`.
        # Through the same check `_rewrap` applies, rather than declared unconditionally.
        # `graph._average_over` answers float64 today, so NaN is always representable -- but a
        # future integer-preserving remap would reintroduce exactly the defect `_rewrap` guards.
        moved_values = np.asarray(moved.data)
        if _representable(float("nan"), moved_values.dtype):
            moved.nodata = float("nan")
        if moved.dimensions:
            # The element axis is the last dimension, by the mesh-variable invariant. Renaming it
            # to the new facet matters because a file-read variable carries real dimension names
            # and the xarray bridge prefers them: leaving the old facet's name on an axis that now
            # has the new facet's length makes the export fail on conflicting dimension sizes.
            mesh_name = self._ds.mesh_name
            moved.dimensions = moved.dimensions[:-1] + (
                f"{mesh_name}_n{location.capitalize()}s",
            )
        new_vars = dict(self._ds._data_variables)
        new_vars[var.name] = moved
        return self._ds._rebuild(new_vars)
