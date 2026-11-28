"""Vector engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.vector``).

Owns the vector hand-off family -- turning a mesh and its values into vector geometry:
``to_geodataframe`` / ``to_feature_collection`` (the mesh as points/lines/polygons), ``contour``
(iso-lines), ``cluster`` (value-aware connected regions), ``polygonize`` (dissolve equal-valued
faces), and the Voronoi pair ``voronoi`` / ``voronoi_polygons`` (the node tessellation, raw and
as a vector layer). Mirrors the raster ``Dataset`` vectorisation surface; the per-method maths
live in :mod:`pyramids.netcdf.ugrid.analysis`, :mod:`~pyramids.netcdf.ugrid.graph` and
:mod:`~pyramids.netcdf.ugrid.vector_ops`.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import shapely

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import analysis as _analysis
from pyramids.netcdf.ugrid import graph as _graph
from pyramids.netcdf.ugrid import vector_ops as _vector_ops
from pyramids.netcdf.ugrid.engines._base import _MeshEngine


class _MeshVector(_MeshEngine):
    """Vector hand-off for a mesh: geodataframe/feature-collection, contour, cluster, voronoi."""

    __slots__ = ()

    def to_geodataframe(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> gpd.GeoDataFrame:
        """Convert mesh to a GeoDataFrame.

        For faces: each row is a Polygon with data columns.
        For nodes: each row is a Point.
        For edges: each row is a LineString.

        Args:
            variable_name: Optional data variable to include as a column.
            location: Mesh location ("face", "node", or "edge").

        Returns:
            geopandas GeoDataFrame.
        """
        geometries = self._ds._build_geometries(location)

        data_dict: dict[str, Any] = {}
        if variable_name is not None:
            var = self._ds[variable_name]
            if var.location == location:
                # For a temporal variable only the first step is tabulated; the windowed read
                # pulls just that slab instead of loading every step to slice `[0]` (#982).
                # `has_data_source` keeps a variable with no readable data to the `None` branch
                # below (checked without forcing a load — review L3).
                if var.has_time:
                    var_data = (
                        var._read_time_window(0, None) if var.has_data_source else None
                    )
                else:
                    var_data = var.data
                # A variable with no readable data becomes a length-correct null column rather than
                # raising — pandas rejects a scalar `None` column ("must pass an index") (review L3).
                if var_data is None:
                    var_data = np.full(len(geometries), np.nan)
                data_dict[variable_name] = var_data

        gdf = gpd.GeoDataFrame(data_dict, geometry=geometries)
        if self._ds.crs is not None:
            gdf = gdf.set_crs(self._ds.crs)

        result = gdf
        return result

    def to_feature_collection(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> FeatureCollection:
        """Convert mesh to a pyramids FeatureCollection.

        Args:
            variable_name: Optional data variable to include.
            location: Mesh location ("face", "node", or "edge").

        Returns:
            pyramids FeatureCollection.
        """
        gdf = self.to_geodataframe(variable_name, location)
        result = FeatureCollection(gdf)
        return result

    def contour(
        self,
        levels: Any = None,
        variable_name: str | None = None,
    ) -> Any:
        """Generate contour lines from face/node values.

        Args:
            levels: Contour levels (int for count, array for explicit).
            variable_name: Variable to contour (default: the only one).

        Returns:
            GeoDataFrame with contour ``LineString`` geometries and a
            ``level`` column.

        Raises:
            ValueError: If the variable has more than one time step;
                select a single step first.

        See Also:
            plot_histogram: Distribution of the same variable as a
                histogram.

        """
        var = self._ds._facade_variable("contour", variable_name)
        data = np.asarray(var.data)
        if data.ndim > 1:
            raise ValueError(
                "contour requires a single time step; select a time "
                "step first to reduce temporal data to 1-D."
            )
        result = _analysis.contour_lines(
            self._ds._mesh,
            data,
            levels,
            var.location,
            crs=self._ds.crs,
        )
        return result

    def cluster(
        self,
        lower: float,
        upper: float,
        variable_name: str | None = None,
    ) -> Any:
        """Label connected groups of faces whose values fall in ``[lower, upper]`` (#1379).

        The value-aware sibling of :meth:`connected_components`, and the mesh counterpart of
        the raster ``cluster``. :meth:`connected_components` labels by **topology alone**, so
        it cannot answer "which separate pools of flooded cells are there"; this restricts the
        face adjacency to the faces inside the range before labelling.

        Faces outside the range — and no-data faces, which cannot be inside one — are labelled
        ``0``, so a label is truthy exactly when the face belongs to a cluster. Clusters are
        numbered ``1..k``.

        Args:
            lower: Inclusive lower bound.
            upper: Inclusive upper bound.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset whose variable carries the per-face cluster labels.

        Raises:
            ValueError: The variable is not one value per face (select a step first).

        See Also:
            connected_components: Topology-only labelling, ignoring values.
            footprint: The dissolved outline of the faces carrying valid values.
        """
        var = self._ds._facade_variable("cluster", variable_name)
        labels = _graph.value_clusters(
            self._ds._mesh,
            np.asarray(var.data),
            lower,
            upper,
            var.nodata,
        )
        return self._ds._rewrap(var, labels, nodata=None)

    def voronoi(self) -> Any:
        """The Voronoi tesselation of the mesh nodes (the mesh dual).

        Returns:
            scipy.spatial.Voronoi: The tesselation.
        """
        return _graph.voronoi(self._ds._mesh)

    def voronoi_polygons(
        self, *, clip: FeatureCollection | None = None
    ) -> FeatureCollection:
        """Voronoi (Thiessen) cells of the mesh nodes as a vector `FeatureCollection`.

        The file-serializable counterpart of :meth:`voronoi`, which returns a bare
        :class:`scipy.spatial.Voronoi` with no geospatial form. This returns the Thiessen
        (Voronoi) cells of the mesh nodes as polygons in the mesh CRS, so unlike the raw
        tessellation it can be written with :meth:`FeatureCollection.to_file` and wrapped as a
        CLI verb.

        The cells come from :meth:`FeatureCollection.voronoi`, so the outer (unbounded) cells
        are bounded to an envelope by default; pass `clip` to intersect every cell with a
        boundary instead (e.g. the mesh domain). Each cell carries a `node` column with the
        index of the node it belongs to.

        One cell per **distinct** node: coincident nodes, and cells emptied by `clip`, are
        dropped, so the result can have fewer rows than the mesh has nodes -- use the `node`
        column, not row position, to map a cell back to its node. A `clip` boundary that
        splits one cell into disjoint pieces yields several rows that share a single `node`
        value, so with such a `clip` the result can instead have *more* rows than nodes and
        `node` is not unique; without `clip` (or with a convex one) each surviving node has
        exactly one row. If the mesh has no CRS (:attr:`crs` is `None`) the output carries
        none either.

        Args:
            clip: A boundary `FeatureCollection` each cell is intersected with, or `None` for
                shapely's default bounded cells.

        Returns:
            FeatureCollection: One polygon per surviving node cell, in the mesh CRS, each
            carrying a `node` column with its source node index.

        Raises:
            InvalidGeometryError: Fewer than two distinct nodes with finite coordinates.

        Examples:
            - Tessellate a four-node mesh and inspect the cells:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     epsg=4326,
                ... )
                >>> cells = mesh.vector.voronoi_polygons()
                >>> len(cells)
                4
                >>> sorted(set(cells.geometry.geom_type))
                ['Polygon']
                >>> cells.epsg
                4326

                ```

        See Also:
            voronoi: The low-level `scipy.spatial.Voronoi` primitive this builds on.
        """
        node_index = np.arange(self._ds._mesh.node_x.size)
        nodes = gpd.GeoDataFrame(
            {"node": node_index},
            geometry=list(shapely.points(self._ds._mesh.node_x, self._ds._mesh.node_y)),
            crs=self._ds.crs,
        )
        # Carry the source node index onto each cell (`values="node"`), so a caller can map a
        # surviving cell back to its node -- positional alignment is NOT safe, because the
        # tessellation drops coincident nodes and any cell emptied by `clip`.
        return FeatureCollection(nodes).voronoi(values="node", clip=clip)

    def polygonize(
        self, variable_name: str | None = None, *, decimals: int | None = None
    ) -> Any:
        """Dissolve adjacent faces of equal value into polygons.

        Args:
            variable_name: The face variable to dissolve by, or ``None`` for the only one.
            decimals: Round to this many decimals before grouping, for approximate equality.

        Returns:
            geopandas.GeoDataFrame: One row per merged region, with its value and geometry.
        """
        var = self._ds._facade_variable("polygonize", variable_name)
        return _vector_ops.polygonize(
            self._ds._mesh,
            np.asarray(var.data),
            decimals=decimals,
            crs=self._ds.crs,
        )
