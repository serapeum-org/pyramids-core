"""Topology engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.topology``).

Owns the unstructured-grid operations that have **no raster analogue** -- they read the mesh's
connectivity rather than a regular grid. The geometry/graph family: ``circumcenters`` (per-face
circumcentres), ``exterior_edges`` / ``exterior_faces`` (the boundary), ``bounding_polygon`` (the
dissolved footprint), ``connected_components`` (topology-only face labelling), the morphology pair
``binary_dilation`` / ``binary_erosion`` (grow/shrink a boolean face mask over the adjacency),
``sparse_adjacency`` (the adjacency as a sparse matrix) and ``active_mask`` (a variable's per-element
wet/dry mask). The per-method maths live in :mod:`pyramids.netcdf.ugrid.graph`; this engine marshals
the mesh and its values and hands them over, exactly as the other mesh engines do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids.netcdf.ugrid import graph as _graph
from pyramids.netcdf.ugrid.engines._base import _MeshEngine


class _MeshTopology(_MeshEngine):
    """Unstructured-grid topology for a mesh with no raster analogue (``mesh.topology``)."""

    __slots__ = ()

    def circumcenters(self) -> np.ndarray:
        """The circumcentre of every face (triangular meshes only).

        Returns:
            numpy.ndarray: ``(n_face, 2)`` circumcentre coordinates.

        Raises:
            ValueError: The mesh is not triangular.
        """
        return _graph.circumcenters(self._ds._mesh)

    def exterior_edges(self) -> np.ndarray:
        """The indices of the edges on the mesh boundary.

        Returns:
            numpy.ndarray: Boundary edge indices.
        """
        return _graph.exterior_edges(self._ds._mesh)

    def exterior_faces(self) -> np.ndarray:
        """The indices of the faces touching the mesh boundary.

        Returns:
            numpy.ndarray: Boundary face indices.
        """
        return _graph.exterior_faces(self._ds._mesh)

    def bounding_polygon(self) -> Any:
        """The mesh footprint as one dissolved polygon, holes preserved.

        Returns:
            shapely.geometry.base.BaseGeometry: The (multi)polygon of the mesh outline.
        """
        return _graph.bounding_polygon(self._ds._mesh)

    def connected_components(self) -> np.ndarray:
        """Label the mesh's connected face components.

        Returns:
            numpy.ndarray: One component label per face.
        """
        return _graph.connected_components(self._ds._mesh)

    def binary_dilation(
        self,
        variable_name: str | None = None,
        *,
        iterations: int = 1,
        border_value: bool = False,
        fixed: np.ndarray | None = None,
    ) -> Any:
        """Grow a boolean face mask over its neighbours.

        Args:
            variable_name: The boolean face variable to dilate, or ``None`` for the only one.
            iterations: How many times to grow. Defaults to 1.
            border_value: What lies outside the mesh. Defaults to False.
            fixed: A boolean per-face mask of the cells the result must preserve, whatever the
                dilation would do to them — scipy's ``mask`` argument. ``None`` leaves every
                face free to change. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with the dilated mask.
        """
        var = self._ds._facade_variable("binary_dilation", variable_name)
        grown = _graph.binary_dilation(
            self._ds._mesh,
            np.asarray(var.data).astype(bool),
            iterations=iterations,
            border_value=border_value,
            fixed=fixed,
        )
        return self._ds._rewrap(var, grown)

    def binary_erosion(
        self,
        variable_name: str | None = None,
        *,
        iterations: int = 1,
        border_value: bool = False,
        fixed: np.ndarray | None = None,
    ) -> Any:
        """Shrink a boolean face mask away from its boundary.

        Args:
            variable_name: The boolean face variable to erode, or ``None`` for the only one.
            iterations: How many times to shrink. Defaults to 1.
            border_value: What lies outside the mesh. Defaults to False, which erodes the
                boundary; pass True to erode only the interior.
            fixed: A boolean per-face mask of the cells the result must preserve, whatever the
                erosion would do to them — scipy's ``mask`` argument. ``None`` leaves every
                face free to change. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with the eroded mask.
        """
        var = self._ds._facade_variable("binary_erosion", variable_name)
        shrunk = _graph.binary_erosion(
            self._ds._mesh,
            np.asarray(var.data).astype(bool),
            iterations=iterations,
            border_value=border_value,
            fixed=fixed,
        )
        return self._ds._rewrap(var, shrunk)

    def sparse_adjacency(self, kind: str = "face") -> Any:
        """The mesh adjacency as a sparse matrix.

        Args:
            kind: ``"face"`` (faces sharing an edge) or ``"node"`` (nodes sharing an edge).

        Returns:
            scipy.sparse.csr_matrix: The symmetric adjacency.
        """
        return _graph.sparse_adjacency(self._ds._mesh, kind)

    def active_mask(self, variable_name: str) -> np.typing.NDArray | None:
        """The variable's per-element active (wet/dry) mask, or ``None`` when it has none.

        The first-class companion of the data (issue #1331): ``True`` marks an active/wet
        element, ``False`` a dry one, kept separate from ``nodata``/NaN so a consumer can tell
        a dry element from a missing value. The mask is element-indexed — ``(n_element,)``
        when it is static or ``(n_time, n_element)`` when it varies per step — and is returned
        as read, never applied to the values. Use :meth:`masked_values` to get the values with
        it applied.

        Args:
            variable_name: Name of the data variable.

        Returns:
            np.ndarray | None: The boolean mask, or ``None`` when the variable carries none.

        Raises:
            KeyError: ``variable_name`` is not a data variable of this dataset.

        Examples:
            - A variable with a mask reports it; one without answers ``None``:
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
                >>> ds.topology.active_mask("depth").tolist()
                [True, False]
                >>> ds.topology.active_mask("depth") is not None
                True

                ```
        """
        var = self._ds[variable_name]
        return None if var.active is None else np.asarray(var.active, dtype=bool)
