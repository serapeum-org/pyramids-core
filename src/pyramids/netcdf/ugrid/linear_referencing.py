"""Linear referencing on a 1D branch network: resolve branch+offset node positions to x/y.

A D-Flow FM / D-HYDRO 1D network stores its computational nodes not as explicit
coordinates but as a **branch index + an along-branch offset** (a chainage). Turning that
pair into x/y is a coordinate derivation from a topology, which per the stack's layering
belongs in pyramids rather than in a file-format driver -- a driver reads the raw arrays,
this places the nodes on the polylines.

:func:`place_nodes_on_network` is the primitive: given each node's branch index and offset,
plus the branches' polyline geometry (a flat vertex list sliced by a per-branch vertex
count), it interpolates each node's x/y along its branch. When the branches declare a
length (``edge_length``) that differs from their digitised polyline's chord length -- a
coarsely digitised branch may -- the offset is treated as a fraction of the declared length,
so a node lands at the same fraction along the polyline rather than at the same raw distance.
"""

from __future__ import annotations

import numpy as np

__all__ = ["place_nodes_on_network"]


def place_nodes_on_network(
    branch_index: np.ndarray,
    offset: np.ndarray,
    geom_x: np.ndarray,
    geom_y: np.ndarray,
    vertices_per_branch: np.ndarray,
    edge_length: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve branch+offset node positions to x/y by linear referencing along the branches.

    Each branch's polyline is sliced from the flat vertex list by the cumulative vertex
    count, the cumulative chord length along its vertices is measured from the branch start,
    and every node on that branch has its x and y linearly interpolated at its offset. A
    single-vertex branch degenerates to that vertex; an offset past a branch's end clamps to
    its last vertex, and a negative offset clamps to its first vertex (:func:`numpy.interp`
    behaviour at both ends).

    In D-HYDRO an offset is a **chainage** in ``[0, edge_length]``, and a branch's declared
    length need not equal its digitised polyline's chord length. When ``edge_length`` is
    given, each branch's distance axis is rescaled from chord length to the declared length,
    so a node's chainage is placed at the same *fraction* along the polyline rather than at
    the same raw distance. Without it, the raw chord length is used (equivalent when the two
    agree, as they do for well-formed files).

    Args:
        branch_index: Per-node branch index, 0-based, shape ``(n_node,)``. (A file's
            ``start_index`` must already have been subtracted by the caller.)
        offset: Per-node distance (chainage) along the branch, shape ``(n_node,)``.
        geom_x: Flat branch-vertex x coordinates, shape ``(n_vertex,)`` -- every branch's
            polyline vertices concatenated in branch order.
        geom_y: Flat branch-vertex y coordinates, shape ``(n_vertex,)``.
        vertices_per_branch: Vertex count of each branch, in branch order, shape
            ``(n_branch,)``. Must sum to ``n_vertex``.
        edge_length: Per-branch declared length used to interpret the chainage, shape
            ``(n_branch,)``, or ``None`` to measure by chord length alone. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: The node ``x`` and ``y`` arrays, each shape
            ``(n_node,)`` and ``float64``.

    Raises:
        ValueError: ``branch_index`` is not a numeric dtype, is not integer-valued (fractional
            or NaN indices), or exceeds the int64 range;
            ``branch_index`` and ``offset`` differ in shape or are not 1-D; ``geom_x`` and
            ``geom_y`` differ in shape or are not 1-D; ``vertices_per_branch`` is not 1-D, is
            negative, or does not sum to ``n_vertex``; ``offset``, ``geom_x`` or ``geom_y`` is
            not finite (NaN/inf); ``edge_length`` is given with a length other than
            ``n_branch``, is not finite, or is negative; a branch index is out of range; or a
            node sits on a branch with no vertices. Malformed input raises rather than yielding
            a silent ``nan`` coordinate.

    Examples:
        Place a node 5 units along a single straight branch running from (0, 0) to (10, 0)::

            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid import place_nodes_on_network
            >>> x, y = place_nodes_on_network(
            ...     np.array([0]),
            ...     np.array([5.0]),
            ...     geom_x=np.array([0.0, 10.0]),
            ...     geom_y=np.array([0.0, 0.0]),
            ...     vertices_per_branch=np.array([2]),
            ... )
            >>> float(x[0]), float(y[0])
            (5.0, 0.0)

        With a declared ``edge_length`` of 20 on a polyline whose chord length is 10, an offset
        of 10 is half the branch -- the polyline midpoint (5, 0), not its end::

            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid import place_nodes_on_network
            >>> x, _ = place_nodes_on_network(
            ...     np.array([0]),
            ...     np.array([10.0]),
            ...     geom_x=np.array([0.0, 10.0]),
            ...     geom_y=np.array([0.0, 0.0]),
            ...     vertices_per_branch=np.array([2]),
            ...     edge_length=np.array([20.0]),
            ... )
            >>> float(x[0])
            5.0
    """
    branch_index = np.asarray(branch_index)
    offset = np.asarray(offset, dtype=np.float64)
    geom_x = np.asarray(geom_x, dtype=np.float64)
    geom_y = np.asarray(geom_y, dtype=np.float64)
    vertices_per_branch = np.asarray(vertices_per_branch)

    if branch_index.ndim != 1 or offset.ndim != 1:
        raise ValueError("branch_index and offset must be 1-D arrays")
    if branch_index.shape != offset.shape:
        raise ValueError(
            f"branch_index and offset must have the same shape; got {branch_index.shape} "
            f"and {offset.shape}"
        )
    # Coerce branch_index to int64 here (after the shape checks, so a 2-D array reports the shape
    # error first). A non-numeric dtype (object/str/bool), a fractional or non-finite value, or a
    # value outside the int64 range is a data error: casting would truncate (0.5 -> 0) or wrap, and
    # the exact-equality branch match would then silently drop the node. Reject it as a ValueError.
    if not np.issubdtype(branch_index.dtype, np.integer):
        if not np.issubdtype(branch_index.dtype, np.floating):
            raise ValueError(
                f"branch_index must be an integer-valued numeric array, got dtype "
                f"{branch_index.dtype}"
            )
        if not np.all(np.isfinite(branch_index)) or not np.all(
            branch_index == np.floor(branch_index)
        ):
            raise ValueError(
                "branch_index must be integer-valued (no fractional or NaN indices)"
            )
        info = np.iinfo(np.int64)
        if branch_index.size and (
            branch_index.min() < info.min or branch_index.max() > info.max
        ):
            raise ValueError("branch_index values exceed the int64 range")
        branch_index = branch_index.astype(np.int64)

    if geom_x.ndim != 1 or geom_y.ndim != 1:
        raise ValueError("geom_x and geom_y must be 1-D arrays")
    if geom_x.shape != geom_y.shape:
        raise ValueError(
            f"geom_x and geom_y must have the same shape; got {geom_x.shape} and "
            f"{geom_y.shape}"
        )
    if vertices_per_branch.ndim != 1:
        raise ValueError("vertices_per_branch must be a 1-D array")
    if np.any(vertices_per_branch < 0):
        raise ValueError("vertices_per_branch must be non-negative")
    if int(vertices_per_branch.sum()) != geom_x.shape[0]:
        raise ValueError(
            f"vertices_per_branch sums to {int(vertices_per_branch.sum())} but the flat "
            f"vertex list has {geom_x.shape[0]} vertices; the counts must tile it exactly"
        )
    # Non-finite data would propagate a silent NaN through np.interp into the returned
    # coordinates; reject it so a bad offset/vertex fails loudly instead.
    if not np.all(np.isfinite(offset)):
        raise ValueError("offset must be finite (no NaN/inf)")
    if not np.all(np.isfinite(geom_x)) or not np.all(np.isfinite(geom_y)):
        raise ValueError("geom_x and geom_y must be finite (no NaN/inf)")
    n_branch = vertices_per_branch.shape[0]
    if edge_length is not None:
        edge_length = np.asarray(edge_length, dtype=np.float64)
        if edge_length.shape != (n_branch,):
            raise ValueError(
                f"edge_length must have shape ({n_branch},) to match vertices_per_branch; "
                f"got {edge_length.shape}"
            )
        if not np.all(np.isfinite(edge_length)):
            raise ValueError("edge_length must be finite (no NaN/inf)")
        if np.any(edge_length < 0):
            raise ValueError("edge_length must be non-negative")
    if branch_index.size and (branch_index.min() < 0 or branch_index.max() >= n_branch):
        raise ValueError(
            f"branch_index out of range: values must be in [0, {n_branch}); got "
            f"[{int(branch_index.min())}, {int(branch_index.max())}]"
        )

    node_x = np.full(branch_index.shape[0], np.nan, dtype=np.float64)
    node_y = np.full(branch_index.shape[0], np.nan, dtype=np.float64)
    starts = np.concatenate(([0], np.cumsum(vertices_per_branch)))
    for index in range(n_branch):
        on_branch = branch_index == index
        if not on_branch.any():
            continue
        begin, end = int(starts[index]), int(starts[index + 1])
        vertex_x, vertex_y = geom_x[begin:end], geom_y[begin:end]
        if vertex_x.size == 0:
            raise ValueError(
                f"node(s) reference branch {index}, which has no geometry vertices"
            )
        segment = np.hypot(np.diff(vertex_x), np.diff(vertex_y))
        distance = np.concatenate(([0.0], np.cumsum(segment)))
        if edge_length is not None and distance[-1] > 0:
            length = float(edge_length[index])
            if length > 0:
                distance = distance * (length / distance[-1])
        here = offset[on_branch]
        node_x[on_branch] = np.interp(here, distance, vertex_x)
        node_y[on_branch] = np.interp(here, distance, vertex_y)
    return node_x, node_y
