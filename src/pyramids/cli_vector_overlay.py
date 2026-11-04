"""Binary/overlay and spatial-predicate-filter commands for ``pyramids vector``.

Thin, scriptable wrappers over the overlay, set-theoretic, and spatial-join
methods of :class:`pyramids.feature.FeatureCollection`, in the same shape as the
other vector commands: an overwrite guard, a one-line-error contract (handlers
raise; :func:`pyramids.cli.main` reports), and ``SRC OUT`` positionals. No
geometry logic lives here — each command maps 1:1 onto an existing
``FeatureCollection`` call and the result is re-wrapped as a new collection
before it is written.
"""

from __future__ import annotations

import argparse

from geopandas import GeoDataFrame, GeoSeries

from pyramids._cli_shared import (
    info_print,
    map_lookup_errors,
    refuse_existing,
    require_matching_crs,
    resolve_vector_driver,
)
from pyramids.feature import FeatureCollection

_HELP_SRC_VECTOR = "source vector path"
_HELP_OUT_VECTOR = "destination vector path"
_HELP_LAYER = "layer name/index for a multi-layer source"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_OTHER = "second vector path (the right-hand operand)"
_HELP_RIGHT = "right-hand vector path to combine with the source"

_OVERLAY_HOW = (
    "intersection",
    "union",
    "identity",
    "symmetric_difference",
    "difference",
)
_JOIN_HOW = ("inner", "left", "right")
_SJOIN_PREDICATES = (
    "intersects",
    "contains",
    "within",
    "covers",
    "covered_by",
    "crosses",
    "overlaps",
    "touches",
)
_FILTER_PREDICATES = (
    "contains",
    "within",
    "intersects",
    "covers",
    "covered_by",
    "crosses",
    "disjoint",
    "overlaps",
    "touches",
    "dwithin",
)


def _to_collection(result: object, src_crs: object) -> FeatureCollection:
    """Re-wrap the result of a ``FeatureCollection`` operation as a collection.

    The underlying geopandas operations return different shapes — a boolean-free
    ``GeoDataFrame`` (``overlay``), a ``GeoSeries`` of geometries (``intersection``,
    ``clip_by_rect``, …), or a single shapely geometry (``union_all``,
    ``intersection_all``). This normalises any of them into a writable
    :class:`FeatureCollection`.

    Args:
        result: The value returned by the operation.
        src_crs: The source collection's CRS, used when ``result`` is a bare
            scalar geometry that carries none of its own.

    Returns:
        FeatureCollection: A collection ready for :meth:`FeatureCollection.to_file`.
    """
    if isinstance(result, FeatureCollection):
        collection = result
    elif isinstance(result, GeoDataFrame):
        collection = FeatureCollection(result)
    elif isinstance(result, GeoSeries):
        collection = FeatureCollection(
            GeoDataFrame(geometry=result.values, crs=result.crs)
        )
    else:
        collection = FeatureCollection(GeoDataFrame(geometry=[result], crs=src_crs))
    return collection


def _write_collection(args: argparse.Namespace, result: object, src_crs: object) -> int:
    """Wrap ``result`` and write it to ``args.output`` under the overwrite guard.

    Args:
        args: Parsed args carrying ``output`` and ``overwrite``.
        result: The value returned by a ``FeatureCollection`` operation.
        src_crs: The source collection's CRS (used for scalar-geometry results).

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, None)
    _to_collection(result, src_crs).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_intersection(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector intersection`` — element-wise geometry intersection.

    Args:
        args: Parsed args with ``src``, ``output``, ``other``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    return _write_collection(args, fc.intersection(other), fc.crs)


def _cmd_vector_difference(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector difference`` — element-wise geometry difference.

    Args:
        args: Parsed args with ``src``, ``output``, ``other``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    return _write_collection(args, fc.difference(other), fc.crs)


def _cmd_vector_symmetric_difference(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector symmetric-difference`` — element-wise symmetric difference.

    Args:
        args: Parsed args with ``src``, ``output``, ``other``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    return _write_collection(args, fc.symmetric_difference(other), fc.crs)


def _cmd_vector_union(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector union`` — element-wise geometry union.

    Args:
        args: Parsed args with ``src``, ``output``, ``other``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    return _write_collection(args, fc.union(other), fc.crs)


def _cmd_vector_intersection_all(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector intersection-all`` — intersection of every feature.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    return _write_collection(args, fc.intersection_all(), fc.crs)


def _cmd_vector_union_all(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector union-all`` — the single union of every feature.

    Args:
        args: Parsed args with ``src``, ``output``, ``method``, ``grid_size``,
            ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.union_all(method=args.method, grid_size=args.grid_size)
    return _write_collection(args, result, fc.crs)


def _cmd_vector_overlay(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector overlay`` — a spatial overlay of two layers.

    Args:
        args: Parsed args with ``src``, ``output``, ``right``, ``how``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    right = FeatureCollection.read_file(args.right)
    require_matching_crs(fc, right)
    return _write_collection(args, fc.overlay(right, how=args.how), fc.crs)


def _cmd_vector_dissolve(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector dissolve`` — merge features, optionally grouped.

    Args:
        args: Parsed args with ``src``, ``output``, ``by``, ``aggfunc``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    # A bad --by column raises KeyError, which main() does not catch; map it to a
    # clean ValueError so a typo'd column is a one-line error, not a crash.
    with map_lookup_errors():
        result = fc.dissolve(by=args.by, aggfunc=args.aggfunc)
    return _write_collection(args, result, fc.crs)


def _cmd_vector_sjoin(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector sjoin`` — a predicate-based spatial join.

    Args:
        args: Parsed args with ``src``, ``output``, ``right``, ``how``,
            ``predicate``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    right = FeatureCollection.read_file(args.right)
    require_matching_crs(fc, right)
    result = fc.sjoin(right, how=args.how, predicate=args.predicate)
    return _write_collection(args, result, fc.crs)


def _cmd_vector_sjoin_nearest(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector sjoin-nearest`` — a nearest-neighbour spatial join.

    Args:
        args: Parsed args with ``src``, ``output``, ``right``, ``how``,
            ``max_distance``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    right = FeatureCollection.read_file(args.right)
    require_matching_crs(fc, right)
    result = fc.sjoin_nearest(right, how=args.how, max_distance=args.max_distance)
    return _write_collection(args, result, fc.crs)


def _cmd_vector_clip_by_rect(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector clip-by-rect`` — clip to an axis-aligned window.

    Args:
        args: Parsed args with ``src``, ``output``, ``rect``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    xmin, ymin, xmax, ymax = args.rect
    return _write_collection(args, fc.clip_by_rect(xmin, ymin, xmax, ymax), fc.crs)


def _cmd_vector_shared_paths(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector shared-paths`` — shared lineal paths of two layers.

    Args:
        args: Parsed args with ``src``, ``output``, ``other``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    return _write_collection(args, fc.shared_paths(other), fc.crs)


def _cmd_vector_shortest_line(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector shortest-line`` — the shortest line between pairs.

    Args:
        args: Parsed args with ``src``, ``output``, ``other``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    return _write_collection(args, fc.shortest_line(other), fc.crs)


def _cmd_vector_explode(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector explode`` — split multi-part geometries into parts.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    return _write_collection(args, fc.explode(), fc.crs)


def _cmd_vector_filter(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector filter`` — keep features matching a mask predicate.

    The mask's geometries are combined into one geometry; the chosen spatial
    predicate is evaluated for every source feature against it, and the features
    whose predicate is ``True`` are kept.

    Args:
        args: Parsed args with ``src``, ``output``, ``predicate``, ``mask``,
            ``distance``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--predicate dwithin`` was chosen without ``--distance``.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    mask = FeatureCollection.read_file(args.mask)
    require_matching_crs(fc, mask)
    mask_geometry = mask.union_all()
    predicate = getattr(fc, args.predicate)
    if args.predicate == "dwithin":
        if args.distance is None:
            raise ValueError("--distance is required for the 'dwithin' predicate.")
        keep = predicate(mask_geometry, args.distance)
    else:
        keep = predicate(mask_geometry)
    return _write_collection(args, FeatureCollection(fc[keep]), fc.crs)


def _add_src_out(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``SRC OUT [--layer] [--overwrite]`` arguments to ``parser``.

    Args:
        parser: The sub-command parser to populate.
    """
    parser.add_argument("src", help=_HELP_SRC_VECTOR)
    parser.add_argument("output", help=_HELP_OUT_VECTOR)
    parser.add_argument("--layer", help=_HELP_LAYER)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)


def add_vector_overlay_commands(vsub: argparse._SubParsersAction) -> None:
    """Register the overlay / set-theoretic / filter commands on the vector group.

    Args:
        vsub: The ``vector`` group's ``add_subparsers`` action.
    """
    for name, handler in (
        ("intersection", _cmd_vector_intersection),
        ("difference", _cmd_vector_difference),
        ("symmetric-difference", _cmd_vector_symmetric_difference),
        ("union", _cmd_vector_union),
        ("shared-paths", _cmd_vector_shared_paths),
        ("shortest-line", _cmd_vector_shortest_line),
    ):
        parser = vsub.add_parser(name, help=f"{name.replace('-', ' ')} with --other")
        _add_src_out(parser)
        parser.add_argument("--other", required=True, help=_HELP_OTHER)
        parser.set_defaults(func=handler)

    v_inter_all = vsub.add_parser(
        "intersection-all", help="intersection of all features"
    )
    _add_src_out(v_inter_all)
    v_inter_all.set_defaults(func=_cmd_vector_intersection_all)

    v_union_all = vsub.add_parser("union-all", help="the single union of all features")
    _add_src_out(v_union_all)
    v_union_all.add_argument(
        "--method",
        choices=("unary", "coverage"),
        default="unary",
        help="union algorithm",
    )
    v_union_all.add_argument(
        "--grid-size", type=float, default=None, help="snapping grid size for the union"
    )
    v_union_all.set_defaults(func=_cmd_vector_union_all)

    v_overlay = vsub.add_parser("overlay", help="spatial overlay of two layers")
    _add_src_out(v_overlay)
    v_overlay.add_argument("--right", required=True, help=_HELP_RIGHT)
    v_overlay.add_argument(
        "--how", choices=_OVERLAY_HOW, default="intersection", help="overlay mode"
    )
    v_overlay.set_defaults(func=_cmd_vector_overlay)

    v_dissolve = vsub.add_parser("dissolve", help="merge features, optionally grouped")
    _add_src_out(v_dissolve)
    v_dissolve.add_argument("--by", help="column to group by before dissolving")
    v_dissolve.add_argument(
        "--aggfunc", default="first", help="aggregation for the non-geometry columns"
    )
    v_dissolve.set_defaults(func=_cmd_vector_dissolve)

    v_sjoin = vsub.add_parser("sjoin", help="predicate-based spatial join")
    _add_src_out(v_sjoin)
    v_sjoin.add_argument("--right", required=True, help=_HELP_RIGHT)
    v_sjoin.add_argument("--how", choices=_JOIN_HOW, default="inner", help="join type")
    v_sjoin.add_argument(
        "--predicate",
        choices=_SJOIN_PREDICATES,
        default="intersects",
        help="spatial predicate for the join",
    )
    v_sjoin.set_defaults(func=_cmd_vector_sjoin)

    v_sjoin_near = vsub.add_parser(
        "sjoin-nearest", help="nearest-neighbour spatial join"
    )
    _add_src_out(v_sjoin_near)
    v_sjoin_near.add_argument("--right", required=True, help=_HELP_RIGHT)
    v_sjoin_near.add_argument(
        "--how", choices=_JOIN_HOW, default="inner", help="join type"
    )
    v_sjoin_near.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="only join within this distance",
    )
    v_sjoin_near.set_defaults(func=_cmd_vector_sjoin_nearest)

    v_clip_rect = vsub.add_parser("clip-by-rect", help="clip to an axis-aligned window")
    _add_src_out(v_clip_rect)
    v_clip_rect.add_argument(
        "--rect",
        nargs=4,
        type=float,
        required=True,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="clip window in the source CRS",
    )
    v_clip_rect.set_defaults(func=_cmd_vector_clip_by_rect)

    v_explode = vsub.add_parser(
        "explode", help="split multi-part geometries into parts"
    )
    _add_src_out(v_explode)
    v_explode.set_defaults(func=_cmd_vector_explode)

    v_filter = vsub.add_parser("filter", help="keep features matching a mask predicate")
    _add_src_out(v_filter)
    v_filter.add_argument(
        "--predicate",
        choices=_FILTER_PREDICATES,
        required=True,
        help="spatial predicate",
    )
    v_filter.add_argument(
        "--mask", required=True, help="vector whose geometry is the mask"
    )
    v_filter.add_argument(
        "--distance",
        type=float,
        default=None,
        help="distance for the 'dwithin' predicate",
    )
    v_filter.set_defaults(func=_cmd_vector_filter)
