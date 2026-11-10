"""Unary geometry-operation commands for the ``pyramids vector`` group.

Thin, scriptable wrappers over the element-wise geometry methods of
:class:`pyramids.feature.FeatureCollection` (buffer, simplify, make-valid, the
affine transforms, …). Each command reads a source vector, applies one operation
that returns a new geometry series, and writes the result to a fresh output —
driver inferred from the extension, guarded by the same overwrite check and
one-line-error contract as the rest of the CLI. No geometry logic lives here;
every command maps 1:1 onto an existing ``FeatureCollection`` call.
"""

from __future__ import annotations

import argparse

import geopandas as gpd

from pyramids._cli_shared import (
    info_print,
    refuse_existing,
    require_matching_crs,
    resolve_vector_driver,
)
from pyramids.feature import FeatureCollection

_HELP_SRC = "source vector path"
_HELP_OUT = "destination vector path"
_HELP_LAYER = "layer name/index for multi-layer sources"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_DRIVER = "OGR driver (default: inferred from extension)"


def _write_result(args: argparse.Namespace, result: object) -> None:
    """Write a geometry-operation result to ``args.output`` as a new vector.

    The element-wise ``FeatureCollection`` operations return a bare
    :class:`geopandas.GeoSeries`; it is wrapped in a ``FeatureCollection`` so the
    class's extension-aware :meth:`to_file` (with its driver catalog) is used, the
    driver being resolved from the output extension exactly as ``vector convert``
    does.

    Args:
        args: Parsed args carrying ``output`` and ``overwrite``.
        result: A ``FeatureCollection`` or ``GeoSeries`` produced by the operation.
    """
    collection = (
        result
        if isinstance(result, FeatureCollection)
        else FeatureCollection(gpd.GeoDataFrame(geometry=result))
    )
    driver = resolve_vector_driver(args.output, None)
    collection.to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")


def _cmd_vector_buffer(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector buffer`` — buffer every geometry by a distance.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``distance``, ``resolution``, ``cap_style``, ``join_style``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.buffer(
        args.distance,
        resolution=args.resolution,
        cap_style=args.cap_style,
        join_style=args.join_style,
    )
    _write_result(args, result)
    return 0


def _cmd_vector_simplify(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector simplify`` — Douglas-Peucker simplification.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``tolerance``, ``no_preserve_topology``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.simplify(
        args.tolerance, preserve_topology=not args.no_preserve_topology
    )
    _write_result(args, result)
    return 0


def _cmd_vector_simplify_coverage(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector simplify-coverage`` — topology-preserving coverage simplify.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``tolerance``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.simplify_coverage(args.tolerance)
    _write_result(args, result)
    return 0


def _cmd_vector_make_valid(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector make-valid`` — repair invalid geometries.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.make_valid()
    _write_result(args, result)
    return 0


def _cmd_vector_normalize(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector normalize`` — normalize geometries to canonical form.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.normalize()
    _write_result(args, result)
    return 0


def _cmd_vector_reverse(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector reverse`` — reverse coordinate (winding) order.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.reverse()
    _write_result(args, result)
    return 0


def _cmd_vector_force_2d(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector force-2d`` — drop the Z coordinate.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.force_2d()
    _write_result(args, result)
    return 0


def _cmd_vector_force_3d(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector force-3d`` — add a constant Z coordinate.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``, ``z``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.force_3d(z=args.z)
    _write_result(args, result)
    return 0


def _cmd_vector_segmentize(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector segmentize`` — densify edges to a max segment length.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``max_length``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.segmentize(args.max_length)
    _write_result(args, result)
    return 0


def _cmd_vector_remove_repeated_points(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector remove-repeated-points`` — drop duplicate vertices.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``tolerance``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.remove_repeated_points(args.tolerance)
    _write_result(args, result)
    return 0


def _cmd_vector_offset_curve(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector offset-curve`` — offset a line by a distance.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``distance``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.offset_curve(args.distance)
    _write_result(args, result)
    return 0


def _cmd_vector_representative_point(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector representative-point`` — a point guaranteed inside each geometry.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.representative_point()
    _write_result(args, result)
    return 0


def _cmd_vector_concave_hull(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector concave-hull`` — the concave hull of each geometry.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``ratio``, ``allow_holes``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.concave_hull(ratio=args.ratio, allow_holes=args.allow_holes)
    _write_result(args, result)
    return 0


def _cmd_vector_minimum_bounding_circle(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector minimum-bounding-circle`` — smallest enclosing circle.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.minimum_bounding_circle()
    _write_result(args, result)
    return 0


def _cmd_vector_minimum_rotated_rectangle(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector minimum-rotated-rectangle`` — smallest enclosing rectangle.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.minimum_rotated_rectangle()
    _write_result(args, result)
    return 0


def _cmd_vector_maximum_inscribed_circle(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector maximum-inscribed-circle`` — largest circle fitting inside.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``tolerance``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.maximum_inscribed_circle(tolerance=args.tolerance)
    _write_result(args, result)
    return 0


def _cmd_vector_affine_transform(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector affine-transform`` — apply a 2-D affine matrix.

    The matrix is the six-number form ``[a, b, d, e, xoff, yoff]`` consumed by
    ``shapely.affinity.affine_transform``.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``matrix``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.affine_transform(args.matrix)
    _write_result(args, result)
    return 0


def _cmd_vector_rotate(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector rotate`` — rotate geometries about an origin.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``angle``, ``origin``, ``radians``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.rotate(args.angle, origin=args.origin, use_radians=args.radians)
    _write_result(args, result)
    return 0


def _cmd_vector_scale(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector scale`` — scale geometries about their center.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``xfact``, ``yfact``, ``zfact``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.scale(xfact=args.xfact, yfact=args.yfact, zfact=args.zfact)
    _write_result(args, result)
    return 0


def _cmd_vector_translate(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector translate`` — shift geometries by an offset.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``xoff``, ``yoff``, ``zoff``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.translate(xoff=args.xoff, yoff=args.yoff, zoff=args.zoff)
    _write_result(args, result)
    return 0


def _cmd_vector_snap(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector snap`` — snap geometries to another layer's vertices.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``other``, ``tolerance``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    other = FeatureCollection.read_file(args.other)
    require_matching_crs(fc, other)
    result = fc.snap(other, args.tolerance)
    _write_result(args, result)
    return 0


def _cmd_vector_build_area(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector build-area`` — build polygonal areas from linework.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.build_area()
    _write_result(args, result)
    return 0


def _cmd_vector_extract_unique_points(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector extract-unique-points`` — the distinct vertices of each geometry.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.extract_unique_points()
    _write_result(args, result)
    return 0


def _cmd_vector_set_precision(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector set-precision`` — snap coordinates to a grid size.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``,
            ``grid_size``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.set_precision(args.grid_size)
    _write_result(args, result)
    return 0


def _cmd_vector_with_coordinates(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector with-coordinates`` — attach per-vertex x/y columns.

    Writes a copy of the source with ``x`` and ``y`` coordinate columns added (scalar
    for points, per-vertex sequences for lines/polygons).

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``driver``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.with_coordinates()
    result.to_file(args.output, driver=resolve_vector_driver(args.output, args.driver))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_with_centroid(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector with-centroid`` — attach per-feature centroid columns.

    Writes a copy of the source with ``x``/``y``/``avg_x``/``avg_y`` and a
    ``center_point`` column (the per-feature center point) added.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``driver``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.with_centroid()
    result.to_file(args.output, driver=resolve_vector_driver(args.output, args.driver))
    info_print(args, f"wrote {args.output}")
    return 0


def _add_io(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``src``/``output``/``--layer``/``--overwrite`` arguments.

    Args:
        parser: The subcommand parser to register the common arguments on.
    """
    parser.add_argument("src", help=_HELP_SRC)
    parser.add_argument("output", help=_HELP_OUT)
    parser.add_argument("--layer", help=_HELP_LAYER)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)


def add_vector_geom_commands(
    vsub: argparse._SubParsersAction, advanced: argparse._SubParsersAction | None = None
) -> None:
    """Register the unary geometry-operation commands on the ``vector`` group.

    Args:
        vsub: The ``vector`` group's ``add_subparsers`` action.
    """
    buf = vsub.add_parser("buffer", help="buffer each geometry by a distance")
    _add_io(buf)
    buf.add_argument("--distance", type=float, required=True, help="buffer distance")
    buf.add_argument("--resolution", type=int, default=16, help="segments per quadrant")
    buf.add_argument(
        "--cap-style",
        choices=("round", "flat", "square"),
        default="round",
        help="end-cap style",
    )
    buf.add_argument(
        "--join-style",
        choices=("round", "mitre", "bevel"),
        default="round",
        help="segment join style",
    )
    buf.set_defaults(func=_cmd_vector_buffer)

    simp = vsub.add_parser("simplify", help="Douglas-Peucker simplification")
    _add_io(simp)
    simp.add_argument(
        "--tolerance", type=float, required=True, help="simplify tolerance"
    )
    simp.add_argument(
        "--no-preserve-topology",
        action="store_true",
        help="allow topology to change (faster)",
    )
    simp.set_defaults(func=_cmd_vector_simplify)

    scov = (advanced or vsub).add_parser(
        "simplify-coverage", help="topology-preserving coverage simplification"
    )
    _add_io(scov)
    scov.add_argument(
        "--tolerance", type=float, required=True, help="simplify tolerance"
    )
    scov.set_defaults(func=_cmd_vector_simplify_coverage)

    valid = vsub.add_parser("make-valid", help="repair invalid geometries")
    _add_io(valid)
    valid.set_defaults(func=_cmd_vector_make_valid)

    norm = (advanced or vsub).add_parser(
        "normalize", help="normalize geometries to canonical form"
    )
    _add_io(norm)
    norm.set_defaults(func=_cmd_vector_normalize)

    rev = (advanced or vsub).add_parser(
        "reverse", help="reverse coordinate (winding) order"
    )
    _add_io(rev)
    rev.set_defaults(func=_cmd_vector_reverse)

    f2d = (advanced or vsub).add_parser("force-2d", help="drop the Z coordinate")
    _add_io(f2d)
    f2d.set_defaults(func=_cmd_vector_force_2d)

    f3d = (advanced or vsub).add_parser("force-3d", help="add a constant Z coordinate")
    _add_io(f3d)
    f3d.add_argument("--z", type=float, default=0.0, help="Z value to assign")
    f3d.set_defaults(func=_cmd_vector_force_3d)

    seg = (advanced or vsub).add_parser(
        "segmentize", help="densify edges to a max segment length"
    )
    _add_io(seg)
    seg.add_argument(
        "--max-length", type=float, required=True, help="maximum segment length"
    )
    seg.set_defaults(func=_cmd_vector_segmentize)

    rrp = (advanced or vsub).add_parser(
        "remove-repeated-points", help="drop duplicate vertices"
    )
    _add_io(rrp)
    rrp.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="merge vertices within this distance",
    )
    rrp.set_defaults(func=_cmd_vector_remove_repeated_points)

    off = (advanced or vsub).add_parser(
        "offset-curve", help="offset a line by a distance"
    )
    _add_io(off)
    off.add_argument("--distance", type=float, required=True, help="offset distance")
    off.set_defaults(func=_cmd_vector_offset_curve)

    rep = (advanced or vsub).add_parser(
        "representative-point", help="a point guaranteed inside each geometry"
    )
    _add_io(rep)
    rep.set_defaults(func=_cmd_vector_representative_point)

    hull = vsub.add_parser("concave-hull", help="the concave hull of each geometry")
    _add_io(hull)
    hull.add_argument(
        "--ratio", type=float, default=0.0, help="hull tightness in [0, 1]"
    )
    hull.add_argument(
        "--allow-holes", action="store_true", help="permit holes in the hull"
    )
    hull.set_defaults(func=_cmd_vector_concave_hull)

    mbc = vsub.add_parser(
        "minimum-bounding-circle", help="the smallest enclosing circle"
    )
    _add_io(mbc)
    mbc.set_defaults(func=_cmd_vector_minimum_bounding_circle)

    mrr = (advanced or vsub).add_parser(
        "minimum-rotated-rectangle", help="the smallest enclosing rectangle"
    )
    _add_io(mrr)
    mrr.set_defaults(func=_cmd_vector_minimum_rotated_rectangle)

    mic = (advanced or vsub).add_parser(
        "maximum-inscribed-circle", help="the largest circle fitting inside"
    )
    _add_io(mic)
    mic.add_argument("--tolerance", type=float, help="search tolerance")
    mic.set_defaults(func=_cmd_vector_maximum_inscribed_circle)

    aff = (advanced or vsub).add_parser(
        "affine-transform", help="apply a 2-D affine matrix"
    )
    _add_io(aff)
    aff.add_argument(
        "--matrix",
        nargs=6,
        type=float,
        required=True,
        metavar=("A", "B", "D", "E", "XOFF", "YOFF"),
        help="affine coefficients [a, b, d, e, xoff, yoff]",
    )
    aff.set_defaults(func=_cmd_vector_affine_transform)

    rot = vsub.add_parser("rotate", help="rotate geometries about an origin")
    _add_io(rot)
    rot.add_argument("--angle", type=float, required=True, help="rotation angle")
    rot.add_argument("--origin", default="center", help="rotation origin")
    rot.add_argument(
        "--radians", action="store_true", help="interpret --angle as radians"
    )
    rot.set_defaults(func=_cmd_vector_rotate)

    sca = vsub.add_parser("scale", help="scale geometries about their center")
    _add_io(sca)
    sca.add_argument("--xfact", type=float, default=1.0, help="X scale factor")
    sca.add_argument("--yfact", type=float, default=1.0, help="Y scale factor")
    sca.add_argument("--zfact", type=float, default=1.0, help="Z scale factor")
    sca.set_defaults(func=_cmd_vector_scale)

    tra = vsub.add_parser("translate", help="shift geometries by an offset")
    _add_io(tra)
    tra.add_argument("--xoff", type=float, default=0.0, help="X offset")
    tra.add_argument("--yoff", type=float, default=0.0, help="Y offset")
    tra.add_argument("--zoff", type=float, default=0.0, help="Z offset")
    tra.set_defaults(func=_cmd_vector_translate)

    snap = (advanced or vsub).add_parser(
        "snap", help="snap geometries to another layer's vertices"
    )
    _add_io(snap)
    snap.add_argument("--other", required=True, help="vector file to snap towards")
    snap.add_argument(
        "--tolerance", type=float, required=True, help="maximum snapping distance"
    )
    snap.set_defaults(func=_cmd_vector_snap)

    area = (advanced or vsub).add_parser(
        "build-area", help="build polygonal areas from linework"
    )
    _add_io(area)
    area.set_defaults(func=_cmd_vector_build_area)

    eup = (advanced or vsub).add_parser(
        "extract-unique-points", help="the distinct vertices of each geometry"
    )
    _add_io(eup)
    eup.set_defaults(func=_cmd_vector_extract_unique_points)

    prec = (advanced or vsub).add_parser(
        "set-precision", help="snap coordinates to a grid size"
    )
    _add_io(prec)
    prec.add_argument(
        "--grid-size", type=float, required=True, help="coordinate grid size"
    )
    prec.set_defaults(func=_cmd_vector_set_precision)

    wc = (advanced or vsub).add_parser(
        "with-coordinates", help="attach per-vertex x/y coordinate columns"
    )
    _add_io(wc)
    wc.add_argument("--driver", help=_HELP_DRIVER)
    wc.set_defaults(func=_cmd_vector_with_coordinates)

    wcen = (advanced or vsub).add_parser(
        "with-centroid", help="attach per-feature centroid columns"
    )
    _add_io(wcen)
    wcen.add_argument("--driver", help=_HELP_DRIVER)
    wcen.set_defaults(func=_cmd_vector_with_centroid)
