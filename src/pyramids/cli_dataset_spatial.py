"""Raster (`Dataset`) spatial / analysis CLI command group.

Holds the top-level raster commands that combine, align, reproject, reshape, or
sample a :class:`~pyramids.dataset.Dataset` and did not fit the core group in
:mod:`pyramids.cli_dataset`: ``align``, ``combine``,
``warped-view``, ``wrap-longitude``, ``translate``, ``transect``, and
``extract``. Each handler is a thin 1:1 wrapper over an existing ``Dataset``
call — no business logic lives here — and registers itself on the top-level
parser via :func:`add_dataset_spatial_commands`.

Two-raster operations whose result is only meaningful when the operands share a
grid (``combine``) guard the pairing with
:func:`pyramids._cli_shared.require_matching_crs` up front, so a CRS mismatch
raises a clean message instead of producing geometrically meaningless output.
``align`` deliberately does **not** guard, because it reprojects a source whose
CRS differs from the reference onto the reference grid — that cross-CRS align is
a supported use, not a mistake.

Deferred: none. Every assigned method maps onto a one-shot verb — the
``Dataset``-returning ones (``align``/``combine``/``warped-view``/
``wrap-longitude``/``translate``) write their result to an output raster, and
the data-returning ones (``transect``/``extract``) write a profile /
table to a CSV or JSON output (or print it with ``--json``).
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from shapely.geometry import LineString

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    map_lookup_errors,
    require_matching_crs,
    resolve_vector_driver,
)
from pyramids._cli_shared import refuse_existing as _refuse_existing
from pyramids.base._utils import DEFAULT_RESAMPLING, numpy_to_gdal_dtype
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

_HELP_SRC_RASTER = "source raster path"
_HELP_DST_RASTER = "destination raster path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_RESAMPLING = "resampling method"
_HELP_JSON = "emit JSON instead of text"
_HELP_BAND = "zero-based band index"

# Named element-wise binary operations for `combine`, each a numpy ufunc taking
# the two operands' matching cells. Kept to the unambiguous, order-independent or
# clearly-ordered arithmetic a one-shot verb can name without a free-form
# expression (that is what `pyramids raster calc` is for).
_COMBINE_OPS = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
    "minimum": np.minimum,
    "maximum": np.maximum,
}


def _cmd_align(args: argparse.Namespace) -> int:
    """Handle `pyramids raster align` — resample a raster onto a reference raster's grid.

    The input is resampled onto the reference's coordinate system, dimensions,
    and cell size (``Dataset.align``). This is *not* guarded by
    ``require_matching_crs``: when the two CRSes differ, ``align`` reprojects the
    input onto the reference grid, which is the intended behaviour.

    Args:
        args: Parsed arguments with ``input``, ``reference``, ``output``,
            ``resampling``, and ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    source = Dataset.read_file(args.input)
    reference = Dataset.read_file(args.reference)
    aligned = source.align(reference, method=args.resampling)
    aligned.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_combine(args: argparse.Namespace) -> int:
    """Handle `pyramids raster combine` — combine two aligned rasters cell by cell.

    Applies the ``--op`` element-wise binary operation to the two rasters'
    matching cells (``Dataset.combine``) and writes the result. The operands must
    already share a grid; ``combine`` does not resample (use ``align`` first).

    Args:
        args: Parsed arguments with ``a``, ``b``, ``output``, ``op``, ``band``,
            and ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    left = Dataset.read_file(args.a)
    right = Dataset.read_file(args.b)
    require_matching_crs(left, right)
    combined = left.combine(right, _COMBINE_OPS[args.op], band=args.band)
    combined.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_warped_view(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced warped-view` — write an on-the-fly reprojected raster.

    Builds the lazy warped VRT view (``Dataset.warped_view``) and materialises it
    to the output path. Unlike ``warp`` this can also set an output ``--cell-size``
    and crop to a target-CRS ``--bbox`` in one step.

    Args:
        args: Parsed arguments with ``input``, ``output``, ``crs``, ``resampling``,
            ``cell_size``, ``bbox``, and ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    view = ds.warped_view(
        args.crs,
        method=args.resampling,
        cell_size=args.cell_size,
        bbox=tuple(args.bbox) if args.bbox else None,
    )
    view.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_wrap_longitude(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced wrap-longitude` — shift a global grid 0..360 -> -180..180.

    A pure column roll (no resampling) via ``Dataset.wrap_longitude``; the input
    must span the whole globe in the 0..360 frame.

    Args:
        args: Parsed arguments with ``input``, ``output``, and ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    wrapped = ds.wrap_longitude()
    wrapped.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced translate` — subset / rescale / retype a raster.

    A ``gdal_translate``-style one-shot over ``Dataset.translate``: extract a
    pixel (``--src-win``) or projected (``--proj-win``) window, rescale values
    (``--scale``), change the output type (``--output-type``), bake in scale/offset
    (``--unscale``), re-order/select bands (``--bands``), or set a no-data value
    (``--nodata``). Only the options the user passes are forwarded.

    Args:
        args: Parsed arguments with ``input``, ``output``, and the optional
            ``unscale`` / ``output_type`` / ``nodata`` / ``src_win`` / ``proj_win``
            / ``scale`` / ``bands`` / ``resampling`` / ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--scale`` is given a length other than 2 or 4.
    """
    _refuse_existing(args.output, args.overwrite)
    kwargs: dict = {}
    if args.unscale:
        kwargs["unscale"] = True
    if args.output_type is not None:
        kwargs["outputType"] = numpy_to_gdal_dtype(args.output_type)
    if args.nodata is not None:
        kwargs["noData"] = args.nodata
    if args.src_win is not None:
        kwargs["srcWin"] = [int(value) for value in args.src_win]
    if args.proj_win is not None:
        # gdal projWin order is [ulx, uly, lrx, lry]; the CLI takes it in that order.
        kwargs["projWin"] = list(args.proj_win)
    if args.scale is not None:
        if len(args.scale) not in (2, 4):
            raise ValueError(
                "--scale takes SRC_MIN SRC_MAX [DST_MIN DST_MAX] (2 or 4 values); "
                f"got {len(args.scale)}."
            )
        kwargs["scaleParams"] = [list(args.scale)]
    if args.bands is not None:
        kwargs["bandList"] = [int(value) for value in args.bands]
    if args.resampling is not None:
        kwargs["resampleAlg"] = args.resampling
    ds = Dataset.read_file(args.input)
    ds.translate(args.output, **kwargs)
    info_print(args, f"wrote {args.output}")
    return 0


def _parse_line(raw: str) -> LineString:
    """Parse a ``"x,y;x,y;..."`` string into a shapely :class:`LineString`.

    Args:
        raw: Semicolon-separated ``x,y`` vertices, e.g. ``"0,0;4,4"``.

    Returns:
        LineString: The line through the parsed vertices.

    Raises:
        ValueError: Fewer than two vertices, or a vertex is not a numeric
            ``x,y`` pair.
    """
    chunks = [chunk for chunk in raw.split(";") if chunk.strip()]
    if len(chunks) < 2:
        raise ValueError("--line needs at least two 'x,y' vertices, e.g. '0,0;4,4'.")
    vertices = []
    for chunk in chunks:
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"bad vertex {chunk.strip()!r}; expected 'x,y'.")
        try:
            vertices.append((float(parts[0]), float(parts[1])))
        except ValueError:
            raise ValueError(
                f"bad vertex {chunk.strip()!r}; coordinates must be numeric."
            ) from None
    return LineString(vertices)


def _write_transect(args: argparse.Namespace, profile) -> None:
    """Write (or print) a ``transect`` profile per the requested output form.

    ``--json`` prints the profile to stdout; a ``.csv`` output writes
    ``distance,value,x,y`` rows; any other extension writes a vector file via the
    OGR driver resolved from it.

    Args:
        args: Parsed arguments (reads ``output``, ``json``, ``driver``).
        profile: The :class:`~pyramids.feature.FeatureCollection` transect result
            (a ``GeoDataFrame`` with ``distance`` / ``value`` columns and point
            geometries).
    """
    records = [
        {
            "distance": float(dist),
            "value": None if value is None or np.isnan(value) else float(value),
            "x": float(geom.x),
            "y": float(geom.y),
        }
        for dist, value, geom in zip(
            profile["distance"], profile["value"], profile.geometry
        )
    ]
    if args.json:
        print(json.dumps(json_sanitize(records)))
    elif args.output.lower().endswith(".csv"):
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["distance", "value", "x", "y"])
            writer.writeheader()
            writer.writerows(records)
        info_print(args, f"wrote {args.output}")
    else:
        profile.to_file(
            args.output, driver=resolve_vector_driver(args.output, args.driver)
        )
        info_print(args, f"wrote {args.output}")


def _cmd_transect(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced transect` — sample a band along a line.

    Densifies ``--line`` into ordered positions (a fixed ``--n`` count or one
    roughly every ``--step`` metres) and samples the band at each
    (``Dataset.transect``), then writes the ``distance`` / ``value`` profile to a
    CSV or vector output (or prints it with ``--json``).

    Args:
        args: Parsed arguments with ``input``, ``output``, ``line``, ``n``,
            ``step``, ``band``, ``driver``, ``json``, and ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if not args.json:
        _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    profile = ds.transect(
        _parse_line(args.line), n=args.n, step=args.step, band=args.band
    )
    _write_transect(args, profile)
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """Handle `pyramids raster extract` — pull a raster's values (optionally at points).

    Reads every cell value (``Dataset.extract``), dropping no-data and any
    ``--exclude-value``; with ``--mask`` (a point vector) it reads the values at
    those points instead. The result (one row per band) is written as CSV or
    printed with ``--json``.

    Args:
        args: Parsed arguments with ``input``, ``output``, ``band``,
            ``exclude_value``, ``mask``, ``json``, and ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if not args.json:
        _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    mask = None
    if args.mask is not None:
        mask = FeatureCollection.read_file(args.mask)
    with map_lookup_errors():
        values = np.atleast_2d(
            np.asarray(
                ds.extract(band=args.band, exclude_value=args.exclude_value, mask=mask)
            )
        )
    rows = [
        [None if np.isnan(value) else float(value) for value in row] for row in values
    ]
    if args.json:
        print(json.dumps(json_sanitize({"values": rows})))
    else:
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for index, row in enumerate(rows):
                writer.writerow([f"band_{index + 1}", *row])
        info_print(args, f"wrote {args.output}")
    return 0


def _add_align(sub: argparse._SubParsersAction) -> None:
    """Register the ``align`` command on ``sub``."""
    align = sub.add_parser("align", help="resample a raster onto a reference grid")
    align.add_argument("input", help=_HELP_SRC_RASTER)
    align.add_argument("reference", help="reference raster whose grid/CRS to adopt")
    align.add_argument("output", help=_HELP_DST_RASTER)
    align.add_argument(
        "--resampling", default=DEFAULT_RESAMPLING, help=_HELP_RESAMPLING
    )
    align.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    align.set_defaults(func=_cmd_align)


def _add_combine(sub: argparse._SubParsersAction) -> None:
    """Register the ``combine`` command on ``sub``."""
    combine = sub.add_parser("combine", help="combine two aligned rasters cell by cell")
    combine.add_argument("a", help="first raster")
    combine.add_argument("b", help="second raster (must share a's grid)")
    combine.add_argument("output", help=_HELP_DST_RASTER)
    combine.add_argument(
        "--op",
        required=True,
        choices=sorted(_COMBINE_OPS),
        help="element-wise binary operation applied to the matching cells",
    )
    combine.add_argument(
        "--band",
        type=int,
        default=None,
        help="combine only this band (default: every band)",
    )
    combine.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    combine.set_defaults(func=_cmd_combine)


def _add_warped_view(sub: argparse._SubParsersAction) -> None:
    """Register the ``warped-view`` command on ``sub``."""
    warped = sub.add_parser(
        "warped-view",
        help="write an on-the-fly reprojected raster (optional grid/window)",
    )
    warped.add_argument("input", help=_HELP_SRC_RASTER)
    warped.add_argument("output", help=_HELP_DST_RASTER)
    warped.add_argument(
        "--crs", required=True, help="target CRS (EPSG code, WKT, PROJ4)"
    )
    warped.add_argument(
        "--resampling", default=DEFAULT_RESAMPLING, help=_HELP_RESAMPLING
    )
    warped.add_argument(
        "--cell-size",
        dest="cell_size",
        type=float,
        default=None,
        help="output cell size in target-CRS units (default: preserve resolution)",
    )
    warped.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="output extent in the target CRS",
    )
    warped.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    warped.set_defaults(func=_cmd_warped_view)


def _add_wrap_longitude(sub: argparse._SubParsersAction) -> None:
    """Register the ``wrap-longitude`` command on ``sub``."""
    wrap = sub.add_parser(
        "wrap-longitude",
        help="shift a global grid from the 0..360 to the -180..180 frame",
    )
    wrap.add_argument("input", help=_HELP_SRC_RASTER)
    wrap.add_argument("output", help=_HELP_DST_RASTER)
    wrap.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    wrap.set_defaults(func=_cmd_wrap_longitude)


def _add_translate(sub: argparse._SubParsersAction) -> None:
    """Register the ``translate`` command on ``sub``."""
    translate = sub.add_parser(
        "translate", help="subset / rescale / retype a raster (gdal_translate-style)"
    )
    translate.add_argument("input", help=_HELP_SRC_RASTER)
    translate.add_argument("output", help=_HELP_DST_RASTER)
    translate.add_argument(
        "--unscale",
        action="store_true",
        help="bake scale/offset into the stored values and clear them",
    )
    translate.add_argument(
        "--output-type",
        dest="output_type",
        help="output numpy dtype (e.g. float32, int16, uint8)",
    )
    translate.add_argument("--nodata", type=float, help="set the output no-data value")
    translate.add_argument(
        "--src-win",
        dest="src_win",
        nargs=4,
        type=int,
        metavar=("XOFF", "YOFF", "XSIZE", "YSIZE"),
        help="pixel subwindow to extract",
    )
    translate.add_argument(
        "--proj-win",
        dest="proj_win",
        nargs=4,
        type=float,
        metavar=("ULX", "ULY", "LRX", "LRY"),
        help="projected-coordinate subwindow to extract",
    )
    translate.add_argument(
        "--scale",
        nargs="+",
        type=float,
        metavar="V",
        help="rescale values: SRC_MIN SRC_MAX [DST_MIN DST_MAX]",
    )
    translate.add_argument(
        "--bands",
        nargs="+",
        type=int,
        metavar="N",
        help="1-based band numbers to keep / re-order",
    )
    translate.add_argument(
        "--resampling", default=None, help="resampling for a resolution change"
    )
    translate.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    translate.set_defaults(func=_cmd_translate)


def _add_transect(sub: argparse._SubParsersAction) -> None:
    """Register the ``transect`` command on ``sub``."""
    transect = sub.add_parser(
        "transect", help="sample a band along a line into a profile"
    )
    transect.add_argument("input", help=_HELP_SRC_RASTER)
    transect.add_argument(
        "output", help="destination profile path (.csv or a vector file)"
    )
    transect.add_argument(
        "--line",
        required=True,
        help="semicolon-separated 'x,y' vertices in the raster CRS, e.g. '0,0;4,4'",
    )
    density = transect.add_mutually_exclusive_group(required=True)
    density.add_argument(
        "--n", type=int, help="number of positions along the line (endpoints included)"
    )
    density.add_argument(
        "--step", type=float, help="ground spacing between positions, in metres"
    )
    transect.add_argument("--band", type=int, default=0, help=_HELP_BAND)
    transect.add_argument(
        "--driver", help="OGR driver for a vector output (default: from extension)"
    )
    transect.add_argument("--json", action="store_true", help=_HELP_JSON)
    transect.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    transect.set_defaults(func=_cmd_transect)


def _add_extract(sub: argparse._SubParsersAction) -> None:
    """Register the ``extract`` command on ``sub``."""
    extract = sub.add_parser(
        "extract", help="pull a raster's values (optionally at point locations)"
    )
    extract.add_argument("input", help=_HELP_SRC_RASTER)
    extract.add_argument("output", help="destination CSV path (one row per band)")
    extract.add_argument("--band", type=int, default=None, help=_HELP_BAND)
    extract.add_argument(
        "--exclude-value",
        dest="exclude_value",
        type=float,
        help="value to drop from the extracted values (in physical units)",
    )
    extract.add_argument(
        "--mask", help="point vector at which to read values instead of every cell"
    )
    extract.add_argument("--json", action="store_true", help=_HELP_JSON)
    extract.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    extract.set_defaults(func=_cmd_extract)


def add_dataset_spatial_commands(
    sub: argparse._SubParsersAction, advanced: argparse._SubParsersAction | None = None
) -> None:
    """Register the raster spatial / analysis commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
        advanced: The cross-cutting ``advanced`` group's subparsers action, or ``None``.
            When given, rarely-one-shot primitive verbs register there instead of on the
            main subsystem; isolation parsers omit it so those verbs stay flat locally.
    """
    _add_align(sub)
    _add_combine(sub)
    _add_warped_view(advanced or sub)
    _add_wrap_longitude(advanced or sub)
    _add_translate(advanced or sub)
    _add_transect(advanced or sub)
    _add_extract(sub)
