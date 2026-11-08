"""Vector (FeatureCollection) CLI command group — ``pyramids vector …`` (M-1).

Thin, scriptable wrappers over :class:`pyramids.feature.FeatureCollection`, in
the same shape as the raster single-shot commands: an overwrite guard, a
one-line-error contract (handlers raise; :func:`pyramids.cli.main` reports), and
``--json`` on the inspection commands. No geometry logic lives here — each
command maps 1:1 onto an existing ``FeatureCollection`` call.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import (
    emit,
    info_print,
    json_safe,
    mark_format_support,
    refuse_existing,
)
from pyramids._cli_shared import (
    resolve_vector_driver as _resolve_vector_driver,
)
from pyramids.cli_vector_geom import add_vector_geom_commands
from pyramids.cli_vector_inspect import add_vector_inspect_commands
from pyramids.cli_vector_io import add_vector_io_commands
from pyramids.cli_vector_overlay import add_vector_overlay_commands
from pyramids.cli_vector_tess import add_vector_tess_commands
from pyramids.feature import FeatureCollection

_HELP_SRC_VECTOR = "source vector path"
_HELP_DST_VECTOR = "destination vector path"
_HELP_LAYER = "layer name/index"
_HELP_DRIVER = "OGR driver (default: inferred from extension)"
_HELP_OVERWRITE = "replace the output if it exists"


def _cmd_vector_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector info`` — print feature count, geometry, CRS, bounds.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    geom_col = fc.geometry.name
    geom_types = sorted({str(t) for t in fc.geometry.geom_type.dropna().unique()})
    minx, miny, maxx, maxy = (float(value) for value in fc.total_bounds)
    payload = {
        "path": args.src,
        "layer": args.layer,
        "features": int(len(fc)),
        "geometry_types": geom_types,
        "epsg": fc.epsg,
        "columns": [column for column in fc.columns if column != geom_col],
        "bounds": [json_safe(value) for value in (minx, miny, maxx, maxy)],
    }
    text = "\n".join(f"{key}: {value}" for key, value in payload.items())
    emit(args, payload, text=text)
    return 0


def _cmd_vector_bounds(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector bounds`` — print total bounds, optionally reprojected.

    Args:
        args: Parsed args with ``src``, ``layer``, ``crs``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    if args.crs:
        fc = fc.to_crs(args.crs)
    minx, miny, maxx, maxy = (float(value) for value in fc.total_bounds)
    payload = {"bounds": [json_safe(value) for value in (minx, miny, maxx, maxy)]}
    emit(args, payload, text=f"{minx} {miny} {maxx} {maxy}")
    return 0


def _cmd_vector_convert(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector convert`` — re-save a vector in another format.

    Args:
        args: Parsed args with ``src``, ``dst``, ``driver``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    driver = _resolve_vector_driver(args.dst, args.driver)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    fc.to_file(args.dst, driver=driver)
    info_print(args, f"wrote {args.dst}")
    return 0


def _cmd_vector_warp(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector warp`` — reproject a vector to ``--crs``.

    Args:
        args: Parsed args with ``src``, ``dst``, ``crs``, ``driver``, ``layer``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The source carries no CRS, so it cannot be reprojected.
    """
    refuse_existing(args.dst, args.overwrite)
    driver = _resolve_vector_driver(args.dst, args.driver)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    if fc.crs is None:
        raise ValueError(
            f"{args.src!r} has no CRS, so it cannot be reprojected; set one first."
        )
    fc.to_crs(args.crs).to_file(args.dst, driver=driver)
    info_print(args, f"wrote {args.dst}")
    return 0


def _cmd_vector_clip(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector clip`` — spatially subset by ``--bbox`` or ``--mask``.

    The ``--bbox`` path reads only the window (an OGR spatial filter); ``--mask``
    geometrically clips to the mask layer's polygons, reprojecting the mask to the
    source CRS first when they differ.

    Args:
        args: Parsed args with ``src``, ``dst``, ``bbox`` / ``mask``, ``driver``,
            ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    driver = _resolve_vector_driver(args.dst, args.driver)
    if args.mask:
        fc = FeatureCollection.read_file(args.src, layer=args.layer)
        mask = FeatureCollection.read_file(args.mask)
        # Reproject the mask only when both layers carry a CRS and they differ.
        # If the source is CRS-less we cannot know its projection, so we clip in
        # the source's raw coordinate space rather than guess a reprojection (N1).
        if mask.crs is not None and fc.crs is not None and mask.epsg != fc.epsg:
            mask = mask.to_crs(fc.epsg)
        clipped = FeatureCollection(fc.clip(mask))
    else:
        clipped = FeatureCollection.read_file(
            args.src, layer=args.layer, bbox=tuple(args.bbox)
        )
    clipped.to_file(args.dst, driver=driver)
    info_print(args, f"wrote {args.dst}")
    return 0


def add_vector_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``vector`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    vector = sub.add_parser("vector", help="vector (FeatureCollection) commands")
    vsub = vector.add_subparsers(dest="command", required=True)

    v_info = vsub.add_parser("info", help="print vector metadata")
    v_info.add_argument("src", help=_HELP_SRC_VECTOR)
    v_info.add_argument("--layer", help="layer name/index for multi-layer sources")
    v_info.add_argument("--json", action="store_true", help="emit JSON")
    v_info.set_defaults(func=_cmd_vector_info)
    mark_format_support(v_info)

    v_bounds = vsub.add_parser("bounds", help="print the vector bounding box")
    v_bounds.add_argument("src", help=_HELP_SRC_VECTOR)
    v_bounds.add_argument("--layer", help=_HELP_LAYER)
    v_bounds.add_argument("--crs", help="reproject bounds to this CRS first")
    v_bounds.add_argument("--json", action="store_true", help="emit JSON")
    v_bounds.set_defaults(func=_cmd_vector_bounds)
    mark_format_support(v_bounds)

    v_conv = vsub.add_parser("convert", help="re-save a vector in another format")
    v_conv.add_argument("src", help=_HELP_SRC_VECTOR)
    v_conv.add_argument("dst", help=_HELP_DST_VECTOR)
    v_conv.add_argument("--driver", help=_HELP_DRIVER)
    v_conv.add_argument("--layer", help=_HELP_LAYER)
    v_conv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_conv.set_defaults(func=_cmd_vector_convert)

    v_warp = vsub.add_parser("warp", help="reproject a vector")
    v_warp.add_argument("src", help=_HELP_SRC_VECTOR)
    v_warp.add_argument("dst", help=_HELP_DST_VECTOR)
    v_warp.add_argument("--crs", required=True, help="target CRS (EPSG, WKT, PROJ4)")
    v_warp.add_argument("--driver", help=_HELP_DRIVER)
    v_warp.add_argument("--layer", help=_HELP_LAYER)
    v_warp.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_warp.set_defaults(func=_cmd_vector_warp)

    v_clip = vsub.add_parser("clip", help="spatially subset a vector")
    v_clip.add_argument("src", help=_HELP_SRC_VECTOR)
    v_clip.add_argument("dst", help=_HELP_DST_VECTOR)
    v_clip_how = v_clip.add_mutually_exclusive_group(required=True)
    v_clip_how.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="keep features intersecting this window (source CRS)",
    )
    v_clip_how.add_argument("--mask", help="vector file whose polygons clip the source")
    v_clip.add_argument("--driver", help=_HELP_DRIVER)
    v_clip.add_argument("--layer", help=_HELP_LAYER)
    v_clip.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_clip.set_defaults(func=_cmd_vector_clip)

    add_vector_inspect_commands(vsub)
    add_vector_geom_commands(vsub)
    add_vector_overlay_commands(vsub)
    add_vector_tess_commands(vsub)
    add_vector_io_commands(vsub)
