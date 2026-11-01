"""Vector (FeatureCollection) CLI command group — ``pyramids vector …`` (M-1).

Thin, scriptable wrappers over :class:`pyramids.feature.FeatureCollection`, in
the same shape as the raster single-shot commands: an overwrite guard, a
one-line-error contract (handlers raise; :func:`pyramids.cli.main` reports), and
``--json`` on the inspection commands. No geometry logic lives here — each
command maps 1:1 onto an existing ``FeatureCollection`` call.
"""

from __future__ import annotations

import argparse
import json
import os

from pyramids._cli_shared import info_print, json_safe, refuse_existing
from pyramids.feature import FeatureCollection

# Output-extension -> OGR driver (alias or literal GDAL name, both accepted by
# FeatureCollection.to_file via its Catalog). FeatureCollection.to_file defaults
# to GeoJSON regardless of extension, so the CLI resolves the driver itself.
_VECTOR_DRIVER_BY_EXT = {
    ".geojson": "geojson",
    ".json": "geojson",
    ".shp": "ESRI Shapefile",
    ".gpkg": "gpkg",
    ".fgb": "FlatGeobuf",
    ".gml": "GML",
    ".kml": "KML",
}


def _resolve_vector_driver(path: str, explicit: str | None) -> str:
    """Return the OGR driver for ``path``: ``explicit`` if given, else by extension.

    Args:
        path: The destination vector path.
        explicit: The value of ``--driver`` (``None`` if not passed).

    Returns:
        str: A driver alias or literal GDAL name for ``FeatureCollection.to_file``.

    Raises:
        ValueError: No ``--driver`` and the extension is not a known vector one.
    """
    if explicit:
        return explicit
    ext = os.path.splitext(path)[1].lower()
    driver = _VECTOR_DRIVER_BY_EXT.get(ext)
    if driver is None:
        raise ValueError(
            f"cannot infer a vector driver from {path!r}; pass --driver "
            f"(known extensions: {sorted(_VECTOR_DRIVER_BY_EXT)})"
        )
    return driver


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
    if args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
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
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"{minx} {miny} {maxx} {maxy}")
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
    v_info.add_argument("src", help="source vector path")
    v_info.add_argument("--layer", help="layer name/index for multi-layer sources")
    v_info.add_argument("--json", action="store_true", help="emit JSON")
    v_info.set_defaults(func=_cmd_vector_info)

    v_bounds = vsub.add_parser("bounds", help="print the vector bounding box")
    v_bounds.add_argument("src", help="source vector path")
    v_bounds.add_argument("--layer", help="layer name/index")
    v_bounds.add_argument("--crs", help="reproject bounds to this CRS first")
    v_bounds.add_argument("--json", action="store_true", help="emit JSON")
    v_bounds.set_defaults(func=_cmd_vector_bounds)

    v_conv = vsub.add_parser("convert", help="re-save a vector in another format")
    v_conv.add_argument("src", help="source vector path")
    v_conv.add_argument("dst", help="destination vector path")
    v_conv.add_argument(
        "--driver", help="OGR driver (default: inferred from extension)"
    )
    v_conv.add_argument("--layer", help="layer name/index")
    v_conv.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    v_conv.set_defaults(func=_cmd_vector_convert)

    v_warp = vsub.add_parser("warp", help="reproject a vector")
    v_warp.add_argument("src", help="source vector path")
    v_warp.add_argument("dst", help="destination vector path")
    v_warp.add_argument("--crs", required=True, help="target CRS (EPSG, WKT, PROJ4)")
    v_warp.add_argument(
        "--driver", help="OGR driver (default: inferred from extension)"
    )
    v_warp.add_argument("--layer", help="layer name/index")
    v_warp.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    v_warp.set_defaults(func=_cmd_vector_warp)

    v_clip = vsub.add_parser("clip", help="spatially subset a vector")
    v_clip.add_argument("src", help="source vector path")
    v_clip.add_argument("dst", help="destination vector path")
    v_clip_how = v_clip.add_mutually_exclusive_group(required=True)
    v_clip_how.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="keep features intersecting this window (source CRS)",
    )
    v_clip_how.add_argument("--mask", help="vector file whose polygons clip the source")
    v_clip.add_argument(
        "--driver", help="OGR driver (default: inferred from extension)"
    )
    v_clip.add_argument("--layer", help="layer name/index")
    v_clip.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    v_clip.set_defaults(func=_cmd_vector_clip)
