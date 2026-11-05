"""Remote / OGC fetch CLI command group — ``pyramids fetch …`` (M-4).

Thin wrappers that pull a bounding-box window from an OGC web service to a local
file: ``fetch wms`` / ``fetch wcs`` / ``fetch wmts`` / ``fetch ogc-coverages``
(raster) and ``fetch wfs`` (vector). Each maps 1:1 onto an existing
``Dataset.from_*`` / ``FeatureCollection.from_*`` classmethod; the transport is
GDAL's native driver.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import (
    info_print,
    refuse_existing,
)
from pyramids._cli_shared import (
    resolve_vector_driver as _resolve_vector_driver,
)
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

_HELP_OVERWRITE = "replace the output if it exists"


def _cmd_fetch_wms(args: argparse.Namespace) -> int:
    """Handle ``pyramids fetch wms`` — render a WMS window to a local raster.

    Args:
        args: Parsed args with ``url``, ``layers``, ``bbox``, ``crs``, ``size`` or
            ``resolution``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {"layers": args.layers, "bbox": tuple(args.bbox)}
    if args.crs:
        kwargs["crs"] = args.crs
    # from_wms requires exactly one of size/resolution; the parser's required
    # mutually-exclusive group guarantees exactly one of these is set (M1).
    if args.size:
        kwargs["size"] = tuple(args.size)
    else:
        kwargs["resolution"] = args.resolution
    ds = Dataset.from_wms(args.url, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_fetch_wcs(args: argparse.Namespace) -> int:
    """Handle ``pyramids fetch wcs`` — pull a WCS coverage window to a raster.

    Args:
        args: Parsed args with ``url``, ``coverage``, ``bbox``, ``crs``,
            ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {"coverage": args.coverage, "bbox": tuple(args.bbox)}
    if args.crs:
        kwargs["crs"] = args.crs
    ds = Dataset.from_wcs(args.url, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_fetch_wmts(args: argparse.Namespace) -> int:
    """Handle ``pyramids fetch wmts`` — crop a WMTS tile-pyramid window to a raster.

    Args:
        args: Parsed args with ``endpoint``, ``layer``, ``bbox``, ``crs``,
            ``tile_matrix_set``, ``resolution``, ``layer_crs``, ``output_crs``,
            ``resample``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {
        "layer": args.layer,
        "bbox": tuple(args.bbox),
        "resample": args.resample,
    }
    if args.crs:
        kwargs["crs"] = args.crs
    if args.tile_matrix_set:
        kwargs["tile_matrix_set"] = args.tile_matrix_set
    if args.resolution is not None:
        kwargs["resolution"] = args.resolution
    if args.layer_crs:
        kwargs["layer_crs"] = args.layer_crs
    if args.output_crs:
        kwargs["output_crs"] = args.output_crs
    ds = Dataset.from_wmts(args.endpoint, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_fetch_ogc_coverages(args: argparse.Namespace) -> int:
    """Handle ``pyramids fetch ogc-coverages`` — read an OGC API Coverages subset.

    Args:
        args: Parsed args with ``endpoint``, ``coverage``, ``bbox``, ``output_crs``,
            ``resolution``, ``coverage_crs``, ``resample``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {
        "coverage": args.coverage,
        "bbox": tuple(args.bbox),
        "resample": args.resample,
    }
    if args.resolution is not None:
        kwargs["resolution"] = args.resolution
    if args.coverage_crs:
        kwargs["coverage_crs"] = args.coverage_crs
    if args.output_crs:
        kwargs["output_crs"] = args.output_crs
    ds = Dataset.from_ogc_coverages(args.endpoint, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_fetch_wfs(args: argparse.Namespace) -> int:
    """Handle ``pyramids fetch wfs`` — pull WFS features to a local vector file.

    Args:
        args: Parsed args with ``url``, ``typename``, ``bbox``, ``driver``,
            ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {"typename": args.typename}
    if args.bbox:
        kwargs["bbox"] = tuple(args.bbox)
    fc = FeatureCollection.from_wfs(args.url, **kwargs)
    # Infer the OGR driver from the output extension (same rule as the vector
    # group) instead of silently defaulting to GeoJSON for any extension (L1).
    fc.to_file(args.output, driver=_resolve_vector_driver(args.output, args.driver))
    info_print(args, f"wrote {args.output}")
    return 0


def _add_bbox(parser: argparse.ArgumentParser, *, required: bool) -> None:
    """Add a shared ``--bbox MINX MINY MAXX MAXY`` option to ``parser``."""
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=required,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="request window (lon/lat unless --crs says otherwise)",
    )


def add_ogc_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``fetch`` group (``wms`` / ``wcs`` / ``wmts`` / ``ogc-coverages`` / ``wfs``) on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    fetch = sub.add_parser("fetch", help="fetch a window from an OGC web service")
    fsub = fetch.add_subparsers(dest="command", required=True)

    wms = fsub.add_parser("wms", help="render a WMS map window to a raster")
    wms.add_argument("url", help="WMS endpoint URL")
    wms.add_argument("output", help="destination raster path")
    wms.add_argument("--layers", required=True, help="WMS layer name")
    _add_bbox(wms, required=True)
    wms.add_argument("--crs", help="request CRS (default: the service default)")
    # from_wms needs exactly one of size/resolution — mirror that as a required
    # mutually-exclusive group so the CLI contract matches the library (M1).
    wms_grid = wms.add_mutually_exclusive_group(required=True)
    wms_grid.add_argument(
        "--size", nargs=2, type=int, metavar=("W", "H"), help="output pixel size"
    )
    wms_grid.add_argument(
        "--resolution", type=float, help="output cell size in CRS units"
    )
    wms.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    wms.set_defaults(func=_cmd_fetch_wms)

    wcs = fsub.add_parser("wcs", help="pull a WCS coverage window to a raster")
    wcs.add_argument("url", help="WCS endpoint URL")
    wcs.add_argument("output", help="destination raster path")
    wcs.add_argument("--coverage", required=True, help="coverage identifier")
    _add_bbox(wcs, required=True)
    wcs.add_argument("--crs", help="bbox CRS (default: the service default)")
    wcs.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    wcs.set_defaults(func=_cmd_fetch_wcs)

    wmts = fsub.add_parser("wmts", help="crop a WMTS tile-pyramid window to a raster")
    wmts.add_argument("endpoint", help="WMTS GetCapabilities URL")
    wmts.add_argument("output", help="destination raster path")
    wmts.add_argument("--layer", required=True, help="WMTS layer identifier")
    _add_bbox(wmts, required=True)
    wmts.add_argument("--crs", help="CRS of --bbox (default: EPSG:4326)")
    wmts.add_argument(
        "--tile-matrix-set", dest="tile_matrix_set", help="tile-matrix-set id to pin"
    )
    wmts.add_argument(
        "--resolution",
        type=float,
        help="output pixel size in the layer's native CRS units",
    )
    wmts.add_argument(
        "--layer-crs", dest="layer_crs", help="layer CRS when the service omits one"
    )
    wmts.add_argument(
        "--output-crs", dest="output_crs", help="reproject the result to this CRS"
    )
    wmts.add_argument(
        "--resample", default="nearest", help="resampling method (default: nearest)"
    )
    wmts.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    wmts.set_defaults(func=_cmd_fetch_wmts)

    cov = fsub.add_parser(
        "ogc-coverages", help="read an OGC API Coverages subset to a raster"
    )
    cov.add_argument("endpoint", help="OGC API landing-page / base URL")
    cov.add_argument("output", help="destination raster path")
    cov.add_argument("--coverage", required=True, help="coverage identifier")
    _add_bbox(cov, required=True)
    cov.add_argument(
        "--resolution",
        type=float,
        help="approximate pixel size in the coverage's native CRS units",
    )
    cov.add_argument(
        "--coverage-crs",
        dest="coverage_crs",
        help="coverage CRS when the service omits one",
    )
    cov.add_argument(
        "--output-crs", dest="output_crs", help="reproject the result to this CRS"
    )
    cov.add_argument(
        "--resample", default="nearest", help="resampling method (default: nearest)"
    )
    cov.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    cov.set_defaults(func=_cmd_fetch_ogc_coverages)

    wfs = fsub.add_parser("wfs", help="pull WFS features to a vector file")
    wfs.add_argument("url", help="WFS endpoint URL")
    wfs.add_argument("output", help="destination vector path")
    wfs.add_argument("--typename", required=True, help="feature type name")
    _add_bbox(wfs, required=False)
    wfs.add_argument("--driver", help="OGR driver (default: geojson)")
    wfs.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    wfs.set_defaults(func=_cmd_fetch_wfs)
