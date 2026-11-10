"""Overview and windowed-read `Dataset` CLI command group.

Holds the raster commands that read a *decimated* or *windowed* view of a
`Dataset` — the overview-pyramid and cloud-native partial-read verbs that the
single-shot `cli_dataset` group does not cover: `recreate-overviews`,
`read-overview-array`, `read-part`, `preview`, `point`, and `read-tile`. Every
handler maps 1:1 onto an existing `Dataset` method; the window/tile reads write
their array to an output raster (georeferenced from the method's own transform),
and `point` prints the sampled value(s).

The top-level `overview` command (in `cli_dataset`) already *builds* pyramids in
place via `create_overviews`; this module never duplicates it — `recreate-overviews`
regenerates the levels that already exist.

Deferred (no sensible one-shot file/stdout result, so intentionally not shipped):

- `block_windows` — the top-level `blocks` command already emits the native
  block/window grid as JSON window geometry, so a `block-windows` verb would
  duplicate it.
- `get_overview` / `get_overview_dataset` — return an in-memory `gdal.Band` /
  `Dataset` object with no standalone file form; overview levels are built by
  `overview` and read by `read-overview-array`.
- `read_windows` / `iter_blocks` — Python generators streaming arrays
  block-by-block (the API-level streaming primitives), not one-shot CLI verbs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from pyramids._cli_shared import info_print, json_safe
from pyramids._cli_shared import refuse_existing as _refuse_existing
from pyramids.base.crs import crs_spec
from pyramids.base.georeference import GeoReference
from pyramids.cli_dataset import (
    _HELP_DST_RASTER,
    _HELP_INSPECT_RASTER,
    _HELP_JSON,
    _HELP_OVERWRITE,
    _HELP_RESAMPLING,
    _HELP_SRC_RASTER,
)
from pyramids.dataset import Dataset
from pyramids.dataset.engines.cog import _xyz_bounds_3857

_HELP_BAND = "0-based band index (default: all bands)"


def _check_band(ds: Dataset, band: int | None) -> None:
    """Raise a clean ``ValueError`` when ``band`` is out of range.

    A negative band would otherwise wrap to a different GDAL band (or an opaque
    GDAL error); refuse it up front with the same message the ``blocks`` command
    uses.

    Args:
        ds: The open dataset.
        band: The requested 0-based band, or ``None`` for all bands.

    Raises:
        ValueError: ``band`` is given and not in ``[0, band_count)``.
    """
    if band is not None and (band < 0 or band >= ds.band_count):
        raise ValueError(f"band {band} out of range for a {ds.band_count}-band raster")


def _output_nodata(ds: Dataset, band: int | None):
    """Return the no-data value(s) to stamp on a window/preview output raster.

    Args:
        ds: The source dataset (``band`` is already range-checked).
        band: The 0-based band read, or ``None`` when all bands were read.

    Returns:
        The source's whole no-data list (all-bands read) or the single band's
        sentinel.
    """
    values = ds.no_data_value
    return values if band is None else values[band]


def _scaled_geotransform(
    ds: Dataset, out_rows: int, out_cols: int
) -> tuple[float, float, float, float, float, float]:
    """Scale the dataset geotransform to a full-extent decimation of a given size.

    A `preview` / `read_overview_array` array covers the whole raster extent at a
    coarser sampling, so the origin and any rotation stay put while the pixel
    spacing grows by the row/column decimation ratio.

    Args:
        ds: The source dataset.
        out_rows: Row count of the decimated array.
        out_cols: Column count of the decimated array.

    Returns:
        The six-tuple geotransform placing the decimated array over the full
        extent.
    """
    x0, pixel_w, row_skew, y0, col_skew, pixel_h = ds.geotransform
    col_scale = ds.columns / out_cols
    row_scale = ds.rows / out_rows
    return (
        x0,
        pixel_w * col_scale,
        row_skew * row_scale,
        y0,
        col_skew * col_scale,
        pixel_h * row_scale,
    )


def _write_raster(
    array: np.ndarray,
    geotransform: tuple[float, float, float, float, float, float],
    epsg,
    nodata,
    output: str,
) -> None:
    """Write a window/preview/tile array to ``output`` as a georeferenced raster.

    Args:
        array: ``(rows, cols)`` or ``(bands, rows, cols)`` pixel values.
        geotransform: The six-tuple placing ``array`` in ``epsg``.
        epsg: The output CRS spec (an EPSG int, WKT string, or ``None``).
        nodata: The no-data value / list to stamp.
        output: Destination raster path.
    """
    Dataset.from_array(
        np.asarray(array),
        geo_ref=GeoReference(geo=geotransform, epsg=epsg),
        no_data_value=nodata,
    ).to_file(output)


def _cmd_recreate_overviews(args: argparse.Namespace) -> int:
    """Handle `pyramids recreate-overviews` — regenerate existing overviews in place.

    Unlike `overview` (which *builds* pyramids via `create_overviews`), this
    regenerates the levels a raster already has; a raster with none is left
    unchanged with a warning from the library.

    Args:
        args: Parsed arguments with `file` and `resampling`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file, read_only=False)
    ds.recreate_overviews(resampling_method=args.resampling)
    info_print(
        args,
        f"regenerated overviews for {args.file} (levels per band: {ds.overview_count})",
    )
    return 0


def _cmd_read_overview_array(args: argparse.Namespace) -> int:
    """Handle `pyramids read-overview-array` — read one overview level.

    Writes the decimated level to `output` as a georeferenced raster, or — with
    `--json` — prints its shape / dtype / min / max / mean instead.

    Args:
        args: Parsed args with `input`, optional `output`, `band`,
            `overview_index`, `json`, `overwrite`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: Neither an `output` nor `--json` was given (or both were), the
            band is out of range, or the band/overview index has no such overview.
    """
    if not args.output and not args.json:
        raise ValueError(
            "read-overview-array needs an OUTPUT path, or --json to print stats instead."
        )
    if args.output and args.json:
        raise ValueError(
            "read-overview-array: pass either an OUTPUT path or --json, not both "
            "(--json prints stats instead of writing the array)."
        )
    ds = Dataset.read_file(args.input)
    _check_band(ds, args.band)
    if args.output and not args.json:
        _refuse_existing(args.output, args.overwrite)
    array = np.asarray(
        ds.read_overview_array(band=args.band, overview_index=args.overview_index)
    )
    if args.json:
        print(
            json.dumps(
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "min": json_safe(float(np.min(array))),
                    "max": json_safe(float(np.max(array))),
                    "mean": json_safe(float(np.mean(array))),
                }
            )
        )
    else:
        geotransform = _scaled_geotransform(ds, array.shape[-2], array.shape[-1])
        _write_raster(
            array,
            geotransform,
            crs_spec(ds.epsg, ds.crs),
            _output_nodata(ds, args.band),
            args.output,
        )
        info_print(args, f"wrote {args.output}")
    return 0


def _cmd_read_part(args: argparse.Namespace) -> int:
    """Handle `pyramids read-part` — read a geographic window to a raster.

    Reads a `--bbox` window decimated from the nearest overview and writes it as
    a georeferenced raster using the window's own transform.

    Args:
        args: Parsed args with `input`, `output`, `bbox`, `dst_width`,
            `dst_height`, `gsd`, `bbox_crs`, `resampling`, `band`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.input)
    _check_band(ds, args.band)
    _refuse_existing(args.output, args.overwrite)
    kwargs: dict = {
        "bbox_crs": args.bbox_crs,
        "resampling": args.resampling,
        "band": args.band,
        "return_transform": True,
    }
    if args.gsd is not None:
        kwargs["gsd"] = args.gsd
    if args.dst_width is not None:
        kwargs["dst_width"] = args.dst_width
    if args.dst_height is not None:
        kwargs["dst_height"] = args.dst_height
    array, geotransform = ds.read_part(tuple(args.bbox), **kwargs)
    _write_raster(
        array,
        geotransform,
        crs_spec(ds.epsg, ds.crs),
        _output_nodata(ds, args.band),
        args.output,
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    """Handle `pyramids preview` — write a whole-image decimated thumbnail.

    Args:
        args: Parsed args with `input`, `output`, `max_size`, `resampling`,
            `band`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.input)
    _check_band(ds, args.band)
    _refuse_existing(args.output, args.overwrite)
    array = np.asarray(
        ds.preview(max_size=args.max_size, resampling=args.resampling, band=args.band)
    )
    geotransform = _scaled_geotransform(ds, array.shape[-2], array.shape[-1])
    _write_raster(
        array,
        geotransform,
        crs_spec(ds.epsg, ds.crs),
        _output_nodata(ds, args.band),
        args.output,
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_point(args: argparse.Namespace) -> int:
    """Handle `pyramids point` — sample band value(s) at a map coordinate.

    Args:
        args: Parsed args with `input`, `x`, `y`, `point_crs`, `band`, `json`.

    Returns:
        int: `0` on success.

    Raises:
        OutOfBoundsError: The point falls outside the raster extent.
    """
    ds = Dataset.read_file(args.input)
    _check_band(ds, args.band)
    sampled = np.asarray(
        ds.point(args.x, args.y, point_crs=args.point_crs, band=args.band)
    )
    values = [json_safe(value) for value in sampled.reshape(-1).tolist()]
    if args.json:
        print(json.dumps({"x": args.x, "y": args.y, "values": values}))
    else:
        print(f"{args.x},{args.y}: {values}")
    return 0


def _cmd_read_tile(args: argparse.Namespace) -> int:
    """Handle `pyramids read-tile` — read a Web-Mercator XYZ tile to a raster.

    Reads slippy-map tile `(z, x, y)` and writes it as an EPSG:3857 raster whose
    geotransform comes from the tile's own Web-Mercator bounds.

    Args:
        args: Parsed args with `input`, `output`, `z`, `x`, `y`, `tilesize`,
            `resampling`, `band`, `overwrite`.

    Returns:
        int: `0` on success.

    Raises:
        OutOfBoundsError: The tile does not intersect the raster.
    """
    ds = Dataset.read_file(args.input)
    _check_band(ds, args.band)
    _refuse_existing(args.output, args.overwrite)
    array = np.asarray(
        ds.read_tile(
            args.z,
            args.x,
            args.y,
            tilesize=args.tilesize,
            resampling=args.resampling,
            band=args.band,
        )
    )
    west, south, east, north = _xyz_bounds_3857(args.z, args.x, args.y)
    geotransform = (
        west,
        (east - west) / args.tilesize,
        0.0,
        north,
        0.0,
        -(north - south) / args.tilesize,
    )
    _write_raster(array, geotransform, 3857, _output_nodata(ds, args.band), args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_dataset_overviews_commands(
    sub: argparse._SubParsersAction, advanced: argparse._SubParsersAction | None = None
) -> None:
    """Register the overview / windowed-read raster commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
        advanced: The cross-cutting ``advanced`` group's subparsers action, or ``None``.
            When given, rarely-one-shot primitive verbs register there instead of on the
            main subsystem; isolation parsers omit it so those verbs stay flat locally.
    """
    recreate = sub.add_parser(
        "recreate-overviews", help="regenerate a raster's existing overviews in place"
    )
    recreate.add_argument("file", help="raster path whose overviews to regenerate")
    recreate.add_argument(
        "--resampling",
        default="nearest",
        help="overview resampling (nearest, average, gauss, cubic, mode, ...)",
    )
    recreate.set_defaults(func=_cmd_recreate_overviews)

    read_overview = (advanced or sub).add_parser(
        "read-overview-array",
        help="read one overview level to a raster or --json stats",
    )
    read_overview.add_argument("input", help=_HELP_INSPECT_RASTER)
    read_overview.add_argument(
        "output",
        nargs="?",
        help="destination raster for the decimated level (omit with --json)",
    )
    read_overview.add_argument("--band", type=int, default=None, help=_HELP_BAND)
    read_overview.add_argument(
        "--overview-index",
        type=int,
        default=0,
        help="0-based overview level to read (default: 0, the coarsest-but-one)",
    )
    read_overview.add_argument(
        "--json", action="store_true", help="print shape/dtype/stats instead of writing"
    )
    read_overview.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    read_overview.set_defaults(func=_cmd_read_overview_array)

    read_part = (advanced or sub).add_parser(
        "read-part", help="read a geographic window to a raster (overview-decimated)"
    )
    read_part.add_argument("input", help=_HELP_SRC_RASTER)
    read_part.add_argument("output", help=_HELP_DST_RASTER)
    read_part.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="window extent in --bbox-crs (default: the raster's own CRS)",
    )
    read_part.add_argument(
        "--dst-width", type=int, help="output width in pixels (decimates the read)"
    )
    read_part.add_argument(
        "--dst-height", type=int, help="output height in pixels (decimates the read)"
    )
    read_part.add_argument(
        "--gsd",
        type=float,
        help="target cell size in --bbox-crs units (instead of --dst-width/--dst-height)",
    )
    read_part.add_argument(
        "--bbox-crs", dest="bbox_crs", help="CRS of --bbox (default: the raster's CRS)"
    )
    read_part.add_argument("--resampling", default="bilinear", help=_HELP_RESAMPLING)
    read_part.add_argument("--band", type=int, default=None, help=_HELP_BAND)
    read_part.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    read_part.set_defaults(func=_cmd_read_part)

    preview = (advanced or sub).add_parser(
        "preview", help="write a whole-image decimated thumbnail raster"
    )
    preview.add_argument("input", help=_HELP_SRC_RASTER)
    preview.add_argument("output", help=_HELP_DST_RASTER)
    preview.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="maximum pixels on the longer edge (default: 1024)",
    )
    preview.add_argument("--resampling", default="bilinear", help=_HELP_RESAMPLING)
    preview.add_argument("--band", type=int, default=None, help=_HELP_BAND)
    preview.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    preview.set_defaults(func=_cmd_preview)

    point = (advanced or sub).add_parser(
        "point", help="sample band value(s) at a map coordinate"
    )
    point.add_argument("input", help="raster path to sample")
    point.add_argument("x", type=float, help="X / longitude / easting in --point-crs")
    point.add_argument("y", type=float, help="Y / latitude / northing in --point-crs")
    point.add_argument(
        "--point-crs",
        dest="point_crs",
        help="CRS of the coordinate (default: the raster's own CRS)",
    )
    point.add_argument("--band", type=int, default=None, help=_HELP_BAND)
    point.add_argument("--json", action="store_true", help=_HELP_JSON)
    point.set_defaults(func=_cmd_point)

    read_tile = (advanced or sub).add_parser(
        "read-tile", help="read a Web-Mercator XYZ tile to an EPSG:3857 raster"
    )
    read_tile.add_argument("input", help=_HELP_SRC_RASTER)
    read_tile.add_argument("output", help=_HELP_DST_RASTER)
    read_tile.add_argument("z", type=int, help="zoom level")
    read_tile.add_argument("x", type=int, help="tile column index")
    read_tile.add_argument("y", type=int, help="tile row index (origin north-west)")
    read_tile.add_argument(
        "--tilesize",
        type=int,
        default=256,
        help="output tile size in pixels (default: 256)",
    )
    read_tile.add_argument("--resampling", default="bilinear", help=_HELP_RESAMPLING)
    read_tile.add_argument("--band", type=int, default=None, help=_HELP_BAND)
    read_tile.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    read_tile.set_defaults(func=_cmd_read_tile)
