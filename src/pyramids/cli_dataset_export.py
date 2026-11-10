"""Raster (`Dataset`) export CLI commands for the ``pyramids`` tool.

Thin 1:1 wrappers over the :class:`pyramids.dataset.Dataset` serialization /
export surface: raw bytes, COG bytes, a pandas table, polygonized regions, a
raster alias write, a colour-mapped image, an XYZ point table, a Zarr store, a
STAC item, and a Mapbox/Terrarium terrain-RGB encode. Each handler maps onto a
single ``Dataset`` call and produces one output artifact.

Optional dependencies (``pyarrow`` for Parquet, ``zarr``/``dask`` for Zarr) are
imported lazily by the underlying methods, so this module itself adds no
optional-extra import at module import time; when a handler needs a missing
extra at call time, the ``ImportError`` is remapped to a clean ``ValueError``
naming the extra. (Note: ``import pyramids.dataset`` already pulls some of these
eagerly on the current build, so this module's cleanliness does not by itself
guarantee ``import pyramids.cli`` is extra-free — that is tracked separately.)

Deferred:
    - ``Dataset.write_array`` — patches an existing raster in place from an
      in-memory ``numpy`` array. A CLI has no array to supply and the call
      produces no new output file (it mutates the input), so it is not a sensible
      one-shot export verb; use ``pyramids calc`` / ``pyramids convert`` to
      produce a new raster instead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.dataset import Dataset
from pyramids.dataset.cog import Layout
from pyramids.feature import FeatureCollection

_HELP_SRC_RASTER = "source raster path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_BAND = "zero-based band index"


@contextlib.contextmanager
def _needs_parquet_extra() -> Iterator[None]:
    """Remap a missing-pyarrow ``ImportError`` to a clean ``ValueError``.

    Writing Parquet imports ``pyarrow`` from the optional ``[parquet]`` extra; its
    absence raises ``ImportError``, which :func:`pyramids.cli.main` does not catch
    cleanly. Turn it into a user error naming the extra.
    """
    try:
        yield
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [parquet] extra (pyarrow); install it "
            "with `pip install pyramids-gis[parquet]`"
        ) from exc


@contextlib.contextmanager
def _needs_lazy_extra() -> Iterator[None]:
    """Remap a missing ``[lazy]`` ``ImportError`` (zarr/dask) to a ``ValueError``.

    ``to_zarr`` imports ``zarr``/``dask`` from the optional ``[lazy]`` extra; its
    absence raises ``ImportError``, which :func:`pyramids.cli.main` does not catch
    cleanly. Turn it into a user error naming the extra.
    """
    try:
        yield
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [lazy] extra (zarr, dask); install it "
            "with `pip install pyramids-gis[lazy]`"
        ) from exc


def _parse_creation_options(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` creation-option tokens into a mapping.

    Args:
        pairs: The raw ``--creation-option`` tokens (``None`` when not passed).

    Returns:
        dict[str, str]: The parsed options (empty when ``pairs`` is ``None``).

    Raises:
        ValueError: A token is not of the form ``KEY=VALUE``.
    """
    options: dict[str, str] = {}
    for token in pairs or []:
        key, sep, value = token.partition("=")
        if not sep or not key:
            raise ValueError(
                f"bad creation option {token!r}; expected KEY=VALUE (e.g. COMPRESS=DEFLATE)."
            )
        options[key] = value
    return options


def _cmd_to_bytes(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-bytes`` — serialize a raster to raw bytes.

    Writes the encoded file contents to ``output`` when given, else to stdout
    (so it can be piped). Only single-file drivers are supported by the method.

    Args:
        args: Parsed args with ``input``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    payload = ds.to_bytes(driver=args.driver)
    if args.output is not None:
        with open(args.output, "wb") as handle:
            handle.write(payload)
        info_print(args, f"wrote {args.output}")
    else:
        sys.stdout.buffer.write(payload)
    return 0


def _cmd_to_cog_bytes(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-cog-bytes`` — serialize a raster to COG bytes.

    Args:
        args: Parsed args with ``input``, ``output``, ``compression``,
            ``blocksize``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    kwargs: dict = {}
    if args.compression:
        kwargs["compression"] = args.compression
    if args.blocksize:
        kwargs["layout"] = Layout(blocksize=args.blocksize)
    payload = ds.to_cog_bytes(**kwargs)
    with open(args.output, "wb") as handle:
        handle.write(payload)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_dataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-dataframe`` — write a cell table as CSV or Parquet.

    The frame carries a ``(band, y, x)`` MultiIndex and a single ``values``
    column; it is written with the index so the coordinates are preserved. The
    output format follows the extension (``.parquet`` → Parquet, else CSV).

    Args:
        args: Parsed args with ``input``, ``output``, ``bands``, ``dropna``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    bands = args.bands if args.bands else None
    df = ds.to_dataframe(bands=bands, dropna=args.dropna)
    if args.output.lower().endswith(".parquet"):
        with _needs_parquet_extra():
            df.to_parquet(args.output)
    else:
        df.to_csv(args.output, index=True)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_polygons(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-polygons`` — dissolve equal-valued regions to polygons.

    Unlike ``pyramids shapes`` (one feature per cell), this merges connected
    equal-valued cells into one polygon per region and writes them to a vector
    file whose driver follows the extension (or ``--driver``).

    Args:
        args: Parsed args with ``input``, ``output``, ``band``, ``driver``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    ds = Dataset.read_file(args.input)
    gdf = ds.to_polygons(band=args.band)
    FeatureCollection(gdf).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_raster(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-raster`` — write a raster via :meth:`Dataset.to_file`.

    Unlike ``pyramids convert`` it exposes the tile length and raw GDAL creation
    options. ``--band`` is forwarded as
    ``to_file``'s ``band`` argument, which selects the single band only for ASCII
    grid drivers; GeoTIFF and the other raster drivers write every band. It is
    range-checked here so an out-of-range index errors cleanly rather than being
    silently ignored.

    Args:
        args: Parsed args with ``input``, ``output``, ``band``, ``tile_length``,
            ``driver``, ``creation_option``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--band`` is negative or not below the raster's band count.
    """
    refuse_existing(args.output, args.overwrite)
    options = _parse_creation_options(args.creation_option)
    ds = Dataset.read_file(args.input)
    if not 0 <= args.band < ds.band_count:
        raise ValueError(
            f"band {args.band} is out of range for a {ds.band_count}-band raster"
        )
    ds.to_file(
        args.output,
        band=args.band,
        tile_length=args.tile_length,
        creation_options=[f"{key}={value}" for key, value in options.items()] or None,
        driver=args.driver,
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_image(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-image`` — write a colour-mapped RGB image of a band.

    Args:
        args: Parsed args with ``input``, ``output``, ``band``, ``cmap``,
            ``exclude_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    image = ds.to_image(
        band=args.band, cmap=args.cmap, exclude_value=args.exclude_value
    )
    image.save(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_xyz(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-xyz`` — write a lon/lat/value point table.

    Args:
        args: Parsed args with ``input``, ``output``, ``bands``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    bands = args.bands if args.bands else None
    ds.to_xyz(bands=bands, path=args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-zarr`` — write the raster to a Zarr store.

    Args:
        args: Parsed args with ``input``, ``output``, ``mode``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    with _needs_lazy_extra():
        ds.to_zarr(args.output, mode=args.mode)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_stac_item(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-stac-item`` — write a STAC Item JSON document.

    The item references the raster as an asset at ``--asset-href`` (defaulting to
    the input path). The item dictionary is scrubbed of non-finite floats before
    serialization so the JSON stays strictly parseable.

    Args:
        args: Parsed args with ``input``, ``output``, ``item_id``, ``asset_href``,
            ``datetime``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    asset_href = args.asset_href or args.input
    item = ds.to_stac_item(args.item_id, asset_href=asset_href, datetime=args.datetime)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(json_sanitize(item), handle)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_to_terrain_rgb(args: argparse.Namespace) -> int:
    """Handle ``pyramids to-terrain-rgb`` — encode a DEM to terrain-RGB.

    Writes one RGB(A) raster by default; with ``--tiles`` it writes an XYZ PNG
    pyramid rooted at ``output`` instead.

    Args:
        args: Parsed args with ``input``, ``output``, ``encoding``, ``band``,
            ``tiles``, ``min_zoom``, ``max_zoom``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    ds.to_terrain_rgb(
        args.output,
        encoding=args.encoding,
        band=args.band,
        tiles=args.tiles,
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom,
    )
    info_print(args, f"wrote {args.output}")
    return 0


def add_dataset_export_commands(sub: argparse._SubParsersAction) -> None:
    """Register the raster (`Dataset`) export commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    to_bytes = sub.add_parser("bytes", help="serialize a raster to raw bytes")
    to_bytes.add_argument("input", help=_HELP_SRC_RASTER)
    to_bytes.add_argument(
        "output", nargs="?", help="destination file (default: write to stdout)"
    )
    to_bytes.add_argument(
        "--driver", default="GTiff", help="GDAL raster driver (default: GTiff)"
    )
    to_bytes.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_bytes.set_defaults(func=_cmd_to_bytes)

    to_cog_bytes = sub.add_parser(
        "cog-bytes", help="serialize a raster to Cloud Optimized GeoTIFF bytes"
    )
    to_cog_bytes.add_argument("input", help=_HELP_SRC_RASTER)
    to_cog_bytes.add_argument("output", help="destination .tif path")
    to_cog_bytes.add_argument(
        "--compression", help="compression method (e.g. DEFLATE, ZSTD)"
    )
    to_cog_bytes.add_argument("--blocksize", type=int, help="internal tile size")
    to_cog_bytes.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_cog_bytes.set_defaults(func=_cmd_to_cog_bytes)

    to_df = sub.add_parser(
        "dataframe", help="write a per-cell table (CSV, or Parquet by extension)"
    )
    to_df.add_argument("input", help=_HELP_SRC_RASTER)
    to_df.add_argument("output", help="destination .csv or .parquet path")
    to_df.add_argument(
        "--bands",
        nargs="+",
        type=int,
        help="zero-based band indices to include (default: all)",
    )
    to_df.add_argument("--dropna", action="store_true", help="drop no-data (NaN) cells")
    to_df.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_df.set_defaults(func=_cmd_to_dataframe)

    to_poly = sub.add_parser(
        "polygons",
        help="dissolve connected equal-valued cells into polygons (vector file)",
    )
    to_poly.add_argument("input", help=_HELP_SRC_RASTER)
    to_poly.add_argument("output", help="destination vector path")
    to_poly.add_argument(
        "--band", type=int, default=0, help=_HELP_BAND + " (default 0)"
    )
    to_poly.add_argument(
        "--driver", help="OGR vector driver (default: inferred from the extension)"
    )
    to_poly.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_poly.set_defaults(func=_cmd_to_polygons)

    to_raster = sub.add_parser(
        "raster", help="write a raster (to_file alias) with band/tiling options"
    )
    to_raster.add_argument("input", help=_HELP_SRC_RASTER)
    to_raster.add_argument("output", help="destination raster path")
    to_raster.add_argument(
        "--band",
        type=int,
        default=0,
        help="band index forwarded to to_file; selects the band only for ASCII grid "
        "drivers (other drivers write every band). Must be in range (default 0)",
    )
    to_raster.add_argument(
        "--tile-length", type=int, default=None, help="tile edge for a windowed write"
    )
    to_raster.add_argument(
        "--driver",
        help="catalog (geotiff) or GDAL (GTiff) driver name (default: from extension)",
    )
    to_raster.add_argument(
        "--creation-option",
        action="append",
        metavar="KEY=VALUE",
        help="GDAL creation option; repeat for each",
    )
    to_raster.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_raster.set_defaults(func=_cmd_to_raster)

    to_image = sub.add_parser(
        "image", help="write a colour-mapped RGB image of a band (PNG/JPEG)"
    )
    to_image.add_argument("input", help=_HELP_SRC_RASTER)
    to_image.add_argument("output", help="destination image path (e.g. out.png)")
    to_image.add_argument(
        "--band", type=int, default=0, help=_HELP_BAND + " (default 0)"
    )
    to_image.add_argument(
        "--cmap", default="viridis", help="matplotlib colormap (default: viridis)"
    )
    to_image.add_argument(
        "--exclude-value",
        type=float,
        default=None,
        help="an extra value to mask out, besides no-data",
    )
    to_image.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_image.set_defaults(func=_cmd_to_image)

    to_xyz = sub.add_parser("xyz", help="write a lon/lat/value XYZ point table")
    to_xyz.add_argument("input", help=_HELP_SRC_RASTER)
    to_xyz.add_argument("output", help="destination .xyz path")
    to_xyz.add_argument(
        "--bands",
        nargs="+",
        type=int,
        help="zero-based band indices to include (default: all)",
    )
    to_xyz.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_xyz.set_defaults(func=_cmd_to_xyz)

    to_zarr = sub.add_parser("zarr", help="write the raster to a Zarr store")
    to_zarr.add_argument("input", help=_HELP_SRC_RASTER)
    to_zarr.add_argument("output", help="destination Zarr store path")
    to_zarr.add_argument("--mode", default="w", help="Zarr open mode (default: w)")
    to_zarr.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_zarr.set_defaults(func=_cmd_to_zarr)

    to_stac = sub.add_parser("stac-item", help="write a STAC Item JSON document")
    to_stac.add_argument("input", help=_HELP_SRC_RASTER)
    to_stac.add_argument("output", help="destination .json path")
    to_stac.add_argument("--item-id", required=True, help="STAC item identifier")
    to_stac.add_argument(
        "--asset-href", help="asset href for the raster (default: the input path)"
    )
    to_stac.add_argument(
        "--datetime", help="item datetime (ISO 8601; default: the item's own default)"
    )
    to_stac.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_stac.set_defaults(func=_cmd_to_stac_item)

    to_terrain = sub.add_parser(
        "terrain-rgb", help="encode a DEM to Mapbox/Terrarium terrain-RGB"
    )
    to_terrain.add_argument("input", help="source elevation raster path")
    to_terrain.add_argument(
        "output", help="destination raster (or tile-root directory with --tiles)"
    )
    to_terrain.add_argument(
        "--encoding",
        choices=["mapbox", "terrarium"],
        default="mapbox",
        help="RGB encoding (default: mapbox)",
    )
    to_terrain.add_argument(
        "--band", type=int, default=0, help="zero-based elevation band (default 0)"
    )
    to_terrain.add_argument(
        "--tiles",
        action="store_true",
        help="write an XYZ PNG pyramid instead of a single raster",
    )
    to_terrain.add_argument(
        "--min-zoom", type=int, default=0, help="lowest XYZ zoom (with --tiles)"
    )
    to_terrain.add_argument(
        "--max-zoom",
        type=int,
        default=None,
        help="highest XYZ zoom (with --tiles; default: from resolution)",
    )
    to_terrain.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    to_terrain.set_defaults(func=_cmd_to_terrain_rgb)
