"""Collection / datacube CLI command group — ``pyramids collection …`` (M-5, #1469).

Thin 1:1 wrappers over :class:`pyramids.dataset.collection.DatasetCollection`:
summarise a temporal stack of co-registered rasters, write time reductions, merge
the stack into one mosaic, reproject / crop / align every timestep, export the cube
to files / COGs / NetCDF / Zarr / kerchunk, and build a cube from a remote STAC
source (``collection import point`` / ``collection import stac``).

Every operating verb opens the stack through :func:`_open_collection`, which supports
three offline sources via ``--source``: ``files`` (default — a glob or directory handed
to :meth:`DatasetCollection.from_files`), ``zarr`` (a store opened with
:meth:`~DatasetCollection.from_zarr`), and ``archive`` (a zip/tar opened with
:meth:`~DatasetCollection.from_archive`). The ``import`` subgroup instead *constructs*
a cube from a remote STAC source and writes it to NetCDF: ``point`` wraps the
point-centred :meth:`~DatasetCollection.from_point` (needs the ``[stac]`` extra and a
live endpoint), and ``stac`` wraps :meth:`~DatasetCollection.from_stac` over a saved
ItemCollection JSON (the shell-expressible subset of its options — callables and the
alignment/fusion/read option objects stay Python-only).

No business logic lives here: each handler marshals arguments and forwards to an
existing library call, returns an ``int`` exit code, and reports expected user
errors as a one-line message through :func:`pyramids.cli.main`.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from pyramids._cli_groups import SuggestingArgumentParser, add_group
from pyramids._cli_shared import (
    ProgressReporter,
    info_print,
    json_safe,
    map_lookup_errors,
    needs_lazy_extra,
    refuse_existing,
    validate_index_pattern,
)
from pyramids.base._utils import DEFAULT_RESAMPLING
from pyramids.base.crs import crs_spec
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.collection import DatasetCollection

_REDUCERS = ("mean", "sum", "min", "max", "std", "var")

# The full reduce_time op vocabulary: the six associative reducers plus the richer
# holistic / trend reductions that only reduce_time exposes.
_TIME_OPS = (
    "mean",
    "sum",
    "min",
    "max",
    "std",
    "var",
    "median",
    "quantile",
    "percentile",
    "mode",
    "trend",
    "sens_slope",
)

# Overlap-resolution rules accepted by DatasetCollection.merge / merge_rasters.
_MERGE_METHODS = ("first", "last", "min", "max", "sum", "count", "mean")
_HELP_DST_DIR = "destination directory (one file per timestep)"

_HELP_SOURCE_PATH = (
    "glob (quote it) or directory of co-registered rasters; with --source zarr it "
    "is a Zarr store path, and with --source archive a zip/tar path"
)
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_RESAMPLING = "resampling method"


def _open_collection(args: argparse.Namespace) -> DatasetCollection:
    """Open a :class:`DatasetCollection` from the source named by ``args``.

    The ``--source`` option selects the opener: ``files`` (the default — the
    positional ``glob`` is a glob or directory of co-registered rasters), ``zarr``
    (the positional is a Zarr store), or ``archive`` (the positional is a zip/tar
    whose members matching ``--member-glob`` are read).

    Args:
        args: Parsed args carrying ``glob`` and, optionally, ``source`` /
            ``member_glob``.

    Returns:
        DatasetCollection: The assembled stack.

    Raises:
        ValueError: ``--source files`` and the pattern matched no rasters.
    """
    source = getattr(args, "source", "files")
    pattern = args.glob
    if source == "zarr":
        with needs_lazy_extra():
            coll = DatasetCollection.from_zarr(pattern)
    elif source == "archive":
        coll = DatasetCollection.from_archive(
            pattern, member_glob=getattr(args, "member_glob", "*")
        )
    else:
        paths = sorted(_glob.glob(pattern, recursive=True))
        if not paths:
            raise ValueError(f"no rasters matched {pattern!r}")
        coll = DatasetCollection.from_files(
            paths, date_format=getattr(args, "date_format", None)
        )
    return coll


def _resolve_times(args: argparse.Namespace, coll: DatasetCollection) -> Any:
    """Return the per-timestep timestamps for :meth:`reduce_time`.

    Prefers an explicit ``--times`` (a comma-separated list, one entry per
    timestep); otherwise falls back to the stack's own parsed time axis. Length
    and parseability are validated downstream by :meth:`reduce_time`.

    Args:
        args: Parsed args carrying ``times``.
        coll: The opened stack (its :attr:`~DatasetCollection.time` is the
            fallback).

    Returns:
        Any: A list of timestamp tokens, or the collection's time axis.

    Raises:
        ValueError: No ``--times`` was given and the stack carries no time axis.
    """
    if args.times:
        resolved: Any = [
            token.strip() for token in args.times.split(",") if token.strip()
        ]
    elif coll.time is not None:
        resolved = list(coll.time)
    else:
        raise ValueError(
            "the stack carries no time axis; pass --times t0,t1,... (one timestamp "
            "per timestep, e.g. --times 2022-01-01,2022-01-02,2022-01-03)"
        )
    return resolved


def _guard_output_dir(directory: str, overwrite: bool) -> None:
    """Refuse a non-empty output directory unless ``--overwrite`` was passed.

    The per-timestep writers (``resample`` / ``to-file`` / ``reproject`` /
    ``crop`` / ``align``) emit many files into one directory, so the single-file
    :func:`~pyramids._cli_shared.refuse_existing` guard does not fit. A missing or
    empty directory is always allowed; an existing one with contents is refused so
    a stale run is not silently mixed with a new one.

    Args:
        directory: The output directory the command is about to populate.
        overwrite: Whether the user passed ``--overwrite``.

    Raises:
        ValueError: ``directory`` exists, is non-empty, and ``overwrite`` is
            ``False``.
    """
    if not overwrite and os.path.isdir(directory) and os.listdir(directory):
        raise ValueError(
            f"output directory {directory!r} is not empty; pass --overwrite to "
            "write into it anyway."
        )


def _cmd_collection_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection info`` — summarise a temporal raster stack.

    Args:
        args: Parsed args with ``glob``, ``source``, ``member_glob``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    coll = _open_collection(args)
    base = coll.base
    time_axis: Any = coll.time
    payload = {
        "rasters": int(coll.time_length),
        "epsg": base.epsg,
        "cell_size": json_safe(float(base.cell_size)),
        "rows": base.rows,
        "columns": base.columns,
        "time": None if time_axis is None else [str(value) for value in time_axis],
    }
    if args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_collection_stats(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection reduce`` — write a time reduction as a raster.

    Args:
        args: Parsed args with ``glob``, ``output``, ``reduce``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    coll = _open_collection(args)
    # The reducers return an ndarray on the stack's shared grid; re-attach the
    # base raster's georeferencing so the result is writable (shape is already
    # (bands, rows, cols), the layout from_array expects).
    array = getattr(coll, args.reduce)(skipna=True)
    base = coll.base
    # crs_spec (not `base.epsg or base.crs`): the `or` idiom prefers an EPSG code
    # even when pyproj cannot resolve it (#943) and leaks an empty CRS string as
    # "Invalid projection: ''"; crs_spec falls back to WKT (M2). Carry the base
    # no-data so skipna-NaN gaps stay flagged rather than taking from_array's
    # -9999 default that no cell matches (L2).
    Dataset.from_array(
        array,
        no_data_value=base.no_data_value[0],
        geo_ref=GeoReference(geo=base.geotransform, epsg=crs_spec(base.epsg, base.crs)),
    ).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_resample(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection resample`` — calendar-frequency time reduction.

    Buckets the timesteps by the ``--freq`` pandas offset alias, reduces each
    window with ``--op`` (:meth:`DatasetCollection.reduce_time`), and writes one
    raster per window into the output directory.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``freq``,
            ``op``, ``q``, ``skipna``, ``times``, ``pattern``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _guard_output_dir(args.output, args.overwrite)
    # Validate --pattern before opening/reducing anything (only {i} is bound here).
    validate_index_pattern(args.pattern)
    coll = _open_collection(args)
    times = _resolve_times(args, coll)
    windows = coll.reduce_time(
        times, freq=args.freq, op=args.op, skipna=args.skipna, q=args.q
    )
    os.makedirs(args.output, exist_ok=True)
    total = len(windows) if hasattr(windows, "__len__") else None
    with ProgressReporter(args, total, label="windows") as progress:
        for index, (label, dataset) in enumerate(windows):
            path = os.path.join(args.output, args.pattern.format(i=index))
            dataset.to_file(path)
            # The progress bar (stderr) replaces the per-window log only when they would
            # collide on one interactive terminal — i.e. progress is drawing AND stdout is
            # itself a tty. When stdout is redirected to a file/pipe, keep the log so the
            # machine-capturable output is not lost.
            if not (progress.enabled and sys.stdout.isatty()):
                info_print(args, f"window {index} ({label}) -> {path}")
            progress.advance()
    return 0


def _cmd_collection_merge(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection merge`` — mosaic every timestep into one raster.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``method``,
            ``nodata``, ``bbox``, ``bbox_crs``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    coll = _open_collection(args)
    kwargs: dict[str, Any] = {"method": args.method}
    if args.nodata is not None:
        kwargs["no_data_value"] = args.nodata
    if args.bbox is not None:
        kwargs["bbox"] = tuple(args.bbox)
        kwargs["bbox_crs"] = args.bbox_crs
    coll.merge(args.output, **kwargs)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_to_file(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection export file`` — write every timestep to a directory.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``driver``,
            ``band``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _guard_output_dir(args.output, args.overwrite)
    coll = _open_collection(args)
    coll.to_file(args.output, driver=args.driver, band=args.band)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_to_cog(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection export cog`` — write each timestep as its own COG.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``pattern``,
            ``name``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    # Refuse a non-empty output dir up front for parity with the other collection
    # writers (to_cog_stack still owns per-file overwrite semantics below).
    _guard_output_dir(args.output, args.overwrite)
    # Validate --pattern up front: to_cog_stack formats it with {name}/{i}, so an
    # unknown placeholder or a missing {i} would otherwise collide/leak mid-loop.
    validate_index_pattern(args.pattern, name=args.name)
    coll = _open_collection(args)
    written = coll.to_cog_stack(
        args.output, pattern=args.pattern, name=args.name, overwrite=args.overwrite
    )
    info_print(args, f"wrote {len(written)} COG(s) to {args.output}")
    return 0


def _cmd_collection_to_netcdf(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection export netcdf`` — write the (T,B,R,C) cube to NetCDF.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``time_dim``,
            ``var_per_band``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    coll = _open_collection(args)
    coll.to_netcdf(args.output, time_dim=args.time_dim, var_per_band=args.var_per_band)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_to_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection export zarr`` — write the (T,B,R,C) cube to a Zarr store.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``mode``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    coll = _open_collection(args)
    with needs_lazy_extra():
        coll.to_zarr(args.output, mode=args.mode)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_to_kerchunk(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection export kerchunk`` — write a kerchunk reference JSON.

    Builds a single kerchunk reference over the stack's source files
    (:meth:`DatasetCollection.to_kerchunk`); the sources must be NetCDF/HDF5.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``,
            ``concat_dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    coll = _open_collection(args)
    with needs_lazy_extra():
        coll.to_kerchunk(args.output, concat_dim=args.concat_dim)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_reproject(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection reproject`` — reproject every timestep.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``crs``,
            ``resampling``, ``maintain_alignment``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _guard_output_dir(args.output, args.overwrite)
    coll = _open_collection(args)
    # Eager collection (opened from files), so to_crs returns a DatasetCollection,
    # never the lazy Delayed / in-place None the union type also allows.
    reprojected = cast(
        DatasetCollection,
        coll.to_crs(
            args.crs,
            method=args.resampling,
            maintain_alignment=args.maintain_alignment,
        ),
    )
    reprojected.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_crop(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection crop`` — crop every timestep to a mask or bbox.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``, ``mask``,
            ``bbox``, ``epsg``, ``touch``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _guard_output_dir(args.output, args.overwrite)
    coll = _open_collection(args)
    mask = Dataset.read_file(args.mask) if args.mask else None
    cropped = cast(
        DatasetCollection,
        coll.crop(
            mask=mask,
            touch=args.touch,
            bbox=tuple(args.bbox) if args.bbox else None,
            epsg=args.epsg,
        ),
    )
    cropped.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_align(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection align`` — align every timestep to a reference.

    Args:
        args: Parsed args with ``glob`` / source options, ``output``,
            ``reference``, ``resampling``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    _guard_output_dir(args.output, args.overwrite)
    coll = _open_collection(args)
    reference = Dataset.read_file(args.reference)
    aligned = cast(DatasetCollection, coll.align(reference, method=args.resampling))
    aligned.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _split_assets(raw: str) -> str | list[str]:
    """Parse a comma-separated asset/band spec into one key or a band-axis sequence.

    Args:
        raw: The raw ``--bands`` / ``--asset`` value.

    Returns:
        str | list[str]: A single key for a one-asset stack, or the list for a band axis.

    Raises:
        ValueError: ``raw`` holds no non-empty key (e.g. ``","`` or whitespace only).
    """
    parts = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not parts:
        raise ValueError(f"no asset/band key found in {raw!r}")
    return parts[0] if len(parts) == 1 else parts


def _load_stac_items(path: str) -> Any:
    """Load a saved STAC ItemCollection / item-list JSON into items for ``from_stac``.

    Accepts a STAC ``ItemCollection`` / GeoJSON ``FeatureCollection`` (the items are its
    ``features``), a bare JSON list of item objects, or a single GeoJSON ``Feature`` (wrapped into
    a one-item list). ``from_stac`` is duck-typed and reads the raw dicts directly, so pystac is
    not needed to parse them.

    Args:
        path: Path to the JSON file.

    Returns:
        Any: The list of STAC item dicts.

    Raises:
        ValueError: The JSON is none of the accepted shapes (ItemCollection, item list, Feature).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        items: Any = data
    elif isinstance(data, dict) and isinstance(data.get("features"), list):
        items = data["features"]
    elif isinstance(data, dict) and data.get("type") == "Feature":
        items = [data]
    else:
        raise ValueError(
            f"{path!r} is not a STAC ItemCollection, a single Feature, or a list of items."
        )
    return items


def _cmd_collection_from_point(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection import point`` — build a point-centred STAC cube.

    Searches ``--collection`` over a square AOI centred on ``(--lat, --lon)`` and stacks the
    ``--bands`` onto the point's local-UTM grid (:meth:`DatasetCollection.from_point`), writing
    the cube to a NetCDF file. Needs the ``[stac]`` extra and a live STAC endpoint.

    Args:
        args: Parsed args with ``lat``, ``lon``, ``collection``, ``bands``, ``start``, ``end``,
            ``edge_size``, ``resolution``, ``units``, ``stac``, ``query``, ``align``, ``output``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with map_lookup_errors():
        coll = DatasetCollection.from_point(
            args.lat,
            args.lon,
            collection=args.collection,
            bands=_split_assets(args.bands),
            start_date=args.start,
            end_date=args.end,
            edge_size=args.edge_size,
            resolution=args.resolution,
            units=args.units,
            stac=args.stac,
            query=json.loads(args.query) if args.query else None,
            align=args.align,
        )
    coll.to_netcdf(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_collection_from_stac(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection import stac`` — build a cube from saved STAC items.

    Reads a STAC ItemCollection JSON (or a bare item list) from ``items``, stacks the ``--asset``
    key(s) across its items (:meth:`DatasetCollection.from_stac`), and writes the cube to a
    NetCDF file. Reading the asset hrefs still needs network access to wherever they point.

    Args:
        args: Parsed args with ``items``, ``output``, ``asset``, ``bbox``, ``max_items``,
            ``groupby``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with map_lookup_errors():
        coll = DatasetCollection.from_stac(
            _load_stac_items(args.items),
            _split_assets(args.asset),
            bbox=tuple(args.bbox) if args.bbox else None,
            max_items=args.max_items,
            groupby=args.groupby,
        )
    coll.to_netcdf(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _add_open_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared stack-open arguments (positional source + ``--source``).

    Args:
        parser: The subcommand parser to extend.
    """
    parser.add_argument("glob", help=_HELP_SOURCE_PATH)
    parser.add_argument(
        "--source",
        choices=("files", "zarr", "archive"),
        default="files",
        help="how to open the stack: 'files' (default, glob/dir), 'zarr' store, "
        "or 'archive' (zip/tar)",
    )
    parser.add_argument(
        "--member-glob",
        dest="member_glob",
        default="*",
        help="with --source archive: glob selecting members inside the archive",
    )
    parser.add_argument(
        "--date-format",
        dest="date_format",
        default=None,
        help="with --source files: strptime format to parse each file's timestamp "
        "from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis",
    )


def _add_info(csub: argparse._SubParsersAction) -> None:
    """Register the ``info`` command."""
    c_info = csub.add_parser("info", help="summarize a raster stack")
    _add_open_args(c_info)
    c_info.add_argument("--json", action="store_true", help="emit JSON")
    c_info.set_defaults(func=_cmd_collection_info)


def _add_stats(csub: argparse._SubParsersAction) -> None:
    """Register the ``stats`` command (the six associative reducers)."""
    c_stats = csub.add_parser("reduce", help="write a time reduction of a stack")
    _add_open_args(c_stats)
    c_stats.add_argument("output", help="destination raster path")
    c_stats.add_argument(
        "--reduce", required=True, choices=_REDUCERS, help="time-reduction function"
    )
    c_stats.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c_stats.set_defaults(func=_cmd_collection_stats)


def _add_resample(csub: argparse._SubParsersAction) -> None:
    """Register the ``resample`` command (calendar-frequency reduce_time)."""
    c = csub.add_parser(
        "resample", help="calendar-frequency time reduction (one raster per window)"
    )
    _add_open_args(c)
    c.add_argument("output", help="destination directory (one raster per window)")
    c.add_argument(
        "--freq", required=True, help="pandas offset alias, e.g. 1MS, 7D, 6h"
    )
    c.add_argument(
        "--op", default="mean", choices=_TIME_OPS, help="reduction operation"
    )
    c.add_argument(
        "--q",
        type=float,
        default=None,
        help="level for --op percentile (0<q<=100) / quantile (0<q<=1)",
    )
    c.add_argument(
        "--no-skipna",
        dest="skipna",
        action="store_false",
        help="propagate NaN instead of ignoring it within each window",
    )
    c.add_argument(
        "--times",
        default=None,
        help="comma-separated timestamps, one per timestep (default: the stack's "
        "own time axis)",
    )
    c.add_argument(
        "--pattern",
        default="window_{i:04d}.tif",
        help="per-window filename template with an {i} index placeholder",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_resample, skipna=True)


def _add_merge(csub: argparse._SubParsersAction) -> None:
    """Register the ``merge`` command (mosaic every timestep into one raster)."""
    c = csub.add_parser("merge", help="mosaic every timestep into one raster")
    _add_open_args(c)
    c.add_argument("output", help="destination raster path")
    c.add_argument(
        "--method",
        default="last",
        choices=_MERGE_METHODS,
        help="overlap-resolution rule (default: last)",
    )
    c.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="no-data value for the mosaic (default: inherit from the timesteps)",
    )
    c.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="restrict the mosaic to this window",
    )
    c.add_argument(
        "--bbox-crs",
        dest="bbox_crs",
        default=None,
        help="CRS of --bbox (default: the mosaic's own CRS)",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_merge)


def _add_to_file(csub: argparse._SubParsersAction) -> None:
    """Register the ``to-file`` command (write every timestep to a directory)."""
    c = csub.add_parser("file", help="write every timestep to a directory")
    _add_open_args(c)
    c.add_argument("output", help=_HELP_DST_DIR)
    c.add_argument(
        "--driver",
        default="geotiff",
        help="output driver / catalog key (default: geotiff)",
    )
    c.add_argument(
        "--band",
        type=int,
        default=0,
        help="band to write for single-band drivers (ignored by GeoTIFF)",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_to_file)


def _add_to_cog(csub: argparse._SubParsersAction) -> None:
    """Register the ``to-cog`` command (each timestep -> its own COG)."""
    c = csub.add_parser("cog", help="write each timestep as its own COG")
    _add_open_args(c)
    c.add_argument("output", help="destination directory (one COG per timestep)")
    c.add_argument(
        "--pattern",
        default="{name}_{i:04d}.tif",
        help="per-slice filename template with {name} and {i} placeholders",
    )
    c.add_argument(
        "--name", default="slice", help="replacement for the {name} placeholder"
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_to_cog)


def _add_to_netcdf(csub: argparse._SubParsersAction) -> None:
    """Register the ``to-netcdf`` command (write the (T,B,R,C) cube to NetCDF)."""
    c = csub.add_parser("netcdf", help="write the (T,B,R,C) cube to a NetCDF file")
    _add_open_args(c)
    c.add_argument("output", help="destination NetCDF path")
    c.add_argument(
        "--time-dim", dest="time_dim", default="time", help="name of the time dimension"
    )
    c.add_argument(
        "--no-var-per-band",
        dest="var_per_band",
        action="store_false",
        help="write one variable for all bands instead of one variable per band",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_to_netcdf, var_per_band=True)


def _add_to_zarr(csub: argparse._SubParsersAction) -> None:
    """Register the ``to-zarr`` command (write the (T,B,R,C) cube to a Zarr store)."""
    c = csub.add_parser("zarr", help="write the (T,B,R,C) cube to a Zarr store")
    _add_open_args(c)
    c.add_argument("output", help="destination Zarr store path")
    c.add_argument("--mode", default="w", help="Zarr write mode (default: w)")
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_to_zarr)


def _add_to_kerchunk(csub: argparse._SubParsersAction) -> None:
    """Register the ``to-kerchunk`` command (write a kerchunk reference JSON)."""
    c = csub.add_parser(
        "kerchunk", help="write a kerchunk reference JSON (NetCDF/HDF5 sources)"
    )
    _add_open_args(c)
    c.add_argument("output", help="destination kerchunk reference JSON path")
    c.add_argument(
        "--concat-dim",
        dest="concat_dim",
        default="time",
        help="name of the concatenation dimension (default: time)",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_to_kerchunk)


def _add_reproject(csub: argparse._SubParsersAction) -> None:
    """Register the ``reproject`` command (reproject every timestep)."""
    c = csub.add_parser("reproject", help="reproject every timestep to a target CRS")
    _add_open_args(c)
    c.add_argument("output", help=_HELP_DST_DIR)
    c.add_argument("--crs", required=True, help="target CRS (EPSG code, WKT, PROJ4)")
    c.add_argument("--resampling", default=DEFAULT_RESAMPLING, help=_HELP_RESAMPLING)
    c.add_argument(
        "--maintain-alignment",
        dest="maintain_alignment",
        action="store_true",
        help="keep the source grid alignment through the reprojection",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_reproject)


def _add_crop(csub: argparse._SubParsersAction) -> None:
    """Register the ``crop`` command (crop every timestep)."""
    c = csub.add_parser("crop", help="crop every timestep to a mask or bbox")
    _add_open_args(c)
    c.add_argument("output", help=_HELP_DST_DIR)
    target = c.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--mask", default=None, help="raster whose footprint crops the stack"
    )
    target.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="crop to this bounding box instead of a mask",
    )
    c.add_argument(
        "--epsg", default=None, help="CRS of --bbox (default: the stack's own CRS)"
    )
    c.add_argument(
        "--no-touch",
        dest="touch",
        action="store_false",
        help="keep only cells whose centre is inside, not every touched cell",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_crop, touch=True)


def _add_align(csub: argparse._SubParsersAction) -> None:
    """Register the ``align`` command (align every timestep to a reference grid)."""
    c = csub.add_parser("align", help="align every timestep to a reference raster grid")
    _add_open_args(c)
    c.add_argument("output", help=_HELP_DST_DIR)
    c.add_argument(
        "--reference", required=True, help="reference raster whose grid to adopt"
    )
    c.add_argument("--resampling", default=DEFAULT_RESAMPLING, help=_HELP_RESAMPLING)
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_align)


def _add_from_point(csub: argparse._SubParsersAction) -> None:
    """Register the ``import point`` command (point-centred STAC cube)."""
    c = csub.add_parser(
        "point",
        help="build a point-centred STAC cube (needs the [stac] extra + a live endpoint)",
    )
    c.add_argument("output", help="destination NetCDF cube path")
    c.add_argument(
        "--lat", type=float, required=True, help="centre latitude (EPSG:4326)"
    )
    c.add_argument(
        "--lon", type=float, required=True, help="centre longitude (EPSG:4326)"
    )
    c.add_argument("--collection", required=True, help="STAC collection id to search")
    c.add_argument(
        "--bands",
        required=True,
        help="asset key, or comma-separated keys for a band axis",
    )
    c.add_argument(
        "--start", required=True, help="search start (YYYY-MM-DD / RFC 3339)"
    )
    c.add_argument("--end", required=True, help="search end (YYYY-MM-DD / RFC 3339)")
    c.add_argument(
        "--edge-size",
        dest="edge_size",
        type=int,
        required=True,
        help="cube side length, in pixels (--units px) or metres (--units m)",
    )
    c.add_argument(
        "--resolution", type=float, required=True, help="pixel size in metres"
    )
    c.add_argument(
        "--units",
        default="px",
        choices=("px", "m"),
        help="unit of --edge-size (default: px)",
    )
    c.add_argument(
        "--stac",
        default=None,
        help="STAC API root URL. The default (Planetary Computer) needs a request signer, which "
        "is Python-only, so point this at an unauthenticated STAC root to use it from the CLI",
    )
    c.add_argument(
        "--query", default=None, help="STAC query-extension filter, as a JSON string"
    )
    c.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        help="skip the multi-asset resolution alignment",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_from_point)


def _add_from_stac(csub: argparse._SubParsersAction) -> None:
    """Register the ``import stac`` command (cube from a saved ItemCollection)."""
    c = csub.add_parser(
        "stac",
        help="build a cube from a saved STAC ItemCollection / item-list JSON file",
    )
    c.add_argument("items", help="path to a STAC ItemCollection / item-list JSON file")
    c.add_argument("output", help="destination NetCDF cube path")
    c.add_argument(
        "--asset",
        required=True,
        help="asset key, or comma-separated keys for a band axis",
    )
    c.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        default=None,
        help="lon/lat (EPSG:4326) item filter: drop items whose footprint misses it",
    )
    c.add_argument(
        "--max-items",
        dest="max_items",
        type=int,
        default=None,
        help="cap the number of items consumed (after --bbox filtering)",
    )
    c.add_argument(
        "--groupby",
        default=None,
        help="collapse items into timesteps: solar_day, id, time, or an item property key",
    )
    c.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    c.set_defaults(func=_cmd_collection_from_stac)


def add_collection_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``collection`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    collection = sub.add_parser("collection", help="temporal raster-stack commands")
    csub = collection.add_subparsers(
        dest="command", required=True, parser_class=SuggestingArgumentParser
    )

    _add_info(csub)
    _add_stats(csub)
    _add_resample(csub)
    _add_merge(csub)
    export_sub = add_group(csub, "export", "write a collection to a file or format")
    _add_to_file(export_sub)
    _add_to_cog(export_sub)
    _add_to_netcdf(export_sub)
    _add_to_zarr(export_sub)
    _add_to_kerchunk(export_sub)
    _add_reproject(csub)
    _add_crop(csub)
    _add_align(csub)
    import_sub = add_group(
        csub, "import", "build a collection from a remote source or saved STAC items"
    )
    _add_from_point(import_sub)
    _add_from_stac(import_sub)
