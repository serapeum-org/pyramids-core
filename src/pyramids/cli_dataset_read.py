"""Raster (`Dataset`) constructor CLI commands for the ``pyramids`` toolkit.

Thin 1:1 wrappers over :class:`pyramids.dataset.Dataset` classmethod constructors
that each read some source (raw bytes, a tabular CSV, a point layer, single-band
files, an archive, a Zarr store, or nothing at all) and write the result to an
output raster. Extracted into its own ``cli_<domain>.py`` so the top-level parser
only wires the group together; every handler maps 1:1 onto an existing ``Dataset``
call and does no business logic of its own.

Commands registered here:

- ``from-bytes`` — open a raster held as raw bytes (a file or stdin).
- ``from-dataframe`` — rebuild a raster from a ``(band, y, x)`` CSV table.
- ``from-points`` — grid a point vector layer (``gdal.Grid``).
- ``from-point-arrays`` — grid raw ``x`` / ``y`` / ``z`` columns of a CSV.
- ``from-band-files`` — stack single-band rasters into one multi-band raster.
- ``from-archive`` — stack the raster members of a zip / tar / gzip archive.
- ``from-zarr`` — read a pyramids-written Zarr store (optional ``zarr`` dep).
- ``create`` — allocate a new raster (optionally no-data filled).
- ``create-empty`` — allocate a header-only raster (out-of-core sibling).
- ``empty-like`` — allocate a header-only raster aligned to a template.
- ``dataset-like`` — stamp a template's georeference onto a ``.npy`` array.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd

from pyramids._cli_shared import info_print, map_lookup_errors, refuse_existing
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

_HELP_OUTPUT = "destination raster path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_ALGORITHM = "gdal.Grid algorithm string (default: inverse-distance weighting)"
_HELP_BBOX = "output extent MINX MINY MAXX MAXY"
_BBOX_METAVAR = ("MINX", "MINY", "MAXX", "MAXY")


@contextlib.contextmanager
def _needs_zarr_extra() -> Iterator[None]:
    """Remap a missing-``zarr`` ``ImportError`` to a clean ``ValueError``.

    ``from_zarr`` imports ``zarr`` from the optional ``[lazy]`` extra; its absence
    raises ``ImportError``, which :func:`pyramids.cli.main` does not catch (it would
    leak the "unexpected failure" branch). Turn it into a clean user error naming
    the extra.
    """
    try:
        yield
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [lazy] extra (zarr); install it with "
            "`pip install pyramids-gis[lazy]`"
        ) from exc


def _parse_nodata(raw: str) -> float | None:
    """Parse a ``--no-data-value`` token into a float, ``NaN``, or ``None``.

    Args:
        raw: The raw ``--no-data-value`` string.

    Returns:
        float | None: ``None`` for ``none``/``unset``, otherwise the float value
        (``nan`` yields ``float('nan')``).

    Raises:
        ValueError: ``raw`` is not a number, ``nan``, nor ``none``/``unset``.
    """
    spec = raw.strip().lower()
    result: float | None
    if spec in {"none", "unset"}:
        result = None
    else:
        try:
            result = float(raw)
        except ValueError:
            raise ValueError(
                f"--no-data-value {raw!r} is not a number, 'nan', or 'none'"
            ) from None
    return result


def _build_geo_ref(args: argparse.Namespace) -> GeoReference | None:
    """Build a :class:`GeoReference` from the shared geo-placement flags.

    Reads ``--geo`` (a six-term affine transform) or ``--top-left`` + ``--cell-size``
    (a north-up grid), plus an optional ``--epsg``. Returns ``None`` when no
    placement flag is given, so an optional ``geo_ref`` constructor keeps its default.

    Args:
        args: Parsed arguments carrying ``geo``, ``top_left``, ``cell_size``, ``epsg``.

    Returns:
        GeoReference | None: The reference, or ``None`` when no placement was given.
    """
    kwargs: dict = {}
    if args.epsg is not None:
        kwargs["epsg"] = args.epsg
    placed = False
    if args.geo is not None:
        kwargs["geo"] = tuple(args.geo)
        placed = True
    if args.top_left is not None:
        kwargs["top_left_corner"] = tuple(args.top_left)
        placed = True
    if args.cell_size is not None:
        kwargs["cell_size"] = args.cell_size
        placed = True
    result: GeoReference | None = None
    if placed or kwargs:
        result = GeoReference(**kwargs)
    return result


def _add_geo_ref_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--geo`` / ``--top-left`` / ``--cell-size`` / ``--epsg`` flags."""
    parser.add_argument(
        "--geo",
        nargs=6,
        type=float,
        metavar=("X", "DX", "RX", "Y", "RY", "DY"),
        help="affine geotransform (x_min, pixel_width, row_rot, y_max, col_rot, pixel_height)",
    )
    parser.add_argument(
        "--top-left",
        dest="top_left",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="top-left corner (used with --cell-size to build a north-up grid)",
    )
    parser.add_argument(
        "--cell-size", dest="cell_size", type=float, help="square pixel size"
    )
    parser.add_argument(
        "--epsg", type=int, default=None, help="EPSG code of the output CRS"
    )


def _cmd_from_bytes(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-bytes`` — open a raster held as raw bytes.

    Reads the bytes from ``input`` (a file) or, with ``--stdin``, from standard
    input, opens them via :meth:`Dataset.from_bytes`, and writes the result.

    Args:
        args: Parsed args with ``input``, ``output``, ``suffix``, ``name``,
            ``stdin``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: Neither ``input`` nor ``--stdin`` was given.
    """
    refuse_existing(args.output, args.overwrite)
    if args.stdin:
        data = sys.stdin.buffer.read()
    elif args.input is not None:
        with open(args.input, "rb") as handle:
            data = handle.read()
    else:
        raise ValueError("from-bytes needs an input path or --stdin.")
    ds = Dataset.from_bytes(data, suffix=args.suffix, name=args.name)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_dataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-dataframe`` — rebuild a raster from a CSV table.

    Reads a CSV whose columns include the ``--x`` / ``--y`` (and optional ``--band``)
    axes and a single value column, sets them as the ``(band, y, x)`` MultiIndex, and
    hands the frame to :meth:`Dataset.from_dataframe` (the inverse of
    ``Dataset.to_dataframe``).

    Args:
        args: Parsed args with ``csv``, ``output``, ``x``, ``y``, ``band``,
            ``value``, ``crs``, ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    frame = pd.read_csv(args.csv)
    index_columns = [name for name in (args.band, args.y, args.x) if name is not None]
    kwargs: dict = {"crs": args.crs, "x": args.x, "y": args.y}
    if args.no_data_value is not None:
        kwargs["no_data_value"] = _parse_nodata(args.no_data_value)
    with map_lookup_errors():
        indexed = frame.set_index(index_columns)
        if args.value is not None:
            indexed = indexed[[args.value]]
        ds = Dataset.from_dataframe(indexed, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _grid_kwargs(args: argparse.Namespace) -> dict:
    """Collect the shared ``gdal.Grid`` sizing kwargs from the point-grid parsers."""
    kwargs: dict = {"algorithm": args.algorithm}
    if args.cell_size is not None:
        kwargs["cell_size"] = args.cell_size
    if args.width is not None:
        kwargs["width"] = args.width
    if args.height is not None:
        kwargs["height"] = args.height
    if args.bbox is not None:
        kwargs["bbox"] = tuple(args.bbox)
    return kwargs


def _cmd_from_points(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-points`` — interpolate a point vector layer to a grid.

    Args:
        args: Parsed args with ``vector``, ``output``, ``value_column``,
            ``algorithm``, ``cell_size``, ``width``, ``height``, ``bbox``, ``epsg``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    points = FeatureCollection.read_file(args.vector)
    kwargs = _grid_kwargs(args)
    if args.epsg is not None:
        kwargs["epsg"] = args.epsg
    ds = Dataset.from_points(points, args.value_column, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_point_arrays(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-point-arrays`` — grid raw x/y/z columns of a CSV.

    Args:
        args: Parsed args with ``csv``, ``output``, ``x``, ``y``, ``z``,
            ``algorithm``, ``cell_size``, ``width``, ``height``, ``bbox``, ``crs``,
            ``epsg``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    frame = pd.read_csv(args.csv)
    kwargs = _grid_kwargs(args)
    if args.crs is not None:
        kwargs["crs"] = args.crs
    if args.epsg is not None:
        kwargs["epsg"] = args.epsg
    with map_lookup_errors():
        x = frame[args.x].to_numpy()
        y = frame[args.y].to_numpy()
        z = frame[args.z].to_numpy()
    ds = Dataset.from_point_arrays(x, y, z, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_band_files(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-band-files`` — stack single-band rasters into one.

    Args:
        args: Parsed args with ``files`` (>= 1), ``output``, ``band_names``,
            ``align``, ``resampling``, ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {"band_names": args.band_names, "align": args.align}
    if args.resampling is not None:
        kwargs["resampling"] = args.resampling
    if args.no_data_value is not None:
        kwargs["no_data_value"] = _parse_nodata(args.no_data_value)
    ds = Dataset.from_band_files(args.files, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_archive(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-archive`` — stack raster members of an archive.

    Args:
        args: Parsed args with ``archive``, ``output``, ``kind``, ``member_glob``,
            ``band_names``, ``align``, ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {
        "kind": args.kind,
        "member_glob": args.member_glob,
        "band_names": args.band_names,
        "align": args.align,
    }
    if args.no_data_value is not None:
        kwargs["no_data_value"] = _parse_nodata(args.no_data_value)
    ds = Dataset.from_archive(args.archive, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids from-zarr`` — read a pyramids-written Zarr store.

    Args:
        args: Parsed args with ``store``, ``output``, ``level``, ``data_name``,
            ``storage_options``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {"level": args.level, "data_name": args.data_name}
    if args.storage_options is not None:
        kwargs["storage_options"] = json.loads(args.storage_options)
    with _needs_zarr_extra():
        ds = Dataset.from_zarr(args.store, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    """Handle ``pyramids create`` — allocate a new raster, optionally no-data filled.

    Args:
        args: Parsed args with ``output``, ``rows``, ``columns``, ``dtype``,
            ``bands``, the geo-placement flags, ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: No geo-placement flag (``--geo`` or ``--top-left``/``--cell-size``)
            was given — ``create`` requires a ``GeoReference``.
    """
    refuse_existing(args.output, args.overwrite)
    placement_given = args.geo is not None or (
        args.top_left is not None and args.cell_size is not None
    )
    geo_ref = _build_geo_ref(args)
    if geo_ref is None or not placement_given:
        raise ValueError(
            "create needs a placement: pass --geo, or --top-left with --cell-size."
        )
    kwargs: dict = {"geo_ref": geo_ref}
    if args.no_data_value is not None:
        kwargs["no_data_value"] = _parse_nodata(args.no_data_value)
    with map_lookup_errors():
        ds = Dataset.create(args.rows, args.columns, args.dtype, args.bands, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_create_empty(args: argparse.Namespace) -> int:
    """Handle ``pyramids create-empty`` — allocate a header-only raster.

    Args:
        args: Parsed args with ``output``, ``rows``, ``cols``, ``bands``, ``dtype``,
            the optional geo-placement flags, ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    kwargs: dict = {"bands": args.bands, "dtype": args.dtype}
    geo_ref = _build_geo_ref(args)
    if geo_ref is not None:
        kwargs["geo_ref"] = geo_ref
    if args.no_data_value is not None:
        kwargs["no_data_value"] = _parse_nodata(args.no_data_value)
    with map_lookup_errors():
        ds = Dataset.create_empty(args.rows, args.cols, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_empty_like(args: argparse.Namespace) -> int:
    """Handle ``pyramids empty-like`` — allocate a header-only raster from a template.

    Args:
        args: Parsed args with ``template``, ``output``, ``dtype``, ``bands``,
            ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    template = Dataset.read_file(args.template)
    kwargs: dict = {"dtype": args.dtype, "bands": args.bands}
    if args.no_data_value is not None:
        kwargs["no_data_value"] = _parse_nodata(args.no_data_value)
    ds = Dataset.empty_like(template, **kwargs)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_dataset_like(args: argparse.Namespace) -> int:
    """Handle ``pyramids dataset-like`` — stamp a template's georeference onto an array.

    Reads a NumPy ``.npy`` array (``allow_pickle=False``) and builds a raster that
    carries ``template``'s geotransform / CRS / no-data while holding the array's
    values (the array must match the template's rows / columns).

    Args:
        args: Parsed args with ``template``, ``array``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    template = Dataset.read_file(args.template)
    array = np.load(args.array, allow_pickle=False)
    ds = Dataset.dataset_like(template, array)
    ds.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _add_from_bytes(sub: argparse._SubParsersAction) -> None:
    """Register ``from-bytes``."""
    parser = sub.add_parser("from-bytes", help="open a raster held as raw bytes")
    parser.add_argument(
        "input", nargs="?", help="file of raw raster bytes (omit with --stdin)"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument(
        "--stdin", action="store_true", help="read the bytes from standard input"
    )
    parser.add_argument(
        "--suffix",
        default=".tif",
        help="extension hint for GDAL driver detection (default: .tif)",
    )
    parser.add_argument("--name", help="cosmetic dataset label")
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_bytes)


def _add_from_dataframe(sub: argparse._SubParsersAction) -> None:
    """Register ``from-dataframe``."""
    parser = sub.add_parser(
        "from-dataframe", help="rebuild a raster from a (band, y, x) CSV table"
    )
    parser.add_argument("csv", help="source CSV path")
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument(
        "--x", required=True, help="column holding the x (column) coordinate"
    )
    parser.add_argument(
        "--y", required=True, help="column holding the y (row) coordinate"
    )
    parser.add_argument(
        "--band", help="column holding the 0-based band axis (single band if omitted)"
    )
    parser.add_argument(
        "--value", help="the value column (required if the CSV has extra columns)"
    )
    parser.add_argument("--crs", help="CRS for the result (EPSG code or CRS string)")
    parser.add_argument(
        "--no-data-value", dest="no_data_value", help="no-data sentinel for gaps"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_dataframe)


def _add_grid_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``gdal.Grid`` sizing options to a point-grid parser."""
    parser.add_argument(
        "--algorithm", default="invdist:power=2.0:smoothing=0.0", help=_HELP_ALGORITHM
    )
    parser.add_argument(
        "--cell-size", dest="cell_size", type=float, help="output pixel size"
    )
    parser.add_argument("--width", type=int, help="output width in pixels")
    parser.add_argument("--height", type=int, help="output height in pixels")
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=_BBOX_METAVAR, help=_HELP_BBOX
    )


def _add_from_points(sub: argparse._SubParsersAction) -> None:
    """Register ``from-points``."""
    parser = sub.add_parser(
        "from-points", help="interpolate a point vector layer to a grid"
    )
    parser.add_argument("vector", help="source point vector path")
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument(
        "--value-column",
        dest="value_column",
        required=True,
        help="numeric attribute to interpolate",
    )
    _add_grid_options(parser)
    parser.add_argument(
        "--epsg", type=int, help="output EPSG code (default: the points' CRS)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_points)


def _add_from_point_arrays(sub: argparse._SubParsersAction) -> None:
    """Register ``from-point-arrays``."""
    parser = sub.add_parser("from-point-arrays", help="grid raw x/y/z columns of a CSV")
    parser.add_argument("csv", help="source CSV path")
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--x", required=True, help="column holding the x coordinates")
    parser.add_argument("--y", required=True, help="column holding the y coordinates")
    parser.add_argument(
        "--z", required=True, help="column holding the values to interpolate"
    )
    _add_grid_options(parser)
    parser.add_argument("--crs", help="CRS to stamp on the output (label only)")
    parser.add_argument(
        "--epsg", type=int, help="output EPSG code (takes precedence over --crs)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_point_arrays)


def _add_from_band_files(sub: argparse._SubParsersAction) -> None:
    """Register ``from-band-files``."""
    parser = sub.add_parser(
        "from-band-files", help="stack single-band rasters into a multiband raster"
    )
    parser.add_argument(
        "files", nargs="+", help="single-band raster paths (one per band, in order)"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument(
        "--band-names",
        dest="band_names",
        nargs="+",
        help="explicit band names, one per file",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="resample mismatched inputs onto the first file's grid",
    )
    parser.add_argument(
        "--resampling", help="alignment resampling method (requires --align)"
    )
    parser.add_argument(
        "--no-data-value",
        dest="no_data_value",
        help="no-data sentinel for the output bands",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_band_files)


def _add_from_archive(sub: argparse._SubParsersAction) -> None:
    """Register ``from-archive``."""
    parser = sub.add_parser(
        "from-archive", help="stack the raster members of a zip/tar/gzip archive"
    )
    parser.add_argument(
        "archive", help="archive path or URL (.zip / .tar / .tar.gz / .gz)"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument(
        "--kind",
        default="auto",
        help="archive kind: zip, tar, gzip, or auto (default: auto)",
    )
    parser.add_argument(
        "--member-glob",
        dest="member_glob",
        default="*",
        help="fnmatch pattern for members (default: *)",
    )
    parser.add_argument(
        "--band-names",
        dest="band_names",
        nargs="+",
        help="explicit band names, one per member",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="resample mismatched members onto the first member's grid",
    )
    parser.add_argument(
        "--no-data-value",
        dest="no_data_value",
        help="no-data sentinel for the output bands",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_archive)


def _add_from_zarr(sub: argparse._SubParsersAction) -> None:
    """Register ``from-zarr``."""
    parser = sub.add_parser("from-zarr", help="read a pyramids-written Zarr store")
    parser.add_argument("store", help="input store (path / fsspec URL)")
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="pyramid downsample factor to read (1 = full resolution)",
    )
    parser.add_argument(
        "--data-name", dest="data_name", help="explicit data-array name to read"
    )
    parser.add_argument(
        "--storage-options",
        dest="storage_options",
        help="fsspec storage options as a JSON object",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_zarr)


def _add_create(sub: argparse._SubParsersAction) -> None:
    """Register ``create``."""
    parser = sub.add_parser(
        "create", help="allocate a new raster, optionally no-data filled"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--rows", type=int, required=True, help="number of rows")
    parser.add_argument("--columns", type=int, required=True, help="number of columns")
    parser.add_argument(
        "--dtype", required=True, help="numpy dtype name (e.g. float32)"
    )
    parser.add_argument("--bands", type=int, required=True, help="number of bands")
    _add_geo_ref_options(parser)
    parser.add_argument(
        "--no-data-value", dest="no_data_value", help="no-data value to stamp and fill"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_create)


def _add_create_empty(sub: argparse._SubParsersAction) -> None:
    """Register ``create-empty``."""
    parser = sub.add_parser(
        "create-empty", help="allocate a header-only raster (out-of-core sibling)"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--rows", type=int, required=True, help="number of rows")
    parser.add_argument("--cols", type=int, required=True, help="number of columns")
    parser.add_argument(
        "--bands", type=int, default=1, help="number of bands (default: 1)"
    )
    parser.add_argument(
        "--dtype", default="float32", help="numpy dtype name (default: float32)"
    )
    _add_geo_ref_options(parser)
    parser.add_argument(
        "--no-data-value",
        dest="no_data_value",
        help="no-data sentinel (default: -9999)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_create_empty)


def _add_empty_like(sub: argparse._SubParsersAction) -> None:
    """Register ``empty-like``."""
    parser = sub.add_parser(
        "empty-like", help="allocate a header-only raster aligned to a template"
    )
    parser.add_argument(
        "template", help="template raster whose footprint the output copies"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--dtype", help="numpy dtype name (default: the template's)")
    parser.add_argument(
        "--bands", type=int, help="number of bands (default: the template's)"
    )
    parser.add_argument(
        "--no-data-value",
        dest="no_data_value",
        help="no-data sentinel (default: inherit)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_empty_like)


def _add_dataset_like(sub: argparse._SubParsersAction) -> None:
    """Register ``dataset-like``."""
    parser = sub.add_parser(
        "dataset-like", help="stamp a template's georeference onto a .npy array"
    )
    parser.add_argument(
        "template", help="template raster whose georeference the output copies"
    )
    parser.add_argument(
        "array", help="NumPy .npy array to store (must match the template's shape)"
    )
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_dataset_like)


def add_dataset_read_commands(sub: argparse._SubParsersAction) -> None:
    """Register the raster (`Dataset`) constructor commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    _add_from_bytes(sub)
    _add_from_dataframe(sub)
    _add_from_points(sub)
    _add_from_point_arrays(sub)
    _add_from_band_files(sub)
    _add_from_archive(sub)
    _add_from_zarr(sub)
    _add_create(sub)
    _add_create_empty(sub)
    _add_empty_like(sub)
    _add_dataset_like(sub)
