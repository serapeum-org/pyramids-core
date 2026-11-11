"""Read-only (plus one in-place editor) raster inspection CLI commands.

A second slice of the raster (`Dataset`) CLI surface, kept apart from
:mod:`pyramids.cli_dataset` so the big single-shot raster/COG group and these
smaller inspection verbs each stay readable. Every command is a thin 1:1 wrapper
over an existing :class:`~pyramids.dataset.Dataset` call and registers itself as a
**top-level** parser on the shared sub-action.

Commands registered by :func:`add_dataset_inspect_commands`:

- ``stats`` → :meth:`Dataset.stats`
- ``histogram`` → :meth:`Dataset.get_histogram`
- ``get-metadata`` → :meth:`Dataset.get_meta_data`
- ``get-attribute-table`` → :meth:`Dataset.get_attribute_table`
- ``set-attribute-table`` → :meth:`Dataset.set_attribute_table` (edits the raster
  **in place** — the attribute table is written into the file, so there is no
  output path and no ``--overwrite``)
- ``mask-flags`` → :meth:`Dataset.mask_flags`
- ``footprint`` → :meth:`Dataset.footprint`
- ``cell-area`` → :meth:`Dataset.cell_area`
- ``domain-area`` → :meth:`Dataset.domain_area`
- ``count-domain-cells`` → :meth:`Dataset.count_domain_cells`
- ``same-grid`` → :meth:`Dataset.same_grid`
- ``checksum`` → :meth:`Dataset.checksum`

No methods are deferred — each of the twelve is a natural one-shot verb.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import cast

import pandas as pd
from pandas import DataFrame

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    map_lookup_errors,
    resolve_vector_driver,
)
from pyramids._cli_shared import refuse_existing as _refuse_existing
from pyramids.cli_dataset import _HELP_INSPECT_RASTER, _HELP_JSON
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

_HELP_BAND = "0-based band index"


def _read_table(path: str) -> DataFrame:
    """Load an attribute table from a ``.csv`` or ``.json`` file into a DataFrame.

    Args:
        path: Path to the table source; the extension picks the reader
            (``.csv`` → :func:`pandas.read_csv`, ``.json`` → :func:`pandas.read_json`).

    Returns:
        DataFrame: The parsed table.

    Raises:
        ValueError: The extension is neither ``.csv`` nor ``.json``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        table = pd.read_csv(path)
    elif ext == ".json":
        table = pd.read_json(path)
    else:
        raise ValueError(
            f"cannot read an attribute table from {path!r}; pass a .csv or .json file."
        )
    return table


def _cmd_stats(args: argparse.Namespace) -> int:
    """Handle `pyramids raster stats` — band statistics (min/max/mean/std).

    Args:
        args: Parsed arguments with `file`, `band`, `exact`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    frame = ds.stats(band=args.band, approx_ok=not args.exact)
    records = [
        {
            "band": str(name),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
        for name, row in frame.iterrows()
    ]
    if args.json:
        print(json.dumps(json_sanitize({"stats": records})))
    else:
        for record in records:
            print(
                f"{record['band']}: min={record['min']} max={record['max']} "
                f"mean={record['mean']} std={record['std']}"
            )
    return 0


def _cmd_histogram(args: argparse.Namespace) -> int:
    """Handle `pyramids raster histogram` — a band's value histogram.

    Args:
        args: Parsed arguments with `file`, `band`, `bins`, `min_value`,
            `max_value`, `include_out_of_range`, `approx`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    with map_lookup_errors():
        counts, ranges = ds.get_histogram(
            band=args.band,
            bins=args.bins,
            min_value=args.min_value,
            max_value=args.max_value,
            include_out_of_range=args.include_out_of_range,
            approx_ok=args.approx,
        )
    payload = {
        "band": args.band,
        "bins": args.bins,
        "counts": [int(count) for count in counts],
        "ranges": [[float(low), float(high)] for low, high in ranges],
    }
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for count, (low, high) in zip(counts, ranges):
            print(f"[{low}, {high}): {int(count)}")
    return 0


def _cmd_get_meta_data(args: argparse.Namespace) -> int:
    """Handle `pyramids raster get-metadata` — a GDAL metadata domain's contents.

    Args:
        args: Parsed arguments with `file`, `domain`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    meta = ds.get_meta_data(domain=args.domain)
    if args.json:
        print(json.dumps(json_sanitize({"domain": args.domain, "metadata": meta})))
    elif isinstance(meta, dict):
        for key, value in meta.items():
            print(f"{key}: {value}")
    else:
        for item in meta:
            print(item)
    return 0


def _cmd_get_attribute_table(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced get-attribute-table` — a band's raster attribute table.

    Args:
        args: Parsed arguments with `file`, `band`, and `json`.

    Returns:
        int: `0` on success (a band without a table prints a notice / emits `null`).
    """
    ds = Dataset.read_file(args.file)
    with map_lookup_errors():
        frame = ds.get_attribute_table(band=args.band)
    records = None if frame is None else frame.to_dict(orient="records")
    if args.json:
        print(
            json.dumps(json_sanitize({"band": args.band, "attribute_table": records}))
        )
    elif records is None:
        print(f"band {args.band} has no attribute table")
    else:
        for record in records:
            print(record)
    return 0


def _cmd_set_attribute_table(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced set-attribute-table` — attach an attribute table in place.

    The table is read from a CSV/JSON file and written **into** the raster (the
    raster is edited in place), so there is no output path and no `--overwrite`.

    Args:
        args: Parsed arguments with `input`, `table`, and `band`.

    Returns:
        int: `0` on success.
    """
    table = _read_table(args.table)
    # Must be opened in update mode and closed for the attribute table to persist
    # into the file (a read-only open keeps it only for this process's lifetime).
    ds = Dataset.read_file(args.input, read_only=False)
    with map_lookup_errors():
        ds.set_attribute_table(table, band=args.band)
    ds.close()
    info_print(args, f"set attribute table on {args.input}")
    return 0


def _cmd_mask_flags(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced mask-flags` — decode a band's GDAL mask flags.

    Args:
        args: Parsed arguments with `file`, `band`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    with map_lookup_errors():
        flags = ds.mask_flags(band=args.band)
    payload = {
        "band": args.band,
        "all_valid": flags.all_valid,
        "per_dataset": flags.per_dataset,
        "alpha": flags.alpha,
        "nodata": flags.nodata,
    }
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_footprint(args: argparse.Namespace) -> int:
    """Handle `pyramids raster footprint` — the covered extent of a band as polygons.

    Writes the polygons to `output` when given; otherwise emits them as GeoJSON
    with `--json`, or prints a one-line summary.

    Args:
        args: Parsed arguments with `file`, `output`, `band`, `exclude_values`,
            `max_samples`, `driver`, `overwrite`, and `json`.

    Returns:
        int: `0` on success (a fully no-data band prints a notice / empty GeoJSON).
    """
    ds = Dataset.read_file(args.file)
    with map_lookup_errors():
        gdf = ds.footprint(
            band=args.band,
            exclude_values=args.exclude_values,
            max_samples=args.max_samples,
        )
    if gdf is None:
        if args.json:
            print(json.dumps({"type": "FeatureCollection", "features": []}))
        else:
            info_print(args, "footprint: no covered cells (band is all no-data)")
    elif args.output:
        _refuse_existing(args.output, args.overwrite)
        FeatureCollection(gdf).to_file(
            args.output, driver=resolve_vector_driver(args.output, args.driver)
        )
        info_print(args, f"wrote {args.output}")
    elif args.json:
        print(gdf.to_json())
    elif gdf.crs is not None and gdf.crs.is_geographic:
        print(
            f"footprint: {len(gdf)} polygon(s) (area omitted — CRS is geographic; "
            f"reproject for a real area, or use --json/--output for the geometry)"
        )
    else:
        total_area = float(gdf.geometry.area.sum())
        print(
            f"footprint: {len(gdf)} polygon(s), total area {total_area} (native CRS units)"
        )
    return 0


def _cmd_cell_area(args: argparse.Namespace) -> int:
    """Handle `pyramids raster cell-area` — per-cell ground area summary.

    The per-cell grid can be large (and is a broadcast view), so the command
    reports its summary (min/max/mean and the full-grid total), not the array.

    Args:
        args: Parsed arguments with `file`, `unit`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    areas = ds.cell_area(unit=args.unit)
    payload = {
        "unit": args.unit,
        "rows": ds.rows,
        "columns": ds.columns,
        "min": float(areas.min()),
        "max": float(areas.max()),
        "mean": float(areas.mean()),
        "total": float(areas.sum()),
    }
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_domain_area(args: argparse.Namespace) -> int:
    """Handle `pyramids raster domain-area` — ground area of a band's valid cells.

    Args:
        args: Parsed arguments with `file`, `band`, `unit`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    area = ds.domain_area(band=args.band, unit=args.unit)
    payload = {"band": args.band, "unit": args.unit, "area": float(area)}
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        print(f"{area} {args.unit}")
    return 0


def _cmd_count_domain_cells(args: argparse.Namespace) -> int:
    """Handle `pyramids raster count-domain-cells` — number of a band's valid cells.

    Args:
        args: Parsed arguments with `file`, `band`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    count = ds.count_domain_cells(band=args.band)
    if args.json:
        print(json.dumps({"band": args.band, "count": int(count)}))
    else:
        print(int(count))
    return 0


def _cmd_same_grid(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced same-grid` — whether two rasters share a grid (and CRS).

    Args:
        args: Parsed arguments with `a`, `b`, `no_compare_crs`, and `json`.

    Returns:
        int: `0` on success (the answer is reported, not encoded in the exit code).
    """
    first = Dataset.read_file(args.a)
    second = Dataset.read_file(args.b)
    same = bool(first.same_grid(second, compare_crs=not args.no_compare_crs))
    if args.json:
        print(json.dumps({"same_grid": same}))
    else:
        print("true" if same else "false")
    return 0


def _cmd_checksum(args: argparse.Namespace) -> int:
    """Handle `pyramids raster advanced checksum` — per-band GDAL checksum for integrity checks.

    Reports the GDAL checksum of every band, or of a single `--band`; two rasters
    that read back the same values share the same checksum(s), so it is a cheap
    integrity / round-trip verification.

    Args:
        args: Parsed arguments with `file`, `band` (default: all bands), and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    with map_lookup_errors():
        result = ds.checksum(band=args.band)
    if args.band is None:
        checksums = [int(value) for value in cast("list[int]", result)]
        if args.json:
            print(json.dumps({"checksums": checksums}))
        else:
            for index, value in enumerate(checksums):
                print(f"band {index}: {value}")
    else:
        single = int(cast("int", result))
        if args.json:
            print(json.dumps({"band": args.band, "checksum": single}))
        else:
            print(single)
    return 0


def add_dataset_inspect_commands(
    sub: argparse._SubParsersAction, advanced: argparse._SubParsersAction | None = None
) -> None:
    """Register the raster inspection commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
        advanced: The cross-cutting ``advanced`` group's subparsers action, or ``None``.
            When given, rarely-one-shot primitive verbs register there instead of on the
            main subsystem; isolation parsers omit it so those verbs stay flat locally.
    """
    stats = sub.add_parser("stats", help="band statistics (min/max/mean/std)")
    stats.add_argument("file", help=_HELP_INSPECT_RASTER)
    stats.add_argument(
        "--band", type=int, default=None, help=f"{_HELP_BAND} (default: all bands)"
    )
    stats.add_argument(
        "--exact",
        action="store_true",
        help="scan every pixel instead of letting GDAL approximate from overviews",
    )
    stats.add_argument("--json", action="store_true", help=_HELP_JSON)
    stats.set_defaults(func=_cmd_stats)

    histogram = sub.add_parser("histogram", help="a band's value histogram")
    histogram.add_argument("file", help=_HELP_INSPECT_RASTER)
    histogram.add_argument(
        "--band", type=int, default=0, help=f"{_HELP_BAND} (default 0)"
    )
    histogram.add_argument(
        "--bins", type=int, default=6, help="number of buckets (default 6)"
    )
    histogram.add_argument(
        "--min", dest="min_value", type=float, help="low edge, physical units"
    )
    histogram.add_argument(
        "--max", dest="max_value", type=float, help="high edge, physical units"
    )
    histogram.add_argument(
        "--include-out-of-range",
        action="store_true",
        help="fold out-of-range values into the first and last buckets",
    )
    histogram.add_argument(
        "--approx", action="store_true", help="approximate from a subsample / overviews"
    )
    histogram.add_argument("--json", action="store_true", help=_HELP_JSON)
    histogram.set_defaults(func=_cmd_histogram)

    meta = sub.add_parser("get-metadata", help="a GDAL metadata domain's contents")
    meta.add_argument("file", help=_HELP_INSPECT_RASTER)
    meta.add_argument(
        "--domain",
        default="",
        help="GDAL metadata domain (default: the default domain)",
    )
    meta.add_argument("--json", action="store_true", help=_HELP_JSON)
    meta.set_defaults(func=_cmd_get_meta_data)

    get_rat = (advanced or sub).add_parser(
        "get-attribute-table", help="a band's raster attribute table"
    )
    get_rat.add_argument("file", help=_HELP_INSPECT_RASTER)
    get_rat.add_argument(
        "--band", type=int, default=0, help=f"{_HELP_BAND} (default 0)"
    )
    get_rat.add_argument("--json", action="store_true", help=_HELP_JSON)
    get_rat.set_defaults(func=_cmd_get_attribute_table)

    set_rat = (advanced or sub).add_parser(
        "set-attribute-table",
        help="attach an attribute table to a raster band (edits the raster in place)",
    )
    set_rat.add_argument("input", help="raster path to edit in place")
    set_rat.add_argument("table", help="attribute-table source (.csv or .json)")
    set_rat.add_argument(
        "--band", type=int, default=None, help=f"{_HELP_BAND} (default: band 0)"
    )
    set_rat.set_defaults(func=_cmd_set_attribute_table)

    mask = (advanced or sub).add_parser(
        "mask-flags", help="decode a band's GDAL mask flags"
    )
    mask.add_argument("file", help=_HELP_INSPECT_RASTER)
    mask.add_argument("--band", type=int, default=0, help=f"{_HELP_BAND} (default 0)")
    mask.add_argument("--json", action="store_true", help=_HELP_JSON)
    mask.set_defaults(func=_cmd_mask_flags)

    footprint = sub.add_parser(
        "footprint", help="the covered extent of a band as polygons"
    )
    footprint.add_argument("file", help=_HELP_INSPECT_RASTER)
    footprint.add_argument(
        "output", nargs="?", help="destination vector path (omit to print GeoJSON)"
    )
    footprint.add_argument(
        "--band", type=int, default=0, help=f"{_HELP_BAND} (default 0)"
    )
    footprint.add_argument(
        "--exclude-values",
        nargs="+",
        type=float,
        help="values treated as uncovered in addition to the band's no-data sentinel",
    )
    footprint.add_argument(
        "--max-samples",
        type=int,
        help="cap pixels read to build the mask (yields an approximate footprint)",
    )
    footprint.add_argument(
        "--driver", help="OGR vector driver for --output (default: from extension)"
    )
    footprint.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output if it already exists",
    )
    footprint.add_argument("--json", action="store_true", help="emit GeoJSON")
    footprint.set_defaults(func=_cmd_footprint)

    cell_area = sub.add_parser("cell-area", help="per-cell ground area summary")
    cell_area.add_argument("file", help=_HELP_INSPECT_RASTER)
    cell_area.add_argument(
        "--unit",
        default="m2",
        choices=["m2", "km2", "ha"],
        help="area unit (default m2)",
    )
    cell_area.add_argument("--json", action="store_true", help=_HELP_JSON)
    cell_area.set_defaults(func=_cmd_cell_area)

    domain_area = sub.add_parser(
        "domain-area", help="ground area of a band's valid cells"
    )
    domain_area.add_argument("file", help=_HELP_INSPECT_RASTER)
    domain_area.add_argument(
        "--band", type=int, default=0, help=f"{_HELP_BAND} (default 0)"
    )
    domain_area.add_argument(
        "--unit",
        default="m2",
        choices=["m2", "km2", "ha"],
        help="area unit (default m2)",
    )
    domain_area.add_argument("--json", action="store_true", help=_HELP_JSON)
    domain_area.set_defaults(func=_cmd_domain_area)

    count = sub.add_parser(
        "count-domain-cells", help="number of a band's valid (non-no-data) cells"
    )
    count.add_argument("file", help=_HELP_INSPECT_RASTER)
    count.add_argument("--band", type=int, default=0, help=f"{_HELP_BAND} (default 0)")
    count.add_argument("--json", action="store_true", help=_HELP_JSON)
    count.set_defaults(func=_cmd_count_domain_cells)

    same = (advanced or sub).add_parser(
        "same-grid", help="whether two rasters share a grid (and CRS)"
    )
    same.add_argument("a", help="first raster path")
    same.add_argument("b", help="second raster path")
    same.add_argument(
        "--no-compare-crs",
        dest="no_compare_crs",
        action="store_true",
        help="compare only the grid (rows/cols/geotransform), not the CRS",
    )
    same.add_argument("--json", action="store_true", help=_HELP_JSON)
    same.set_defaults(func=_cmd_same_grid)

    checksum = (advanced or sub).add_parser(
        "checksum", help="per-band GDAL checksum for integrity verification"
    )
    checksum.add_argument("file", help=_HELP_INSPECT_RASTER)
    checksum.add_argument(
        "--band", type=int, default=None, help=f"{_HELP_BAND} (default: all bands)"
    )
    checksum.add_argument("--json", action="store_true", help=_HELP_JSON)
    checksum.set_defaults(func=_cmd_checksum)
