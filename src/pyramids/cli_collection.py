"""Collection / datacube CLI command group — ``pyramids collection …`` (M-5).

Thin wrappers over :class:`pyramids.dataset.collection.DatasetCollection`:
summarise a temporal stack of co-registered rasters, and write a time reduction
(mean / sum / min / max / std / var) as a single raster.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
from typing import Any

from pyramids._cli_shared import info_print, json_safe, refuse_existing
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.collection import DatasetCollection

_REDUCERS = ("mean", "sum", "min", "max", "std", "var")


def _open_collection(pattern: str) -> DatasetCollection:
    """Open a :class:`DatasetCollection` from a glob or directory ``pattern``.

    Args:
        pattern: A glob (``tiles/*.tif``) or a directory of co-registered rasters.

    Returns:
        DatasetCollection: The assembled stack.

    Raises:
        ValueError: The pattern matched no rasters.
    """
    paths = sorted(_glob.glob(pattern, recursive=True))
    if not paths:
        raise ValueError(f"no rasters matched {pattern!r}")
    return DatasetCollection.from_files(paths)


def _cmd_collection_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids collection info`` — summarise a temporal raster stack.

    Args:
        args: Parsed args with ``glob`` and ``json``.

    Returns:
        int: ``0`` on success.
    """
    coll = _open_collection(args.glob)
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
    """Handle ``pyramids collection stats`` — write a time reduction as a raster.

    Args:
        args: Parsed args with ``glob``, ``output``, ``reduce``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    coll = _open_collection(args.glob)
    # The reducers return an ndarray on the stack's shared grid; re-attach the
    # base raster's georeferencing so the result is writable (shape is already
    # (bands, rows, cols), the layout from_array expects).
    array = getattr(coll, args.reduce)(skipna=True)
    base = coll.base
    Dataset.from_array(
        array,
        geo_ref=GeoReference(geo=base.geotransform, epsg=base.epsg or base.crs),
    ).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_collection_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``collection`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    collection = sub.add_parser("collection", help="temporal raster-stack commands")
    csub = collection.add_subparsers(dest="command", required=True)

    c_info = csub.add_parser("info", help="summarize a raster stack")
    c_info.add_argument(
        "glob", help="glob (quote it) or directory of co-registered rasters"
    )
    c_info.add_argument("--json", action="store_true", help="emit JSON")
    c_info.set_defaults(func=_cmd_collection_info)

    c_stats = csub.add_parser("stats", help="write a time reduction of a stack")
    c_stats.add_argument("glob", help="glob (quote it) of co-registered rasters")
    c_stats.add_argument("output", help="destination raster path")
    c_stats.add_argument(
        "--reduce", required=True, choices=_REDUCERS, help="time-reduction function"
    )
    c_stats.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    c_stats.set_defaults(func=_cmd_collection_stats)
