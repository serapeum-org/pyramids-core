"""NetCDF inspection and global-attribute commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`: per-variable statistics
(`summary`), the full metadata record (`metadata`), a dimension's coordinate values
(`dim-values`), the formatted time axis (`time-values`), and setting/deleting a
global attribute (`set-attr`/`del-attr`). Reads go through the ``NetCDF`` class; no
xarray is imported here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

from pyramids._cli_shared import (
    info_print,
    json_safe,
    json_sanitize,
    map_lookup_errors,
    refuse_existing,
)
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_JSON = "emit JSON"


def _metadata_payload(metadata: object) -> object:
    """Convert a metadata record to a JSON-friendly structure.

    Args:
        metadata: The object returned by ``NetCDF.refresh_metadata``.

    Returns:
        object: A dataclass converted with :func:`dataclasses.asdict`, else the
        object's ``__dict__``, else its string form.
    """
    if dataclasses.is_dataclass(metadata) and not isinstance(metadata, type):
        return dataclasses.asdict(metadata)
    if hasattr(metadata, "__dict__"):
        return vars(metadata)
    return str(metadata)


def _cmd_nc_summary(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc summary`` — per-variable statistics.

    Args:
        args: Parsed args with ``file``, ``vars``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    with map_lookup_errors():
        frame = nc.summary(variables=args.vars or None)
    if args.json:
        print(json.dumps(json_sanitize(frame.to_dict(orient="index"))))
    else:
        print(frame.to_string())
    return 0


def _cmd_nc_metadata(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc metadata`` — the full metadata record.

    Args:
        args: Parsed args with ``file``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    payload = _metadata_payload(NetCDF.read_file(args.file).refresh_metadata())
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        print(payload)
    return 0


def _cmd_nc_dim_values(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc dim-values`` — a dimension's coordinate values.

    Args:
        args: Parsed args with ``file``, ``dim``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    values = NetCDF.read_file(args.file).get_dimension_values(args.dim)
    listed = [] if values is None else [json_safe(v) for v in values.tolist()]
    if args.json:
        print(json.dumps({"dim": args.dim, "values": listed}))
    elif values is None:
        print(f"{args.dim}: no coordinate values")
    else:
        print(f"{args.dim}: {listed}")
    return 0


def _cmd_nc_time_values(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc time-values`` — the time axis values.

    The default emits the decoded, ``--format``-formatted labels (``get_time_variable``).
    ``--raw`` instead emits the stored numeric offsets (``get_time_values``), i.e. the
    axis as the file holds it before CF decoding.

    Args:
        args: Parsed args with ``file``, ``var``, ``format``, ``raw``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    listed: list[Any]
    if args.raw:
        values = nc.get_time_values(var_name=args.var)
        listed = [] if values is None else [json_safe(v) for v in values.tolist()]
    else:
        listed = nc.get_time_variable(var_name=args.var, time_format=args.format) or []
    if args.json:
        print(json.dumps({"var": args.var, "values": listed}))
    else:
        print(f"{args.var}: {listed}")
    return 0


def _cmd_nc_set_attr(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc set-attr`` — set a global attribute into a new file.

    The input is copied to ``output`` first, then the attribute is written on the
    writable copy, so the source file is never modified.

    Args:
        args: Parsed args with ``file``, ``output``, ``name``, ``value``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    NetCDF.read_file(args.file).to_file(args.output)
    NetCDF.read_file(args.output, read_only=False).set_global_attribute(
        args.name, args.value
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_del_attr(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc del-attr`` — delete a global attribute into a new file.

    The input is copied to ``output`` first, then the attribute is removed from the
    writable copy (deleting a missing attribute is a no-op), so the source file is
    never modified.

    Args:
        args: Parsed args with ``file``, ``output``, ``name``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    NetCDF.read_file(args.file).to_file(args.output)
    NetCDF.read_file(args.output, read_only=False).delete_global_attribute(args.name)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_inspect_commands(ncsub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` inspection and global-attribute commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_sum = ncsub.add_parser("summary", help="per-variable statistics")
    nc_sum.add_argument("file", help=_HELP_NC_FILE)
    nc_sum.add_argument(
        "--vars", nargs="+", help="variables to summarize (default: all)"
    )
    nc_sum.add_argument("--json", action="store_true", help=_HELP_JSON)
    nc_sum.set_defaults(func=_cmd_nc_summary)

    nc_meta = ncsub.add_parser("metadata", help="the full metadata record")
    nc_meta.add_argument("file", help=_HELP_NC_FILE)
    nc_meta.add_argument("--json", action="store_true", help=_HELP_JSON)
    nc_meta.set_defaults(func=_cmd_nc_metadata)

    nc_dim = ncsub.add_parser("dim-values", help="a dimension's coordinate values")
    nc_dim.add_argument("file", help=_HELP_NC_FILE)
    nc_dim.add_argument("--dim", required=True, help="dimension name")
    nc_dim.add_argument("--json", action="store_true", help=_HELP_JSON)
    nc_dim.set_defaults(func=_cmd_nc_dim_values)

    nc_time = ncsub.add_parser("time-values", help="the formatted time-axis labels")
    nc_time.add_argument("file", help=_HELP_NC_FILE)
    nc_time.add_argument(
        "--var",
        default="time",
        help="time variable name (empty result if it is not a time axis)",
    )
    nc_time.add_argument(
        "--format", default="%Y-%m-%d", help="strftime format for the labels"
    )
    nc_time.add_argument(
        "--raw",
        action="store_true",
        help="emit the stored numeric offsets instead of decoded labels",
    )
    nc_time.add_argument("--json", action="store_true", help=_HELP_JSON)
    nc_time.set_defaults(func=_cmd_nc_time_values)

    nc_set = ncsub.add_parser("set-attr", help="set a global attribute")
    nc_set.add_argument("file", help=_HELP_NC_FILE)
    nc_set.add_argument("output", help="destination NetCDF path")
    nc_set.add_argument("--name", required=True, help="attribute name")
    nc_set.add_argument("--value", required=True, help="attribute value")
    nc_set.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_set.set_defaults(func=_cmd_nc_set_attr)

    nc_del = ncsub.add_parser("del-attr", help="delete a global attribute")
    nc_del.add_argument("file", help=_HELP_NC_FILE)
    nc_del.add_argument("output", help="destination NetCDF path")
    nc_del.add_argument("--name", required=True, help="attribute name")
    nc_del.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_del.set_defaults(func=_cmd_nc_del_attr)
