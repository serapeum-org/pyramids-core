"""NetCDF I/O export commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF` export methods: round-trip to a
NetCDF (`convert`), tabular export (`to-dataframe`), a structured dump (`to-dict`),
and a per-variable Zarr store (`to-zarr`). No xarray is imported here — the
underlying methods own that; the CLI only marshals paths and scalar options.
"""

from __future__ import annotations

import argparse
import json

from pyramids._cli_shared import info_print, json_sanitize, refuse_existing
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"


def _cmd_nc_convert(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc convert`` — re-write a file as a NetCDF.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    NetCDF.read_file(args.file).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_to_dataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-dataframe`` — export variables to a CSV table.

    Args:
        args: Parsed args with ``file``, ``output``, ``vars``, ``dropna``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    frame = nc.to_dataframe(variables=args.vars or None, dropna=args.dropna)
    frame.to_csv(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_to_dict(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-dict`` — print the file's structure as a dict.

    ``--data`` includes the variable arrays (large); the default is structure and
    metadata only.

    Args:
        args: Parsed args with ``file``, ``json``, ``data``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    payload = nc.to_dict(data=args.data)
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_nc_to_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-zarr`` — write one variable to a Zarr store.

    Zarr is written per variable (the container itself is not spatial), so ``--var``
    selects the gridded variable to store.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` names a variable that is not a gridded raster, or the
            store already exists without ``--overwrite``.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    variable = nc.get_variable(args.var)
    if not hasattr(variable, "to_zarr"):
        raise ValueError(
            f"variable {args.var!r} is not exposable as a raster plane "
            f"(GDAL returned a labeled array); it cannot be written to Zarr."
        )
    variable.to_zarr(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_io_commands(ncsub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` I/O export commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_conv = ncsub.add_parser("convert", help="re-write a file as a NetCDF")
    nc_conv.add_argument("file", help=_HELP_NC_FILE)
    nc_conv.add_argument("output", help="destination NetCDF path")
    nc_conv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_conv.set_defaults(func=_cmd_nc_convert)

    nc_df = ncsub.add_parser("to-dataframe", help="export variables to a CSV table")
    nc_df.add_argument("file", help=_HELP_NC_FILE)
    nc_df.add_argument("output", help="destination CSV path")
    nc_df.add_argument("--vars", nargs="+", help="variables to export (default: all)")
    nc_df.add_argument(
        "--dropna", action="store_true", help="drop rows with any missing value"
    )
    nc_df.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_df.set_defaults(func=_cmd_nc_to_dataframe)

    nc_dict = ncsub.add_parser("to-dict", help="print the file's structure as a dict")
    nc_dict.add_argument("file", help=_HELP_NC_FILE)
    nc_dict.add_argument("--json", action="store_true", help="emit JSON")
    nc_dict.add_argument(
        "--data", action="store_true", help="include variable arrays (large)"
    )
    nc_dict.set_defaults(func=_cmd_nc_to_dict)

    nc_zarr = ncsub.add_parser("to-zarr", help="write one variable to a Zarr store")
    nc_zarr.add_argument("file", help=_HELP_NC_FILE)
    nc_zarr.add_argument("output", help="destination .zarr store path")
    nc_zarr.add_argument("--var", required=True, help="variable to store")
    nc_zarr.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_zarr.set_defaults(func=_cmd_nc_to_zarr)
