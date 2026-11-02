"""NetCDF / multidim CLI command group — ``pyramids nc …`` (M-2).

Thin wrappers over :class:`pyramids.netcdf.NetCDF`: list the variables GDAL can
expose as raster planes, and extract one of them to a georeferenced raster. No
multidim logic lives here — reads go through GDAL as the ``NetCDF`` class
already does (never xarray).
"""

from __future__ import annotations

import argparse
import json

from pyramids._cli_shared import info_print, refuse_existing
from pyramids.netcdf import NetCDF


def _cmd_nc_list_vars(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc list-vars`` — list a file's raster-exposable variables.

    Args:
        args: Parsed args with ``file`` and ``json``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    names = list(nc.variable_names)
    if args.json:
        print(json.dumps({"path": args.file, "variables": names}))
    else:
        print(f"file: {args.file}")
        for name in names:
            print(f"  - {name}")
    return 0


def _cmd_nc_extract(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc extract`` — write one variable to a raster.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The variable is not exposable as a raster plane (GDAL
            returned a labeled array rather than a raster).
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    variable = nc.get_variable(args.var)
    if not hasattr(variable, "to_file"):
        raise ValueError(
            f"variable {args.var!r} is not exposable as a raster plane "
            f"(GDAL returned a labeled array); it cannot be written to "
            f"{args.output!r}."
        )
    variable.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_netcdf_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    nc = sub.add_parser("nc", help="NetCDF / multidim commands")
    ncsub = nc.add_subparsers(dest="command", required=True)

    nc_list = ncsub.add_parser("list-vars", help="list variables/subdatasets")
    nc_list.add_argument("file", help="NetCDF path")
    nc_list.add_argument("--json", action="store_true", help="emit JSON")
    nc_list.set_defaults(func=_cmd_nc_list_vars)

    nc_ex = ncsub.add_parser("extract", help="extract one variable to a raster")
    nc_ex.add_argument("file", help="NetCDF path")
    nc_ex.add_argument("output", help="destination raster path")
    nc_ex.add_argument(
        "--var", required=True, help="variable name (supports 'group/var')"
    )
    nc_ex.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    nc_ex.set_defaults(func=_cmd_nc_extract)
