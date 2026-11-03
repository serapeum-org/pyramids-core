"""NetCDF / multidim CLI command group — ``pyramids nc …``.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`: summarise a file (`info`), list
the variables GDAL can expose as raster planes (`list-vars`), extract one to a
raster (`extract`), select one step of the CF ``time`` axis (`sel`), and spatially
subset (`crop`). ``info`` and ``sel`` assume the time axis is the CF-standard one
named ``time``. No multidim logic lives here — reads go through GDAL as the
``NetCDF`` class already does (never xarray).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from pyramids._cli_shared import info_print, json_safe, refuse_existing
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"


def _cmd_nc_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc info`` — print variables, dimensions, CRS, bounds, time.

    Args:
        args: Parsed args with ``file`` and ``json``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    time_var = nc.get_time_variable()
    minx, miny, maxx, maxy = (float(value) for value in nc.bbox)
    payload = {
        "path": args.file,
        "variables": list(nc.variable_names),
        "dimensions": dict(nc.dimension_sizes),
        "epsg": nc.epsg,
        "bounds": [json_safe(value) for value in (minx, miny, maxx, maxy)],
        "time_steps": 0 if time_var is None else len(time_var),
        "time_start": None if not time_var else time_var[0],
        "time_end": None if not time_var else time_var[-1],
    }
    if args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


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


def _cmd_nc_sel(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc sel`` — select one ``time`` step into a new NetCDF.

    Selects exactly by the ``time`` label (as printed by ``nc info``); ``--var``
    optionally narrows to one variable first (group-qualified names like
    ``group/var`` are accepted, as in ``extract``). Writes the selection to
    ``output`` (a ``.nc`` path keeps it a NetCDF; another extension re-encodes it).
    The time axis must be the CF-standard one named ``time``.

    Args:
        args: Parsed args with ``file``, ``output``, ``time``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` names an unknown variable (raised by ``get_variable``),
            names one GDAL exposes as a non-selectable labeled array, or the
            selection cannot be written to ``output``.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    # get_variable may return a NetCDF or a LabeledArray; `obj` is duck-typed for
    # .sel/.to_file and guarded below, so keep it Any to satisfy the union.
    obj: Any = nc
    if args.var is not None:
        obj = nc.get_variable(args.var)
        if not hasattr(obj, "sel"):
            raise ValueError(
                f"variable {args.var!r} is not a selectable raster variable"
            )
    selected = obj.sel(time=args.time)
    if not hasattr(selected, "to_file"):
        raise ValueError(
            f"the selection of {args.file!r} cannot be written to {args.output!r}"
        )
    selected.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_crop(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc crop`` — spatially subset to ``--bbox`` into a new NetCDF.

    Args:
        args: Parsed args with ``file``, ``output``, ``bbox``, ``overwrite``.

    Returns:
        int: ``0`` on success (``crop`` raises a clear error when the bbox does not
        overlap any valid data).
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    cropped = nc.crop(bbox=tuple(args.bbox))
    cropped.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_netcdf_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    nc = sub.add_parser("nc", help="NetCDF / multidim commands")
    ncsub = nc.add_subparsers(dest="command", required=True)

    nc_info = ncsub.add_parser("info", help="summarize a NetCDF file")
    nc_info.add_argument("file", help=_HELP_NC_FILE)
    nc_info.add_argument("--json", action="store_true", help="emit JSON")
    nc_info.set_defaults(func=_cmd_nc_info)

    nc_list = ncsub.add_parser("list-vars", help="list variables/subdatasets")
    nc_list.add_argument("file", help=_HELP_NC_FILE)
    nc_list.add_argument("--json", action="store_true", help="emit JSON")
    nc_list.set_defaults(func=_cmd_nc_list_vars)

    nc_ex = ncsub.add_parser("extract", help="extract one variable to a raster")
    nc_ex.add_argument("file", help=_HELP_NC_FILE)
    nc_ex.add_argument("output", help="destination raster path")
    nc_ex.add_argument(
        "--var", required=True, help="variable name (supports 'group/var')"
    )
    nc_ex.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    nc_ex.set_defaults(func=_cmd_nc_extract)

    nc_sel = ncsub.add_parser("sel", help="select along time into a new NetCDF")
    nc_sel.add_argument("file", help=_HELP_NC_FILE)
    nc_sel.add_argument("output", help="destination path (.nc keeps it a NetCDF)")
    nc_sel.add_argument(
        "--time", required=True, help="time label to select (see 'nc info')"
    )
    nc_sel.add_argument("--var", help="narrow to one variable before selecting")
    nc_sel.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    nc_sel.set_defaults(func=_cmd_nc_sel)

    nc_crop = ncsub.add_parser("crop", help="spatially subset to a bbox")
    nc_crop.add_argument("file", help=_HELP_NC_FILE)
    nc_crop.add_argument("output", help="destination path (.nc keeps it a NetCDF)")
    nc_crop.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="crop window in the file's CRS",
    )
    nc_crop.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    nc_crop.set_defaults(func=_cmd_nc_crop)
