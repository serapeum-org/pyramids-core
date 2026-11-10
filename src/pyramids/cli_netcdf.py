"""NetCDF / multidim CLI command group — ``pyramids nc …``.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`. This module owns the core
verbs — ``info``, ``list-vars``, ``extract``, ``sel``, ``crop``, ``groups``,
``get-group`` — and registers the rest of the group from sibling modules via
:func:`add_netcdf_commands`, each of which documents its own commands:
``cli_netcdf_inspect`` (inspection + global attributes), ``cli_netcdf_vars``
(variable & dimension management), ``cli_netcdf_select`` (positional/label selection
& subsetting), ``cli_netcdf_cftime`` (calendar / time / calculus),
``cli_netcdf_io`` (export, construction & combine), and ``cli_netcdf_plot``
(headless plotting). ``info`` and ``sel`` assume the time axis is the CF-standard one
named ``time``. No multidim logic lives here — reads go through GDAL as the ``NetCDF``
class already does (never xarray).
"""

from __future__ import annotations

import argparse
import io
import json
from typing import Any

from pyramids._cli_groups import SuggestingArgumentParser, add_group
from pyramids._cli_shared import (
    emit,
    info_print,
    json_safe,
    mark_format_support,
    refuse_existing,
)
from pyramids.cli_netcdf_cftime import add_nc_cftime_commands
from pyramids.cli_netcdf_inspect import add_nc_inspect_commands
from pyramids.cli_netcdf_io import add_nc_io_commands
from pyramids.cli_netcdf_plot import add_nc_plot_commands
from pyramids.cli_netcdf_select import add_nc_select_commands
from pyramids.cli_netcdf_vars import add_nc_vars_commands
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_DST_NC = "destination path (.nc keeps it a NetCDF)"


def _cmd_nc_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc info`` — print variables, dimensions, CRS, bounds, time.

    The default is a compact summary (optionally ``--json``). ``--full`` instead prints
    :meth:`NetCDF.info`'s complete ncdump-style report — every dimension, variable with
    its axes, and global/variable attribute — as text. ``--full`` and ``--json`` are
    mutually exclusive (the full report is text-only), so the parser rejects both.

    Args:
        args: Parsed args with ``file``, ``json``, and ``full``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    if args.full:
        report = io.StringIO()
        nc.info(report)
        print(report.getvalue(), end="")
    else:
        time_var = nc.get_time_variable()
        minx, miny, maxx, maxy = (float(value) for value in nc.total_bounds)
        payload = {
            "path": args.file,
            "variables": list(nc.variable_names),
            "dimensions": dict(nc.dimension_sizes),
            "epsg": nc.epsg,
            "bounds": [json_safe(value) for value in (minx, miny, maxx, maxy)],
            "time_steps": 0 if not time_var else len(time_var),
            "time_start": None if not time_var else time_var[0],
            "time_end": None if not time_var else time_var[-1],
        }
        if args.json:
            print(json.dumps(payload))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
    return 0


def _cmd_nc_groups(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc groups`` — list the file's root-level sub-group names.

    Args:
        args: Parsed args with ``file`` and ``json``.

    Returns:
        int: ``0`` on success (an empty list for a flat / classic store).
    """
    names = list(NetCDF.read_file(args.file).group_names)
    text = "\n".join([f"file: {args.file}", *(f"  - {name}" for name in names)])
    emit(args, {"path": args.file, "groups": names}, text=text)
    return 0


def _cmd_nc_get_group(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc get-group`` — write one sub-group out as its own NetCDF.

    Opens ``--group`` as a zero-copy view (nested paths like ``forecast/surface`` are
    accepted) and writes it to ``output`` as a standalone container.

    Args:
        args: Parsed args with ``file``, ``output``, ``group``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The group does not exist, or the file has no root group (raised by
            ``get_group``).
    """
    refuse_existing(args.output, args.overwrite)
    NetCDF.read_file(args.file).get_group(args.group).to_file(args.output)
    info_print(args, f"wrote {args.output}")
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
    text = "\n".join([f"file: {args.file}", *(f"  - {name}" for name in names)])
    emit(args, {"path": args.file, "variables": names}, text=text)
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

    Selects by a ``time`` date label; matching is by date, so on a sub-daily axis
    one date can match several steps. ``nc info`` shows the step count and the first
    and last labels (not every interior label). ``--var`` optionally narrows to one
    variable first (group-qualified names like ``group/var`` are accepted, as in
    ``extract``). Writes the selection to ``output`` (a ``.nc`` path keeps it a
    NetCDF; another extension re-encodes it). The time axis must be the CF-standard
    one named ``time``.

    Args:
        args: Parsed args with ``file``, ``output``, ``time``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` names an unknown variable (raised by ``get_variable``),
            names one GDAL exposes as a labeled array rather than a raster plane, or
            the selection cannot be written to ``output``.
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
                f"variable {args.var!r} is not exposable as a raster plane "
                f"(GDAL returned a labeled array); it cannot be selected."
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
    ncsub = nc.add_subparsers(dest="command", required=True, parser_class=SuggestingArgumentParser)

    nc_info = ncsub.add_parser("info", help="summarize a NetCDF file")
    nc_info.add_argument("file", help=_HELP_NC_FILE)
    info_fmt = nc_info.add_mutually_exclusive_group()
    info_fmt.add_argument(
        "--json", action="store_true", help="emit the compact summary as JSON"
    )
    info_fmt.add_argument(
        "--full",
        action="store_true",
        help="print the full ncdump-style report (dims/vars/attrs) as text",
    )
    nc_info.set_defaults(func=_cmd_nc_info)

    nc_groups = ncsub.add_parser("groups", help="list root-level sub-group names")
    nc_groups.add_argument("file", help=_HELP_NC_FILE)
    nc_groups.add_argument("--json", action="store_true", help="emit JSON")
    nc_groups.set_defaults(func=_cmd_nc_groups)
    mark_format_support(nc_groups)

    nc_gg = ncsub.add_parser("get-group", help="write one sub-group out as a NetCDF")
    nc_gg.add_argument("file", help=_HELP_NC_FILE)
    nc_gg.add_argument("output", help=_HELP_DST_NC)
    nc_gg.add_argument(
        "--group", required=True, help="sub-group name (nested 'a/b' accepted)"
    )
    nc_gg.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_gg.set_defaults(func=_cmd_nc_get_group)

    nc_list = ncsub.add_parser("list-vars", help="list variables/subdatasets")
    nc_list.add_argument("file", help=_HELP_NC_FILE)
    nc_list.add_argument("--json", action="store_true", help="emit JSON")
    nc_list.set_defaults(func=_cmd_nc_list_vars)
    mark_format_support(nc_list)

    nc_ex = ncsub.add_parser("extract", help="extract one variable to a raster")
    nc_ex.add_argument("file", help=_HELP_NC_FILE)
    nc_ex.add_argument("output", help="destination raster path")
    nc_ex.add_argument(
        "--var", required=True, help="variable name (supports 'group/var')"
    )
    nc_ex.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_ex.set_defaults(func=_cmd_nc_extract)

    nc_sel = ncsub.add_parser("sel", help="select along time into a new NetCDF")
    nc_sel.add_argument("file", help=_HELP_NC_FILE)
    nc_sel.add_argument("output", help=_HELP_DST_NC)
    nc_sel.add_argument(
        "--time",
        required=True,
        help="a date label on the time axis, e.g. 2020-06-15 "
        "('nc info' shows the step count and endpoints)",
    )
    nc_sel.add_argument("--var", help="narrow to one variable before selecting")
    nc_sel.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_sel.set_defaults(func=_cmd_nc_sel)

    nc_crop = ncsub.add_parser("crop", help="spatially subset to a bbox")
    nc_crop.add_argument("file", help=_HELP_NC_FILE)
    nc_crop.add_argument("output", help=_HELP_DST_NC)
    nc_crop.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="crop window in the file's CRS",
    )
    nc_crop.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_crop.set_defaults(func=_cmd_nc_crop)

    # Power-user primitives (array/coord/dim mechanics rarely run one-shot) collect under a
    # single cross-cutting `advanced` subgroup, fed by the vars/select/calc registrars.
    nc_advanced = add_group(
        ncsub, "advanced", "power-user primitives (array/coord/dim mechanics)"
    )
    add_nc_inspect_commands(
        add_group(ncsub, "inspect", "summaries, metadata, dim/time values")
    )
    add_nc_vars_commands(
        add_group(ncsub, "vars", "variable and dimension management"), nc_advanced
    )
    add_nc_select_commands(
        add_group(ncsub, "select", "positional/label selection and subsetting"),
        nc_advanced,
    )
    add_nc_cftime_commands(
        add_group(ncsub, "calc", "CF/time/calendar and calculus"), nc_advanced
    )
    add_nc_io_commands(add_group(ncsub, "io", "format I/O and conversion"))
    add_nc_plot_commands(add_group(ncsub, "plot", "headless plotting"))
