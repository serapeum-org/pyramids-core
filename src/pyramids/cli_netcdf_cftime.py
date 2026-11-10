"""CF-time, calendar, and calculus commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`: calendar conversion
(`convert-calendar`, `interp-calendar`), binned / rolling / windowed reductions
(`groupby-bins`, `rolling`, `rolling-exp`, `shift`, `reduce`, `coarsen`, `weighted`),
the along-dimension calculus (`integrate`, `cumulative-integrate`, `differentiate`,
`polyfit`, `cumulative`), and coordinate interpolation (`interp`, `interpolate-na`,
`interp-like`). Each command reads a NetCDF, applies one
:class:`~pyramids.netcdf.NetCDF` method along a dimension, and writes the resulting
NetCDF to a new file. Reads go through the ``NetCDF`` class; no xarray is imported here
(the methods may use it internally).
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import info_print, refuse_existing
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OUTPUT = "destination path (.nc keeps it a NetCDF)"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_DIM = "the dimension to operate along"
_HELP_HOW = "aggregation to apply (mean, sum, min, max, std, median, ...)"
_HELP_METHOD = "interpolation method (default: linear)"
_WEIGHTS_AREA = "area"


def _parse_coords(items: list[str]) -> dict[str, float]:
    """Parse ``DIM=VALUE`` tokens into a ``{dim: float(value)}`` mapping.

    Args:
        items: ``--coord`` tokens, each ``DIM=VALUE`` (e.g. ``time=5.0``).

    Returns:
        dict[str, float]: Dimension name mapped to its target coordinate value.

    Raises:
        ValueError: A token has no ``=``, an empty dimension name, or a value that
            is not a float.
    """
    coords: dict[str, float] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--coord expects DIM=VALUE tokens; got {item!r} (no '=')")
        if not name:
            raise ValueError(f"--coord token {item!r} has an empty dimension name")
        try:
            coords[name] = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"--coord {item!r} has a non-numeric value {raw!r}"
            ) from exc
    return coords


def _cmd_nc_convert_calendar(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc convert-calendar`` — recast the time axis to a calendar.

    Args:
        args: Parsed args with ``file``, ``output``, ``calendar``, ``dim``,
            ``align_on``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--calendar`` is unknown to cftime, or ``--dim`` is not a
            dimension of the file.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).convert_calendar(
        args.calendar, dim=args.dim, align_on=args.align_on
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_interp_calendar(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc interp-calendar`` — interpolate onto a target's time axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``target``, ``dim``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, or the target's time
            axis is incompatible.
    """
    refuse_existing(args.output, args.overwrite)
    target = NetCDF.read_file(args.target)
    result = NetCDF.read_file(args.file).interp_calendar(target, dim=args.dim)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_groupby_bins(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc groupby-bins`` — bin a dimension and reduce each bin.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``bins``, ``how``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, or ``--how`` is not a
            known aggregation.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).groupby_bins(args.dim, args.bins, args.how)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_rolling(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc rolling`` — a moving-window reduction along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``window``, ``how``,
            ``center``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, or ``--how`` is not a
            known aggregation.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).rolling(
        args.dim, args.window, how=args.how, center=args.center
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_rolling_exp(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc rolling-exp`` — an exponentially weighted moving reduction.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``alpha``, ``how``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, or ``--how`` is not a
            known aggregation.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).rolling_exp(args.dim, args.alpha, how=args.how)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_shift(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc shift`` — shift values along a dimension by ``--periods``.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``periods``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).shift(args.dim, args.periods)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_integrate(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc integrate`` — integrate over a dimension (consumes it).

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).integrate(args.dim)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_cumulative_integrate(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc cumulative-integrate`` — running integral along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).cumulative_integrate(args.dim)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_differentiate(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc differentiate`` — derivative of values along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).differentiate(args.dim)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_polyfit(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc polyfit`` — least-squares polynomial fit along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``deg``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, or ``--deg`` is
            negative.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).polyfit(args.dim, args.deg)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_interp(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc interp`` — interpolate onto new coordinate values.

    Only non-spatial (band) dimensions can be interpolated; for a new spatial grid
    use the raster warp/resample commands instead.

    Args:
        args: Parsed args with ``file``, ``output``, ``coord``, ``method``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: A ``--coord`` token is malformed, names a dimension the file
            lacks, or names a spatial axis.
    """
    refuse_existing(args.output, args.overwrite)
    coords = _parse_coords(args.coord)
    result = NetCDF.read_file(args.file).interp(args.method, **coords)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_interpolate_na(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc interpolate-na`` — fill gaps along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``method``, ``limit``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a non-spatial dimension of the file.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).interpolate_na(
        args.dim, args.method, limit=args.limit
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_reduce(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc reduce`` — collapse a dimension with an aggregation.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``how``, ``groupby``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, or ``--how`` is not a
            known aggregation.
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).reduce(
        args.dim, args.how, groupby=args.groupby
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_interp_like(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc interp-like`` — interpolate onto a target cube's band axes.

    ``interp_like`` interpolates this file's variables onto the band coordinates of the
    ``--target`` cube (the band dimensions the two share), and writes the result to
    ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``target``, ``method``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The two cubes share no band dimension, or a shared dimension is
            incompatible (raised by ``interp_like``).
    """
    refuse_existing(args.output, args.overwrite)
    target = NetCDF.read_file(args.target)
    result = NetCDF.read_file(args.file).interp_like(target, method=args.method)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_coarsen(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc coarsen`` — block-reduce a dimension by a window.

    ``coarsen`` groups ``--dim`` into non-overlapping blocks of ``--window`` steps and
    reduces each block with ``--how``; ``--boundary`` decides how a trailing partial
    block is handled, and ``--q`` supplies the quantile when ``--how quantile``. The
    result is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``window``, ``how``,
            ``boundary``, ``q``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a dimension of the file, ``--how`` is unknown, or a
            trailing partial block is left with ``--boundary exact`` (raised by
            ``coarsen``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).coarsen(
        args.dim, args.window, how=args.how, boundary=args.boundary, q=args.q
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_cumulative(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc cumulative`` — a running total or product along a dimension.

    Builds the ``cumulative(dim)`` accessor and applies ``--how`` (``sum`` for the running
    total, ``prod`` for the running product), writing the accumulated cube to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``how``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not an accumulable band dimension (raised by
            ``cumulative``).
    """
    refuse_existing(args.output, args.overwrite)
    # Keep a strong reference to the container: the accessor holds only a weak one, so a
    # temporary `NetCDF.read_file(...).cumulative(dim)` would be collected before .sum().
    nc = NetCDF.read_file(args.file)
    accessor = nc.cumulative(args.dim)
    result = accessor.sum() if args.how == "sum" else accessor.prod()
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_weighted(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc weighted`` — a weighted reduction over one or more dimensions.

    ``--weights`` is either the literal keyword ``area`` (``cos(latitude)`` cell weights
    for a regional/global mean over the spatial axes) or a path to a raster of weights on
    the same grid. ``--dims`` names the dimension(s) to reduce (default: both spatial
    axes), ``--how`` the statistic, and ``--q`` the quantile when ``--how quantile``. The
    reduced cube is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``weights``, ``dims``, ``how``,
            ``q``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--how`` is unknown, ``area`` weights are asked of a non-geographic
            grid, or the weights raster is not on the file's grid (raised by
            ``weighted``).
    """
    refuse_existing(args.output, args.overwrite)
    weights = (
        args.weights
        if args.weights == _WEIGHTS_AREA
        else Dataset.read_file(args.weights)
    )
    result = NetCDF.read_file(args.file).weighted(
        weights, args.dims, how=args.how, q=args.q
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_cftime_commands(
    ncsub: argparse._SubParsersAction,
    advanced: argparse._SubParsersAction | None = None,
) -> None:
    """Register the ``nc`` CF-time / calendar / calculus commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_cc = ncsub.add_parser(
        "convert-calendar", help="recast the time axis to a calendar"
    )
    nc_cc.add_argument("file", help=_HELP_NC_FILE)
    nc_cc.add_argument("output", help=_HELP_OUTPUT)
    nc_cc.add_argument(
        "--calendar", required=True, help="target calendar, e.g. noleap, 360_day"
    )
    nc_cc.add_argument("--dim", default="time", help=_HELP_DIM)
    nc_cc.add_argument(
        "--align-on", default="date", dest="align_on", help="date alignment rule"
    )
    nc_cc.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_cc.set_defaults(func=_cmd_nc_convert_calendar)

    nc_ic = ncsub.add_parser(
        "interp-calendar", help="interpolate onto a target's time axis"
    )
    nc_ic.add_argument("file", help=_HELP_NC_FILE)
    nc_ic.add_argument("output", help=_HELP_OUTPUT)
    nc_ic.add_argument(
        "--target", required=True, help="NetCDF whose time axis to interpolate onto"
    )
    nc_ic.add_argument("--dim", default="time", help=_HELP_DIM)
    nc_ic.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_ic.set_defaults(func=_cmd_nc_interp_calendar)

    nc_gb = (advanced or ncsub).add_parser(
        "groupby-bins", help="bin a dimension and reduce each bin"
    )
    nc_gb.add_argument("file", help=_HELP_NC_FILE)
    nc_gb.add_argument("output", help=_HELP_OUTPUT)
    nc_gb.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_gb.add_argument(
        "--bins", type=int, required=True, help="number of equal-width bins"
    )
    nc_gb.add_argument("--how", default="mean", help=_HELP_HOW)
    nc_gb.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_gb.set_defaults(func=_cmd_nc_groupby_bins)

    nc_rl = (advanced or ncsub).add_parser(
        "rolling", help="moving-window reduction along a dimension"
    )
    nc_rl.add_argument("file", help=_HELP_NC_FILE)
    nc_rl.add_argument("output", help=_HELP_OUTPUT)
    nc_rl.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_rl.add_argument(
        "--window", type=int, required=True, help="window length in steps"
    )
    nc_rl.add_argument("--how", default="mean", help=_HELP_HOW)
    nc_rl.add_argument(
        "--center", action="store_true", help="center the window on each step"
    )
    nc_rl.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rl.set_defaults(func=_cmd_nc_rolling)

    nc_re = (advanced or ncsub).add_parser(
        "rolling-exp", help="exponentially weighted moving reduction"
    )
    nc_re.add_argument("file", help=_HELP_NC_FILE)
    nc_re.add_argument("output", help=_HELP_OUTPUT)
    nc_re.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_re.add_argument(
        "--alpha", type=float, required=True, help="smoothing factor in (0, 1]"
    )
    nc_re.add_argument("--how", default="mean", help=_HELP_HOW)
    nc_re.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_re.set_defaults(func=_cmd_nc_rolling_exp)

    nc_sh = (advanced or ncsub).add_parser(
        "shift", help="shift values along a dimension"
    )
    nc_sh.add_argument("file", help=_HELP_NC_FILE)
    nc_sh.add_argument("output", help=_HELP_OUTPUT)
    nc_sh.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_sh.add_argument(
        "--periods", type=int, default=1, help="number of steps to shift"
    )
    nc_sh.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_sh.set_defaults(func=_cmd_nc_shift)

    nc_in = (advanced or ncsub).add_parser(
        "integrate", help="integrate over a dimension"
    )
    nc_in.add_argument("file", help=_HELP_NC_FILE)
    nc_in.add_argument("output", help=_HELP_OUTPUT)
    nc_in.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_in.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_in.set_defaults(func=_cmd_nc_integrate)

    nc_ci = (advanced or ncsub).add_parser(
        "cumulative-integrate", help="running integral along a dimension"
    )
    nc_ci.add_argument("file", help=_HELP_NC_FILE)
    nc_ci.add_argument("output", help=_HELP_OUTPUT)
    nc_ci.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_ci.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_ci.set_defaults(func=_cmd_nc_cumulative_integrate)

    nc_df = (advanced or ncsub).add_parser(
        "differentiate", help="derivative of values along a dimension"
    )
    nc_df.add_argument("file", help=_HELP_NC_FILE)
    nc_df.add_argument("output", help=_HELP_OUTPUT)
    nc_df.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_df.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_df.set_defaults(func=_cmd_nc_differentiate)

    nc_pf = ncsub.add_parser(
        "polyfit", help="least-squares polynomial fit along a dimension"
    )
    nc_pf.add_argument("file", help=_HELP_NC_FILE)
    nc_pf.add_argument("output", help=_HELP_OUTPUT)
    nc_pf.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_pf.add_argument(
        "--deg", type=int, required=True, help="polynomial degree (0 or more)"
    )
    nc_pf.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_pf.set_defaults(func=_cmd_nc_polyfit)

    nc_ip = ncsub.add_parser("interp", help="interpolate onto new coordinate values")
    nc_ip.add_argument("file", help=_HELP_NC_FILE)
    nc_ip.add_argument("output", help=_HELP_OUTPUT)
    nc_ip.add_argument(
        "--coord",
        nargs="+",
        required=True,
        metavar="DIM=VALUE",
        help="target coordinate(s), e.g. time=5 (non-spatial dimensions only)",
    )
    nc_ip.add_argument("--method", default="linear", help=_HELP_METHOD)
    nc_ip.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_ip.set_defaults(func=_cmd_nc_interp)

    nc_na = ncsub.add_parser("interpolate-na", help="fill gaps along a dimension")
    nc_na.add_argument("file", help=_HELP_NC_FILE)
    nc_na.add_argument("output", help=_HELP_OUTPUT)
    nc_na.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_na.add_argument("--method", default="linear", help=_HELP_METHOD)
    nc_na.add_argument(
        "--limit", type=int, default=None, help="max consecutive gaps to fill"
    )
    nc_na.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_na.set_defaults(func=_cmd_nc_interpolate_na)

    nc_rd = ncsub.add_parser("reduce", help="collapse a dimension with an aggregation")
    nc_rd.add_argument("file", help=_HELP_NC_FILE)
    nc_rd.add_argument("output", help=_HELP_OUTPUT)
    nc_rd.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_rd.add_argument("--how", default="mean", help=_HELP_HOW)
    nc_rd.add_argument(
        "--groupby", default=None, help="optional coordinate to group by first"
    )
    nc_rd.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rd.set_defaults(func=_cmd_nc_reduce)

    nc_il = (advanced or ncsub).add_parser(
        "interp-like", help="interpolate onto a target cube's band axes"
    )
    nc_il.add_argument("file", help=_HELP_NC_FILE)
    nc_il.add_argument("output", help=_HELP_OUTPUT)
    nc_il.add_argument(
        "--target",
        required=True,
        help="NetCDF whose band coordinates to interpolate onto",
    )
    nc_il.add_argument("--method", default="linear", help=_HELP_METHOD)
    nc_il.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_il.set_defaults(func=_cmd_nc_interp_like)

    nc_co = (advanced or ncsub).add_parser(
        "coarsen", help="block-reduce a dimension by a window"
    )
    nc_co.add_argument("file", help=_HELP_NC_FILE)
    nc_co.add_argument("output", help=_HELP_OUTPUT)
    nc_co.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_co.add_argument(
        "--window", type=int, required=True, help="number of steps per block"
    )
    nc_co.add_argument("--how", default="mean", help=_HELP_HOW)
    nc_co.add_argument(
        "--boundary",
        default="exact",
        choices=("exact", "trim", "pad"),
        help="how to treat a trailing partial block (default: exact)",
    )
    nc_co.add_argument(
        "--q", type=float, default=None, help="quantile in [0, 1] when --how quantile"
    )
    nc_co.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_co.set_defaults(func=_cmd_nc_coarsen)

    nc_cu = (advanced or ncsub).add_parser(
        "cumulative", help="running total/product along a dimension"
    )
    nc_cu.add_argument("file", help=_HELP_NC_FILE)
    nc_cu.add_argument("output", help=_HELP_OUTPUT)
    nc_cu.add_argument("--dim", required=True, help=_HELP_DIM)
    nc_cu.add_argument(
        "--how",
        default="sum",
        choices=("sum", "prod"),
        help="'sum' for a running total (default) or 'prod' for a running product",
    )
    nc_cu.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_cu.set_defaults(func=_cmd_nc_cumulative)

    nc_we = (advanced or ncsub).add_parser(
        "weighted", help="weighted reduction over dimension(s)"
    )
    nc_we.add_argument("file", help=_HELP_NC_FILE)
    nc_we.add_argument("output", help=_HELP_OUTPUT)
    nc_we.add_argument(
        "--weights",
        required=True,
        help="the keyword 'area' (cos-latitude cell weights) or a weights-raster path",
    )
    nc_we.add_argument(
        "--dims",
        nargs="+",
        default=None,
        help="dimension(s) to reduce over (default: both spatial axes)",
    )
    nc_we.add_argument("--how", default="mean", help=_HELP_HOW)
    nc_we.add_argument(
        "--q", type=float, default=None, help="quantile in [0, 1] when --how quantile"
    )
    nc_we.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_we.set_defaults(func=_cmd_nc_weighted)
