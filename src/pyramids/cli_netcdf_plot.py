"""Headless NetCDF plotting commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF` plotting, which goes through
cleopatra. The Agg backend is forced before any plot import so a headless CLI/CI
run never pops a blocking GUI window. ``plot`` covers filled/contour renders via
``--kind`` (including ``contour``); ``plot-histogram`` and ``animate`` cover the
value distribution and a time animation. Vector-field plots need paired u/v
components and are better driven from Python, so they are not exposed here.
"""

from __future__ import annotations

import argparse
import os

from pyramids._cli_shared import info_print, refuse_existing
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OVERWRITE = "replace the output if it exists"
_HELP_VAR = "variable to plot"


def _force_agg() -> None:
    """Force the non-interactive Agg matplotlib backend.

    Assigned (not ``setdefault``) so a pre-exported interactive backend such as
    TkAgg cannot win and block on a popup in a headless run.
    """
    os.environ["MPLBACKEND"] = "Agg"


def _cmd_nc_plot(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc plot`` — save a map of one variable (via cleopatra).

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``kind``, ``title``,
            ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    _force_agg()
    nc = NetCDF.read_file(args.file)
    glyph = nc.plot(variable=args.var, kind=args.kind, title=args.title)
    glyph.fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_plot_histogram(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc plot-histogram`` — save a variable's value histogram.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` names a variable that cannot be plotted (no figure).
    """
    refuse_existing(args.output, args.overwrite)
    _force_agg()
    variable = NetCDF.read_file(args.file).get_variable(args.var)
    if not hasattr(variable, "plot_histogram"):
        raise ValueError(f"variable {args.var!r} cannot be plotted as a histogram")
    figure = variable.plot_histogram()[0]
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_animate(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc animate`` — save an animation over a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``dim``, ``fps``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    _force_agg()
    nc = NetCDF.read_file(args.file)
    glyph = nc.animate(variable=args.var, dim=args.dim)
    glyph.save_animation(args.output, fps=args.fps)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_plot_commands(ncsub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` plotting commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_plot = ncsub.add_parser("plot", help="save a map of one variable (PNG)")
    nc_plot.add_argument("file", help=_HELP_NC_FILE)
    nc_plot.add_argument("output", help="destination image path")
    nc_plot.add_argument("--var", required=True, help=_HELP_VAR)
    nc_plot.add_argument(
        "--kind",
        default="auto",
        choices=("auto", "imshow", "pcolormesh", "contour"),
        help="render kind (default: auto)",
    )
    nc_plot.add_argument("--title", help="plot title")
    nc_plot.add_argument("--dpi", type=int, default=150, help="output DPI")
    nc_plot.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_plot.set_defaults(func=_cmd_nc_plot)

    nc_hist = ncsub.add_parser(
        "plot-histogram", help="save a variable's value histogram (PNG)"
    )
    nc_hist.add_argument("file", help=_HELP_NC_FILE)
    nc_hist.add_argument("output", help="destination image path")
    nc_hist.add_argument("--var", required=True, help=_HELP_VAR)
    nc_hist.add_argument("--dpi", type=int, default=150, help="output DPI")
    nc_hist.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_hist.set_defaults(func=_cmd_nc_plot_histogram)

    nc_anim = ncsub.add_parser("animate", help="save an animation over a dimension")
    nc_anim.add_argument("file", help=_HELP_NC_FILE)
    nc_anim.add_argument("output", help="destination animation path (e.g. .gif)")
    nc_anim.add_argument("--var", required=True, help=_HELP_VAR)
    nc_anim.add_argument("--dim", default="time", help="dimension to animate over")
    nc_anim.add_argument("--fps", type=int, default=2, help="frames per second")
    nc_anim.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_anim.set_defaults(func=_cmd_nc_animate)
