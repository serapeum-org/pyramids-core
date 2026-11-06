"""Raster (`Dataset`) plotting CLI commands for the ``pyramids`` tool.

Thin 1:1 wrappers over the :class:`pyramids.dataset.Dataset` plotting surface
that all delegate to cleopatra: a value histogram, a two-band vector field, and a
multi-band animation. Every handler forces matplotlib's non-interactive ``Agg``
backend before any plot import so a headless CLI / CI run never pops a blocking
GUI window, then saves the figure to the requested output file.

``cleopatra`` is imported lazily by the underlying methods (guarded by the
``[viz]`` extra), so importing this module — and ``pyramids.cli`` — never requires
it; a missing extra surfaces as the framework's own clean error.

Deferred:
    - ``Dataset.contour`` — owned by the analysis command group, not re-wrapped
      here.
    - ``Dataset.plot`` — already exposed as the top-level ``render`` command.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import force_agg_backend, info_print, refuse_existing
from pyramids.dataset import Dataset

_HELP_SRC_RASTER = "source raster path"
_HELP_OUTPUT = "destination image path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_DPI = "output DPI (default: 150)"


def _cmd_plot_histogram(args: argparse.Namespace) -> int:
    """Handle ``pyramids plot-histogram`` — save a band's value histogram.

    Args:
        args: Parsed args with ``input``, ``output``, ``band``, ``bins``,
            ``exclude_value``, ``max_samples``, ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    ds = Dataset.read_file(args.input)
    fig, _ax, _hist = ds.plot_histogram(
        band=args.band,
        bins=args.bins,
        exclude_value=args.exclude_value,
        max_samples=args.max_samples,
    )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_plot_vector_field(args: argparse.Namespace) -> int:
    """Handle ``pyramids plot-vector-field`` — save a two-band vector-field plot.

    Args:
        args: Parsed args with ``input``, ``output``, ``u_band``, ``v_band``,
            ``kind``, ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    ds = Dataset.read_file(args.input)
    fig, _ax, _im = ds.plot_vector_field(
        u_band=args.u_band, v_band=args.v_band, kind=args.kind
    )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_animate(args: argparse.Namespace) -> int:
    """Handle ``pyramids animate`` — save a multi-band animation (GIF/MP4).

    The animation needs at least two frames; the writer is chosen by matplotlib
    from the output extension (``.gif`` → pillow, ``.mp4`` → ffmpeg).

    Args:
        args: Parsed args with ``input``, ``output``, ``bands``, ``interval``,
            ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    ds = Dataset.read_file(args.input)
    bands = args.bands if args.bands else None
    glyph = ds.animate(bands=bands, interval=args.interval)
    glyph.anim.save(args.output, dpi=args.dpi)
    info_print(args, f"wrote {args.output}")
    return 0


def add_dataset_plot_commands(sub: argparse._SubParsersAction) -> None:
    """Register the raster (`Dataset`) plotting commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    hist = sub.add_parser(
        "plot-histogram", help="save a band's value-distribution histogram (PNG)"
    )
    hist.add_argument("input", help=_HELP_SRC_RASTER)
    hist.add_argument("output", help=_HELP_OUTPUT + " (e.g. hist.png)")
    hist.add_argument("--band", type=int, default=0, help="zero-based band (default 0)")
    hist.add_argument(
        "--bins", type=int, default=15, help="number of bins (default 15)"
    )
    hist.add_argument(
        "--exclude-value",
        type=float,
        default=None,
        help="an extra value to drop before binning, besides no-data",
    )
    hist.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap on pixels read (decimates a large band; default: read all)",
    )
    hist.add_argument("--dpi", type=int, default=150, help=_HELP_DPI)
    hist.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    hist.set_defaults(func=_cmd_plot_histogram)

    vfield = sub.add_parser(
        "plot-vector-field", help="save a two-band (u, v) vector-field plot (PNG)"
    )
    vfield.add_argument("input", help=_HELP_SRC_RASTER)
    vfield.add_argument("output", help=_HELP_OUTPUT + " (e.g. field.png)")
    vfield.add_argument(
        "--u-band", type=int, default=0, help="x-component band index (default 0)"
    )
    vfield.add_argument(
        "--v-band", type=int, default=1, help="y-component band index (default 1)"
    )
    vfield.add_argument(
        "--kind",
        choices=["quiver", "barbs", "streamplot"],
        default="quiver",
        help="render kind (default: quiver)",
    )
    vfield.add_argument("--dpi", type=int, default=150, help=_HELP_DPI)
    vfield.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    vfield.set_defaults(func=_cmd_plot_vector_field)

    animate = sub.add_parser("animate", help="save a multi-band animation (GIF/MP4)")
    animate.add_argument("input", help=_HELP_SRC_RASTER)
    animate.add_argument("output", help="destination animation path (e.g. out.gif)")
    animate.add_argument(
        "--bands",
        nargs="+",
        type=int,
        help="zero-based band indices to animate, in order (default: all, >= 2)",
    )
    animate.add_argument(
        "--interval", type=int, default=200, help="milliseconds between frames"
    )
    animate.add_argument(
        "--dpi", type=int, default=100, help="output DPI (default: 100)"
    )
    animate.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    animate.set_defaults(func=_cmd_animate)
