"""UGRID mesh plotting CLI commands — ``pyramids mesh plot …``.

Thin 1:1 wrappers over the :class:`pyramids.netcdf.ugrid.MeshDataset` plotting
surface, which delegates to cleopatra: a filled render of a variable, a wireframe
outline, a two-variable vector field, a value histogram, and a temporal
animation. Every handler forces matplotlib's non-interactive ``Agg`` backend
before any plot import so a headless CLI / CI run never pops a blocking GUI
window, then saves the figure (or animation) to the requested output file.

``cleopatra`` is imported lazily by the underlying methods (guarded by the
``[viz]`` extra), so importing this module — and ``pyramids.cli`` — never requires
it; a missing extra surfaces as the framework's own clean error.

The ``plot`` facade returns a cleopatra ``MeshGlyph`` whose ``fig`` is saved;
``plot_vector_field`` and ``plot_histogram`` return a matplotlib ``(fig, ax, …)``
tuple instead, and ``animate`` returns the ``FuncAnimation`` itself (unlike the
raster ``Dataset.animate``, which wraps it in a glyph), so it is saved with
``anim.save(...)``.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import (
    force_agg_backend,
    info_print,
    map_lookup_errors,
    refuse_existing,
)
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "source UGRID NetCDF path"
_HELP_OUTPUT = "destination image path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_DPI = "output DPI (default: 150)"


def _cmd_mesh_plot(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh plot`` — save a filled plot of a mesh variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``cmap``, ``title``,
            ``basemap``, ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        glyph = mesh.plot(
            args.var, cmap=args.cmap, title=args.title, basemap=args.basemap
        )
    glyph.fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_plot_outline(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh plot-outline`` — save the mesh wireframe.

    Args:
        args: Parsed args with ``file``, ``output``, ``basemap``, ``dpi``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    mesh = MeshDataset.read_file(args.file)
    glyph = mesh.plot_outline(basemap=args.basemap)
    glyph.fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_plot_vector_field(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh plot-vector-field`` — save a two-variable field plot.

    The two components must share a ``face`` / ``node`` location; an edge-located
    component is refused by the underlying method.

    Args:
        args: Parsed args with ``file``, ``output``, ``u``, ``v``, ``kind``,
            ``thin``, ``time_index``, ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        fig, _ax, _im = mesh.plot_vector_field(
            args.u,
            args.v,
            kind=args.kind,
            thin=args.thin,
            time_index=args.time_index,
        )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_plot_histogram(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh plot-histogram`` — save a variable's value histogram.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``bins``, ``dpi``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        fig, _ax, _hist = mesh.plot_histogram(bins=args.bins, variable_name=args.var)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_animate(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh animate`` — save a temporal variable's animation.

    The animation needs a time axis; the writer is chosen by matplotlib from the
    output extension (``.gif`` → pillow, ``.mp4`` → ffmpeg).

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``interval``,
            ``dpi``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        anim = mesh.animate(variable_name=args.var, interval=args.interval)
    anim.save(args.output, dpi=args.dpi)
    info_print(args, f"wrote {args.output}")
    return 0


def add_ugrid_plot_commands(msub: argparse._SubParsersAction) -> None:
    """Register the mesh plotting commands on the ``mesh`` sub-parser.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    m_plot = msub.add_parser("plot", help="save a filled plot of a mesh variable (PNG)")
    m_plot.add_argument("file", help=_HELP_MESH_FILE)
    m_plot.add_argument("output", help=_HELP_OUTPUT + " (e.g. mesh.png)")
    m_plot.add_argument("--var", required=True, help="mesh variable to plot")
    m_plot.add_argument(
        "--cmap", default="viridis", help="matplotlib colormap (default: viridis)"
    )
    m_plot.add_argument("--title", default=None, help="figure title (default: none)")
    m_plot.add_argument(
        "--basemap",
        default=None,
        help="web-tile basemap provider name (default: none; needs a CRS)",
    )
    m_plot.add_argument("--dpi", type=int, default=150, help=_HELP_DPI)
    m_plot.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_plot.set_defaults(func=_cmd_mesh_plot)

    m_outline = msub.add_parser(
        "plot-outline", help="save the mesh wireframe/outline (PNG)"
    )
    m_outline.add_argument("file", help=_HELP_MESH_FILE)
    m_outline.add_argument("output", help=_HELP_OUTPUT + " (e.g. outline.png)")
    m_outline.add_argument(
        "--basemap",
        default=None,
        help="web-tile basemap provider name (default: none; needs a CRS)",
    )
    m_outline.add_argument("--dpi", type=int, default=150, help=_HELP_DPI)
    m_outline.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_outline.set_defaults(func=_cmd_mesh_plot_outline)

    m_vfield = msub.add_parser(
        "plot-vector-field", help="save a two-variable (u, v) vector-field plot (PNG)"
    )
    m_vfield.add_argument("file", help=_HELP_MESH_FILE)
    m_vfield.add_argument("output", help=_HELP_OUTPUT + " (e.g. field.png)")
    m_vfield.add_argument("--u", required=True, help="x-component variable name")
    m_vfield.add_argument("--v", required=True, help="y-component variable name")
    m_vfield.add_argument(
        "--kind",
        choices=["quiver", "barbs", "streamplot"],
        default="quiver",
        help="render kind (default: quiver)",
    )
    m_vfield.add_argument(
        "--thin",
        type=int,
        default=None,
        help="keep every n-th arrow to declutter (default: keep all)",
    )
    m_vfield.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="time step to draw for a temporal component (default: 0)",
    )
    m_vfield.add_argument("--dpi", type=int, default=150, help=_HELP_DPI)
    m_vfield.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_vfield.set_defaults(func=_cmd_mesh_plot_vector_field)

    m_hist = msub.add_parser(
        "plot-histogram", help="save a variable's value-distribution histogram (PNG)"
    )
    m_hist.add_argument("file", help=_HELP_MESH_FILE)
    m_hist.add_argument("output", help=_HELP_OUTPUT + " (e.g. hist.png)")
    m_hist.add_argument(
        "--var",
        default=None,
        help="mesh variable to histogram (default: the only data variable)",
    )
    m_hist.add_argument(
        "--bins", type=int, default=15, help="number of bins (default 15)"
    )
    m_hist.add_argument("--dpi", type=int, default=150, help=_HELP_DPI)
    m_hist.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_hist.set_defaults(func=_cmd_mesh_plot_histogram)

    m_anim = msub.add_parser(
        "animate", help="save an animation of a temporal mesh variable (GIF/MP4)"
    )
    m_anim.add_argument("file", help=_HELP_MESH_FILE)
    m_anim.add_argument("output", help="destination animation path (e.g. out.gif)")
    m_anim.add_argument(
        "--var",
        default=None,
        help="temporal variable to animate (default: the only temporal one)",
    )
    m_anim.add_argument(
        "--interval", type=int, default=200, help="milliseconds between frames"
    )
    m_anim.add_argument(
        "--dpi", type=int, default=100, help="output DPI (default: 100)"
    )
    m_anim.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_anim.set_defaults(func=_cmd_mesh_animate)
