"""Selection / subset commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`'s index/label selection members, each
returning a new NetCDF written to a fresh path (the source is only read). Positional
selection (`isel`), windowing (`thin`/`head`/`tail`), dropping by position or label
(`drop-isel`/`drop-sel`), de-duplication (`drop-duplicates`), gap removal (`dropna`), and
constant padding (`pad`). Reads go through the ``NetCDF`` class; no xarray is imported here.

Two shapes of receiver are involved. ``isel``, ``dropna`` and ``pad`` operate on the
file's container and keep its multidimensional structure (so the output keeps its band
dimension). The band-windowing members (``thin``/``head``/``tail``/``drop-isel``/
``drop-sel``/``drop-duplicates``) cut one variable's bands and refuse a container, so these
commands resolve the file's single data variable first and refuse a multi-variable file
with a clear message.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from pyramids._cli_shared import info_print, map_lookup_errors, refuse_existing
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OUTPUT = "destination path (.nc keeps it a NetCDF)"
_HELP_OVERWRITE = "replace the output if it already exists"
_METAVAR_DIM_N = "DIM=N"


def _dim_value_map(
    tokens: list[str], *, cast: Callable[[str], Any] = str
) -> dict[str, Any]:
    """Parse ``DIM=VALUE`` tokens into a ``{dimension: cast(value)}`` mapping.

    Args:
        tokens: The raw ``DIM=VALUE`` strings from a ``nargs="+"`` flag.
        cast: How to coerce each value. ``int`` for a band index (raising ``ValueError``
            on a non-integer, as ``int("x")`` does); the default ``str`` keeps a label
            verbatim.

    Returns:
        dict[str, Any]: One entry per token, the value coerced by ``cast``.

    Raises:
        ValueError: A token is not of the form ``DIM=VALUE`` (no ``=`` or an empty
            dimension), or ``cast`` rejects the value.
    """
    indexers: dict[str, Any] = {}
    for token in tokens:
        dim, sep, value = token.partition("=")
        if not sep or not dim:
            raise ValueError(f"expected DIM=VALUE, got {token!r}")
        indexers[dim] = cast(value)
    return indexers


def _single_variable(nc: NetCDF) -> Any:
    """Return the file's sole data variable, refusing a multi-variable file.

    The band-windowing members cut one variable's bands and have nothing to cut on a
    container, so these commands need exactly one data variable to act on.

    Args:
        nc: The container read from the input file.

    Returns:
        NetCDF: The one variable, ready for a band-windowing member.

    Raises:
        ValueError: The file does not expose exactly one data variable.
    """
    names = list(nc.variable_names)
    if len(names) != 1:
        raise ValueError(
            f"this command operates on a single-variable file, but it exposes "
            f"{len(names)} variables ({names}); extract the one you want first."
        )
    return nc.get_variable(names[0])


def _selected(build: Callable[[], NetCDF]) -> NetCDF:
    """Run a selection and map a bad index/label to a caught ``ValueError``.

    The members raise ``IndexError`` for an out-of-range position and ``KeyError`` for an
    absent label; neither is in :func:`pyramids.cli.main`'s caught set, so they would be
    reported as an "unexpected failure". This remaps them to ``ValueError`` so the CLI
    user sees a clean ``error:`` line.

    Args:
        build: A thunk that performs the selection and returns the new NetCDF.

    Returns:
        NetCDF: Whatever ``build`` returned.

    Raises:
        ValueError: ``build`` raised ``KeyError`` or ``IndexError``.
    """
    with map_lookup_errors():
        return build()


def _cmd_nc_isel(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc isel`` — select bands by position into a new NetCDF.

    Args:
        args: Parsed args with ``file``, ``output``, ``index``, ``drop``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    indexers = _dim_value_map(args.index, cast=int)
    nc = NetCDF.read_file(args.file)
    result = _selected(lambda: nc.isel(drop=args.drop, **indexers))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_thin(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc thin`` — keep every ``N``-th band of the file's variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``step``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    steps = _dim_value_map(args.step, cast=int)
    variable = _single_variable(NetCDF.read_file(args.file))
    result = _selected(lambda: variable.thin(**steps))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_head(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc head`` — keep the first ``N`` bands of the file's variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``count``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    counts = _dim_value_map(args.count, cast=int)
    variable = _single_variable(NetCDF.read_file(args.file))
    result = _selected(lambda: variable.head(**counts))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_tail(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc tail`` — keep the last ``N`` bands of the file's variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``count``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    counts = _dim_value_map(args.count, cast=int)
    variable = _single_variable(NetCDF.read_file(args.file))
    result = _selected(lambda: variable.tail(**counts))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_drop_isel(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc drop-isel`` — drop bands by position from the file's variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``index``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    indexers = _dim_value_map(args.index, cast=int)
    variable = _single_variable(NetCDF.read_file(args.file))
    result = _selected(lambda: variable.drop_isel(**indexers))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_drop_sel(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc drop-sel`` — drop bands by coordinate label from the variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``label``, ``errors``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    labels = _dim_value_map(args.label, cast=str)
    variable = _single_variable(NetCDF.read_file(args.file))
    result = _selected(lambda: variable.drop_sel(errors=args.errors, **labels))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_drop_duplicates(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc drop-duplicates`` — drop bands whose stamp repeats.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``keep``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    variable = _single_variable(NetCDF.read_file(args.file))
    result = _selected(lambda: variable.drop_duplicates(args.dim, keep=args.keep))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_dropna(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc dropna`` — drop a dimension's steps that hold missing cells.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``how``, ``thresh``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    result = _selected(lambda: nc.dropna(args.dim, how=args.how, thresh=args.thresh))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_pad(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc pad`` — grow dimensions with a constant fill into a new file.

    Args:
        args: Parsed args with ``file``, ``output``, ``width``, ``mode``, ``value``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    pad_width = _dim_value_map(args.width, cast=int)
    nc = NetCDF.read_file(args.file)
    result = _selected(
        lambda: nc.pad(mode=args.mode, constant_values=args.value, **pad_width)
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_subset(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc subset`` — read a windowed `(variable, time, bbox)` slice.

    Reads only the requested window of ``--var`` from the gridded cube — a time step via
    ``--time``, a spatial window via ``--bbox`` (in ``--crs``), and an index per extra
    non-spatial dimension via ``--dim DIM=IDX`` — and writes the georeferenced single-
    variable result to ``output``. Selection is by integer index, as the underlying
    ``subset`` does.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``time``, ``bbox``,
            ``crs``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` is absent or not gridded, a ``--dim`` token is malformed or
            names a non-selectable dimension, a required non-spatial dimension is
            unselected, or the bbox selects no cells (raised by ``subset``).
    """
    refuse_existing(args.output, args.overwrite)
    dims = _dim_value_map(args.dim, cast=int) if args.dim else {}
    bbox = tuple(args.bbox) if args.bbox else None
    nc = NetCDF.read_file(args.file)
    result = _selected(
        lambda: nc.subset(args.var, time=args.time, bbox=bbox, crs=args.crs, **dims)
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_select_commands(ncsub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` selection / subset commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_isel = ncsub.add_parser("isel", help="select bands by position")
    nc_isel.add_argument("file", help=_HELP_NC_FILE)
    nc_isel.add_argument("output", help=_HELP_OUTPUT)
    nc_isel.add_argument(
        "--index",
        nargs="+",
        required=True,
        metavar=_METAVAR_DIM_N,
        help="DIM=N pairs, N a 0-based band index",
    )
    nc_isel.add_argument(
        "--drop", action="store_true", help="drop axes a scalar index collapsed"
    )
    nc_isel.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_isel.set_defaults(func=_cmd_nc_isel)

    nc_thin = ncsub.add_parser("thin", help="keep every N-th band")
    nc_thin.add_argument("file", help=_HELP_NC_FILE)
    nc_thin.add_argument("output", help=_HELP_OUTPUT)
    nc_thin.add_argument(
        "--step",
        nargs="+",
        required=True,
        metavar=_METAVAR_DIM_N,
        help="DIM=N pairs, N the stride (at least 1)",
    )
    nc_thin.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_thin.set_defaults(func=_cmd_nc_thin)

    nc_head = ncsub.add_parser("head", help="keep the first N bands")
    nc_head.add_argument("file", help=_HELP_NC_FILE)
    nc_head.add_argument("output", help=_HELP_OUTPUT)
    nc_head.add_argument(
        "--count",
        nargs="+",
        required=True,
        metavar=_METAVAR_DIM_N,
        help="DIM=N pairs, N how many leading bands to keep",
    )
    nc_head.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_head.set_defaults(func=_cmd_nc_head)

    nc_tail = ncsub.add_parser("tail", help="keep the last N bands")
    nc_tail.add_argument("file", help=_HELP_NC_FILE)
    nc_tail.add_argument("output", help=_HELP_OUTPUT)
    nc_tail.add_argument(
        "--count",
        nargs="+",
        required=True,
        metavar=_METAVAR_DIM_N,
        help="DIM=N pairs, N how many trailing bands to keep",
    )
    nc_tail.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_tail.set_defaults(func=_cmd_nc_tail)

    nc_di = ncsub.add_parser("drop-isel", help="drop bands by position")
    nc_di.add_argument("file", help=_HELP_NC_FILE)
    nc_di.add_argument("output", help=_HELP_OUTPUT)
    nc_di.add_argument(
        "--index",
        nargs="+",
        required=True,
        metavar=_METAVAR_DIM_N,
        help="DIM=N pairs, N a 0-based band index to drop",
    )
    nc_di.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_di.set_defaults(func=_cmd_nc_drop_isel)

    nc_ds = ncsub.add_parser("drop-sel", help="drop bands by coordinate label")
    nc_ds.add_argument("file", help=_HELP_NC_FILE)
    nc_ds.add_argument("output", help=_HELP_OUTPUT)
    nc_ds.add_argument(
        "--label",
        nargs="+",
        required=True,
        metavar="DIM=VALUE",
        help="DIM=VALUE pairs (a CF date label resolves on a time axis)",
    )
    nc_ds.add_argument(
        "--errors",
        choices=("raise", "ignore"),
        default="raise",
        help="'raise' (default) refuses an absent label; 'ignore' skips it",
    )
    nc_ds.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_ds.set_defaults(func=_cmd_nc_drop_sel)

    nc_dd = ncsub.add_parser("drop-duplicates", help="drop repeated-stamp bands")
    nc_dd.add_argument("file", help=_HELP_NC_FILE)
    nc_dd.add_argument("output", help=_HELP_OUTPUT)
    nc_dd.add_argument("--dim", required=True, help="band dimension to de-duplicate")
    nc_dd.add_argument(
        "--keep",
        choices=("first", "last"),
        default="first",
        help="which plane of a repeated stamp to keep",
    )
    nc_dd.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_dd.set_defaults(func=_cmd_nc_drop_duplicates)

    nc_dn = ncsub.add_parser("dropna", help="drop steps that hold missing cells")
    nc_dn.add_argument("file", help=_HELP_NC_FILE)
    nc_dn.add_argument("output", help=_HELP_OUTPUT)
    nc_dn.add_argument(
        "--dim", required=True, help="non-spatial dimension to drop from"
    )
    nc_dn.add_argument(
        "--how",
        choices=("any", "all"),
        default="any",
        help="'any' drops a step with any gap; 'all' only a fully empty step",
    )
    nc_dn.add_argument(
        "--thresh",
        type=int,
        default=None,
        help="keep a step with at least this many valid cells (overrides --how)",
    )
    nc_dn.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_dn.set_defaults(func=_cmd_nc_dropna)

    nc_pad = ncsub.add_parser("pad", help="grow dimensions with a constant fill")
    nc_pad.add_argument("file", help=_HELP_NC_FILE)
    nc_pad.add_argument("output", help=_HELP_OUTPUT)
    nc_pad.add_argument(
        "--width",
        nargs="+",
        required=True,
        metavar=_METAVAR_DIM_N,
        help="DIM=N pairs, N steps added on each side (band or x/y/lon/lat)",
    )
    nc_pad.add_argument(
        "--mode", default="constant", help="pad mode (only 'constant' is supported)"
    )
    nc_pad.add_argument(
        "--value",
        type=float,
        default=None,
        help="constant fill (default: the variable's no-data value)",
    )
    nc_pad.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_pad.set_defaults(func=_cmd_nc_pad)

    nc_sub = ncsub.add_parser(
        "subset", help="read a windowed (variable, time, bbox) slice"
    )
    nc_sub.add_argument("file", help=_HELP_NC_FILE)
    nc_sub.add_argument("output", help=_HELP_OUTPUT)
    nc_sub.add_argument("--var", required=True, help="data variable to window")
    nc_sub.add_argument(
        "--time", type=int, default=None, help="0-based time-step index (default: all)"
    )
    nc_sub.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="crop window in --crs (default: the full grid)",
    )
    nc_sub.add_argument(
        "--crs",
        default="EPSG:4326",
        help="CRS of --bbox: an EPSG code or a CRS string (e.g. EPSG:4326); default EPSG:4326",
    )
    nc_sub.add_argument(
        "--dim",
        nargs="+",
        default=None,
        metavar=_METAVAR_DIM_N,
        help="DIM=N index per extra non-spatial dimension (e.g. plev=0)",
    )
    nc_sub.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_sub.set_defaults(func=_cmd_nc_subset)
