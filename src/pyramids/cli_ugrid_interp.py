"""UGRID mesh interpolation / fill / time / calculus commands — ``pyramids mesh …``.

Thin wrappers over :class:`pyramids.netcdf.ugrid.MeshDataset` for the theme that reads a
mesh, applies one along-time / along-value transform, and writes a new mesh file:

- interpolation onto another mesh or new time coordinates (``interp-like``, ``interp-time``);
- calendar handling (``convert-calendar``, ``interp-calendar``);
- the fill family (``interpolate-na``, ``fillna``, ``ffill``, ``bfill``, ``fill-na-spatial``,
  ``laplace-interpolate``) — names confirmed against the live class after the #1463 rename;
- windowed reductions (``resample``, ``rolling``, ``coarsen``);
- the along-time calculus (``integrate``, ``cumulative-integrate``, ``differentiate``);
- and the per-variable value transforms (``convert-units``, ``shift``, ``dropna``).

Each command maps 1:1 onto an existing ``MeshDataset`` method; no mesh logic lives here. The
output is written with :meth:`MeshDataset.to_file`. Two theme methods have no CLI surface:
``curvefit`` takes a Python callable (``func``) a shell cannot supply, and ``polyfit`` returns
a variable on a ``coefficient`` dimension that ``MeshDataset.to_file`` cannot yet serialize
(it raises ``RuntimeError: Illegal numpy array rank 2``), so there is no mesh file to write.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from pyramids._cli_shared import info_print, map_lookup_errors, refuse_existing
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "UGRID NetCDF path"
_HELP_OUTPUT = "destination mesh path (.nc keeps it a UGRID NetCDF)"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_VAR = "data variable to act on; required only when the mesh has >1 data variable"
_HELP_DIM = "the dimension to operate along (default: the time dimension)"
_HELP_LIMIT = "maximum number of consecutive gaps to fill (default: no limit)"


def _apply(
    args: argparse.Namespace, transform: Callable[[MeshDataset], MeshDataset]
) -> int:
    """Read the input mesh, apply ``transform``, and write the result to ``args.output``.

    Centralises the overwrite guard, the ``KeyError``/``IndexError`` remap (so a typo'd
    ``--var`` / ``--dim`` is a clean user error rather than an "unexpected failure"), and the
    quiet-aware success line that every command in this group shares.

    Args:
        args: Parsed args carrying ``file``, ``output``, ``overwrite`` (and the quiet flag).
        transform: A callable mapping the opened mesh to the mesh to write.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``args.output`` exists without ``--overwrite``, or a variable/dimension
            token is not part of the mesh (remapped from ``KeyError``/``IndexError``).
    """
    refuse_existing(args.output, args.overwrite)
    with map_lookup_errors():
        mesh = MeshDataset.read_file(args.file)
        result = transform(mesh)
        result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_interp_like(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh interp-like`` — interpolate onto another mesh's grid.

    Args:
        args: Parsed args with ``file``, ``output``, ``other``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args, lambda mesh: mesh.interp_like(MeshDataset.read_file(args.other))
    )


def _cmd_mesh_interp_time(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh interp-time`` — resample variables onto new time coordinates.

    Args:
        args: Parsed args with ``file``, ``output``, ``time``, ``method``, ``bounds``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.interp_time(
            args.time, method=args.method, bounds=args.bounds
        ),
    )


def _cmd_mesh_interp_calendar(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh interp-calendar`` — interpolate onto a target's time axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``target``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args, lambda mesh: mesh.interp_calendar(MeshDataset.read_file(args.target))
    )


def _cmd_mesh_convert_calendar(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh convert-calendar`` — recast the time axis to a calendar.

    Args:
        args: Parsed args with ``file``, ``output``, ``calendar``, ``align_on``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args, lambda mesh: mesh.convert_calendar(args.calendar, align_on=args.align_on)
    )


def _cmd_mesh_interpolate_na(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh interpolate-na`` — fill gaps by interpolating along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``method``, ``limit``,
            ``use_coordinate``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.interpolate_na(
            method=args.method,
            limit=args.limit,
            use_coordinate=args.use_coordinate,
            dim=args.dim,
        ),
    )


def _cmd_mesh_fillna(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh fillna`` — replace missing values with a constant.

    Args:
        args: Parsed args with ``file``, ``output``, ``value``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.fillna(args.value, variable_name=args.var))


def _cmd_mesh_ffill(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh ffill`` — forward-fill gaps along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``limit``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.ffill(limit=args.limit, dim=args.dim))


def _cmd_mesh_bfill(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh bfill`` — back-fill gaps along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``limit``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.bfill(limit=args.limit, dim=args.dim))


def _cmd_mesh_fill_na_spatial(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh fill-na-spatial`` — fill gaps from nearby cells in space.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``max_distance``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.fill_na_spatial(
            variable_name=args.var, max_distance=args.max_distance
        ),
    )


def _cmd_mesh_laplace_interpolate(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh laplace-interpolate`` — fill gaps by solving Laplace's equation.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``use_weights``,
            ``direct_solve``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.laplace_interpolate(
            variable_name=args.var,
            use_weights=args.use_weights,
            direct_solve=args.direct_solve,
        ),
    )


def _cmd_mesh_resample(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh resample`` — bin the time axis to a frequency and reduce.

    Args:
        args: Parsed args with ``file``, ``output``, ``freq``, ``how``, ``skipna``, ``q``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.resample(args.freq, args.how, skipna=args.skipna, q=args.q),
    )


def _cmd_mesh_rolling(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh rolling`` — a moving-window reduction along a dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``window``, ``how``, ``center``,
            ``min_periods``, ``q``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.rolling(
            args.window,
            args.how,
            center=args.center,
            min_periods=args.min_periods,
            q=args.q,
            dim=args.dim,
        ),
    )


def _cmd_mesh_coarsen(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh coarsen`` — block-reduce the time axis by a window.

    Args:
        args: Parsed args with ``file``, ``output``, ``window``, ``how``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args,
        lambda mesh: mesh.coarsen(args.window, args.how, variable_name=args.var),
    )


def _cmd_mesh_integrate(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh integrate`` — integrate a variable over the time axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.integrate(variable_name=args.var))


def _cmd_mesh_cumulative_integrate(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh cumulative-integrate`` — running integral along the time axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.cumulative_integrate(variable_name=args.var))


def _cmd_mesh_differentiate(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh differentiate`` — derivative of a variable along the time axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.differentiate(variable_name=args.var))


def _cmd_mesh_convert_units(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh convert-units`` — convert a variable to target units.

    Args:
        args: Parsed args with ``file``, ``output``, ``target``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args, lambda mesh: mesh.convert_units(args.target, variable_name=args.var)
    )


def _cmd_mesh_shift(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh shift`` — shift values along the time axis by ``--periods``.

    Args:
        args: Parsed args with ``file``, ``output``, ``periods``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(args, lambda mesh: mesh.shift(args.periods))


def _cmd_mesh_dropna(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh dropna`` — drop time steps that carry missing values.

    Args:
        args: Parsed args with ``file``, ``output``, ``how``, ``thresh``, ``dim``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    return _apply(
        args, lambda mesh: mesh.dropna(args.how, thresh=args.thresh, dim=args.dim)
    )


def _add_writer_io(parser: argparse.ArgumentParser) -> None:
    """Add the ``file`` / ``output`` / ``--overwrite`` triad shared by every command.

    Args:
        parser: The subcommand parser to extend.
    """
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)


def add_ugrid_interp_commands(
    msub: argparse._SubParsersAction, advanced: argparse._SubParsersAction | None = None
) -> None:
    """Register the interp/fill/time/calculus ``mesh`` commands on ``msub``.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action (shared with the Phase 1
            ``mesh`` commands).
    """
    m_il = msub.add_parser("interp-like", help="interpolate onto another mesh's grid")
    _add_writer_io(m_il)
    m_il.add_argument(
        "--other", required=True, help="mesh whose grid/time axis the output adopts"
    )
    m_il.set_defaults(func=_cmd_mesh_interp_like)

    m_it = msub.add_parser(
        "interp-time", help="resample variables onto new time coordinates"
    )
    _add_writer_io(m_it)
    m_it.add_argument(
        "--time",
        nargs="+",
        type=float,
        required=True,
        help="target time coordinate(s) on the existing axis's scale",
    )
    m_it.add_argument(
        "--method",
        default="linear",
        choices=("linear", "nearest"),
        help="interpolation method (default: linear)",
    )
    m_it.add_argument(
        "--bounds",
        default="nan",
        choices=("nan", "clamp"),
        help="value for targets outside the axis (default: nan)",
    )
    m_it.set_defaults(func=_cmd_mesh_interp_time)

    m_ic = msub.add_parser(
        "interp-calendar", help="interpolate onto a target's time axis"
    )
    _add_writer_io(m_ic)
    m_ic.add_argument(
        "--target", required=True, help="mesh whose time axis to interpolate onto"
    )
    m_ic.set_defaults(func=_cmd_mesh_interp_calendar)

    m_cc = msub.add_parser(
        "convert-calendar", help="recast the time axis to a calendar"
    )
    _add_writer_io(m_cc)
    m_cc.add_argument(
        "--calendar", required=True, help="target calendar, e.g. noleap, 360_day"
    )
    m_cc.add_argument(
        "--align-on", default="date", dest="align_on", help="date alignment rule"
    )
    m_cc.set_defaults(func=_cmd_mesh_convert_calendar)

    m_in = msub.add_parser(
        "interpolate-na", help="fill gaps by interpolating along a dimension"
    )
    _add_writer_io(m_in)
    m_in.add_argument(
        "--method", default="linear", help="interpolation method (default: linear)"
    )
    m_in.add_argument("--limit", type=int, default=None, help=_HELP_LIMIT)
    m_in.add_argument("--dim", default=None, help=_HELP_DIM)
    m_in.add_argument(
        "--no-use-coordinate",
        dest="use_coordinate",
        action="store_false",
        help="interpolate against index positions, not the coordinate values",
    )
    m_in.set_defaults(func=_cmd_mesh_interpolate_na)

    m_fn = msub.add_parser("fillna", help="replace missing values with a constant")
    _add_writer_io(m_fn)
    m_fn.add_argument(
        "--value", type=float, required=True, help="the constant to fill with"
    )
    m_fn.add_argument("--var", default=None, help=_HELP_VAR)
    m_fn.set_defaults(func=_cmd_mesh_fillna)

    m_ff = (advanced or msub).add_parser(
        "ffill", help="forward-fill gaps along a dimension"
    )
    _add_writer_io(m_ff)
    m_ff.add_argument("--limit", type=int, default=None, help=_HELP_LIMIT)
    m_ff.add_argument("--dim", default=None, help=_HELP_DIM)
    m_ff.set_defaults(func=_cmd_mesh_ffill)

    m_bf = (advanced or msub).add_parser(
        "bfill", help="back-fill gaps along a dimension"
    )
    _add_writer_io(m_bf)
    m_bf.add_argument("--limit", type=int, default=None, help=_HELP_LIMIT)
    m_bf.add_argument("--dim", default=None, help=_HELP_DIM)
    m_bf.set_defaults(func=_cmd_mesh_bfill)

    m_fs = msub.add_parser(
        "fill-na-spatial", help="fill gaps from nearby cells in space"
    )
    _add_writer_io(m_fs)
    m_fs.add_argument("--var", default=None, help=_HELP_VAR)
    m_fs.add_argument(
        "--max-distance",
        dest="max_distance",
        type=float,
        default=None,
        help="only fill from donors within this distance (default: no limit)",
    )
    m_fs.set_defaults(func=_cmd_mesh_fill_na_spatial)

    m_lp = msub.add_parser(
        "laplace-interpolate", help="fill gaps by solving Laplace's equation"
    )
    _add_writer_io(m_lp)
    m_lp.add_argument("--var", default=None, help=_HELP_VAR)
    m_lp.add_argument(
        "--no-weights",
        dest="use_weights",
        action="store_false",
        help="use an unweighted (uniform) graph Laplacian",
    )
    m_lp.add_argument(
        "--direct-solve",
        dest="direct_solve",
        action="store_true",
        help="use a direct sparse solve instead of the iterative one",
    )
    m_lp.set_defaults(func=_cmd_mesh_laplace_interpolate)

    m_rs = msub.add_parser(
        "resample", help="bin the time axis to a frequency and reduce"
    )
    _add_writer_io(m_rs)
    m_rs.add_argument(
        "--freq", required=True, help="pandas offset alias, e.g. 1D, 2W, MS"
    )
    m_rs.add_argument(
        "--how", default="mean", help="aggregation per bin (mean, sum, min, max, ...)"
    )
    m_rs.add_argument(
        "--q", type=float, default=None, help="quantile in [0, 1] when --how quantile"
    )
    m_rs.add_argument(
        "--no-skipna",
        dest="skipna",
        action="store_false",
        help="let missing values propagate instead of skipping them",
    )
    m_rs.set_defaults(func=_cmd_mesh_resample)

    m_rl = (advanced or msub).add_parser(
        "rolling", help="moving-window reduction along a dimension"
    )
    _add_writer_io(m_rl)
    m_rl.add_argument(
        "--window", type=int, required=True, help="window length in steps"
    )
    m_rl.add_argument(
        "--how",
        default="mean",
        help="aggregation per window (mean, sum, min, max, ...)",
    )
    m_rl.add_argument(
        "--center", action="store_true", help="center the window on each step"
    )
    m_rl.add_argument(
        "--min-periods",
        dest="min_periods",
        type=int,
        default=None,
        help="minimum valid steps in a window to produce a value",
    )
    m_rl.add_argument(
        "--q", type=float, default=None, help="quantile in [0, 1] when --how quantile"
    )
    m_rl.add_argument("--dim", default=None, help=_HELP_DIM)
    m_rl.set_defaults(func=_cmd_mesh_rolling)

    m_co = (advanced or msub).add_parser(
        "coarsen", help="block-reduce the time axis by a window"
    )
    _add_writer_io(m_co)
    m_co.add_argument(
        "--window", type=int, required=True, help="number of steps per block"
    )
    m_co.add_argument(
        "--how", default="mean", help="aggregation per block (mean, sum, min, max, ...)"
    )
    m_co.add_argument("--var", default=None, help=_HELP_VAR)
    m_co.set_defaults(func=_cmd_mesh_coarsen)

    m_ig = (advanced or msub).add_parser(
        "integrate", help="integrate a variable over the time axis"
    )
    _add_writer_io(m_ig)
    m_ig.add_argument("--var", default=None, help=_HELP_VAR)
    m_ig.set_defaults(func=_cmd_mesh_integrate)

    m_ci = (advanced or msub).add_parser(
        "cumulative-integrate", help="running integral along the time axis"
    )
    _add_writer_io(m_ci)
    m_ci.add_argument("--var", default=None, help=_HELP_VAR)
    m_ci.set_defaults(func=_cmd_mesh_cumulative_integrate)

    m_df = (advanced or msub).add_parser(
        "differentiate", help="derivative of a variable along the time axis"
    )
    _add_writer_io(m_df)
    m_df.add_argument("--var", default=None, help=_HELP_VAR)
    m_df.set_defaults(func=_cmd_mesh_differentiate)

    m_cu = msub.add_parser("convert-units", help="convert a variable to target units")
    _add_writer_io(m_cu)
    m_cu.add_argument(
        "--to", dest="target", required=True, help="target units, e.g. km, m/s, degC"
    )
    m_cu.add_argument("--var", default=None, help=_HELP_VAR)
    m_cu.set_defaults(func=_cmd_mesh_convert_units)

    m_sh = (advanced or msub).add_parser(
        "shift", help="shift values along the time axis"
    )
    _add_writer_io(m_sh)
    m_sh.add_argument(
        "--periods", type=int, default=1, help="number of steps to shift (default: 1)"
    )
    m_sh.set_defaults(func=_cmd_mesh_shift)

    m_dn = (advanced or msub).add_parser(
        "dropna", help="drop time steps that carry missing values"
    )
    _add_writer_io(m_dn)
    m_dn.add_argument(
        "--how",
        default="any",
        choices=("any", "all"),
        help="drop a step with 'any' missing value (default) or only when 'all' are",
    )
    m_dn.add_argument(
        "--thresh",
        type=int,
        default=None,
        help="keep a step with at least this many valid values",
    )
    m_dn.add_argument("--dim", default=None, help=_HELP_DIM)
    m_dn.set_defaults(func=_cmd_mesh_dropna)
