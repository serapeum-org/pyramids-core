"""Selection / subset commands for the ``pyramids mesh`` group.

Thin wrappers over :class:`pyramids.netcdf.ugrid.MeshDataset`'s selection and subset
members. Each command reads a mesh with :meth:`MeshDataset.read_file`, applies one
member, and writes the resulting mesh to a fresh path with :meth:`MeshDataset.to_file`
(the source is only read) — so every command is a writer guarded by ``refuse_existing`` /
``--overwrite``. No mesh logic lives here; each verb maps 1:1 onto an existing method.

Three axes of selection are covered: **time** (``sel``/``isel``/``sel-time``/
``sel-time-range``/``thin``/``head``/``tail``/``drop-sel``/``drop-isel``/
``drop-duplicates``/``squeeze``), **layer** on a ragged 3-D volume mesh (``sel-layer``/
``isel-layer``/``collapse-layer``), and **space** (``subset-by-bounds``/``crop``/``clip``).
``clip-values`` clamps values and ``where`` masks elements by a comparison;
``drop-variables`` drops whole variables. A bad index/label raised as
``KeyError``/``IndexError`` is remapped to ``ValueError`` via ``map_lookup_errors`` so the
CLI reports it as a clean ``error:`` line rather than an "unexpected failure".

``sel-time`` collapses the time axis down to a single step by position — the serialising
counterpart of ``isel --time N`` (which also covers the same need). An earlier kernel bug
left a time-collapsed variable's ``dimensions`` stale so the result could not be written
back; that has since been fixed, and the round-trip is covered by a test.
"""

from __future__ import annotations

import argparse
import operator
import re
from collections.abc import Callable
from typing import Any

from pyramids._cli_shared import info_print, map_lookup_errors, refuse_existing
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "source mesh path (UGRID NetCDF)"
_HELP_OUTPUT = "destination mesh path (.nc keeps it a UGRID NetCDF)"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_VAR = "variable to act on (default: the mesh's only data variable)"
_BBOX_METAVAR = ("MINX", "MINY", "MAXX", "MAXY")

_WHERE_OPERATORS: dict[str, Callable[[Any, float], Any]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
_COND_PATTERN = re.compile(
    r"^\s*(?:(?P<var>[A-Za-z_]\w*)\s*)?(?P<op>>=|<=|==|!=|>|<)\s*(?P<value>[-+0-9.eE]+)\s*$"
)


def _scalar_or_list(values: list[Any] | None) -> Any:
    """Return a lone value as a scalar, several as a list, and ``None`` unchanged.

    A scalar time/element index collapses that axis in ``sel``/``isel``, whereas a list
    keeps it; passing a one-element list straight through would wrongly keep the axis, so
    unwrap it first.

    Args:
        values: The parsed ``nargs="+"`` list, or ``None`` when the flag was omitted.

    Returns:
        Any: ``None``, the sole element, or the list itself.
    """
    if values is None:
        result: Any = None
    elif len(values) == 1:
        result = values[0]
    else:
        result = values
    return result


def _resolve_variable(mesh: MeshDataset, name: str | None) -> str:
    """Return the variable to act on, defaulting to the mesh's sole data variable.

    Args:
        mesh: The opened mesh.
        name: An explicit variable name, or ``None`` to use the only one.

    Returns:
        str: The resolved variable name.

    Raises:
        ValueError: ``name`` is ``None`` and the mesh does not expose exactly one data
            variable, so the command cannot pick one on the user's behalf.
    """
    if name is not None:
        return name
    names = list(mesh.data_variable_names)
    if len(names) != 1:
        raise ValueError(
            f"this mesh exposes {len(names)} data variables ({names}); pass --var to choose one"
        )
    return names[0]


def _build_condition(mesh: MeshDataset, variable_name: str, expr: str) -> Any:
    """Turn a ``[VAR] OP NUMBER`` comparison into a boolean mask over a variable's values.

    ``MeshDataset.where`` takes a boolean array, not a Python callable, so a comparison
    expression is enough to drive it from the CLI (a callable-taking form would be out of
    scope). The comparison is evaluated against ``variable_name``'s values.

    Args:
        mesh: The opened mesh.
        variable_name: The variable whose values the comparison is applied to.
        expr: A comparison such as ``"depth > 3"`` or ``"> 3"``.

    Returns:
        Any: A boolean array broadcastable to the variable's values.

    Raises:
        ValueError: ``expr`` is not a ``[VAR] OP NUMBER`` comparison.
    """
    match = _COND_PATTERN.match(expr)
    if match is None:
        raise ValueError(
            f"--cond must be '[VAR] OP NUMBER' with OP one of {sorted(_WHERE_OPERATORS)}, "
            f"got {expr!r}"
        )
    values = mesh.get_data(variable_name).data
    return _WHERE_OPERATORS[match.group("op")](values, float(match.group("value")))


def _write(args: argparse.Namespace, result: MeshDataset) -> int:
    """Write a selected mesh to ``args.output`` and print the quiet-aware success line.

    Args:
        args: Parsed args carrying ``output`` and the global ``quiet`` flag.
        result: The mesh produced by the selection.

    Returns:
        int: ``0`` on success.
    """
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_sel(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh sel`` — select time steps by coordinate value.

    Args:
        args: Parsed args with ``file``, ``output``, ``time``, ``method``, ``tolerance``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    time = _scalar_or_list(args.time)
    with map_lookup_errors():
        result = mesh.sel(time, method=args.method, tolerance=args.tolerance)
    return _write(args, result)


def _cmd_mesh_isel(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh isel`` — select by position along the time and/or element axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``time``, ``element``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.isel(
            time=_scalar_or_list(args.time), element=_scalar_or_list(args.element)
        )
    return _write(args, result)


def _cmd_mesh_sel_time(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh sel-time`` — select a single time step by position.

    Collapses the time axis down to the step at ``--index``; non-temporal variables are
    left unchanged. This is the serialising counterpart of ``isel --time N``.

    Args:
        args: Parsed args with ``file``, ``output``, ``index``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.sel_time(args.index)
    return _write(args, result)


def _cmd_mesh_sel_layer(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh sel-layer`` — select a layer by coordinate value.

    Args:
        args: Parsed args with ``file``, ``output``, ``value``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.sel_layer(args.value)
    return _write(args, result)


def _cmd_mesh_isel_layer(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh isel-layer`` — select a layer by position.

    Args:
        args: Parsed args with ``file``, ``output``, ``index``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.isel_layer(args.index)
    return _write(args, result)


def _cmd_mesh_sel_time_range(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh sel-time-range`` — select a half-open range of time steps.

    Args:
        args: Parsed args with ``file``, ``output``, ``start``, ``stop``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.sel_time_range(args.start, args.stop)
    return _write(args, result)


def _cmd_mesh_collapse_layer(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh collapse-layer`` — collapse a 3-D volume to a 2-D face field.

    Args:
        args: Parsed args with ``file``, ``output``, ``layer``, ``location``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.collapse_layer(args.layer, location=args.location)
    return _write(args, result)


def _cmd_mesh_subset_by_bounds(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh subset-by-bounds`` — keep the faces inside a bounding box.

    Args:
        args: Parsed args with ``file``, ``output``, ``bbox``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    minx, miny, maxx, maxy = args.bbox
    with map_lookup_errors():
        result = mesh.subset_by_bounds(minx, miny, maxx, maxy)
    return _write(args, result)


def _cmd_mesh_crop(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh crop`` — keep faces inside a polygon mask or a bbox.

    Args:
        args: Parsed args with ``file``, ``output``, ``mask``, ``bbox``, ``epsg``,
            ``touch``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mask = None if args.mask is None else FeatureCollection.read_file(args.mask)
    bbox = None if args.bbox is None else tuple(args.bbox)
    with map_lookup_errors():
        result = mesh.crop(mask=mask, touch=args.touch, bbox=bbox, epsg=args.epsg)
    return _write(args, result)


def _cmd_mesh_clip(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh clip`` — clip the mesh to a polygon mask.

    Args:
        args: Parsed args with ``file``, ``output``, ``mask``, ``touch``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mask = FeatureCollection.read_file(args.mask)
    with map_lookup_errors():
        result = mesh.clip(mask, touch=args.touch)
    return _write(args, result)


def _cmd_mesh_clip_values(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh clip-values`` — clamp a variable's values to ``[lower, upper]``.

    Args:
        args: Parsed args with ``file``, ``output``, ``lower``, ``upper``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.clip_values(args.lower, args.upper, variable_name=args.var)
    return _write(args, result)


def _cmd_mesh_thin(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh thin`` — keep every ``N``-th time step.

    Args:
        args: Parsed args with ``file``, ``output``, ``step``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.thin(args.step)
    return _write(args, result)


def _cmd_mesh_where(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh where`` — mask the elements a comparison does not select.

    Args:
        args: Parsed args with ``file``, ``output``, ``cond``, ``other``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    match = _COND_PATTERN.match(args.cond)
    cond_var = None if match is None else match.group("var")
    variable_name = _resolve_variable(mesh, cond_var or args.var)
    cond = _build_condition(mesh, variable_name, args.cond)
    with map_lookup_errors():
        result = mesh.where(cond, other=args.other, variable_name=variable_name)
    return _write(args, result)


def _cmd_mesh_drop_sel(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh drop-sel`` — drop time steps by coordinate value.

    Args:
        args: Parsed args with ``file``, ``output``, ``values``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.drop_sel(_scalar_or_list(args.values))
    return _write(args, result)


def _cmd_mesh_drop_isel(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh drop-isel`` — drop time steps by position.

    Args:
        args: Parsed args with ``file``, ``output``, ``indices``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.drop_isel(_scalar_or_list(args.indices))
    return _write(args, result)


def _cmd_mesh_drop_variables(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh drop-variables`` — drop whole data variables.

    Args:
        args: Parsed args with ``file``, ``output``, ``names``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.drop_variables(args.names)
    return _write(args, result)


def _cmd_mesh_drop_duplicates(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh drop-duplicates`` — keep the first of each repeated time stamp.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.drop_duplicates()
    return _write(args, result)


def _cmd_mesh_head(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh head`` — keep the first ``N`` time steps.

    Args:
        args: Parsed args with ``file``, ``output``, ``n``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.head(args.n)
    return _write(args, result)


def _cmd_mesh_tail(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh tail`` — keep the last ``N`` time steps.

    Args:
        args: Parsed args with ``file``, ``output``, ``n``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.tail(args.n)
    return _write(args, result)


def _cmd_mesh_squeeze(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh squeeze`` — drop a length-1 time dimension.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.squeeze()
    return _write(args, result)


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``file``/``output``/``--overwrite`` arguments to a writer subparser.

    Args:
        parser: The subparser to extend.
    """
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)


def add_ugrid_select_commands(msub: argparse._SubParsersAction) -> None:
    """Register the ``mesh`` selection / subset commands on ``msub``.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    m_sel = msub.add_parser("sel", help="select time steps by coordinate value")
    _add_io_arguments(m_sel)
    m_sel.add_argument(
        "--time",
        nargs="+",
        type=float,
        required=True,
        help="coordinate value(s); one value collapses the time axis, several keep it",
    )
    m_sel.add_argument(
        "--method",
        choices=("nearest",),
        default=None,
        help="'nearest' snaps each value to the closest coordinate (default: exact)",
    )
    m_sel.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="furthest a --method nearest snap may reach",
    )
    m_sel.set_defaults(func=_cmd_mesh_sel)

    m_isel = msub.add_parser("isel", help="select by time and/or element position")
    _add_io_arguments(m_isel)
    m_isel.add_argument(
        "--time",
        nargs="+",
        type=int,
        default=None,
        help="0-based time step(s); one collapses the axis",
    )
    m_isel.add_argument(
        "--element",
        nargs="+",
        type=int,
        default=None,
        help="0-based face index/indices to keep (renumbers the mesh)",
    )
    m_isel.set_defaults(func=_cmd_mesh_isel)

    m_sl = msub.add_parser("sel-layer", help="select a layer by coordinate value")
    _add_io_arguments(m_sl)
    m_sl.add_argument(
        "--value", type=float, required=True, help="layer coordinate value"
    )
    m_sl.set_defaults(func=_cmd_mesh_sel_layer)

    m_il = msub.add_parser("isel-layer", help="select a layer by position")
    _add_io_arguments(m_il)
    m_il.add_argument("--index", type=int, required=True, help="0-based layer position")
    m_il.set_defaults(func=_cmd_mesh_isel_layer)

    m_st = msub.add_parser(
        "sel-time", help="select a single time step by position (collapses the axis)"
    )
    _add_io_arguments(m_st)
    m_st.add_argument(
        "--index",
        type=int,
        required=True,
        help="0-based time step position (positional, not a coordinate value; "
        "use 'sel --time' to select by coordinate)",
    )
    m_st.set_defaults(func=_cmd_mesh_sel_time)

    m_str = msub.add_parser(
        "sel-time-range", help="select a half-open range of time steps"
    )
    _add_io_arguments(m_str)
    m_str.add_argument(
        "--start", type=int, required=True, help="start step index (inclusive)"
    )
    m_str.add_argument(
        "--stop", type=int, required=True, help="stop step index (exclusive)"
    )
    m_str.set_defaults(func=_cmd_mesh_sel_time_range)

    m_cl = msub.add_parser(
        "collapse-layer", help="collapse a 3-D volume to a 2-D face field"
    )
    _add_io_arguments(m_cl)
    m_cl.add_argument(
        "--layer",
        type=int,
        required=True,
        help="0-based layer position to take per column",
    )
    m_cl.add_argument(
        "--location",
        default="face",
        help="target mesh location (only 'face' is supported)",
    )
    m_cl.set_defaults(func=_cmd_mesh_collapse_layer)

    m_sbb = msub.add_parser(
        "subset-by-bounds", help="keep the faces inside a bounding box"
    )
    _add_io_arguments(m_sbb)
    m_sbb.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=_BBOX_METAVAR,
        help="bounding box in the mesh's own CRS",
    )
    m_sbb.set_defaults(func=_cmd_mesh_subset_by_bounds)

    m_crop = msub.add_parser("crop", help="keep faces inside a polygon mask or a bbox")
    _add_io_arguments(m_crop)
    crop_region = m_crop.add_mutually_exclusive_group(required=True)
    crop_region.add_argument("--mask", help="polygon vector file whose faces to keep")
    crop_region.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=_BBOX_METAVAR,
        help="bounding box to crop to",
    )
    m_crop.add_argument(
        "--epsg", type=int, default=None, help="EPSG of --bbox (default: the mesh CRS)"
    )
    m_crop.add_argument(
        "--no-touch",
        dest="touch",
        action="store_false",
        help="keep only faces fully inside the mask (default: also those touching it)",
    )
    m_crop.set_defaults(func=_cmd_mesh_crop)

    m_clip = msub.add_parser("clip", help="clip the mesh to a polygon mask")
    _add_io_arguments(m_clip)
    m_clip.add_argument("--mask", required=True, help="polygon vector file to clip to")
    m_clip.add_argument(
        "--no-touch",
        dest="touch",
        action="store_false",
        help="keep only faces fully inside the mask (default: also those touching it)",
    )
    m_clip.set_defaults(func=_cmd_mesh_clip)

    m_cv = msub.add_parser(
        "clip-values", help="clamp a variable's values to [lower, upper]"
    )
    _add_io_arguments(m_cv)
    m_cv.add_argument(
        "--lower", type=float, required=True, help="minimum allowed value"
    )
    m_cv.add_argument(
        "--upper", type=float, required=True, help="maximum allowed value"
    )
    m_cv.add_argument("--var", default=None, help=_HELP_VAR)
    m_cv.set_defaults(func=_cmd_mesh_clip_values)

    m_thin = msub.add_parser("thin", help="keep every N-th time step")
    _add_io_arguments(m_thin)
    m_thin.add_argument("--step", type=int, required=True, help="stride (>= 1)")
    m_thin.set_defaults(func=_cmd_mesh_thin)

    m_where = msub.add_parser(
        "where", help="mask elements a comparison does not select"
    )
    _add_io_arguments(m_where)
    m_where.add_argument(
        "--cond",
        required=True,
        help="comparison '[VAR] OP NUMBER', e.g. 'depth > 3' or '> 3'",
    )
    m_where.add_argument(
        "--other",
        type=float,
        default=float("nan"),
        help="value for unselected elements (default: NaN)",
    )
    m_where.add_argument("--var", default=None, help=_HELP_VAR)
    m_where.set_defaults(func=_cmd_mesh_where)

    m_dsel = msub.add_parser("drop-sel", help="drop time steps by coordinate value")
    _add_io_arguments(m_dsel)
    m_dsel.add_argument(
        "--values",
        nargs="+",
        type=float,
        required=True,
        help="coordinate value(s) to drop",
    )
    m_dsel.set_defaults(func=_cmd_mesh_drop_sel)

    m_disel = msub.add_parser("drop-isel", help="drop time steps by position")
    _add_io_arguments(m_disel)
    m_disel.add_argument(
        "--indices",
        nargs="+",
        type=int,
        required=True,
        help="0-based step position(s) to drop",
    )
    m_disel.set_defaults(func=_cmd_mesh_drop_isel)

    m_dvar = msub.add_parser("drop-variables", help="drop whole data variables")
    _add_io_arguments(m_dvar)
    m_dvar.add_argument(
        "--names", nargs="+", required=True, help="variable name(s) to drop"
    )
    m_dvar.set_defaults(func=_cmd_mesh_drop_variables)

    m_dd = msub.add_parser(
        "drop-duplicates", help="keep the first of each repeated time stamp"
    )
    _add_io_arguments(m_dd)
    m_dd.set_defaults(func=_cmd_mesh_drop_duplicates)

    m_head = msub.add_parser("head", help="keep the first N time steps")
    _add_io_arguments(m_head)
    m_head.add_argument(
        "--n", type=int, default=5, help="how many leading steps to keep (default: 5)"
    )
    m_head.set_defaults(func=_cmd_mesh_head)

    m_tail = msub.add_parser("tail", help="keep the last N time steps")
    _add_io_arguments(m_tail)
    m_tail.add_argument(
        "--n", type=int, default=5, help="how many trailing steps to keep (default: 5)"
    )
    m_tail.set_defaults(func=_cmd_mesh_tail)

    m_sq = msub.add_parser("squeeze", help="drop a length-1 time dimension")
    _add_io_arguments(m_sq)
    m_sq.set_defaults(func=_cmd_mesh_squeeze)
