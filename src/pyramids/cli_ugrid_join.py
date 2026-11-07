"""UGRID multi-mesh join CLI commands — ``pyramids mesh concat`` / ``merge``.

Thin 1:1 wrappers over :meth:`pyramids.netcdf.ugrid.MeshDataset.concat` and
:meth:`~pyramids.netcdf.ugrid.MeshDataset.merge`. Both join several same-topology
meshes: ``concat`` stacks matching variables, ``merge`` unions distinct variables
onto the shared topology. Each handler reads every ``--inputs`` path, joins the
first with the rest, and writes the single result to ``output``.

``merge`` refuses a variable that appears in more than one input under the default
``no_conflicts`` compatibility; pass ``--compat override`` to let a later input win.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import info_print, map_lookup_errors, refuse_existing
from pyramids.netcdf.ugrid import MeshDataset

_HELP_INPUTS = "input UGRID NetCDF paths to join (at least two)"
_HELP_OUTPUT = "destination UGRID NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"


def _load_join_inputs(paths: list[str]) -> tuple[MeshDataset, list[MeshDataset]]:
    """Read the join inputs and split them into the receiver and the rest.

    Args:
        paths: The ``--inputs`` mesh paths.

    Returns:
        tuple[MeshDataset, list[MeshDataset]]: The first mesh and the remaining ones.

    Raises:
        ValueError: Fewer than two input paths were given.
    """
    if len(paths) < 2:
        raise ValueError(f"a join needs at least two input meshes; got {len(paths)}")
    datasets = [MeshDataset.read_file(path) for path in paths]
    return datasets[0], datasets[1:]


def _cmd_mesh_concat(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh concat`` — concatenate same-topology meshes.

    Args:
        args: Parsed args with ``inputs``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    first, rest = _load_join_inputs(args.inputs)
    with map_lookup_errors():
        result = first.concat(rest)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_merge(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh merge`` — merge variables of same-topology meshes.

    Args:
        args: Parsed args with ``inputs``, ``output``, ``compat``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    first, rest = _load_join_inputs(args.inputs)
    with map_lookup_errors():
        result = first.merge(rest, compat=args.compat)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_ugrid_join_commands(msub: argparse._SubParsersAction) -> None:
    """Register the multi-mesh join commands on the ``mesh`` sub-parser.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    m_concat = msub.add_parser(
        "concat", help="concatenate several same-topology meshes into one"
    )
    m_concat.add_argument("output", help=_HELP_OUTPUT)
    m_concat.add_argument(
        "--inputs", nargs="+", required=True, metavar="MESH", help=_HELP_INPUTS
    )
    m_concat.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_concat.set_defaults(func=_cmd_mesh_concat)

    m_merge = msub.add_parser(
        "merge", help="merge the variables of several same-topology meshes into one"
    )
    m_merge.add_argument("output", help=_HELP_OUTPUT)
    m_merge.add_argument(
        "--inputs", nargs="+", required=True, metavar="MESH", help=_HELP_INPUTS
    )
    m_merge.add_argument(
        "--compat",
        default="no_conflicts",
        help="conflict policy for shared variables (default: no_conflicts; "
        "use 'override' to let a later input win)",
    )
    m_merge.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_merge.set_defaults(func=_cmd_mesh_merge)
