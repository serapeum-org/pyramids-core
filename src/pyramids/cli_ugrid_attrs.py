"""UGRID mesh global-attribute CLI commands — ``pyramids mesh set-attr`` / ``del-attr``.

Thin 1:1 wrappers over :meth:`pyramids.netcdf.ugrid.MeshDataset.set_global_attribute`
and :meth:`~pyramids.netcdf.ugrid.MeshDataset.delete_global_attribute`. A UGRID
derivation is immutable, so both methods answer a *new* dataset rather than mutating
in place; each handler reads the input mesh, applies the change, and writes the
returned dataset to ``output`` — the source file is never touched. Deleting an absent
attribute is a no-op, mirroring the method's own contract and the ``nc del-attr`` verb.

Attribute values are taken verbatim as strings, matching the ``nc set-attr`` verb; no
type coercion is attempted.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import info_print, refuse_existing
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "source UGRID NetCDF path"
_HELP_OUTPUT = "destination UGRID NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"


def _cmd_mesh_set_attr(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh set-attr`` — set a global attribute into a new file.

    Args:
        args: Parsed args with ``file``, ``output``, ``name``, ``value``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mesh.set_global_attribute(args.name, args.value).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_del_attr(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh del-attr`` — delete a global attribute into a new file.

    Dropping an attribute that is not present is not an error (it writes an
    unchanged copy), mirroring the method's own contract.

    Args:
        args: Parsed args with ``file``, ``output``, ``name``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mesh.delete_global_attribute(args.name).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_ugrid_attrs_commands(msub: argparse._SubParsersAction) -> None:
    """Register the mesh global-attribute commands on the ``mesh`` sub-parser.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    m_set = msub.add_parser(
        "set-attr", help="set a global attribute into a new mesh file"
    )
    m_set.add_argument("file", help=_HELP_MESH_FILE)
    m_set.add_argument("output", help=_HELP_OUTPUT)
    m_set.add_argument("--name", required=True, help="attribute name")
    m_set.add_argument(
        "--value", required=True, help="attribute value (stored as a string)"
    )
    m_set.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_set.set_defaults(func=_cmd_mesh_set_attr)

    m_del = msub.add_parser(
        "del-attr", help="delete a global attribute into a new mesh file"
    )
    m_del.add_argument("file", help=_HELP_MESH_FILE)
    m_del.add_argument("output", help=_HELP_OUTPUT)
    m_del.add_argument("--name", required=True, help="attribute name")
    m_del.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    m_del.set_defaults(func=_cmd_mesh_del_attr)
