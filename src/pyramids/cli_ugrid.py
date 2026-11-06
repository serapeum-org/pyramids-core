"""UGRID mesh CLI command group — ``pyramids mesh …``.

Thin wrappers over :class:`pyramids.netcdf.ugrid.MeshDataset`: inspect a mesh,
rasterize a mesh variable onto a raster grid, and convert the mesh to a vector
layer. Phase 1 of the full MeshDataset surface (see the CLI-coverage tracker);
no mesh logic lives here — each command maps 1:1 onto an existing method.
"""

from __future__ import annotations

import argparse
import json

from pyramids._cli_shared import (
    info_print,
    json_safe,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.cli_ugrid_analysis import add_ugrid_analysis_commands
from pyramids.cli_ugrid_attrs import add_ugrid_attrs_commands
from pyramids.cli_ugrid_export import add_ugrid_export_commands
from pyramids.cli_ugrid_inspect import add_ugrid_inspect_commands
from pyramids.cli_ugrid_interp import add_ugrid_interp_commands
from pyramids.cli_ugrid_join import add_ugrid_join_commands
from pyramids.cli_ugrid_plot import add_ugrid_plot_commands
from pyramids.cli_ugrid_read import add_ugrid_read_commands
from pyramids.cli_ugrid_select import add_ugrid_select_commands
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "UGRID NetCDF path"


def _validate_variable(mesh: MeshDataset, var: str | None) -> None:
    """Raise a clean ``ValueError`` when ``var`` is not a data variable of ``mesh``.

    ``rasterize_like`` / ``to_feature_collection`` raise ``KeyError`` for an unknown
    variable, which ``main()`` reports as an "unexpected failure" rather than a clean
    user error (the project intentionally maps global ``KeyError`` to that). Validating
    here keeps a typo'd ``--var`` in the expected-error class and gives one consistent
    message across ``extract`` and ``convert`` (M1).

    Args:
        mesh: The opened mesh.
        var: The requested variable name, or ``None``.

    Raises:
        ValueError: ``var`` is given but is not among the mesh's data variables.
    """
    if var is not None and var not in mesh.data_variable_names:
        raise ValueError(
            f"unknown mesh variable {var!r}; available: {list(mesh.data_variable_names)}"
        )


def _cmd_mesh_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh info`` — print node/face/edge counts, CRS, bounds.

    Args:
        args: Parsed args with ``file`` and ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    bounds = mesh.bounds
    time_values = mesh.time_values
    payload = {
        "path": args.file,
        "mesh_name": mesh.mesh_name,
        "n_node": mesh.n_node,
        "n_edge": mesh.n_edge,
        "n_face": mesh.n_face,
        "n_layers": mesh.n_layers,
        "epsg": mesh.epsg,
        "bounds": None if bounds is None else [json_safe(float(v)) for v in bounds],
        "variables": list(mesh.data_variable_names),
        "time_steps": 0 if time_values is None else len(time_values),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_mesh_extract(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh extract`` — rasterize a mesh variable onto a grid.

    Args:
        args: Parsed args with ``file``, ``output``, ``like``, ``var``, ``method``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    _validate_variable(mesh, args.var)
    template = Dataset.read_file(args.like)
    raster = mesh.rasterize_like(template, variable_name=args.var, method=args.method)
    raster.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_convert(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh convert`` — write the mesh as a vector file.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``location``,
            ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    mesh = MeshDataset.read_file(args.file)
    _validate_variable(mesh, args.var)
    fc = mesh.to_feature_collection(variable_name=args.var, location=args.location)
    fc.to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def add_ugrid_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``mesh`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    mesh = sub.add_parser("mesh", help="UGRID mesh commands")
    msub = mesh.add_subparsers(dest="command", required=True)

    m_info = msub.add_parser("info", help="print mesh metadata")
    m_info.add_argument("file", help=_HELP_MESH_FILE)
    m_info.add_argument("--json", action="store_true", help="emit JSON")
    m_info.set_defaults(func=_cmd_mesh_info)

    m_ex = msub.add_parser("extract", help="rasterize a mesh variable onto a grid")
    m_ex.add_argument("file", help=_HELP_MESH_FILE)
    m_ex.add_argument("output", help="destination raster path")
    m_ex.add_argument(
        "--like", required=True, help="template raster whose grid/CRS the output adopts"
    )
    m_ex.add_argument(
        "--var",
        help="mesh variable to rasterize; required when the mesh has >1 data variable",
    )
    m_ex.add_argument(
        "--method", default="nearest", help="rasterization method (default: nearest)"
    )
    m_ex.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    m_ex.set_defaults(func=_cmd_mesh_extract)

    m_conv = msub.add_parser("convert", help="convert a mesh to a vector file")
    m_conv.add_argument("file", help=_HELP_MESH_FILE)
    m_conv.add_argument("output", help="destination vector path")
    m_conv.add_argument(
        "--var", help="mesh variable to carry onto each cell (optional)"
    )
    m_conv.add_argument(
        "--location",
        default="face",
        help="mesh location to convert: face / node / edge (default: face)",
    )
    m_conv.add_argument(
        "--driver", help="OGR driver (default: inferred from extension)"
    )
    m_conv.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    m_conv.set_defaults(func=_cmd_mesh_convert)

    add_ugrid_read_commands(msub)
    add_ugrid_inspect_commands(msub)
    add_ugrid_select_commands(msub)
    add_ugrid_export_commands(msub)
    add_ugrid_analysis_commands(msub)
    add_ugrid_interp_commands(msub)
    add_ugrid_plot_commands(msub)
    add_ugrid_attrs_commands(msub)
    add_ugrid_join_commands(msub)
