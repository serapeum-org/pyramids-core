"""UGRID mesh CLI command group — ``pyramids mesh …``.

Thin wrappers over :class:`pyramids.netcdf.ugrid.UgridDataset`: inspect a mesh,
rasterize a mesh variable onto a raster grid, and convert the mesh to a vector
layer. Phase 1 of the full UgridDataset surface (see the CLI-coverage tracker);
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
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid import UgridDataset


def _cmd_mesh_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh info`` — print node/face/edge counts, CRS, bounds.

    Args:
        args: Parsed args with ``file`` and ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = UgridDataset.read_file(args.file)
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
    mesh = UgridDataset.read_file(args.file)
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
    mesh = UgridDataset.read_file(args.file)
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
    m_info.add_argument("file", help="UGRID NetCDF path")
    m_info.add_argument("--json", action="store_true", help="emit JSON")
    m_info.set_defaults(func=_cmd_mesh_info)

    m_ex = msub.add_parser("extract", help="rasterize a mesh variable onto a grid")
    m_ex.add_argument("file", help="UGRID NetCDF path")
    m_ex.add_argument("output", help="destination raster path")
    m_ex.add_argument(
        "--like", required=True, help="template raster whose grid/CRS the output adopts"
    )
    m_ex.add_argument(
        "--var", help="mesh variable to rasterize (default: the first data variable)"
    )
    m_ex.add_argument(
        "--method", default="nearest", help="rasterization method (default: nearest)"
    )
    m_ex.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    m_ex.set_defaults(func=_cmd_mesh_extract)

    m_conv = msub.add_parser("convert", help="convert a mesh to a vector file")
    m_conv.add_argument("file", help="UGRID NetCDF path")
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
