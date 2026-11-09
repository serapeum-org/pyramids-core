"""UGRID mesh inspection CLI commands — ``pyramids mesh <verb>``.

Thin 1:1 wrappers over the read-only inspection surface of
:class:`pyramids.netcdf.ugrid.MeshDataset`. Three kinds of verb:

- **print verbs** (``summary``, ``mesh-names``, ``keys``, ``items``, ``values``,
  ``get``) describe the mesh or a variable to stdout, with ``--json`` for a
  machine-readable form.
- **geometry verbs** (``exterior-edges``, ``exterior-faces``, ``circumcenters``,
  ``bounding-polygon``) derive vector geometry and write it to a vector file whose
  driver is resolved from the extension (or ``--driver``).
- **array verbs** (``sparse-adjacency``, ``connected-components``, ``active-mask``,
  ``masked-values``) print a numeric summary and, with ``--output``, persist the
  array (``.npy``, or ``.npz`` for the sparse matrix).

No mesh logic lives here — every handler maps 1:1 onto an existing ``MeshDataset``
call. Registered by :func:`add_ugrid_inspect_commands` onto the ``mesh`` sub-action.
"""

from __future__ import annotations

import argparse
import json

import geopandas as gpd
import numpy as np
from scipy.sparse import save_npz
from shapely.geometry import LineString, Point, Polygon

from pyramids._cli_shared import (
    emit,
    info_print,
    json_sanitize,
    map_lookup_errors,
    mark_format_support,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset, MeshVariable

_HELP_MESH_FILE = "UGRID NetCDF path"
_HELP_JSON = "emit JSON"
_HELP_VAR = "mesh variable name"


def _describe_variable(variable: MeshVariable) -> dict:
    """Return the JSON-safe metadata of a :class:`MeshVariable`.

    Args:
        variable: A mesh variable (the value returned by ``get``/``items``/``values``).

    Returns:
        dict: Its name, location, dtype, shape, units, and time metadata.
    """
    return {
        "name": variable.name,
        "location": variable.location,
        "mesh_name": variable.mesh_name,
        "dtype": str(variable.dtype),
        "shape": list(variable.shape),
        "units": variable.units,
        "standard_name": variable.standard_name,
        "nodata": variable.nodata,
        "n_elements": variable.n_elements,
        "n_time_steps": variable.n_time_steps,
        "has_time": variable.has_time,
    }


def _write_vector(
    rows: list, geometry: list, crs: object, args: argparse.Namespace
) -> int:
    """Write derived geometry to the output vector file and report it.

    Args:
        rows: The attribute records, one per geometry (may be empty dicts).
        geometry: The shapely geometries to write.
        crs: The CRS to stamp on the layer (possibly ``None``).
        args: Parsed args carrying ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    gdf = gpd.GeoDataFrame(rows, geometry=geometry, crs=crs)
    FeatureCollection(gdf).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh summary`` — per-variable summary statistics.

    Args:
        args: Parsed args with ``file``, ``variables``, ``metrics``, ``no_skipna``,
            ``ddof``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        frame = mesh.summary(
            variables=args.variables,
            metrics=tuple(args.metrics),
            skipna=not args.no_skipna,
            ddof=args.ddof,
        )
    records = frame.reset_index().to_dict(orient="records")
    if args.json:
        print(json.dumps(json_sanitize({"summary": records})))
    else:
        for record in records:
            print(record)
    return 0


def _cmd_mesh_names(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh mesh-names`` — the topology names in a file.

    Args:
        args: Parsed args with ``file``, ``driver``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    names = MeshDataset.mesh_names(args.file, driver=args.driver)
    emit(args, {"mesh_names": names}, text="\n".join(names))
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh get`` — describe a single data variable.

    Wraps :meth:`MeshDataset.get_data` (the strict lookup shared with ``get``); an
    unknown name is surfaced as a clean error.

    Args:
        args: Parsed args with ``file``, ``variable``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        variable = mesh.get_data(args.variable)
    payload = _describe_variable(variable)
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh keys`` — list the data-variable names.

    Args:
        args: Parsed args with ``file``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    names = mesh.keys()
    if args.json:
        print(json.dumps({"keys": names}))
    else:
        for name in names:
            print(name)
    return 0


def _cmd_items(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh items`` — describe every (name, variable) pair.

    Args:
        args: Parsed args with ``file``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    payload = [_describe_variable(variable) for _, variable in mesh.items()]
    if args.json:
        print(json.dumps(json_sanitize({"items": payload})))
    else:
        for record in payload:
            print(record)
    return 0


def _cmd_values(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh values`` — describe every variable (no keys).

    Args:
        args: Parsed args with ``file``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    payload = [_describe_variable(variable) for variable in mesh.values()]
    if args.json:
        print(json.dumps(json_sanitize({"values": payload})))
    else:
        for record in payload:
            print(record)
    return 0


def _cmd_exterior_edges(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh exterior-edges`` — write boundary edges as lines.

    Args:
        args: Parsed args with ``file``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    topology = mesh.mesh
    with map_lookup_errors():
        edges = mesh.exterior_edges()
        geometry = [
            LineString(list(zip(*topology.get_edge_coords(int(edge)))))
            for edge in edges
        ]
    rows = [{"edge": int(edge)} for edge in edges]
    return _write_vector(rows, geometry, mesh.crs, args)


def _cmd_exterior_faces(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh exterior-faces`` — write boundary faces as polygons.

    Args:
        args: Parsed args with ``file``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    topology = mesh.mesh
    with map_lookup_errors():
        faces = mesh.exterior_faces()
        geometry = [Polygon(topology.get_face_polygon(int(face))) for face in faces]
    rows = [{"face": int(face)} for face in faces]
    return _write_vector(rows, geometry, mesh.crs, args)


def _cmd_circumcenters(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh circumcenters`` — write face circumcenters as points.

    Defined for triangular meshes only; a non-triangular mesh raises a clean error.

    Args:
        args: Parsed args with ``file``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    centers = mesh.circumcenters()
    geometry = [Point(float(x), float(y)) for x, y in centers]
    rows = [{"face": index} for index in range(len(geometry))]
    return _write_vector(rows, geometry, mesh.crs, args)


def _cmd_bounding_polygon(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh bounding-polygon`` — write the mesh outline polygon.

    Args:
        args: Parsed args with ``file``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    polygon = mesh.bounding_polygon()
    return _write_vector([{"mesh": mesh.mesh_name}], [polygon], mesh.crs, args)


def _cmd_sparse_adjacency(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh sparse-adjacency`` — the adjacency matrix summary.

    Prints the matrix shape and non-zero count; with ``--output`` the matrix is
    written to a ``.npz`` file (``scipy.sparse.save_npz``).

    Args:
        args: Parsed args with ``file``, ``kind``, ``output``, ``overwrite``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        matrix = mesh.sparse_adjacency(kind=args.kind)
    payload = {"kind": args.kind, "shape": list(matrix.shape), "nnz": int(matrix.nnz)}
    if args.output:
        refuse_existing(args.output, args.overwrite)
        save_npz(args.output, matrix)
        info_print(args, f"wrote {args.output}")
    elif args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_connected_components(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh connected-components`` — per-node component labels.

    Prints the component count and the number of labelled elements; with ``--output``
    the label array is written to a ``.npy`` file.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    labels = mesh.connected_components()
    payload = {
        "n_elements": int(labels.size),
        "n_components": int(np.unique(labels).size),
    }
    if args.output:
        refuse_existing(args.output, args.overwrite)
        np.save(args.output, labels)
        info_print(args, f"wrote {args.output}")
    elif args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_active_mask(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh active-mask`` — a variable's active-cell mask.

    Prints how many cells are active (or reports that the variable has no mask);
    with ``--output`` the boolean mask is written to a ``.npy`` file.

    Args:
        args: Parsed args with ``file``, ``variable``, ``output``, ``overwrite``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        mask = mesh.active_mask(args.variable)
    if mask is None:
        payload = {"variable": args.variable, "active_mask": None}
    else:
        payload = {
            "variable": args.variable,
            "n_elements": int(mask.size),
            "n_active": int(np.count_nonzero(mask)),
        }
    if args.output and mask is not None:
        refuse_existing(args.output, args.overwrite)
        np.save(args.output, mask)
        info_print(args, f"wrote {args.output}")
    elif args.output:
        raise ValueError(f"variable {args.variable!r} has no active mask to write")
    elif args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_masked_values(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh masked-values`` — a variable's no-data-masked values.

    Prints the element count and finite-value count; with ``--output`` the values
    are written to a ``.npy`` file.

    Args:
        args: Parsed args with ``file``, ``variable``, ``output``, ``overwrite``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.masked_values(args.variable)
    finite = np.isfinite(values)
    payload = {
        "variable": args.variable,
        "n_elements": int(values.size),
        "n_finite": int(np.count_nonzero(finite)),
    }
    if args.output:
        refuse_existing(args.output, args.overwrite)
        np.save(args.output, values)
        info_print(args, f"wrote {args.output}")
    elif args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _add_summary(msub: argparse._SubParsersAction) -> None:
    """Register ``summary``."""
    parser = msub.add_parser("summary", help="per-variable summary statistics")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument(
        "--variables", nargs="+", help="variables to summarise (default: all)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["count", "min", "max", "mean", "std"],
        help="statistics to report (default: count min max mean std)",
    )
    parser.add_argument(
        "--no-skipna",
        dest="no_skipna",
        action="store_true",
        help="count no-data cells instead of skipping them",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="delta degrees of freedom for std (default: 0)",
    )
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_summary)


def _add_mesh_names(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh-names``."""
    parser = msub.add_parser("mesh-names", help="list the topology names in a file")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("--driver", help="mesh driver name (default: detect)")
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_mesh_names)
    mark_format_support(parser)


def _add_get(msub: argparse._SubParsersAction) -> None:
    """Register ``get``."""
    parser = msub.add_parser("get", help="describe a single data variable")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("variable", help=_HELP_VAR)
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_get)


def _add_keys(msub: argparse._SubParsersAction) -> None:
    """Register ``keys``."""
    parser = msub.add_parser("keys", help="list the data-variable names")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_keys)


def _add_items(msub: argparse._SubParsersAction) -> None:
    """Register ``items``."""
    parser = msub.add_parser("items", help="describe every (name, variable) pair")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_items)


def _add_values(msub: argparse._SubParsersAction) -> None:
    """Register ``values``."""
    parser = msub.add_parser("values", help="describe every data variable")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_values)


def _add_geometry_verb(
    msub: argparse._SubParsersAction, name: str, help_text: str, func: object
) -> None:
    """Register a geometry verb that writes a vector file.

    Args:
        msub: The ``mesh`` group's sub-action.
        name: The verb name.
        help_text: The verb's help string.
        func: The handler to bind.
    """
    parser = msub.add_parser(name, help=help_text)
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help="destination vector path")
    parser.add_argument(
        "--driver", help="OGR driver (default: inferred from extension)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    parser.set_defaults(func=func)


def _add_array_verb(
    msub: argparse._SubParsersAction,
    name: str,
    help_text: str,
    func: object,
    *,
    with_variable: bool,
    output_ext: str,
) -> None:
    """Register an array verb that prints a summary and optionally saves the array.

    Args:
        msub: The ``mesh`` group's sub-action.
        name: The verb name.
        help_text: The verb's help string.
        func: The handler to bind.
        with_variable: Whether the verb takes a positional ``variable``.
        output_ext: The expected extension named in ``--output`` help.
    """
    parser = msub.add_parser(name, help=help_text)
    parser.add_argument("file", help=_HELP_MESH_FILE)
    if with_variable:
        parser.add_argument("variable", help=_HELP_VAR)
    parser.add_argument("--output", help=f"write the array to this {output_ext} file")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace --output if it exists"
    )
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=func)


def add_ugrid_inspect_commands(msub: argparse._SubParsersAction) -> None:
    """Register the mesh inspection verbs on the ``mesh`` sub-action.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    _add_summary(msub)
    _add_mesh_names(msub)
    _add_get(msub)
    _add_keys(msub)
    _add_items(msub)
    _add_values(msub)
    _add_geometry_verb(
        msub,
        "exterior-edges",
        "write the boundary edges as a vector",
        _cmd_exterior_edges,
    )
    _add_geometry_verb(
        msub,
        "exterior-faces",
        "write the boundary faces as a vector",
        _cmd_exterior_faces,
    )
    _add_geometry_verb(
        msub,
        "circumcenters",
        "write triangle circumcenters as points",
        _cmd_circumcenters,
    )
    _add_geometry_verb(
        msub,
        "bounding-polygon",
        "write the mesh outline polygon",
        _cmd_bounding_polygon,
    )
    sparse = msub.add_parser(
        "sparse-adjacency", help="the mesh adjacency matrix (summary / .npz)"
    )
    sparse.add_argument("file", help=_HELP_MESH_FILE)
    sparse.add_argument(
        "--kind",
        default="face",
        help="adjacency kind: face / node / edge (default: face)",
    )
    sparse.add_argument("--output", help="write the matrix to this .npz file")
    sparse.add_argument(
        "--overwrite", action="store_true", help="replace --output if it exists"
    )
    sparse.add_argument("--json", action="store_true", help=_HELP_JSON)
    sparse.set_defaults(func=_cmd_sparse_adjacency)
    _add_array_verb(
        msub,
        "connected-components",
        "label the mesh's connected components",
        _cmd_connected_components,
        with_variable=False,
        output_ext=".npy",
    )
    _add_array_verb(
        msub,
        "active-mask",
        "a variable's active-cell mask",
        _cmd_active_mask,
        with_variable=True,
        output_ext=".npy",
    )
    _add_array_verb(
        msub,
        "masked-values",
        "a variable's no-data-masked values",
        _cmd_masked_values,
        with_variable=True,
        output_ext=".npy",
    )
