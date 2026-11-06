"""UGRID mesh read / construct CLI commands — ``pyramids mesh <verb>``.

Thin 1:1 wrappers over the :class:`pyramids.netcdf.ugrid.MeshDataset` constructor
classmethods: each reads some source (a multi-mesh file, raw NumPy arrays, a vector
layer of polygons, a structured raster, a Zarr store, raw bytes, or several files)
and writes the resulting mesh to an output UGRID NetCDF via ``MeshDataset.to_file``.
No mesh logic lives here — every handler maps 1:1 onto an existing ``MeshDataset``
call. Registered by :func:`add_ugrid_read_commands` onto the ``mesh`` sub-action.

Commands registered here:

- ``read-all`` → :meth:`MeshDataset.read_all` (writes one NetCDF per topology)
- ``from-arrays`` → :meth:`MeshDataset.from_arrays` (node/face ``.npy`` arrays)
- ``from-geodataframe`` → :meth:`MeshDataset.from_geodataframe` (a polygon vector layer)
- ``from-structured`` → :meth:`MeshDataset.from_structured` (a raster, one quad per cell)
- ``from-zarr`` → :meth:`MeshDataset.from_zarr` (optional ``[lazy]`` extra)
- ``from-bytes`` → :meth:`MeshDataset.from_bytes` (a file or stdin)
- ``open-mf`` → :meth:`MeshDataset.open_mfdataset` (concatenate several files)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from pyramids._cli_shared import info_print, map_lookup_errors, refuse_existing
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_OUTPUT = "destination UGRID NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_MESH_NAME = "topology name to stamp on the mesh (default: mesh2d)"
_HELP_DRIVER = "mesh driver name, bypassing detection (default: detect)"


@contextlib.contextmanager
def _needs_lazy_extra() -> Iterator[None]:
    """Remap a missing-``zarr`` ``ImportError`` to a clean ``ValueError``.

    ``from_zarr`` imports ``zarr`` from the optional ``[lazy]`` extra; its absence
    raises ``ImportError``, which :func:`pyramids.cli.main` does not catch (it would
    leak the "unexpected failure" branch). Turn it into a clean user error naming
    the extra.

    Yields:
        None: Control returns to the wrapped block.

    Raises:
        ValueError: ``zarr`` (the ``[lazy]`` extra) is not installed.
    """
    try:
        yield
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [lazy] extra (zarr); install it with "
            "`pip install pyramids-gis[lazy]`"
        ) from exc


def _parse_kv(pairs: list[str] | None) -> dict[str, str]:
    """Parse a list of ``NAME=VALUE`` tokens into a dictionary.

    Args:
        pairs: Raw ``NAME=VALUE`` strings (or ``None`` for an empty mapping).

    Returns:
        dict[str, str]: The parsed mapping (empty when ``pairs`` is ``None``).

    Raises:
        ValueError: A token does not contain a single ``=`` separator.
    """
    mapping: dict[str, str] = {}
    for token in pairs or []:
        if "=" not in token:
            raise ValueError(f"expected NAME=VALUE but got {token!r}")
        name, _, value = token.partition("=")
        mapping[name] = value
    return mapping


def _cmd_read_all(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh read-all`` — write every topology to its own NetCDF.

    Reads all meshes in a (possibly multi-mesh) file and writes one
    ``<output_dir>/<mesh_name>.nc`` per topology, so nothing is silently dropped.

    Args:
        args: Parsed args with ``file``, ``output_dir``, ``driver``, ``eager``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    meshes = MeshDataset.read_all(args.file, driver=args.driver, lazy=not args.eager)
    written = []
    for name, mesh in meshes.items():
        out = str(args.output_dir / f"{name}.nc")
        refuse_existing(out, args.overwrite)
        mesh.to_file(out)
        written.append(out)
    for out in written:
        info_print(args, f"wrote {out}")
    return 0


def _cmd_from_arrays(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh from-arrays`` — build a mesh from ``.npy`` arrays.

    Reads the node coordinates and face-node connectivity from NumPy ``.npy`` files
    and, optionally, one ``.npy`` per data variable (``--data NAME=PATH``) with its
    mesh location (``--data-location NAME=face|node|edge``).

    Args:
        args: Parsed args with ``node_x``, ``node_y``, ``faces``, ``output``,
            ``data``, ``data_location``, ``epsg``, ``mesh_name``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    data_paths = _parse_kv(args.data)
    data = {
        name: np.load(path, allow_pickle=False) for name, path in data_paths.items()
    }
    kwargs: dict = {"epsg": args.epsg, "mesh_name": args.mesh_name}
    if data:
        kwargs["data"] = data
        kwargs["data_locations"] = _parse_kv(args.data_location)
    mesh = MeshDataset.from_arrays(
        np.load(args.node_x, allow_pickle=False),
        np.load(args.node_y, allow_pickle=False),
        np.load(args.faces, allow_pickle=False),
        **kwargs,
    )
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_geodataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh from-geodataframe`` — build a mesh from polygons.

    Reads a polygon vector layer; each polygon becomes one face and the layer's
    attribute columns become per-face variables.

    Args:
        args: Parsed args with ``vector``, ``output``, ``mesh_name``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    layer = FeatureCollection.read_file(args.vector)
    mesh = MeshDataset.from_geodataframe(layer, mesh_name=args.mesh_name)
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_structured(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh from-structured`` — build a quad mesh from a raster.

    Each raster cell becomes one quad face (shared corners deduplicated) and each
    band becomes a per-face variable.

    Args:
        args: Parsed args with ``raster``, ``output``, ``band_names``, ``mesh_name``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    raster = Dataset.read_file(args.raster)
    mesh = MeshDataset.from_structured(
        raster, band_names=args.band_names, mesh_name=args.mesh_name
    )
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh from-zarr`` — read a pyramids-written Zarr store.

    Args:
        args: Parsed args with ``store``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with _needs_lazy_extra():
        mesh = MeshDataset.from_zarr(args.store)
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_from_bytes(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh from-bytes`` — open a mesh held as raw bytes.

    Reads the bytes from ``input`` (a file) or, with ``--stdin``, from standard
    input, opens them via :meth:`MeshDataset.from_bytes`, and writes the result.

    Args:
        args: Parsed args with ``input``, ``output``, ``suffix``, ``stdin``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: Neither ``input`` nor ``--stdin`` was given.
    """
    refuse_existing(args.output, args.overwrite)
    if args.stdin:
        data = sys.stdin.buffer.read()
    elif args.input is not None:
        with open(args.input, "rb") as handle:
            data = handle.read()
    else:
        raise ValueError("from-bytes needs an input path or --stdin.")
    mesh = MeshDataset.from_bytes(data, suffix=args.suffix)
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_open_mf(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh open-mf`` — concatenate several mesh files into one.

    Args:
        args: Parsed args with ``files`` (>= 1), ``output``, ``driver``,
            ``mesh_name``, ``no_sort``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with map_lookup_errors():
        mesh = MeshDataset.open_mfdataset(
            args.files,
            driver=args.driver,
            mesh_name=args.mesh_name,
            sort=not args.no_sort,
        )
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _add_read_all(msub: argparse._SubParsersAction) -> None:
    """Register ``read-all``."""
    parser = msub.add_parser(
        "read-all", help="write every topology in a file to its own NetCDF"
    )
    parser.add_argument("file", help="source mesh file path")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory to write one <mesh_name>.nc per topology into",
    )
    parser.add_argument("--driver", help=_HELP_DRIVER)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="read variable values eagerly instead of lazily",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_read_all)


def _add_from_arrays(msub: argparse._SubParsersAction) -> None:
    """Register ``from-arrays``."""
    parser = msub.add_parser(
        "from-arrays", help="build a mesh from node/face .npy arrays"
    )
    parser.add_argument("node_x", help=".npy of node x coordinates")
    parser.add_argument("node_y", help=".npy of node y coordinates")
    parser.add_argument(
        "faces", help=".npy of face-node connectivity (n_face, max_nodes)"
    )
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--data",
        action="append",
        metavar="NAME=PATH",
        help="a per-variable .npy of values (repeatable)",
    )
    parser.add_argument(
        "--data-location",
        dest="data_location",
        action="append",
        metavar="NAME=LOC",
        help="mesh location for a --data variable: face / node / edge (repeatable)",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=4326,
        help="EPSG code of the mesh CRS (default: 4326)",
    )
    parser.add_argument(
        "--mesh-name", dest="mesh_name", default="mesh2d", help=_HELP_MESH_NAME
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_arrays)


def _add_from_geodataframe(msub: argparse._SubParsersAction) -> None:
    """Register ``from-geodataframe``."""
    parser = msub.add_parser(
        "from-geodataframe", help="build a mesh from a polygon vector layer"
    )
    parser.add_argument("vector", help="source polygon vector path")
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--mesh-name", dest="mesh_name", default="mesh2d", help=_HELP_MESH_NAME
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_geodataframe)


def _add_from_structured(msub: argparse._SubParsersAction) -> None:
    """Register ``from-structured``."""
    parser = msub.add_parser(
        "from-structured", help="build a quad mesh from a raster (one quad per cell)"
    )
    parser.add_argument("raster", help="source raster path")
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--band-names",
        dest="band_names",
        nargs="+",
        help="names for the per-face variables, one per band",
    )
    parser.add_argument(
        "--mesh-name", dest="mesh_name", default="mesh2d", help=_HELP_MESH_NAME
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_structured)


def _add_from_zarr(msub: argparse._SubParsersAction) -> None:
    """Register ``from-zarr``."""
    parser = msub.add_parser(
        "from-zarr", help="read a pyramids-written Zarr store (needs the [lazy] extra)"
    )
    parser.add_argument("store", help="input Zarr store (path / fsspec URL)")
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_zarr)


def _add_from_bytes(msub: argparse._SubParsersAction) -> None:
    """Register ``from-bytes``."""
    parser = msub.add_parser("from-bytes", help="open a mesh held as raw bytes")
    parser.add_argument(
        "input", nargs="?", help="file of raw mesh bytes (omit with --stdin)"
    )
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--stdin", action="store_true", help="read the bytes from standard input"
    )
    parser.add_argument(
        "--suffix",
        default=".nc",
        help="extension hint for driver detection (default: .nc)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_from_bytes)


def _add_open_mf(msub: argparse._SubParsersAction) -> None:
    """Register ``open-mf``."""
    parser = msub.add_parser(
        "open-mf", help="concatenate several mesh files into one dataset"
    )
    parser.add_argument("files", nargs="+", help="source mesh file paths (in order)")
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument("--driver", help=_HELP_DRIVER)
    parser.add_argument(
        "--mesh-name",
        dest="mesh_name",
        help="topology to read from each file (default: the first)",
    )
    parser.add_argument(
        "--no-sort",
        dest="no_sort",
        action="store_true",
        help="keep the given file order instead of sorting by time",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_open_mf)


def add_ugrid_read_commands(msub: argparse._SubParsersAction) -> None:
    """Register the mesh read / construct verbs on the ``mesh`` sub-action.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    _add_read_all(msub)
    _add_from_arrays(msub)
    _add_from_geodataframe(msub)
    _add_from_structured(msub)
    _add_from_zarr(msub)
    _add_from_bytes(msub)
    _add_open_mf(msub)
