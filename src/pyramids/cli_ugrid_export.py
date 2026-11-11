"""UGRID mesh convert/export CLI commands — ``pyramids mesh …``.

Thin 1:1 wrappers over the convert/export surface of
:class:`pyramids.netcdf.ugrid.MeshDataset` (issue #1470, the "convert / export"
theme). Each command reads a mesh, maps onto a single ``MeshDataset`` method, and
writes one output artifact — a mesh file, a vector layer, a tabular file, a
raster, a Zarr store, an image, or a raw byte buffer. No mesh logic lives here.

These sit alongside the Phase 1 ``mesh info|extract|convert`` verbs in
:mod:`pyramids.cli_ugrid`; the registrator here is wired onto the same ``mesh``
sub-parser by the top-level CLI. The existing ``mesh convert`` already wraps
``to_feature_collection``; the ``export geodataframe`` verb here is its distinct
sibling (``MeshDataset.to_geodataframe``) and does not duplicate it.

Voronoi:
    - ``MeshDataset.voronoi`` returns a raw :class:`scipy.spatial.Voronoi` (vertices,
      ridges, regions) with no geospatial file form, so it has no verb of its own.
      Its serialising counterpart ``MeshDataset.voronoi_polygons`` — the Voronoi cells
      of the nodes as a vector layer — is what the ``voronoi-polygons`` verb here wraps.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pyramids._cli_shared import (
    force_agg_backend,
    info_print,
    map_lookup_errors,
    needs_parquet_extra,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "source UGRID NetCDF path"
_HELP_MESH_OUTPUT = "destination UGRID NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_VAR = "mesh variable to operate on (default: the only data variable)"
_HELP_VECTOR_OUTPUT = "destination vector path"
_HELP_DRIVER = "OGR driver (default: inferred from the extension)"


def _cmd_mesh_to_file(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export file`` — re-write a mesh through a writer.

    Round-trips the mesh to UGRID NetCDF, or, with ``--driver``, through a
    registered third-party mesh driver (2DM, Selafin, PLY, …).

    Args:
        args: Parsed args with ``file``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mesh.to_file(args.output, driver=args.driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_geodataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export geodataframe`` — write the mesh as a vector file.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``location``,
            ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        gdf = mesh.to_geodataframe(variable_name=args.var, location=args.location)
    FeatureCollection(gdf).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_voronoi_polygons(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh voronoi-polygons`` — Voronoi cells of the mesh nodes.

    Writes the dual cells as a vector file. The serialising counterpart of
    :meth:`MeshDataset.voronoi`, whose bare :class:`scipy.spatial.Voronoi` has no file form.

    Args:
        args: Parsed args with ``file``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        cells = mesh.voronoi_polygons()
    cells.to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_dataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export dataframe`` — write a tidy table (CSV or Parquet).

    The frame carries one row per element value with ``variable``/``element``/
    coordinate columns; the output format follows the extension (``.parquet`` →
    Parquet, else CSV).

    Args:
        args: Parsed args with ``file``, ``output``, ``variables``, ``dropna``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    variables = args.variables if args.variables else None
    with map_lookup_errors():
        df = mesh.to_dataframe(variables=variables, dropna=args.dropna)
    if args.output.lower().endswith(".parquet"):
        with needs_parquet_extra():
            df.to_parquet(args.output)
    else:
        df.to_csv(args.output, index=False)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_dataset(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export dataset`` — rasterize a variable to a GeoTIFF.

    Interpolates a mesh variable onto a regular grid and writes the resulting
    raster to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``cell_size``,
            ``method``, ``bounds``, ``epsg``, ``nodata``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    bounds = tuple(args.bounds) if args.bounds else None
    with map_lookup_errors():
        grid = mesh.to_dataset(
            args.var,
            args.cell_size,
            method=args.method,
            bounds=bounds,
            epsg=args.epsg,
            nodata=args.nodata,
        )
    grid.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export zarr`` — write the mesh to a Zarr store.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mesh.to_zarr(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_image(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export image`` — write a colour-mapped image of a variable.

    The mesh is rasterised, then the raster's colour mapping (through cleopatra)
    produces the image. The Agg backend is forced first so a headless run never
    opens a GUI window.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``cell_size``,
            ``cmap``, ``method``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    force_agg_backend()
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        image = mesh.to_image(
            variable_name=args.var,
            cell_size=args.cell_size,
            cmap=args.cmap,
            method=args.method,
        )
    image.save(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_bytes(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export bytes`` — serialise the mesh to a UGRID NetCDF buffer.

    Writes the encoded bytes to ``output`` when given, else to stdout so they can
    be piped.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    payload = mesh.to_bytes()
    if args.output is not None:
        with open(args.output, "wb") as handle:
            handle.write(payload)
        info_print(args, f"wrote {args.output}")
    else:
        sys.stdout.buffer.write(payload)
    return 0


def _cmd_mesh_to_location(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export location`` — remap a variable onto another facet.

    Averages the contributing elements, so a face field asked for on the nodes
    becomes each node's mean of the surrounding faces, and writes the new mesh.

    Args:
        args: Parsed args with ``file``, ``output``, ``location``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        moved = mesh.to_location(args.location, variable_name=args.var)
    moved.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_to_crs(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh export crs`` — reproject the mesh to a target CRS.

    Args:
        args: Parsed args with ``file``, ``output``, ``epsg``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    reprojected = mesh.to_crs(args.epsg)
    reprojected.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_set_crs(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh advanced set-crs`` — declare the mesh's CRS (no reprojection).

    Exactly one of ``--epsg`` / ``--crs`` must be given; the underlying method
    enforces that and raises a clean error otherwise.

    Args:
        args: Parsed args with ``file``, ``output``, ``epsg``, ``crs``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    mesh.set_crs(crs=args.crs, epsg=args.epsg)
    mesh.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_burn(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh burn`` — burn a vector layer onto the faces.

    Every face the geometry covers takes ``--value``; the rest take ``--fill``
    (NaN by default). Writes the mesh carrying the new variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``geometry``, ``value``,
            ``name``, ``fill``, ``all_touched``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    geometry = FeatureCollection.read_file(args.geometry)
    fill = float("nan") if args.fill is None else args.fill
    burned = mesh.burn(
        geometry,
        args.value,
        name=args.name,
        fill=fill,
        all_touched=args.all_touched,
    )
    burned.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_regrid(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh regrid`` — regrid a variable onto another mesh.

    Args:
        args: Parsed args with ``file``, ``output``, ``target``, ``method``,
            ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    target = MeshDataset.read_file(args.target)
    with map_lookup_errors():
        regridded = mesh.regrid(target, method=args.method, variable_name=args.var)
    regridded.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_polygonize(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh polygonize`` — dissolve equal-valued faces to polygons.

    Merges adjacent faces of equal value into one polygon per region and writes
    them to a vector file whose driver follows the extension (or ``--driver``).

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``decimals``,
            ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        gdf = mesh.polygonize(variable_name=args.var, decimals=args.decimals)
    FeatureCollection(gdf).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_attach_data(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh attach-data`` — attach a results-only file's variables.

    Reads values from a topology-less results file (SMS ``.dat``, XMDF, XDMF, …)
    and merges them onto the mesh, then writes the combined mesh.

    Args:
        args: Parsed args with ``file``, ``output``, ``data``, ``driver``,
            ``lazy``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    attached = mesh.attach_data(args.data, driver=args.driver, lazy=args.lazy)
    attached.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_with_variable(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh advanced with-variable`` — add a variable from a ``.npy`` array.

    The values are loaded from a NumPy ``.npy`` file: ``(n_elements,)`` for a
    static variable or ``(n_time, n_elements)`` for a temporal one.

    Args:
        args: Parsed args with ``file``, ``output``, ``name``, ``data``,
            ``location``, ``nodata``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    values = np.load(args.data)
    updated = mesh.with_variable(
        args.name, values, location=args.location, nodata=args.nodata
    )
    updated.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_mesh_rename_variable(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh advanced rename-variable`` — rename a data variable.

    Args:
        args: Parsed args with ``file``, ``output``, ``old``, ``new``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        renamed = mesh.rename_variable(args.old, args.new)
    renamed.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _add_to_file(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export file``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser("file", help="re-write a mesh (UGRID NetCDF or a driver)")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--driver",
        help="registered mesh driver (default: built-in UGRID NetCDF writer)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_file)


def _add_to_geodataframe(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export geodataframe``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "geodataframe", help="write the mesh elements as a vector file"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_VECTOR_OUTPUT)
    parser.add_argument("--var", help=_HELP_VAR + " (carried as a column)")
    parser.add_argument(
        "--location",
        default="face",
        help="mesh location: face / node / edge (default: face)",
    )
    parser.add_argument("--driver", help=_HELP_DRIVER)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_geodataframe)


def _add_to_dataframe(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export dataframe``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "dataframe", help="write a tidy value table (CSV, or Parquet by extension)"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help="destination .csv or .parquet path")
    parser.add_argument(
        "--variables", nargs="+", help="variable names to include (default: all)"
    )
    parser.add_argument(
        "--dropna", action="store_true", help="drop rows whose value is no-data (NaN)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_dataframe)


def _add_to_dataset(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export dataset``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "dataset", help="rasterize a mesh variable to a regular-grid GeoTIFF"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help="destination raster path")
    parser.add_argument("--var", required=True, help="mesh variable to rasterize")
    parser.add_argument(
        "--cell-size",
        type=float,
        required=True,
        help="grid cell size in coordinate units",
    )
    parser.add_argument(
        "--method", default="nearest", help="interpolation method (default: nearest)"
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="target bounds (default: the mesh bounds)",
    )
    parser.add_argument("--epsg", type=int, help="target EPSG code (default: mesh CRS)")
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="output no-data value (default: -9999)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_dataset)


def _add_to_zarr(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export zarr``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser("zarr", help="write the mesh to a Zarr store")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help="destination Zarr store path")
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_zarr)


def _add_to_image(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export image``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "image", help="write a colour-mapped image of a mesh variable (PNG/JPEG)"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help="destination image path (e.g. out.png)")
    parser.add_argument("--var", help=_HELP_VAR)
    parser.add_argument(
        "--cell-size",
        type=float,
        default=None,
        help="rasterisation cell size (default: ~512 px across the longer side)",
    )
    parser.add_argument(
        "--cmap", default="viridis", help="matplotlib colormap (default: viridis)"
    )
    parser.add_argument(
        "--method", default="nearest", help="rasterisation method (default: nearest)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_image)


def _add_to_bytes(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export bytes``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "bytes", help="serialise the mesh to a UGRID NetCDF byte buffer"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument(
        "output", nargs="?", help="destination file (default: write to stdout)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_bytes)


def _add_to_location(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export location``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "location", help="remap a variable onto another mesh facet (node/edge/face)"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--location", required=True, help="target facet: node / edge / face"
    )
    parser.add_argument("--var", help=_HELP_VAR)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_location)


def _add_to_crs(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh export crs``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser("crs", help="reproject the mesh to a target CRS")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--epsg", type=int, required=True, help="target EPSG code (e.g. 4326)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_to_crs)


def _add_set_crs(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh set-crs``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "set-crs", help="declare the mesh's CRS without reprojecting"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument("--epsg", type=int, help="EPSG code to assign")
    parser.add_argument("--crs", help="WKT CRS string to assign")
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_set_crs)


def _add_burn(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh burn``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser("burn", help="burn a vector layer onto the mesh faces")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--geometry", required=True, help="vector file whose geometry is burned"
    )
    parser.add_argument(
        "--value", type=float, required=True, help="value burned into covered faces"
    )
    parser.add_argument(
        "--name", default="burned", help="name for the new variable (default: burned)"
    )
    parser.add_argument(
        "--fill", type=float, default=None, help="value for missed faces (default: NaN)"
    )
    parser.add_argument(
        "--no-all-touched",
        dest="all_touched",
        action="store_false",
        help="only burn faces whose centroid falls inside (default: every touched face)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_burn, all_touched=True)


def _add_regrid(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh regrid``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser("regrid", help="regrid a variable onto another mesh")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--target", required=True, help="destination mesh whose topology is the target"
    )
    parser.add_argument(
        "--method",
        default="overlap",
        help="overlap / conservative / centroid / barycentric (default: overlap)",
    )
    parser.add_argument("--var", help=_HELP_VAR)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_regrid)


def _add_polygonize(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh polygonize``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "polygonize", help="dissolve equal-valued faces into polygons (vector file)"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_VECTOR_OUTPUT)
    parser.add_argument("--var", help=_HELP_VAR + " (the face field to dissolve by)")
    parser.add_argument(
        "--decimals", type=int, help="round to this many decimals before grouping"
    )
    parser.add_argument("--driver", help=_HELP_DRIVER)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_polygonize)


def _add_attach_data(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh attach-data``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "attach-data", help="attach a results-only file's variables to the mesh"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument(
        "--data", required=True, help="results file to read values from"
    )
    parser.add_argument(
        "--driver", help="registered driver to use (default: detected from the suffix)"
    )
    parser.add_argument(
        "--no-lazy",
        dest="lazy",
        action="store_false",
        help="read variable values eagerly (default: defer until first access)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_attach_data, lazy=True)


def _add_with_variable(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh with-variable``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "with-variable", help="add a data variable from a NumPy .npy array"
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument("--name", required=True, help="name for the new variable")
    parser.add_argument(
        "--data", required=True, help=".npy file holding the values array"
    )
    parser.add_argument(
        "--location",
        default="face",
        help="mesh location: face / node / edge (default: face)",
    )
    parser.add_argument(
        "--nodata", type=float, default=None, help="no-data value for the new variable"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_with_variable)


def _add_rename_variable(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh rename-variable``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser("rename-variable", help="rename a data variable")
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_MESH_OUTPUT)
    parser.add_argument("--old", required=True, help="current variable name")
    parser.add_argument("--new", required=True, help="new variable name")
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_rename_variable)


def _add_voronoi_polygons(msub: argparse._SubParsersAction) -> None:
    """Register ``mesh voronoi-polygons``.

    Args:
        msub: The ``mesh`` sub-parser action.
    """
    parser = msub.add_parser(
        "voronoi-polygons",
        help="Voronoi (Thiessen) cells of the mesh nodes, as a vector file",
    )
    parser.add_argument("file", help=_HELP_MESH_FILE)
    parser.add_argument("output", help=_HELP_VECTOR_OUTPUT)
    parser.add_argument("--driver", help=_HELP_DRIVER)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_mesh_voronoi_polygons)


def add_ugrid_export_commands(
    msub: argparse._SubParsersAction, advanced: argparse._SubParsersAction | None = None
) -> None:
    """Register the mesh convert/export commands on the ``mesh`` sub-parser.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action (shared with the
        advanced: The cross-cutting ``advanced`` group's subparsers action, or ``None``.
            When given, rarely-one-shot primitive verbs register there instead of on the
            main subsystem; isolation parsers omit it so those verbs stay flat locally.
            Phase 1 ``info``/``extract``/``convert`` verbs).
    """
    _add_to_file(msub)
    _add_to_geodataframe(msub)
    _add_voronoi_polygons(msub)
    _add_to_dataframe(msub)
    _add_to_dataset(msub)
    _add_to_zarr(msub)
    _add_to_image(msub)
    _add_to_bytes(msub)
    _add_to_location(msub)
    _add_to_crs(msub)
    _add_set_crs(advanced or msub)
    _add_burn(msub)
    _add_regrid(msub)
    _add_polygonize(msub)
    _add_attach_data(msub)
    _add_with_variable(advanced or msub)
    _add_rename_variable(advanced or msub)
