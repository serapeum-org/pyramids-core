"""UGRID mesh analysis / terrain CLI commands — ``pyramids mesh <verb>``.

Thin 1:1 wrappers over the analysis surface of
:class:`pyramids.netcdf.ugrid.MeshDataset`: terrain derivatives (``slope``,
``aspect``, ``hillshade``), focal statistics (``focal-mean``, ``focal-std``),
morphology (``binary-dilation``, ``binary-erosion``), connected-value grouping
(``cluster``, ``sieve``), topological distance (``proximity``), domain summaries
(``zonal-stats``, ``histogram``, ``domain-area``, ``count-domain-cells``,
``footprint``), point / line / random sampling (``sample-point``, ``sample``,
``sample-line``, ``sample-random``), and iso-value tracing (``contour``). No mesh
logic lives here — each handler maps onto one existing method; the top-level
parser only wires them together.

The method :meth:`MeshDataset.extract` samples variable values at arbitrary
``(x, y)`` coordinates, but the ``mesh`` group already ships an ``extract`` verb
that *rasterizes* a mesh variable onto a grid (``cli_ugrid.py``). To avoid the
clash, the coordinate sampler is exposed here as ``sample-point``.

Deferred:
    - ``focal_apply`` — takes a Python callable (the per-neighbourhood reducer)
      that has no file/expression form, so it is not CLI-serialisable. The two
      built-in reducers are already surfaced as ``focal-mean`` / ``focal-std``.
"""

from __future__ import annotations

import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    map_lookup_errors,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

_HELP_MESH_FILE = "UGRID NetCDF path"
_HELP_MESH_OUT = "destination mesh (NetCDF) path"
_HELP_VECTOR_OUT = "destination vector path"
_HELP_VAR = "mesh variable to act on; defaults to the only data variable"
_HELP_CSV_OUT = "write x/y/value to this CSV path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_JSON = "emit JSON"
_HELP_DRIVER = "OGR vector driver (default: inferred from the output extension)"


def _print_payload(args: argparse.Namespace, payload: dict) -> None:
    """Emit a scalar/summary payload as JSON or as ``key: value`` lines.

    Args:
        args: Parsed args carrying the ``json`` flag.
        payload: The mapping to print.
    """
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _emit_records(args: argparse.Namespace, frame: pd.DataFrame) -> None:
    """Write a table to CSV (``--output``), dump JSON records, or print them.

    Args:
        args: Parsed args carrying ``output`` and ``json``.
        frame: The table to emit.
    """
    if getattr(args, "output", None) is not None:
        frame.to_csv(args.output, index=False)
        info_print(args, f"wrote {args.output}")
    else:
        records = json_sanitize(frame.to_dict(orient="records"))
        if args.json:
            print(json.dumps(records))
        else:
            for record in records:
                print(record)


def _cmd_slope(args: argparse.Namespace) -> int:
    """Handle ``mesh slope`` — per-face slope magnitude, written as a mesh file.

    Args:
        args: Parsed args with ``file``, ``output``, ``units``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.slope(units=args.units, variable_name=args.var)
    mesh.with_variable("slope", np.asarray(values), location="face").to_file(
        args.output
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_aspect(args: argparse.Namespace) -> int:
    """Handle ``mesh aspect`` — per-face aspect, written as a mesh file.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.aspect(variable_name=args.var)
    mesh.with_variable("aspect", np.asarray(values), location="face").to_file(
        args.output
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_hillshade(args: argparse.Namespace) -> int:
    """Handle ``mesh hillshade`` — per-face shaded relief, written as a mesh file.

    Args:
        args: Parsed args with ``file``, ``output``, ``azimuth``, ``altitude``,
            ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.hillshade(
            azimuth=args.azimuth, altitude=args.altitude, variable_name=args.var
        )
    mesh.with_variable("hillshade", np.asarray(values), location="face").to_file(
        args.output
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_focal_mean(args: argparse.Namespace) -> int:
    """Handle ``mesh focal-mean`` — k-ring neighbourhood mean, written as a mesh file.

    Args:
        args: Parsed args with ``file``, ``output``, ``rings``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.focal_mean(rings=args.rings, variable_name=args.var)
    mesh.with_variable("focal_mean", np.asarray(values), location="face").to_file(
        args.output
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_focal_std(args: argparse.Namespace) -> int:
    """Handle ``mesh focal-std`` — k-ring neighbourhood std, written as a mesh file.

    Args:
        args: Parsed args with ``file``, ``output``, ``rings``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.focal_std(rings=args.rings, variable_name=args.var)
    mesh.with_variable("focal_std", np.asarray(values), location="face").to_file(
        args.output
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_proximity(args: argparse.Namespace) -> int:
    """Handle ``mesh proximity`` — topological hop distance to target faces.

    Writes a mesh carrying the per-face hop count to the nearest target face
    (``0`` at a target, ``-1`` where unreachable).

    Args:
        args: Parsed args with ``file``, ``output``, ``target_faces``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    distances = mesh.proximity(np.asarray(args.target_faces))
    mesh.with_variable("proximity", np.asarray(distances), location="face").to_file(
        args.output
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_sieve(args: argparse.Namespace) -> int:
    """Handle ``mesh sieve`` — drop value-connected face groups below a size.

    Args:
        args: Parsed args with ``file``, ``output``, ``threshold``, ``var``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.sieve(args.threshold, variable_name=args.var)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Handle ``mesh cluster`` — label connected faces whose values fall in a range.

    Args:
        args: Parsed args with ``file``, ``output``, ``lower``, ``upper``,
            ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.cluster(args.lower, args.upper, variable_name=args.var)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_binary_dilation(args: argparse.Namespace) -> int:
    """Handle ``mesh binary-dilation`` — grow a boolean face mask over neighbours.

    Args:
        args: Parsed args with ``file``, ``output``, ``iterations``,
            ``border_value``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.binary_dilation(
            args.var, iterations=args.iterations, border_value=args.border_value
        )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_binary_erosion(args: argparse.Namespace) -> int:
    """Handle ``mesh binary-erosion`` — shrink a boolean face mask at its border.

    Args:
        args: Parsed args with ``file``, ``output``, ``iterations``,
            ``border_value``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.binary_erosion(
            args.var, iterations=args.iterations, border_value=args.border_value
        )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_zonal_stats(args: argparse.Namespace) -> int:
    """Handle ``mesh zonal-stats`` — per-zone statistics of a mesh variable.

    The zones vector is reprojected onto the mesh CRS when both declare one and
    they differ. Writes a CSV with ``--output``, emits JSON records with
    ``--json``, or prints the records otherwise.

    Args:
        args: Parsed args with ``file``, ``zones``, ``var``, ``stats``,
            ``unweighted``, ``time_index``, ``output``, ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    zones = FeatureCollection.read_file(args.zones)
    if zones.crs is not None and mesh.crs is not None and zones.epsg != mesh.epsg:
        zones = zones.to_crs(mesh.epsg)
    with map_lookup_errors():
        table = mesh.zonal_stats(
            zones,
            variable_name=args.var,
            stats=tuple(args.stats),
            weighted=not args.unweighted,
            time_index=args.time_index,
        )
    _emit_records(args, table.reset_index(drop=True))
    return 0


def _cmd_histogram(args: argparse.Namespace) -> int:
    """Handle ``mesh histogram`` — the variable's value distribution as data.

    Writes a CSV of ``bin_left, bin_right, count`` with ``--output``, emits JSON
    records with ``--json``, or prints the records otherwise.

    Args:
        args: Parsed args with ``file``, ``bins``, ``range``, ``var``,
            ``output``, ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    value_range = None if args.range is None else (args.range[0], args.range[1])
    with map_lookup_errors():
        counts, edges = mesh.get_histogram(
            bins=args.bins, value_range=value_range, variable_name=args.var
        )
    frame = pd.DataFrame(
        {
            "bin_left": np.asarray(edges[:-1]),
            "bin_right": np.asarray(edges[1:]),
            "count": np.asarray(counts),
        }
    )
    _emit_records(args, frame)
    return 0


def _cmd_domain_area(args: argparse.Namespace) -> int:
    """Handle ``mesh domain-area`` — total area of the mesh domain.

    Args:
        args: Parsed args with ``file`` and ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    _print_payload(args, {"area": float(mesh.domain_area())})
    return 0


def _cmd_count_domain_cells(args: argparse.Namespace) -> int:
    """Handle ``mesh count-domain-cells`` — number of active mesh faces.

    Args:
        args: Parsed args with ``file``, ``var``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        count = mesh.count_domain_cells(variable_name=args.var)
    _print_payload(args, {"count": int(count)})
    return 0


def _cmd_footprint(args: argparse.Namespace) -> int:
    """Handle ``mesh footprint`` — the dissolved outline of the valid-value faces.

    Writes the outline polygon as a one-row vector file with ``--output``, else
    prints its WKT, bounds, and area.

    Args:
        args: Parsed args with ``file``, ``var``, ``output``, ``driver``,
            ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        polygon = mesh.footprint(variable_name=args.var)
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
        driver = resolve_vector_driver(args.output, args.driver)
        frame = gpd.GeoDataFrame(geometry=[polygon], crs=mesh.crs)
        FeatureCollection(frame).to_file(args.output, driver=driver)
        info_print(args, f"wrote {args.output}")
    else:
        _print_payload(
            args,
            {
                "wkt": polygon.wkt,
                "bounds": [float(v) for v in polygon.bounds],
                "area": float(polygon.area),
            },
        )
    return 0


def _points_frame(x: list[float], y: list[float], values: np.ndarray) -> pd.DataFrame:
    """Build a ``x, y, value`` table from parallel coordinate and value arrays.

    Args:
        x: Query x-coordinates.
        y: Query y-coordinates.
        values: Sampled values, one per point.

    Returns:
        pd.DataFrame: A table with ``x``, ``y`` and ``value`` columns.
    """
    return pd.DataFrame(
        {"x": np.asarray(x), "y": np.asarray(y), "value": np.asarray(values)}
    )


def _cmd_sample_point(args: argparse.Namespace) -> int:
    """Handle ``mesh sample-point`` — values at arbitrary ``(x, y)`` coordinates.

    Wraps :meth:`MeshDataset.extract`; named ``sample-point`` because the ``mesh``
    group's ``extract`` verb already rasterizes a variable onto a grid.

    Args:
        args: Parsed args with ``file``, ``x``, ``y``, ``var``, ``output``,
            ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.extract(
            np.asarray(args.x), np.asarray(args.y), variable_name=args.var
        )
    _emit_records(args, _points_frame(args.x, args.y, values))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    """Handle ``mesh sample`` — element-resolved values at point locations.

    Args:
        args: Parsed args with ``file``, ``var``, ``x``, ``y``, ``method``,
            ``time_index``, ``output``, ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        values = mesh.sample(
            args.var,
            np.asarray(args.x),
            np.asarray(args.y),
            method=args.method,
            time_index=args.time_index,
        )
    _emit_records(args, _points_frame(args.x, args.y, values))
    return 0


def _cmd_sample_line(args: argparse.Namespace) -> int:
    """Handle ``mesh sample-line`` — one sample per face crossing along a line.

    The line is given as flat ``--coords x0 y0 x1 y1 …`` pairs.

    Args:
        args: Parsed args with ``file``, ``coords``, ``var``, ``output``,
            ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--coords`` is not an even-length list of at least two
            ``(x, y)`` pairs.
    """
    if len(args.coords) < 4 or len(args.coords) % 2 != 0:
        raise ValueError(
            "--coords needs an even number of values (>= 2 x/y pairs), e.g. 0 0 5 5"
        )
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    line = list(zip(args.coords[0::2], args.coords[1::2]))
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        distances, values, face_indices = mesh.sample_line(line, variable_name=args.var)
    frame = pd.DataFrame(
        {
            "distance": np.asarray(distances),
            "value": np.asarray(values),
            "face_index": np.asarray(face_indices),
        }
    )
    _emit_records(args, frame)
    return 0


def _cmd_sample_random(args: argparse.Namespace) -> int:
    """Handle ``mesh sample-random`` — draw ``n`` spatially random samples.

    Args:
        args: Parsed args with ``file``, ``n``, ``var``, ``seed``, ``output``,
            ``json``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        x, y, values = mesh.sample_random(
            args.n, variable_name=args.var, seed=args.seed
        )
    _emit_records(args, _points_frame(list(x), list(y), values))
    return 0


def _cmd_contour(args: argparse.Namespace) -> int:
    """Handle ``mesh contour`` — trace iso-value contour lines into a vector file.

    ``--levels`` is a single integer (a level count) or several explicit levels.

    Args:
        args: Parsed args with ``file``, ``output``, ``levels``, ``var``,
            ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, args.driver)
    levels = None
    if args.levels is not None:
        levels = int(args.levels[0]) if len(args.levels) == 1 else list(args.levels)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        gdf = mesh.contour(levels=levels, variable_name=args.var)
    FeatureCollection(gdf).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _add_var(parser: argparse.ArgumentParser, required: bool = False) -> None:
    """Add the shared ``--var`` option to ``parser``.

    Args:
        parser: The sub-command parser to extend.
        required: Whether the variable must be given.
    """
    parser.add_argument("--var", required=required, help=_HELP_VAR)


def _add_overwrite(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--overwrite`` flag to ``parser``.

    Args:
        parser: The sub-command parser to extend.
    """
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)


def _register_terrain(msub: argparse._SubParsersAction) -> None:
    """Register the terrain / focal / proximity mesh-writing verbs.

    Args:
        msub: The ``mesh`` sub-command action to register on.
    """
    slope = msub.add_parser("slope", help="per-face slope magnitude")
    slope.add_argument("file", help=_HELP_MESH_FILE)
    slope.add_argument("output", help=_HELP_MESH_OUT)
    slope.add_argument(
        "--units",
        choices=["degrees", "radians"],
        default="degrees",
        help="slope units (default: degrees)",
    )
    _add_var(slope)
    _add_overwrite(slope)
    slope.set_defaults(func=_cmd_slope)

    aspect = msub.add_parser("aspect", help="per-face aspect (clockwise from north)")
    aspect.add_argument("file", help=_HELP_MESH_FILE)
    aspect.add_argument("output", help=_HELP_MESH_OUT)
    _add_var(aspect)
    _add_overwrite(aspect)
    aspect.set_defaults(func=_cmd_aspect)

    hill = msub.add_parser("hillshade", help="per-face shaded relief")
    hill.add_argument("file", help=_HELP_MESH_FILE)
    hill.add_argument("output", help=_HELP_MESH_OUT)
    hill.add_argument(
        "--azimuth", type=float, default=315.0, help="light azimuth (default: 315)"
    )
    hill.add_argument(
        "--altitude", type=float, default=45.0, help="light altitude (default: 45)"
    )
    _add_var(hill)
    _add_overwrite(hill)
    hill.set_defaults(func=_cmd_hillshade)

    fmean = msub.add_parser("focal-mean", help="k-ring neighbourhood mean")
    fmean.add_argument("file", help=_HELP_MESH_FILE)
    fmean.add_argument("output", help=_HELP_MESH_OUT)
    fmean.add_argument(
        "--rings",
        type=int,
        default=1,
        help="neighbourhood radius in rings (default: 1)",
    )
    _add_var(fmean)
    _add_overwrite(fmean)
    fmean.set_defaults(func=_cmd_focal_mean)

    fstd = msub.add_parser("focal-std", help="k-ring neighbourhood standard deviation")
    fstd.add_argument("file", help=_HELP_MESH_FILE)
    fstd.add_argument("output", help=_HELP_MESH_OUT)
    fstd.add_argument(
        "--rings",
        type=int,
        default=1,
        help="neighbourhood radius in rings (default: 1)",
    )
    _add_var(fstd)
    _add_overwrite(fstd)
    fstd.set_defaults(func=_cmd_focal_std)

    prox = msub.add_parser("proximity", help="topological hop distance to target faces")
    prox.add_argument("file", help=_HELP_MESH_FILE)
    prox.add_argument("output", help=_HELP_MESH_OUT)
    prox.add_argument(
        "--target-faces",
        dest="target_faces",
        nargs="+",
        type=int,
        required=True,
        help="zero-based face indices to measure distance from",
    )
    _add_overwrite(prox)
    prox.set_defaults(func=_cmd_proximity)


def _register_morphology(msub: argparse._SubParsersAction) -> None:
    """Register the value-grouping and boolean-morphology mesh-writing verbs.

    Args:
        msub: The ``mesh`` sub-command action to register on.
    """
    sieve = msub.add_parser("sieve", help="drop value-connected groups below a size")
    sieve.add_argument("file", help=_HELP_MESH_FILE)
    sieve.add_argument("output", help=_HELP_MESH_OUT)
    sieve.add_argument(
        "--threshold", type=int, required=True, help="minimum group size to keep"
    )
    _add_var(sieve)
    _add_overwrite(sieve)
    sieve.set_defaults(func=_cmd_sieve)

    cluster = msub.add_parser(
        "cluster", help="label connected faces whose values fall in a range"
    )
    cluster.add_argument("file", help=_HELP_MESH_FILE)
    cluster.add_argument("output", help=_HELP_MESH_OUT)
    cluster.add_argument(
        "--lower", type=float, required=True, help="lower bound of the value range"
    )
    cluster.add_argument(
        "--upper", type=float, required=True, help="upper bound of the value range"
    )
    _add_var(cluster)
    _add_overwrite(cluster)
    cluster.set_defaults(func=_cmd_cluster)

    dilate = msub.add_parser(
        "binary-dilation", help="grow a boolean face mask over its neighbours"
    )
    dilate.add_argument("file", help=_HELP_MESH_FILE)
    dilate.add_argument("output", help=_HELP_MESH_OUT)
    dilate.add_argument(
        "--iterations", type=int, default=1, help="how many times to grow (default: 1)"
    )
    dilate.add_argument(
        "--border-value",
        dest="border_value",
        action="store_true",
        help="treat the region outside the mesh as True",
    )
    _add_var(dilate)
    _add_overwrite(dilate)
    dilate.set_defaults(func=_cmd_binary_dilation)

    erode = msub.add_parser(
        "binary-erosion", help="shrink a boolean face mask at its border"
    )
    erode.add_argument("file", help=_HELP_MESH_FILE)
    erode.add_argument("output", help=_HELP_MESH_OUT)
    erode.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="how many times to shrink (default: 1)",
    )
    erode.add_argument(
        "--border-value",
        dest="border_value",
        action="store_true",
        help="treat the region outside the mesh as True",
    )
    _add_var(erode)
    _add_overwrite(erode)
    erode.set_defaults(func=_cmd_binary_erosion)


def _register_summaries(msub: argparse._SubParsersAction) -> None:
    """Register the printing / vector-summary verbs.

    Args:
        msub: The ``mesh`` sub-command action to register on.
    """
    zonal = msub.add_parser(
        "zonal-stats", help="per-zone statistics of a mesh variable"
    )
    zonal.add_argument("file", help=_HELP_MESH_FILE)
    zonal.add_argument("zones", help="polygon vector file defining the zones")
    _add_var(zonal, required=True)
    zonal.add_argument(
        "--stats",
        nargs="+",
        default=["mean"],
        help="stats to compute: mean sum min max std … (default: mean)",
    )
    zonal.add_argument(
        "--unweighted",
        action="store_true",
        help="weight each face equally instead of by its area",
    )
    zonal.add_argument(
        "--time-index",
        dest="time_index",
        type=int,
        default=0,
        help="time step to summarise (default: 0)",
    )
    zonal.add_argument("--output", help="write the table to this CSV path")
    zonal.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(zonal)
    zonal.set_defaults(func=_cmd_zonal_stats)

    hist = msub.add_parser(
        "histogram", help="the variable's value distribution as data"
    )
    hist.add_argument("file", help=_HELP_MESH_FILE)
    hist.add_argument(
        "--bins",
        type=int,
        default=256,
        help="number of equal-width bins (default: 256)",
    )
    hist.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="value range to bin within (default: the data range)",
    )
    _add_var(hist)
    hist.add_argument(
        "--output", help="write bin_left/bin_right/count to this CSV path"
    )
    hist.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(hist)
    hist.set_defaults(func=_cmd_histogram)

    area = msub.add_parser("domain-area", help="total area of the mesh domain")
    area.add_argument("file", help=_HELP_MESH_FILE)
    area.add_argument("--json", action="store_true", help=_HELP_JSON)
    area.set_defaults(func=_cmd_domain_area)

    count = msub.add_parser("count-domain-cells", help="number of active mesh faces")
    count.add_argument("file", help=_HELP_MESH_FILE)
    _add_var(count)
    count.add_argument("--json", action="store_true", help=_HELP_JSON)
    count.set_defaults(func=_cmd_count_domain_cells)

    footprint = msub.add_parser(
        "footprint", help="dissolved outline of the valid-value faces"
    )
    footprint.add_argument("file", help=_HELP_MESH_FILE)
    _add_var(footprint)
    footprint.add_argument(
        "--output", help="write the outline polygon to this vector path"
    )
    footprint.add_argument("--driver", help=_HELP_DRIVER)
    footprint.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(footprint)
    footprint.set_defaults(func=_cmd_footprint)


def _register_sampling(msub: argparse._SubParsersAction) -> None:
    """Register the point / line / random sampling and contour verbs.

    Args:
        msub: The ``mesh`` sub-command action to register on.
    """
    sample_point = msub.add_parser(
        "sample-point", help="variable values at arbitrary x/y coordinates"
    )
    sample_point.add_argument("file", help=_HELP_MESH_FILE)
    sample_point.add_argument(
        "--x", nargs="+", type=float, required=True, help="query x-coordinate(s)"
    )
    sample_point.add_argument(
        "--y", nargs="+", type=float, required=True, help="query y-coordinate(s)"
    )
    _add_var(sample_point)
    sample_point.add_argument("--output", help=_HELP_CSV_OUT)
    sample_point.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(sample_point)
    sample_point.set_defaults(func=_cmd_sample_point)

    sample = msub.add_parser(
        "sample", help="element-resolved values at point locations"
    )
    sample.add_argument("file", help=_HELP_MESH_FILE)
    _add_var(sample, required=True)
    sample.add_argument(
        "--x", nargs="+", type=float, required=True, help="query x-coordinate(s)"
    )
    sample.add_argument(
        "--y", nargs="+", type=float, required=True, help="query y-coordinate(s)"
    )
    sample.add_argument(
        "--method",
        choices=["contains", "nearest"],
        default="contains",
        help="how a point resolves to an element (default: contains)",
    )
    sample.add_argument(
        "--time-index",
        dest="time_index",
        type=int,
        default=0,
        help="time step to sample (default: 0)",
    )
    sample.add_argument("--output", help=_HELP_CSV_OUT)
    sample.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(sample)
    sample.set_defaults(func=_cmd_sample)

    sample_line = msub.add_parser(
        "sample-line", help="one sample per face crossing along a line"
    )
    sample_line.add_argument("file", help=_HELP_MESH_FILE)
    sample_line.add_argument(
        "--coords",
        nargs="+",
        type=float,
        required=True,
        help="flat x/y pairs defining the line, e.g. 0 0 5 5",
    )
    _add_var(sample_line)
    sample_line.add_argument(
        "--output", help="write distance/value/face_index to this CSV path"
    )
    sample_line.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(sample_line)
    sample_line.set_defaults(func=_cmd_sample_line)

    sample_random = msub.add_parser(
        "sample-random", help="draw spatially random samples from the domain"
    )
    sample_random.add_argument("file", help=_HELP_MESH_FILE)
    sample_random.add_argument(
        "-n", "--n", type=int, required=True, help="number of samples"
    )
    _add_var(sample_random)
    sample_random.add_argument("--seed", type=int, help="RNG seed for reproducibility")
    sample_random.add_argument("--output", help=_HELP_CSV_OUT)
    sample_random.add_argument("--json", action="store_true", help=_HELP_JSON)
    _add_overwrite(sample_random)
    sample_random.set_defaults(func=_cmd_sample_random)

    contour = msub.add_parser(
        "contour", help="trace iso-value contour lines to a vector"
    )
    contour.add_argument("file", help=_HELP_MESH_FILE)
    contour.add_argument("output", help=_HELP_VECTOR_OUT)
    contour.add_argument(
        "--levels",
        nargs="+",
        type=float,
        help="a single level count, or several explicit levels (default: 10)",
    )
    _add_var(contour)
    contour.add_argument("--driver", help=_HELP_DRIVER)
    _add_overwrite(contour)
    contour.set_defaults(func=_cmd_contour)


def _cmd_polyfit(args: argparse.Namespace) -> int:
    """Handle ``pyramids mesh polyfit`` — per-element polynomial trend along time.

    Fits a degree-``d`` polynomial to each element's time series and writes the coefficients
    (highest power first) as a mesh file, the time axis replaced by a ``coefficient`` axis.

    Args:
        args: Parsed args with ``file``, ``output``, ``degree``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    mesh = MeshDataset.read_file(args.file)
    with map_lookup_errors():
        result = mesh.polyfit(args.degree, variable_name=args.var)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _register_fitting(msub: argparse._SubParsersAction) -> None:
    """Register the along-time fitting verbs on the ``mesh`` sub-command action.

    Args:
        msub: The ``mesh`` group's ``add_subparsers`` action.
    """
    pf = msub.add_parser(
        "polyfit", help="per-element polynomial trend along time (writes coefficients)"
    )
    pf.add_argument("file", help=_HELP_MESH_FILE)
    pf.add_argument("output", help=_HELP_MESH_OUT)
    pf.add_argument(
        "--degree", "-d", type=int, required=True, help="polynomial degree (>= 0)"
    )
    _add_var(pf)
    _add_overwrite(pf)
    pf.set_defaults(func=_cmd_polyfit)


def add_ugrid_analysis_commands(msub: argparse._SubParsersAction) -> None:
    """Register the mesh analysis / terrain verbs on the ``mesh`` sub-command action.

    Args:
        msub: The ``add_subparsers`` action of the ``mesh`` group parser, onto
            which each analysis verb is registered.
    """
    _register_terrain(msub)
    _register_morphology(msub)
    _register_summaries(msub)
    _register_sampling(msub)
    _register_fitting(msub)
