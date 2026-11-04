"""Tessellation, binning, sampling and measurement commands for ``pyramids vector``.

A second slice of the ``pyramids vector`` group, registered by
:func:`add_vector_tess_commands`. Two families live here, both thin wrappers over
:class:`pyramids.feature.FeatureCollection`:

* **Tessellation / binning / sampling** — ``voronoi-polygons``, ``delaunay-triangles``,
  ``constrained-delaunay-triangles``, ``to-h3``, ``h3-bin``, ``sample-points``,
  ``polygonize`` and the ``fishnet`` classmethod. Each produces a new vector layer,
  writes it under the overwrite guard, and re-reads cleanly. The geopandas-delegated
  tessellators return a bare ``GeoSeries``; it is wrapped back into a
  ``FeatureCollection`` before writing.
* **Measurement** — ``geodesic-area``/``-length``/``-distance``, ``distance``,
  ``hausdorff-distance``, ``frechet-distance``, ``minimum-clearance``,
  ``hilbert-distance`` and ``get-coordinates``. These return a ``pandas`` Series or
  frame; the command prints a one-line summary, or the raw values as JSON with
  ``--json`` (``get-coordinates`` writes a CSV instead).

No geometry logic lives here — each command maps 1:1 onto an existing
``FeatureCollection`` call, with the same one-line-error contract as the rest of the
CLI (handlers raise; :func:`pyramids.cli.main` reports).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import geopandas as gpd

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    map_lookup_errors,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.feature import FeatureCollection

_HELP_SRC_VECTOR = "source vector path"
_HELP_DST_VECTOR = "destination vector path"
_HELP_LAYER = "layer name/index for multi-layer sources"
_HELP_OVERWRITE = "replace the output if it exists"
_HELP_JSON = "emit the raw per-feature values as JSON"
_HELP_RESOLUTION = "H3 resolution, 0 (coarsest) to 15 (finest)"
_HELP_TOLERANCE = "snap input coordinates to this grid before tessellating"
_HELP_ONLY_EDGES = "return the tessellation edges as lines instead of polygons"
_HELP_OTHER = "second vector file, compared element-wise against the source"


def _as_collection(result: Any) -> FeatureCollection:
    """Return ``result`` as a ``FeatureCollection``, wrapping a bare ``GeoSeries``.

    The pyramids tessellators (``to_h3``/``h3_bin``/``fishnet``) already return a
    ``FeatureCollection``; the geopandas-delegated ones (``voronoi_polygons`` …)
    return a ``GeoSeries`` that has to be wrapped before it can be written.

    Args:
        result: A ``FeatureCollection`` or a ``geopandas`` ``GeoSeries``.

    Returns:
        FeatureCollection: ``result`` unchanged, or a collection built around it.
    """
    if isinstance(result, FeatureCollection):
        return result
    return FeatureCollection(gpd.GeoDataFrame(geometry=result))


def _write_collection(args: argparse.Namespace, result: Any) -> None:
    """Write a tessellation result to ``args.dst`` and print a ``wrote …`` line.

    Args:
        args: Parsed args carrying ``dst`` and the global ``quiet`` flag.
        result: A ``FeatureCollection`` or ``GeoSeries`` to persist.
    """
    driver = resolve_vector_driver(args.dst, None)
    _as_collection(result).to_file(args.dst, driver=driver)
    info_print(args, f"wrote {args.dst}")


def _series_summary(series: Any, label: str) -> str:
    """Build a one-line summary of a measurement Series.

    Args:
        series: A ``pandas`` Series of per-feature values.
        label: The measurement name used as the line's leading token.

    Returns:
        str: ``"<label>: count=… min=… max=… mean=… sum=…"``.
    """
    return (
        f"{label}: count={int(series.count())} min={series.min()} "
        f"max={series.max()} mean={series.mean()} sum={series.sum()}"
    )


def _emit_series(args: argparse.Namespace, series: Any, label: str) -> None:
    """Print a measurement Series as a summary line, or as a JSON value list.

    Args:
        args: Parsed args carrying the ``json`` flag.
        series: A ``pandas`` Series of per-feature values.
        label: The measurement name used in the plain summary.
    """
    if args.json:
        print(json.dumps(json_sanitize(series.tolist())))
    else:
        print(_series_summary(series, label))


def _cmd_vector_voronoi_polygons(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector voronoi-polygons`` — Voronoi cells of the inputs.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``tolerance``,
            ``only_edges``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.voronoi_polygons(tolerance=args.tolerance, only_edges=args.only_edges)
    _write_collection(args, result)
    return 0


def _cmd_vector_delaunay_triangles(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector delaunay-triangles`` — Delaunay triangulation.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``tolerance``,
            ``only_edges``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.delaunay_triangles(tolerance=args.tolerance, only_edges=args.only_edges)
    _write_collection(args, result)
    return 0


def _cmd_vector_constrained_delaunay_triangles(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector constrained-delaunay-triangles``.

    The constrained Delaunay triangulation of each polygon, one geometry collection
    of triangles per input feature.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.constrained_delaunay_triangles()
    _write_collection(args, result)
    return 0


def _cmd_vector_to_h3(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector to-h3`` — attach each point's H3 cell index.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``resolution``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.to_h3(args.resolution)
    _write_collection(args, result)
    return 0


def _cmd_vector_h3_bin(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector h3-bin`` — aggregate points into H3 hexagon cells.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``resolution``, ``agg``,
            ``column``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    with map_lookup_errors():
        result = fc.h3_bin(args.resolution, agg=args.agg, column=args.column)
    _write_collection(args, result)
    return 0


def _cmd_vector_sample_points(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector sample-points`` — random points within each geometry.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``size``, ``method``,
            ``seed``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    result = fc.sample_points(size=args.size, method=args.method, rng=args.seed)
    _write_collection(args, result)
    return 0


def _cmd_vector_polygonize(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector polygonize`` — build polygons from noded line work.

    With ``--full`` the leftover categories (cut edges, dangles, invalid ring lines)
    are counted in the ``wrote …`` line; only the polygons are written either way.

    Args:
        args: Parsed args with ``src``, ``dst``, ``layer``, ``full``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    extra = ""
    if args.full:
        polygons, cuts, dangles, invalid = fc.polygonize(full=True)
        extra = f" (cuts={len(cuts)}, dangles={len(dangles)}, invalid={len(invalid)})"
    else:
        polygons = fc.polygonize()
    driver = resolve_vector_driver(args.dst, None)
    _as_collection(polygons).to_file(args.dst, driver=driver)
    info_print(args, f"wrote {args.dst}{extra}")
    return 0


def _cmd_vector_fishnet(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector fishnet`` — a grid of square cell polygons.

    A classmethod wrapper: the grid is built from ``--bounds`` and ``--cell-size``
    directly, so this command takes no source layer.

    Args:
        args: Parsed args with ``dst``, ``bounds``, ``cell_size``, ``epsg``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.dst, args.overwrite)
    grid = FeatureCollection.fishnet(
        tuple(args.bounds), args.cell_size, crs=f"EPSG:{args.epsg}"
    )
    _write_collection(args, grid)
    return 0


def _cmd_vector_geodesic_area(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector geodesic-area`` — ellipsoidal area per feature.

    Args:
        args: Parsed args with ``src``, ``layer``, ``unit``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    _emit_series(args, fc.geodesic_area(unit=args.unit), "geodesic_area")
    return 0


def _cmd_vector_geodesic_length(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector geodesic-length`` — ellipsoidal length per feature.

    Args:
        args: Parsed args with ``src``, ``unit``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    _emit_series(args, fc.geodesic_length(unit=args.unit), "geodesic_length")
    return 0


def _cmd_vector_geodesic_distance(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector geodesic-distance`` — ground distance to ``--other``.

    Args:
        args: Parsed args with ``src``, ``other``, ``unit``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    other = FeatureCollection.read_file(args.other)
    series = fc.geodesic_distance(other, unit=args.unit)
    _emit_series(args, series, "geodesic_distance")
    return 0


def _cmd_vector_distance(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector distance`` — planar distance to ``--other``.

    Args:
        args: Parsed args with ``src``, ``other``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    other = FeatureCollection.read_file(args.other)
    _emit_series(args, fc.distance(other), "distance")
    return 0


def _cmd_vector_hausdorff_distance(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector hausdorff-distance`` — Hausdorff distance to ``--other``.

    Args:
        args: Parsed args with ``src``, ``other``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    other = FeatureCollection.read_file(args.other)
    _emit_series(args, fc.hausdorff_distance(other), "hausdorff_distance")
    return 0


def _cmd_vector_frechet_distance(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector frechet-distance`` — Frechet distance to ``--other``.

    Args:
        args: Parsed args with ``src``, ``other``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    other = FeatureCollection.read_file(args.other)
    _emit_series(args, fc.frechet_distance(other), "frechet_distance")
    return 0


def _cmd_vector_minimum_clearance(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector minimum-clearance`` — minimum clearance per feature.

    Args:
        args: Parsed args with ``src``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    _emit_series(args, fc.minimum_clearance(), "minimum_clearance")
    return 0


def _cmd_vector_hilbert_distance(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector hilbert-distance`` — Hilbert-curve index per feature.

    Args:
        args: Parsed args with ``src``, ``level``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    _emit_series(args, fc.hilbert_distance(level=args.level), "hilbert_distance")
    return 0


def _cmd_vector_get_coordinates(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector get-coordinates`` — the vertex coordinate table.

    Writes the ``x``/``y`` table to ``dst`` as CSV, or prints it as JSON with
    ``--json`` (in which case ``dst`` is not touched).

    Args:
        args: Parsed args with ``src``, ``dst``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src)
    frame = fc.get_coordinates()
    if args.json:
        print(json.dumps(json_sanitize(frame.to_dict(orient="list"))))
    else:
        refuse_existing(args.dst, overwrite=False)
        frame.to_csv(args.dst, index=False)
        info_print(args, f"wrote {args.dst}")
    return 0


def _add_write_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``src``/``dst``/``--layer``/``--overwrite`` arguments.

    Args:
        parser: The tessellation sub-command parser to extend.
    """
    parser.add_argument("src", help=_HELP_SRC_VECTOR)
    parser.add_argument("dst", help=_HELP_DST_VECTOR)
    parser.add_argument("--layer", help=_HELP_LAYER)
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)


def _add_tessellation_commands(vsub: argparse._SubParsersAction) -> None:
    """Register the tessellation / binning / sampling commands.

    Args:
        vsub: The ``vector`` group's ``add_subparsers`` action.
    """
    v_vor = vsub.add_parser("voronoi-polygons", help="Voronoi cells of the inputs")
    _add_write_args(v_vor)
    v_vor.add_argument("--tolerance", type=float, default=0.0, help=_HELP_TOLERANCE)
    v_vor.add_argument("--only-edges", action="store_true", help=_HELP_ONLY_EDGES)
    v_vor.set_defaults(func=_cmd_vector_voronoi_polygons)

    v_del = vsub.add_parser("delaunay-triangles", help="Delaunay triangulation")
    _add_write_args(v_del)
    v_del.add_argument("--tolerance", type=float, default=0.0, help=_HELP_TOLERANCE)
    v_del.add_argument("--only-edges", action="store_true", help=_HELP_ONLY_EDGES)
    v_del.set_defaults(func=_cmd_vector_delaunay_triangles)

    v_cdt = vsub.add_parser(
        "constrained-delaunay-triangles",
        help="constrained Delaunay triangulation of each polygon",
    )
    _add_write_args(v_cdt)
    v_cdt.set_defaults(func=_cmd_vector_constrained_delaunay_triangles)

    v_h3 = vsub.add_parser("to-h3", help="attach each point's H3 cell index")
    _add_write_args(v_h3)
    v_h3.add_argument("--resolution", type=int, required=True, help=_HELP_RESOLUTION)
    v_h3.set_defaults(func=_cmd_vector_to_h3)

    v_bin = vsub.add_parser("h3-bin", help="aggregate points into H3 hexagon cells")
    _add_write_args(v_bin)
    v_bin.add_argument("--resolution", type=int, required=True, help=_HELP_RESOLUTION)
    v_bin.add_argument(
        "--agg", default="count", help="per-cell reducer applied to --column"
    )
    v_bin.add_argument("--column", help="numeric column to aggregate (default: count)")
    v_bin.set_defaults(func=_cmd_vector_h3_bin)

    v_samp = vsub.add_parser("sample-points", help="random points within each geometry")
    _add_write_args(v_samp)
    v_samp.add_argument("--size", type=int, required=True, help="points per geometry")
    v_samp.add_argument("--method", default="uniform", help="sampling method")
    v_samp.add_argument("--seed", type=int, help="random seed for reproducibility")
    v_samp.set_defaults(func=_cmd_vector_sample_points)

    v_poly = vsub.add_parser("polygonize", help="build polygons from noded line work")
    _add_write_args(v_poly)
    v_poly.add_argument(
        "--full", action="store_true", help="also report cut/dangle/invalid counts"
    )
    v_poly.set_defaults(func=_cmd_vector_polygonize)

    v_fish = vsub.add_parser("fishnet", help="a grid of square cell polygons")
    v_fish.add_argument("dst", help=_HELP_DST_VECTOR)
    v_fish.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="extent the grid covers, in the units of --epsg",
    )
    v_fish.add_argument(
        "--cell-size", type=float, required=True, help="side length of each square cell"
    )
    v_fish.add_argument(
        "--epsg", type=int, required=True, help="EPSG code for the grid"
    )
    v_fish.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_fish.set_defaults(func=_cmd_vector_fishnet)


def _add_measurement_commands(vsub: argparse._SubParsersAction) -> None:
    """Register the geodesic and geometry-measurement commands.

    Args:
        vsub: The ``vector`` group's ``add_subparsers`` action.
    """
    v_area = vsub.add_parser("geodesic-area", help="ellipsoidal area per feature")
    v_area.add_argument("src", help=_HELP_SRC_VECTOR)
    v_area.add_argument("--layer", help=_HELP_LAYER)
    v_area.add_argument("--unit", default="m2", help="m2 (default), km2 or ha")
    v_area.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_area.set_defaults(func=_cmd_vector_geodesic_area)

    v_len = vsub.add_parser("geodesic-length", help="ellipsoidal length per feature")
    v_len.add_argument("src", help=_HELP_SRC_VECTOR)
    v_len.add_argument("--unit", default="m", help="m (default), km, mi or nmi")
    v_len.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_len.set_defaults(func=_cmd_vector_geodesic_length)

    v_gdist = vsub.add_parser("geodesic-distance", help="ground distance to --other")
    v_gdist.add_argument("src", help=_HELP_SRC_VECTOR)
    v_gdist.add_argument("--other", required=True, help=_HELP_OTHER)
    v_gdist.add_argument("--unit", default="m", help="m (default), km, mi or nmi")
    v_gdist.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_gdist.set_defaults(func=_cmd_vector_geodesic_distance)

    v_dist = vsub.add_parser("distance", help="planar distance to --other")
    v_dist.add_argument("src", help=_HELP_SRC_VECTOR)
    v_dist.add_argument("--other", required=True, help=_HELP_OTHER)
    v_dist.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_dist.set_defaults(func=_cmd_vector_distance)

    v_haus = vsub.add_parser("hausdorff-distance", help="Hausdorff distance to --other")
    v_haus.add_argument("src", help=_HELP_SRC_VECTOR)
    v_haus.add_argument("--other", required=True, help=_HELP_OTHER)
    v_haus.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_haus.set_defaults(func=_cmd_vector_hausdorff_distance)

    v_fre = vsub.add_parser("frechet-distance", help="Frechet distance to --other")
    v_fre.add_argument("src", help=_HELP_SRC_VECTOR)
    v_fre.add_argument("--other", required=True, help=_HELP_OTHER)
    v_fre.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_fre.set_defaults(func=_cmd_vector_frechet_distance)

    v_clear = vsub.add_parser("minimum-clearance", help="minimum clearance per feature")
    v_clear.add_argument("src", help=_HELP_SRC_VECTOR)
    v_clear.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_clear.set_defaults(func=_cmd_vector_minimum_clearance)

    v_hil = vsub.add_parser("hilbert-distance", help="Hilbert-curve index per feature")
    v_hil.add_argument("src", help=_HELP_SRC_VECTOR)
    v_hil.add_argument("--level", type=int, default=16, help="Hilbert-curve level")
    v_hil.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_hil.set_defaults(func=_cmd_vector_hilbert_distance)

    v_coords = vsub.add_parser("get-coordinates", help="the vertex coordinate table")
    v_coords.add_argument("src", help=_HELP_SRC_VECTOR)
    v_coords.add_argument("dst", help="destination CSV path")
    v_coords.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_coords.set_defaults(func=_cmd_vector_get_coordinates)


def add_vector_tess_commands(vsub: argparse._SubParsersAction) -> None:
    """Register the tessellation, binning, sampling and measurement commands.

    Args:
        vsub: The ``vector`` group's ``add_subparsers`` action, shared with
            :func:`pyramids.cli_vector.add_vector_commands`.
    """
    _add_tessellation_commands(vsub)
    _add_measurement_commands(vsub)
