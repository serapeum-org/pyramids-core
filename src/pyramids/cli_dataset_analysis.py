"""Raster (`Dataset`) analysis CLI command group.

Holds the single-shot raster *analysis* commands that sit alongside the core
raster verbs in :mod:`pyramids.cli_dataset`: ``proximity``, ``cluster``,
``zonal-stats``, ``match-histogram``, ``percentile-stretch``, ``pan-sharpen``,
``fillna``, ``where``, ``convert-units`` and ``contour``. Each handler is a thin
1:1 wrapper over an existing :class:`~pyramids.dataset.Dataset` method; the
top-level parser only wires them together.

Deferred:
    - ``cluster2`` — a deprecated alias for :meth:`Dataset.to_polygons`; a new
      CLI surface should not expose a method already slated for removal (and the
      polygonize verb belongs to the vector group, not this one).
    - ``fill_gaps`` — a low-level helper used internally by ``crop``: it takes a
      raw NumPy ``src_array`` plus a mask and returns an ndarray, not a
      file-in / file-out one-shot verb. User-facing gap filling is reachable via
      ``fillna`` (constant fill) or the ``apply fill`` command (neighbour fill).
"""

from __future__ import annotations

import argparse
import ast
import json

import numpy as np

from pyramids._cli_shared import info_print, json_sanitize, refuse_existing
from pyramids._cli_shared import resolve_vector_driver as _resolve_vector_driver
from pyramids.base.crs import crs_spec
from pyramids.base.georeference import GeoReference
from pyramids.cli_dataset import _safe_calc_eval
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

_HELP_SRC_RASTER = "source raster path"
_HELP_DST_RASTER = "destination raster path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_BAND = "zero-based band index (default: 0)"
_HELP_JSON = "emit JSON"


def _georef_like(ds: Dataset) -> GeoReference:
    """Build a :class:`GeoReference` carrying ``ds``'s full geotransform and CRS.

    Uses the whole geotransform (not ``top_left_corner`` + ``cell_size``) so a
    rotation/skew or anisotropic cell size survives, and ``crs_spec`` so a
    CRS-less source yields an ungeoreferenced result rather than an opaque empty
    projection string.

    Args:
        ds: The source dataset to copy georeferencing from.

    Returns:
        GeoReference: A georeference matching ``ds``.
    """
    return GeoReference(geo=ds.geotransform, epsg=crs_spec(ds.epsg, ds.crs))


def _cmd_proximity(args: argparse.Namespace) -> int:
    """Handle `pyramids proximity` — distance-to-target raster.

    Args:
        args: Parsed args with `input`, `output`, `band`, `target_values`,
            `distance_units`, `max_distance`, `nodata`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    out = ds.proximity(
        band=args.band,
        target_values=args.target_values,
        distance_units=args.distance_units,
        max_distance=args.max_distance,
        nodata=args.nodata,
    )
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Handle `pyramids cluster` — label connected cells within a value range.

    Writes the per-cell cluster-label array as a raster sharing the source grid,
    and reports the cluster count returned by :meth:`Dataset.cluster`.

    Args:
        args: Parsed args with `input`, `output`, `lower`, `upper`, `json`,
            `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    labels, count, _position, _values = ds.cluster(args.lower, args.upper)
    Dataset.from_array(np.asarray(labels), geo_ref=_georef_like(ds)).to_file(
        args.output
    )
    if args.json:
        print(json.dumps({"output": args.output, "count": int(count)}))
    else:
        info_print(args, f"wrote {args.output} ({int(count)} cluster label(s))")
    return 0


def _cmd_zonal_stats(args: argparse.Namespace) -> int:
    """Handle `pyramids zonal-stats` — statistics per polygon zone.

    Writes a CSV when `--output` is given, emits JSON records with `--json`, or
    prints the table otherwise. The zones vector is reprojected onto the raster's
    CRS when both declare one and they differ.

    Args:
        args: Parsed args with `raster`, `zones`, `stats`, `band`, `output`,
            `json`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    if args.output is not None:
        refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.raster)
    zones = FeatureCollection.read_file(args.zones)
    if zones.crs is not None and ds.crs and zones.epsg != ds.epsg:
        zones = zones.to_crs(ds.epsg)
    table = ds.zonal_stats(zones, stats=tuple(args.stats), band=args.band)
    if args.output is not None:
        table.to_csv(args.output)
        info_print(args, f"wrote {args.output}")
        return 0
    records = json_sanitize(table.reset_index().to_dict(orient="records"))
    if args.json:
        print(json.dumps(records))
    else:
        for record in records:
            print(record)
    return 0


def _cmd_match_histogram(args: argparse.Namespace) -> int:
    """Handle `pyramids match-histogram` — match a band onto a reference's distribution.

    Args:
        args: Parsed args with `input`, `reference`, `output`, `band`,
            `reference_band`, `bins`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    reference = Dataset.read_file(args.reference)
    out = ds.match_histogram(
        reference,
        band=args.band,
        reference_band=args.reference_band,
        bins=args.bins,
    )
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_percentile_stretch(args: argparse.Namespace) -> int:
    """Handle `pyramids percentile-stretch` — stretch a band between two percentiles.

    Args:
        args: Parsed args with `input`, `output`, `lower`, `upper`, `band`,
            `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    out = ds.percentile_stretch(args.lower, args.upper, band=args.band)
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_pan_sharpen(args: argparse.Namespace) -> int:
    """Handle `pyramids pan-sharpen` — fuse a multispectral raster with a pan band.

    Args:
        args: Parsed args with `multispectral`, `pan`, `output`, `algorithm`,
            `weights`, `resampling`, `bands`, `nodata`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ms = Dataset.read_file(args.multispectral)
    pan = Dataset.read_file(args.pan)
    out = ms.pan_sharpen(
        pan,
        algorithm=args.algorithm,
        weights=args.weights,
        resampling=args.resampling,
        bands=args.bands,
        nodata=args.nodata,
    )
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_fillna(args: argparse.Namespace) -> int:
    """Handle `pyramids fillna` — give every gap a constant value.

    Args:
        args: Parsed args with `input`, `output`, `value`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    ds.fillna(args.value).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _parse_other(raw: str | None) -> tuple[bool, float | None]:
    """Parse a `where --other` token into (given, value).

    Accepts a number, ``nan``, or ``none`` (an explicit NaN whatever the raster
    declares, per :meth:`Dataset.where`). ``None`` means the flag was not passed,
    so the method's own default (the raster's no-data value) is used.

    Args:
        raw: The raw ``--other`` string, or ``None`` when not passed.

    Returns:
        tuple[bool, float | None]: ``(given, value)`` — ``given`` is ``False``
        when the flag was absent; otherwise ``value`` is the parsed fill.

    Raises:
        ValueError: ``raw`` is not a number, ``nan``, nor ``none``.
    """
    if raw is None:
        return False, None
    spec = raw.strip().lower()
    if spec == "none":
        return True, None
    if spec == "nan":
        return True, float("nan")
    try:
        return True, float(raw)
    except ValueError:
        raise ValueError(f"--other {raw!r} is not a number, 'nan', or 'none'") from None


def _cmd_where(args: argparse.Namespace) -> int:
    """Handle `pyramids where` — keep the cells a condition selects, mask the rest.

    The condition is an expression over the input raster bound to ``A`` (e.g.
    ``'A > 5'``), evaluated by the same AST whitelist `calc` uses — never
    ``eval``. Unselected cells take ``--other`` (the raster's no-data value by
    default).

    Args:
        args: Parsed args with `input`, `output`, `cond`, `other`, `drop`,
            `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    values = np.asarray(ds.read_array(squeeze=True))
    mask = np.asarray(_safe_calc_eval(ast.parse(args.cond, mode="eval"), {"A": values}))
    given, other = _parse_other(args.other)
    if given:
        out = ds.where(mask, other, drop=args.drop)
    else:
        out = ds.where(mask, drop=args.drop)
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_convert_units(args: argparse.Namespace) -> int:
    """Handle `pyramids convert-units` — transform band values to another unit.

    Args:
        args: Parsed args with `input`, `output`, `target`, `band`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    ds.convert_units(args.target, band=args.band).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_contour(args: argparse.Namespace) -> int:
    """Handle `pyramids contour` — trace iso-value contours into a vector file.

    Emits contour lines at each level, or filled polygons between levels with
    `--polygonize`. Exactly one of `--interval` / `--fixed-levels` is required.

    Args:
        args: Parsed args with `input`, `output`, `interval`, `fixed_levels`,
            `base`, `band`, `attribute`, `polygonize`, `driver`, `overwrite`.

    Returns:
        int: `0` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = _resolve_vector_driver(args.output, args.driver)
    ds = Dataset.read_file(args.input)
    contours = ds.contour(
        interval=args.interval,
        fixed_levels=args.fixed_levels,
        base=args.base,
        band=args.band,
        attribute=args.attribute,
        polygonize=args.polygonize,
    )
    contours.to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def add_dataset_analysis_commands(sub: argparse._SubParsersAction) -> None:
    """Register the raster-analysis commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    proximity = sub.add_parser("proximity", help="distance-to-target raster")
    proximity.add_argument("input", help=_HELP_SRC_RASTER)
    proximity.add_argument("output", help=_HELP_DST_RASTER)
    proximity.add_argument("--band", type=int, default=0, help=_HELP_BAND)
    proximity.add_argument(
        "--target-values",
        dest="target_values",
        nargs="+",
        type=int,
        help="pixel values that count as targets (default: every non-zero pixel)",
    )
    proximity.add_argument(
        "--distance-units",
        dest="distance_units",
        choices=["GEO", "PIXEL"],
        default="GEO",
        help="measure distance in georeferenced units (GEO) or pixels (PIXEL)",
    )
    proximity.add_argument(
        "--max-distance",
        dest="max_distance",
        type=float,
        help="stop searching beyond this distance",
    )
    proximity.add_argument(
        "--nodata",
        type=float,
        help="no-data value, also filling pixels beyond --max-distance",
    )
    proximity.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    proximity.set_defaults(func=_cmd_proximity)

    cluster = sub.add_parser(
        "cluster", help="label connected cells within a value range"
    )
    cluster.add_argument("input", help=_HELP_SRC_RASTER)
    cluster.add_argument("output", help="destination label raster path")
    cluster.add_argument(
        "--lower", type=float, required=True, help="lower bound of the value range"
    )
    cluster.add_argument(
        "--upper", type=float, required=True, help="upper bound of the value range"
    )
    cluster.add_argument("--json", action="store_true", help=_HELP_JSON)
    cluster.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    cluster.set_defaults(func=_cmd_cluster)

    zonal = sub.add_parser("zonal-stats", help="statistics per polygon zone")
    zonal.add_argument("raster", help=_HELP_SRC_RASTER)
    zonal.add_argument("zones", help="polygon vector file defining the zones")
    zonal.add_argument(
        "--stats",
        nargs="+",
        default=["mean"],
        help="stats to compute: mean sum min max std var count (default: mean)",
    )
    zonal.add_argument("--band", type=int, default=0, help=_HELP_BAND)
    zonal.add_argument("--output", help="write the table to this CSV path")
    zonal.add_argument("--json", action="store_true", help=_HELP_JSON)
    zonal.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    zonal.set_defaults(func=_cmd_zonal_stats)

    match = sub.add_parser(
        "match-histogram", help="match a band onto a reference's distribution"
    )
    match.add_argument("input", help=_HELP_SRC_RASTER)
    match.add_argument("reference", help="reference raster whose distribution to match")
    match.add_argument("output", help=_HELP_DST_RASTER)
    match.add_argument("--band", type=int, default=0, help=_HELP_BAND)
    match.add_argument(
        "--reference-band",
        dest="reference_band",
        type=int,
        help="zero-based band of the reference (default: same as --band)",
    )
    match.add_argument(
        "--bins", type=int, default=256, help="quantile levels sampled (default: 256)"
    )
    match.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    match.set_defaults(func=_cmd_match_histogram)

    stretch = sub.add_parser(
        "percentile-stretch", help="stretch a band between two percentiles into [0, 1]"
    )
    stretch.add_argument("input", help=_HELP_SRC_RASTER)
    stretch.add_argument("output", help=_HELP_DST_RASTER)
    stretch.add_argument(
        "--lower",
        type=float,
        default=2.0,
        help="lower percentile mapped to 0 (default: 2)",
    )
    stretch.add_argument(
        "--upper",
        type=float,
        default=98.0,
        help="upper percentile mapped to 1 (default: 98)",
    )
    stretch.add_argument("--band", type=int, default=0, help=_HELP_BAND)
    stretch.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    stretch.set_defaults(func=_cmd_percentile_stretch)

    pan = sub.add_parser(
        "pan-sharpen", help="fuse a multispectral raster with a finer pan band"
    )
    pan.add_argument("multispectral", help="coarse multispectral raster path")
    pan.add_argument("pan", help="fine single-band panchromatic raster path")
    pan.add_argument("output", help=_HELP_DST_RASTER)
    pan.add_argument(
        "--algorithm",
        default="brovey",
        help="pan-sharpening algorithm (default: brovey)",
    )
    pan.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="per-band Brovey weights (one per selected band)",
    )
    pan.add_argument(
        "--resampling",
        default="cubic",
        help="spectral upsampling: nearest, bilinear, cubic, cubicspline, lanczos, average",
    )
    pan.add_argument(
        "--bands",
        nargs="+",
        type=int,
        help="zero-based multispectral bands to sharpen (default: all)",
    )
    pan.add_argument(
        "--nodata", type=float, help="no-data value set on the fused output"
    )
    pan.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    pan.set_defaults(func=_cmd_pan_sharpen)

    fillna = sub.add_parser("fillna", help="give every gap a constant value")
    fillna.add_argument("input", help=_HELP_SRC_RASTER)
    fillna.add_argument("output", help=_HELP_DST_RASTER)
    fillna.add_argument(
        "--value", type=float, required=True, help="value written into every gap"
    )
    fillna.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    fillna.set_defaults(func=_cmd_fillna)

    where = sub.add_parser(
        "where", help="keep the cells a condition selects and mask the rest"
    )
    where.add_argument("input", help=_HELP_SRC_RASTER)
    where.add_argument("output", help=_HELP_DST_RASTER)
    where.add_argument(
        "--cond",
        required=True,
        help="condition over the input bound to A, e.g. 'A > 5'",
    )
    where.add_argument(
        "--other",
        help="value for unselected cells: a number, 'nan', or 'none' "
        "(default: the raster's no-data value)",
    )
    where.add_argument(
        "--drop",
        action="store_true",
        help="trim the result to the rows/columns/bands the condition selected",
    )
    where.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    where.set_defaults(func=_cmd_where)

    convert_units = sub.add_parser(
        "convert-units", help="transform band values to another unit"
    )
    convert_units.add_argument("input", help=_HELP_SRC_RASTER)
    convert_units.add_argument("output", help=_HELP_DST_RASTER)
    convert_units.add_argument(
        "--target", required=True, help="target unit label (e.g. celsius, hPa, knots)"
    )
    convert_units.add_argument(
        "--band",
        type=int,
        default=None,
        help="band to convert (default: every band)",
    )
    convert_units.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    convert_units.set_defaults(func=_cmd_convert_units)

    contour = sub.add_parser(
        "contour", help="trace iso-value contours into a vector file"
    )
    contour.add_argument("input", help=_HELP_SRC_RASTER)
    contour.add_argument("output", help="destination vector path")
    levels = contour.add_mutually_exclusive_group(required=True)
    levels.add_argument(
        "--interval",
        type=float,
        help="spacing between contour levels (anchored at --base)",
    )
    levels.add_argument(
        "--fixed-levels",
        dest="fixed_levels",
        nargs="+",
        type=float,
        help="explicit levels to contour",
    )
    contour.add_argument(
        "--base",
        type=float,
        default=0.0,
        help="level the interval is anchored to (default: 0)",
    )
    contour.add_argument("--band", type=int, default=0, help=_HELP_BAND)
    contour.add_argument(
        "--attribute", default="elev", help="elevation attribute name (default: elev)"
    )
    contour.add_argument(
        "--polygonize",
        action="store_true",
        help="emit filled polygons between levels instead of lines",
    )
    contour.add_argument(
        "--driver",
        help="OGR vector driver (default: inferred from the output extension)",
    )
    contour.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    contour.set_defaults(func=_cmd_contour)
