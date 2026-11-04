"""Read-only inspection commands for the ``pyramids vector`` group.

Thin wrappers over :class:`pyramids.feature.FeatureCollection`: feature/field
metadata (`feature-count`, `feature-info`, `list-layers`), the estimated UTM CRS
(`estimate-utm`), per-feature validity (`is-valid-reason`), polygon-coverage checks
(`is-valid-coverage`, `invalid-coverage-edges`), and per-feature structure counts
(`count-coordinates`, `count-geometries`, `count-interior-rings`). Each command maps
1:1 onto an existing ``FeatureCollection`` call; no geometry logic lives here. The
one-line-error contract matches the other groups — handlers raise and
:func:`pyramids.cli.main` reports — and ``--json`` emits strict-valid JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    map_lookup_errors,
    refuse_existing,
)
from pyramids._cli_shared import (
    resolve_vector_driver as _resolve_vector_driver,
)
from pyramids.feature import FeatureCollection

_HELP_SRC = "source vector path"
_HELP_LAYER = "layer name/index for multi-layer sources"
_HELP_JSON = "emit JSON"
_HELP_GAP = "coverage gap-width tolerance in CRS units"


def _info_payload(info: Any) -> Any:
    """Convert a :class:`~pyramids.feature.VectorInfo` record to a JSON-friendly dict.

    Args:
        info: The object returned by ``FeatureCollection.feature_info``.

    Returns:
        object: A dataclass converted with :func:`dataclasses.asdict`, else the
        object's ``__dict__``, else its string form.
    """
    if dataclasses.is_dataclass(info) and not isinstance(info, type):
        return dataclasses.asdict(info)
    if hasattr(info, "__dict__"):
        return vars(info)
    return str(info)


def _cmd_vector_feature_count(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector feature-count`` — count features without loading geometry.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    with map_lookup_errors():
        count = FeatureCollection.feature_count(args.src, layer=args.layer)
    if args.json:
        print(json.dumps(json_sanitize({"path": args.src, "features": count})))
    else:
        print(count)
    return 0


def _cmd_vector_feature_info(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector feature-info`` — the metadata record read without geometry.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    with map_lookup_errors():
        info = FeatureCollection.feature_info(args.src, layer=args.layer)
    payload = _info_payload(info)
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_vector_list_layers(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector list-layers`` — the vector-layer names in the source.

    Args:
        args: Parsed args with ``src``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    layers = FeatureCollection.list_layers(args.src)
    if args.json:
        print(json.dumps(json_sanitize({"layers": layers})))
    else:
        for layer in layers:
            print(layer)
    return 0


def _cmd_vector_estimate_utm(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector estimate-utm`` — the UTM CRS best covering the data.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    crs = fc.estimate_utm_crs()
    epsg = crs.to_epsg()
    if args.json:
        print(json.dumps(json_sanitize({"epsg": epsg, "crs": str(crs)})))
    else:
        print(epsg if epsg is not None else str(crs))
    return 0


def _cmd_vector_is_valid_reason(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector is-valid-reason`` — the per-feature validity reason.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    reasons = fc.is_valid_reason()
    if args.json:
        print(json.dumps(json_sanitize({"reasons": reasons.tolist()})))
    else:
        print(reasons.to_string())
    return 0


def _cmd_vector_is_valid_coverage(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector is-valid-coverage`` — whether polygons form a valid coverage.

    Args:
        args: Parsed args with ``src``, ``layer``, ``gap_width``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    valid = fc.is_valid_coverage(gap_width=args.gap_width)
    print(str(bool(valid)).lower())
    return 0


def _cmd_vector_invalid_coverage_edges(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector invalid-coverage-edges`` — write the invalid coverage edges.

    The edges GeoSeries returned by ``invalid_coverage_edges`` is wrapped in a
    ``FeatureCollection`` and written to ``out`` through the resolved OGR driver.

    Args:
        args: Parsed args with ``src``, ``out``, ``layer``, ``gap_width``,
            ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.out, args.overwrite)
    driver = _resolve_vector_driver(args.out, args.driver)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    edges = fc.invalid_coverage_edges(gap_width=args.gap_width)
    FeatureCollection(edges).to_file(args.out, driver=driver)
    info_print(args, f"wrote {args.out}")
    return 0


def _cmd_vector_count(args: argparse.Namespace) -> int:
    """Handle the ``vector count-*`` commands — a per-feature structure count.

    The method to call and the JSON key are carried on the parser via
    ``set_defaults(count_method=…, count_key=…)``, so one handler serves
    ``count-coordinates``, ``count-geometries`` and ``count-interior-rings``.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``, ``count_method``,
            ``count_key``.

    Returns:
        int: ``0`` on success.
    """
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    counts = getattr(fc, args.count_method)()
    if args.json:
        print(json.dumps(json_sanitize({args.count_key: counts.tolist()})))
    else:
        print(counts.to_string())
    return 0


def _add_src_layer_json(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``src``/``--layer``/``--json`` arguments to an inspection parser.

    Args:
        parser: The sub-command parser to extend.
    """
    parser.add_argument("src", help=_HELP_SRC)
    parser.add_argument("--layer", help=_HELP_LAYER)
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)


def add_vector_inspect_commands(vsub: argparse._SubParsersAction) -> None:
    """Register the read-only ``vector`` inspection commands on ``vsub``.

    Args:
        vsub: The ``vector`` group's ``add_subparsers`` action.
    """
    v_fcount = vsub.add_parser("feature-count", help="count features")
    _add_src_layer_json(v_fcount)
    v_fcount.set_defaults(func=_cmd_vector_feature_count)

    v_finfo = vsub.add_parser("feature-info", help="metadata read without geometry")
    _add_src_layer_json(v_finfo)
    v_finfo.set_defaults(func=_cmd_vector_feature_info)

    v_layers = vsub.add_parser("list-layers", help="list the vector-layer names")
    v_layers.add_argument("src", help=_HELP_SRC)
    v_layers.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_layers.set_defaults(func=_cmd_vector_list_layers)

    v_utm = vsub.add_parser("estimate-utm", help="estimate the best UTM CRS")
    _add_src_layer_json(v_utm)
    v_utm.set_defaults(func=_cmd_vector_estimate_utm)

    v_reason = vsub.add_parser(
        "is-valid-reason", help="per-feature geometry-validity reason"
    )
    _add_src_layer_json(v_reason)
    v_reason.set_defaults(func=_cmd_vector_is_valid_reason)

    v_cov = vsub.add_parser(
        "is-valid-coverage", help="whether polygons form a valid coverage"
    )
    v_cov.add_argument("src", help=_HELP_SRC)
    v_cov.add_argument("--layer", help=_HELP_LAYER)
    v_cov.add_argument("--gap-width", type=float, default=0.0, help=_HELP_GAP)
    v_cov.set_defaults(func=_cmd_vector_is_valid_coverage)

    v_edges = vsub.add_parser(
        "invalid-coverage-edges", help="write the invalid coverage edges"
    )
    v_edges.add_argument("src", help=_HELP_SRC)
    v_edges.add_argument("out", help="destination vector path")
    v_edges.add_argument("--layer", help=_HELP_LAYER)
    v_edges.add_argument("--gap-width", type=float, default=0.0, help=_HELP_GAP)
    v_edges.add_argument(
        "--driver", help="OGR driver (default: inferred from extension)"
    )
    v_edges.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    v_edges.set_defaults(func=_cmd_vector_invalid_coverage_edges)

    v_coords = vsub.add_parser("count-coordinates", help="per-feature coordinate count")
    _add_src_layer_json(v_coords)
    v_coords.set_defaults(
        func=_cmd_vector_count,
        count_method="count_coordinates",
        count_key="coordinates",
    )

    v_geoms = vsub.add_parser("count-geometries", help="per-feature geometry count")
    _add_src_layer_json(v_geoms)
    v_geoms.set_defaults(
        func=_cmd_vector_count,
        count_method="count_geometries",
        count_key="geometries",
    )

    v_rings = vsub.add_parser(
        "count-interior-rings", help="per-feature interior-ring count"
    )
    _add_src_layer_json(v_rings)
    v_rings.set_defaults(
        func=_cmd_vector_count,
        count_method="count_interior_rings",
        count_key="interior_rings",
    )
