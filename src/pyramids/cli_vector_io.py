"""Vector I/O, export, and remote-service reader commands for ``pyramids vector``.

Thin wrappers over :class:`pyramids.feature.FeatureCollection` construction/export
methods: build from a bbox, round-trip Parquet/Feather, emit GeoJSON/WKT/WKB, write
vector tiles (MVT/PMTiles), and read from OGC API Features / Esri FeatureServer /
vector-tile services. Network/DB readers are thin forwarders; parquet/tile/service
dependencies stay lazily imported by the underlying methods, so importing this
module (and ``pyramids.cli``) requires no optional extra.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator

import pandas as pd
import shapely

from pyramids._cli_shared import (
    ProgressReporter,
    info_print,
    json_sanitize,
    map_lookup_errors,
    refuse_existing,
    resolve_vector_driver,
)
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.feature import FeatureCollection

_HELP_SRC = "source vector path"
_HELP_OUTPUT = "destination path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_LAYER = "layer name or index in a multi-layer source"
_HELP_JSON = "emit JSON"
_HELP_BBOX = "bounding box as MINX MINY MAXX MAXY"
_HELP_DRIVER = "OGR driver (default: inferred from extension)"
_HELP_CRS = "CRS to attach to the result (EPSG int, 'EPSG:4326', WKT, PROJ4)"


@contextlib.contextmanager
def _needs_parquet_extra() -> Iterator[None]:
    """Remap a missing-pyarrow ``ImportError`` to a clean ``ValueError``.

    ``to_parquet``/``to_feather`` import ``pyarrow`` from the optional ``[parquet]``
    extra; its absence raises ``ImportError``, which :func:`pyramids.cli.main` does
    not catch (it would leak the "unexpected failure" branch). Turn it into a clean
    user error naming the extra.
    """
    try:
        yield
    except OptionalPackageDoesNotExist as exc:
        # Pass the seam's own message through. It already names the distribution that provides the
        # backend and how to install it; replacing it with an extras hint hands the user advice
        # that cannot fix the problem (the extras install the stack, not the backend).
        raise ValueError(str(exc)) from exc
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [parquet] extra (pyarrow); install it "
            "with `pip install pyramids-core[parquet]`"
        ) from exc


def _cmd_vector_from_bbox(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import bbox`` — write a one-feature bbox polygon.

    Args:
        args: Parsed args with ``output``, ``bbox``, ``epsg``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, None)
    FeatureCollection.from_bbox(tuple(args.bbox), epsg=args.epsg).to_file(
        args.output, driver=driver
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_read_parquet(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import parquet`` — convert a GeoParquet to a vector file.

    Args:
        args: Parsed args with ``src``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    driver = resolve_vector_driver(args.output, None)
    FeatureCollection.read_parquet(args.src).to_file(args.output, driver=driver)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_to_parquet(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export parquet`` — write a GeoParquet file.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with _needs_parquet_extra():
        FeatureCollection.read_file(args.src, layer=args.layer).to_parquet(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_to_feather(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export feather`` — write a GeoFeather file.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with _needs_parquet_extra():
        FeatureCollection.read_file(args.src, layer=args.layer).to_feather(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_to_json(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export json`` — emit the GeoJSON string.

    Writes to ``--out`` when given, else prints to stdout (for piping).

    Args:
        args: Parsed args with ``src``, ``layer``, ``out``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    text = FeatureCollection.read_file(args.src, layer=args.layer).to_json()
    if args.out:
        refuse_existing(args.out, args.overwrite)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        info_print(args, f"wrote {args.out}")
    else:
        print(text)
    return 0


def _cmd_vector_to_geo_dict(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export geo-dict`` — print the GeoJSON mapping.

    Args:
        args: Parsed args with ``src``, ``layer``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    payload = FeatureCollection.read_file(args.src, layer=args.layer).to_geo_dict()
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        print(f"type: {payload.get('type')}")
        print(f"features: {len(payload.get('features', []))}")
    return 0


def _cmd_vector_to_wkt(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export wkt`` — write a WKT table to CSV.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    FeatureCollection.read_file(args.src, layer=args.layer).to_wkt().to_csv(
        args.output, index=False
    )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_to_wkb(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export wkb`` — write a WKB table to CSV.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``hex``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    fc.to_wkb(hex=args.hex).to_csv(args.output, index=False)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_to_mvt(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export mvt`` — write a Mapbox Vector Tiles archive.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``min_zoom``,
            ``max_zoom``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    fc.to_mvt(args.output, min_zoom=args.min_zoom, max_zoom=args.max_zoom)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_to_pmtiles(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector export pmtiles`` — write a PMTiles archive.

    Args:
        args: Parsed args with ``src``, ``output``, ``layer``, ``min_zoom``,
            ``max_zoom``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.read_file(args.src, layer=args.layer)
    fc.to_pmtiles(args.output, min_zoom=args.min_zoom, max_zoom=args.max_zoom)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_from_ogc_features(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import ogc-features`` — read an OGC API Features service.

    Args:
        args: Parsed args with ``endpoint``, ``collection``, ``bbox``, ``output``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    bbox = tuple(args.bbox) if args.bbox else None
    fc = FeatureCollection.from_ogc_features(
        args.endpoint, collection=args.collection, bbox=bbox
    )
    fc.to_file(args.output, driver=resolve_vector_driver(args.output, None))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_from_featureserver(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import featureserver`` — read an Esri FeatureServer layer.

    Args:
        args: Parsed args with ``url``, ``where``, ``out_fields``, ``output``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    fc = FeatureCollection.from_featureserver(
        args.url, where=args.where, out_fields=args.out_fields
    )
    fc.to_file(args.output, driver=resolve_vector_driver(args.output, None))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_from_vectortileserver(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import vectortileserver`` — read a vector-tile service.

    Args:
        args: Parsed args with ``url``, ``layer``, ``bbox``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    bbox = tuple(args.bbox) if args.bbox else None
    fc = FeatureCollection.from_vectortileserver(args.url, layer=args.layer, bbox=bbox)
    fc.to_file(args.output, driver=resolve_vector_driver(args.output, None))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_from_xyz(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import xyz`` — build a point layer from CSV columns.

    Reads the ``--x`` / ``--y`` (and optional ``--z``) columns of a CSV as parallel
    coordinate arrays and writes the resulting point FeatureCollection.

    Args:
        args: Parsed args with ``csv``, ``output``, ``x``, ``y``, ``z``, ``crs``,
            ``z_column``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    frame = pd.read_csv(args.csv)
    with map_lookup_errors():
        x = frame[args.x].to_numpy()
        y = frame[args.y].to_numpy()
        z = frame[args.z].to_numpy() if args.z else None
    fc = FeatureCollection.from_xyz(x, y, z, crs=args.crs, z_column=args.z_column)
    fc.to_file(args.output, driver=resolve_vector_driver(args.output, args.driver))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_from_features(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import features`` — build an FC from GeoJSON features.

    Reads a JSON file holding either a GeoJSON ``FeatureCollection`` mapping or a bare
    list of GeoJSON Feature dicts, and writes the resulting FeatureCollection.

    Args:
        args: Parsed args with ``src``, ``output``, ``crs``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with open(args.src, encoding="utf-8") as handle:
        data = json.load(handle)
    features = (
        data["features"] if isinstance(data, dict) and "features" in data else data
    )
    fc = FeatureCollection.from_features(features, crs=args.crs)
    fc.to_file(args.output, driver=resolve_vector_driver(args.output, args.driver))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_from_records(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import records`` — build an FC from a CSV with a WKT column.

    Reads a CSV whose ``--geometry`` column holds WKT geometries; every other column
    becomes an attribute. The WKT is parsed to shapely geometries before the
    FeatureCollection is assembled and written.

    Args:
        args: Parsed args with ``csv``, ``output``, ``geometry``, ``crs``, ``driver``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    frame = pd.read_csv(args.csv)
    with map_lookup_errors():
        geometries = shapely.from_wkt(frame[args.geometry].to_list())
    columns = {name: frame[name].to_list() for name in frame.columns}
    columns[args.geometry] = geometries
    fc = FeatureCollection.from_records(
        columns, geometry=args.geometry, crs=args.crs, orient="list"
    )
    fc.to_file(args.output, driver=resolve_vector_driver(args.output, args.driver))
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_vector_read_gpx(args: argparse.Namespace) -> int:
    """Handle ``pyramids vector import gpx`` — write every non-empty GPX sub-layer.

    Reads all non-empty sub-layers of a GPX file (waypoints, routes, tracks,
    route_points, track_points) and writes each as a named layer in a single
    multi-layer output (e.g. a GeoPackage).

    Args:
        args: Parsed args with ``src``, ``output``, ``driver``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The GPX file has no non-empty sub-layers to write.
    """
    refuse_existing(args.output, args.overwrite)
    layers = FeatureCollection.read_gpx_layers(args.src)
    if not layers:
        raise ValueError(f"{args.src!r} has no non-empty GPX sub-layers to write")
    driver = resolve_vector_driver(args.output, args.driver)
    names = sorted(layers)
    with ProgressReporter(args, len(names), label="layers") as progress:
        for index, name in enumerate(names):
            layers[name].to_file(
                args.output, driver=driver, layer=name, mode="w" if index == 0 else "a"
            )
            progress.advance()
    info_print(args, f"wrote {args.output} (layers: {', '.join(names)})")
    return 0


def add_vector_io_commands(
    export_sub: argparse._SubParsersAction,
    import_sub: argparse._SubParsersAction,
) -> None:
    """Register the ``vector`` export (writer) and import (reader) commands.

    Args:
        export_sub: The ``vector export`` subgroup's ``add_subparsers`` action (writers).
        import_sub: The ``vector import`` subgroup's ``add_subparsers`` action (readers).
    """
    v_bbox = import_sub.add_parser("bbox", help="write a one-feature bbox polygon")
    v_bbox.add_argument("output", help=_HELP_OUTPUT)
    v_bbox.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help=_HELP_BBOX,
    )
    v_bbox.add_argument("--epsg", type=int, required=True, help="EPSG code of the bbox")
    v_bbox.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_bbox.set_defaults(func=_cmd_vector_from_bbox)

    v_rpq = import_sub.add_parser(
        "parquet", help="convert a GeoParquet to a vector file"
    )
    v_rpq.add_argument("src", help="source GeoParquet path")
    v_rpq.add_argument("output", help=_HELP_OUTPUT)
    v_rpq.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_rpq.set_defaults(func=_cmd_vector_read_parquet)

    v_tpq = export_sub.add_parser("parquet", help="write a GeoParquet file")
    v_tpq.add_argument("src", help=_HELP_SRC)
    v_tpq.add_argument("output", help="destination .parquet path")
    v_tpq.add_argument("--layer", help=_HELP_LAYER)
    v_tpq.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_tpq.set_defaults(func=_cmd_vector_to_parquet)

    v_tf = export_sub.add_parser("feather", help="write a GeoFeather file")
    v_tf.add_argument("src", help=_HELP_SRC)
    v_tf.add_argument("output", help="destination .feather path")
    v_tf.add_argument("--layer", help=_HELP_LAYER)
    v_tf.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_tf.set_defaults(func=_cmd_vector_to_feather)

    v_tj = export_sub.add_parser(
        "json", help="emit the GeoJSON string (stdout or --out)"
    )
    v_tj.add_argument("src", help=_HELP_SRC)
    v_tj.add_argument("--layer", help=_HELP_LAYER)
    v_tj.add_argument("--out", help="write to this path instead of stdout")
    v_tj.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_tj.set_defaults(func=_cmd_vector_to_json)

    v_gd = export_sub.add_parser("geo-dict", help="print the GeoJSON mapping")
    v_gd.add_argument("src", help=_HELP_SRC)
    v_gd.add_argument("--layer", help=_HELP_LAYER)
    v_gd.add_argument("--json", action="store_true", help=_HELP_JSON)
    v_gd.set_defaults(func=_cmd_vector_to_geo_dict)

    v_wkt = export_sub.add_parser("wkt", help="write a WKT table to CSV")
    v_wkt.add_argument("src", help=_HELP_SRC)
    v_wkt.add_argument("output", help="destination .csv path")
    v_wkt.add_argument("--layer", help=_HELP_LAYER)
    v_wkt.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_wkt.set_defaults(func=_cmd_vector_to_wkt)

    v_wkb = export_sub.add_parser("wkb", help="write a WKB table to CSV")
    v_wkb.add_argument("src", help=_HELP_SRC)
    v_wkb.add_argument("output", help="destination .csv path")
    v_wkb.add_argument("--layer", help=_HELP_LAYER)
    v_wkb.add_argument("--hex", action="store_true", help="encode geometries as hex")
    v_wkb.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_wkb.set_defaults(func=_cmd_vector_to_wkb)

    v_mvt = export_sub.add_parser("mvt", help="write a Mapbox Vector Tiles archive")
    v_mvt.add_argument("src", help=_HELP_SRC)
    v_mvt.add_argument("output", help="destination .mvt path")
    v_mvt.add_argument("--layer", help=_HELP_LAYER)
    v_mvt.add_argument("--min-zoom", type=int, default=0, help="minimum zoom level")
    v_mvt.add_argument("--max-zoom", type=int, default=None, help="maximum zoom level")
    v_mvt.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_mvt.set_defaults(func=_cmd_vector_to_mvt)

    v_pm = export_sub.add_parser("pmtiles", help="write a PMTiles archive")
    v_pm.add_argument("src", help=_HELP_SRC)
    v_pm.add_argument("output", help="destination .pmtiles path")
    v_pm.add_argument("--layer", help=_HELP_LAYER)
    v_pm.add_argument("--min-zoom", type=int, default=0, help="minimum zoom level")
    v_pm.add_argument("--max-zoom", type=int, default=None, help="maximum zoom level")
    v_pm.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_pm.set_defaults(func=_cmd_vector_to_pmtiles)

    v_ogc = import_sub.add_parser(
        "ogc-features", help="read an OGC API Features service"
    )
    v_ogc.add_argument("output", help=_HELP_OUTPUT)
    v_ogc.add_argument("--endpoint", required=True, help="service base URL")
    v_ogc.add_argument("--collection", required=True, help="collection id")
    v_ogc.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help=_HELP_BBOX,
    )
    v_ogc.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_ogc.set_defaults(func=_cmd_vector_from_ogc_features)

    v_fs = import_sub.add_parser(
        "featureserver", help="read an Esri FeatureServer layer"
    )
    v_fs.add_argument("output", help=_HELP_OUTPUT)
    v_fs.add_argument("--url", required=True, help="FeatureServer layer URL")
    v_fs.add_argument("--where", default="1=1", help="attribute filter")
    v_fs.add_argument("--out-fields", default="*", help="comma-separated fields")
    v_fs.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_fs.set_defaults(func=_cmd_vector_from_featureserver)

    v_vts = import_sub.add_parser("vectortileserver", help="read a vector-tile service")
    v_vts.add_argument("output", help=_HELP_OUTPUT)
    v_vts.add_argument("--url", required=True, help="vector-tile service URL")
    v_vts.add_argument("--layer", help="layer name")
    v_vts.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help=_HELP_BBOX,
    )
    v_vts.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_vts.set_defaults(func=_cmd_vector_from_vectortileserver)

    v_xyz = import_sub.add_parser(
        "xyz", help="build a point layer from CSV x/y/z columns"
    )
    v_xyz.add_argument("csv", help="source CSV path")
    v_xyz.add_argument("output", help=_HELP_OUTPUT)
    v_xyz.add_argument("--x", required=True, help="name of the X-coordinate column")
    v_xyz.add_argument("--y", required=True, help="name of the Y-coordinate column")
    v_xyz.add_argument("--z", help="name of the optional value column")
    v_xyz.add_argument("--crs", help=_HELP_CRS)
    v_xyz.add_argument("--z-column", default="z", help="name to store --z under")
    v_xyz.add_argument("--driver", help=_HELP_DRIVER)
    v_xyz.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_xyz.set_defaults(func=_cmd_vector_from_xyz)

    v_ff = import_sub.add_parser(
        "features", help="build an FC from a GeoJSON features file"
    )
    v_ff.add_argument(
        "src", help="source JSON path (FeatureCollection or feature list)"
    )
    v_ff.add_argument("output", help=_HELP_OUTPUT)
    v_ff.add_argument("--crs", help=_HELP_CRS)
    v_ff.add_argument("--driver", help=_HELP_DRIVER)
    v_ff.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_ff.set_defaults(func=_cmd_vector_from_features)

    v_fr = import_sub.add_parser(
        "records", help="build an FC from a CSV with a WKT geometry column"
    )
    v_fr.add_argument("csv", help="source CSV path")
    v_fr.add_argument("output", help=_HELP_OUTPUT)
    v_fr.add_argument(
        "--geometry", default="geometry", help="name of the WKT geometry column"
    )
    v_fr.add_argument("--crs", help=_HELP_CRS)
    v_fr.add_argument("--driver", help=_HELP_DRIVER)
    v_fr.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_fr.set_defaults(func=_cmd_vector_from_records)

    v_gpx = import_sub.add_parser(
        "gpx", help="write every non-empty GPX sub-layer to a multi-layer output"
    )
    v_gpx.add_argument("src", help="source .gpx path")
    v_gpx.add_argument("output", help="destination multi-layer path (e.g. .gpkg)")
    v_gpx.add_argument("--driver", help=_HELP_DRIVER)
    v_gpx.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    v_gpx.set_defaults(func=_cmd_vector_read_gpx)
