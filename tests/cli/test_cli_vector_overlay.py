"""Tests for the ``pyramids vector`` overlay / set-theoretic / filter commands.

Each command is driven through a local parser built only from
:func:`pyramids.cli_vector_overlay.add_vector_overlay_commands`, with a ``run``
helper that mirrors :func:`pyramids.cli.main`'s one-line-error contract. Every
command gets one happy-path test (writes the output, re-reads it, asserts an
observable effect) and one error-path test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon

from pyramids._cli_groups import add_group
from pyramids.base._errors import _PyramidsError
from pyramids.cli import main
from pyramids.cli_vector_overlay import add_vector_overlay_commands
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

_DATA = Path(__file__).resolve().parents[1] / "data"
_COELLO_POLYGONS = str(_DATA / "coello_polygons.geojson")
_G_POLYGONS = str(_DATA / "geometries" / "polygons.geojson")
_GAUGES = str(_DATA / "coello-gauges.geojson")


def _build_parser():
    """Build a local ``pyramids vector`` parser carrying only the overlay commands.

    Returns:
        argparse.ArgumentParser: The parser with the ``vector`` group registered.
    """
    parser = argparse.ArgumentParser(prog="pyramids")
    sub = parser.add_subparsers(dest="group", required=True)
    vector = sub.add_parser("vector")
    vsub = vector.add_subparsers(dest="command", required=True)
    advanced = add_group(vsub, "advanced", "power-user primitives (geometry micro-ops)")
    add_vector_overlay_commands(
        add_group(vsub, "overlay", "overlay / set ops, joins, filter"), advanced
    )
    return parser


def run(argv):
    """Parse ``argv`` and dispatch, mirroring ``pyramids.cli.main``'s error contract.

    Args:
        argv: The argument list (without the program name).

    Returns:
        int: ``0`` on success, ``1`` on an expected user error.
    """
    args = _build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except (ValueError, TypeError, OSError, RuntimeError, _PyramidsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        result = 1
    return result


@pytest.fixture
def pair_a(tmp_path) -> str:
    """Two overlapping unit squares sharing index ``0,1`` (left operand).

    Returns:
        str: Path to the written vector.
    """
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(4, 0), (6, 0), (6, 2), (4, 2)]),
        ],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "pair_a.geojson")
    FeatureCollection(gdf).to_file(path)
    return path


@pytest.fixture
def pair_b(tmp_path) -> str:
    """Two squares overlapping ``pair_a`` element-wise (right operand).

    Returns:
        str: Path to the written vector.
    """
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]),
            Polygon([(5, 0), (7, 0), (7, 2), (5, 2)]),
        ],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "pair_b.geojson")
    FeatureCollection(gdf).to_file(path)
    return path


@pytest.fixture
def lines_a(tmp_path) -> str:
    """A single L-shaped line (left operand for shared-paths / shortest-line).

    Returns:
        str: Path to the written vector.
    """
    gdf = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (2, 0), (2, 2)])], crs="EPSG:4326"
    )
    path = str(tmp_path / "lines_a.geojson")
    FeatureCollection(gdf).to_file(path)
    return path


@pytest.fixture
def lines_b(tmp_path) -> str:
    """A line sharing the ``(0,0)->(2,0)`` segment with ``lines_a``.

    Returns:
        str: Path to the written vector.
    """
    gdf = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (2, 0), (2, -2)])], crs="EPSG:4326"
    )
    path = str(tmp_path / "lines_b.geojson")
    FeatureCollection(gdf).to_file(path)
    return path


@pytest.fixture
def multipart(tmp_path) -> str:
    """A one-row layer holding a two-part MultiPolygon (for explode).

    Returns:
        str: Path to the written vector.
    """
    geom = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
    )
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")
    path = str(tmp_path / "multipart.geojson")
    FeatureCollection(gdf).to_file(path)
    return path


class TestIntersection:
    """`pyramids vector intersection`."""

    def test_writes_overlap(self, pair_a, pair_b, tmp_path):
        """The element-wise intersection writes the two overlap strips."""
        out = str(tmp_path / "x.geojson")
        assert (
            run(["vector", "overlay", "intersection", pair_a, out, "--other", pair_b])
            == 0
        )
        back = FeatureCollection.read_file(out)
        assert len(back) == 2
        assert back.total_bounds[0] == 1.0, (
            "feature 0 clipped to the overlap strip (minx moves 0->1)"
        )

    def test_missing_other_exits_one(self, pair_a, tmp_path, capsys):
        """A missing --other file exits 1 with a clean error."""
        rc = run(
            [
                "vector",
                "overlay",
                "intersection",
                pair_a,
                str(tmp_path / "o.geojson"),
                "--other",
                str(tmp_path / "nope.geojson"),
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "Traceback" not in err


class TestDifference:
    """`pyramids vector difference`."""

    def test_writes_remainder(self, pair_a, pair_b, tmp_path):
        """The difference keeps the non-overlapping remainder of each feature."""
        out = str(tmp_path / "d.geojson")
        assert (
            run(["vector", "overlay", "difference", pair_a, out, "--other", pair_b])
            == 0
        )
        back = FeatureCollection.read_file(out)
        assert len(back) == 2
        assert back.total_bounds[0] == 0.0, "the left edge of feature 0 survives"

    def test_missing_other_exits_one(self, pair_a, tmp_path, capsys):
        """A missing --other file exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "difference",
                pair_a,
                str(tmp_path / "o.geojson"),
                "--other",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestSymmetricDifference:
    """`pyramids vector symmetric-difference`."""

    def test_writes_symmetric(self, pair_a, pair_b, tmp_path):
        """The symmetric difference writes both exclusive parts per pair."""
        out = str(tmp_path / "s.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "symmetric-difference",
                pair_a,
                out,
                "--other",
                pair_b,
            ]
        )
        assert rc == 0
        back = FeatureCollection.read_file(out)
        assert len(back) == 2
        geom_types = {str(t) for t in back.geometry.geom_type.dropna().unique()}
        assert "MultiPolygon" in geom_types

    def test_missing_other_exits_one(self, pair_a, tmp_path, capsys):
        """A missing --other file exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "symmetric-difference",
                pair_a,
                str(tmp_path / "o.geojson"),
                "--other",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestUnion:
    """`pyramids vector union`."""

    def test_writes_union(self, pair_a, pair_b, tmp_path):
        """The element-wise union merges each overlapping pair into one polygon."""
        out = str(tmp_path / "u.geojson")
        assert run(["vector", "overlay", "union", pair_a, out, "--other", pair_b]) == 0
        back = FeatureCollection.read_file(out)
        assert len(back) == 2
        assert back.total_bounds[2] == 7.0, "feature 1's union reaches x=7"

    def test_missing_other_exits_one(self, pair_a, tmp_path, capsys):
        """A missing --other file exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "union",
                pair_a,
                str(tmp_path / "o.geojson"),
                "--other",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestIntersectionAll:
    """`pyramids vector intersection-all`."""

    def test_disjoint_gives_empty(self, pair_a, tmp_path):
        """Intersecting all disjoint features yields a single (empty) geometry."""
        out = str(tmp_path / "ia.geojson")
        assert run(["vector", "overlay", "intersection-all", pair_a, out]) == 0
        back = FeatureCollection.read_file(out)
        assert len(back) == 1

    def test_missing_src_exits_one(self, tmp_path, capsys):
        """A missing source exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "intersection-all",
                str(tmp_path / "nope.geojson"),
                str(tmp_path / "o.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestUnionAll:
    """`pyramids vector union-all`."""

    def test_writes_single_union(self, pair_a, tmp_path):
        """union-all collapses the layer to one multi-part geometry."""
        out = str(tmp_path / "ua.geojson")
        assert (
            run(["vector", "overlay", "union-all", pair_a, out, "--method", "unary"])
            == 0
        )
        back = FeatureCollection.read_file(out)
        assert len(back) == 1

    def test_grid_size_accepted(self, pair_a, tmp_path):
        """--grid-size is forwarded and still produces one geometry."""
        out = str(tmp_path / "uag.geojson")
        rc = run(["vector", "overlay", "union-all", pair_a, out, "--grid-size", "0.5"])
        assert rc == 0
        assert len(FeatureCollection.read_file(out)) == 1

    def test_bad_method_rejected(self, pair_a, tmp_path):
        """An out-of-choices --method is rejected by argparse."""
        with pytest.raises(SystemExit):
            run(
                [
                    "vector",
                    "overlay",
                    "union-all",
                    pair_a,
                    str(tmp_path / "o.geojson"),
                    "--method",
                    "bogus",
                ]
            )


class TestOverlay:
    """`pyramids vector overlay`."""

    def test_intersection_overlay(self, tmp_path):
        """overlay intersection of two real overlapping polygon layers writes parts."""
        out = str(tmp_path / "ov.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "apply",
                _COELLO_POLYGONS,
                out,
                "--right",
                _G_POLYGONS,
                "--how",
                "intersection",
            ]
        )
        assert rc == 0
        back = FeatureCollection.read_file(out)
        assert len(back) > 0

    def test_bad_how_rejected(self, tmp_path):
        """An out-of-choices --how is rejected by argparse."""
        with pytest.raises(SystemExit):
            run(
                [
                    "vector",
                    "overlay",
                    "apply",
                    _COELLO_POLYGONS,
                    str(tmp_path / "o.geojson"),
                    "--right",
                    _G_POLYGONS,
                    "--how",
                    "bogus",
                ]
            )

    def test_missing_right_exits_one(self, tmp_path, capsys):
        """A missing --right file exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "apply",
                _COELLO_POLYGONS,
                str(tmp_path / "o.geojson"),
                "--right",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestDissolve:
    """`pyramids vector dissolve`."""

    def test_dissolve_all(self, pair_a, tmp_path):
        """A plain dissolve merges every feature into one row."""
        out = str(tmp_path / "di.geojson")
        assert run(["vector", "overlay", "dissolve", pair_a, out]) == 0
        assert len(FeatureCollection.read_file(out)) == 1

    def test_dissolve_by_keeps_groups(self, pair_a, tmp_path):
        """Dissolving by a distinct column keeps one row per group."""
        out = str(tmp_path / "dib.geojson")
        assert run(["vector", "overlay", "dissolve", pair_a, out, "--by", "id"]) == 0
        assert len(FeatureCollection.read_file(out)) == 2

    def test_missing_src_exits_one(self, tmp_path, capsys):
        """A missing source exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "dissolve",
                str(tmp_path / "nope.geojson"),
                str(tmp_path / "o.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestSjoin:
    """`pyramids vector sjoin`."""

    def test_inner_join(self, tmp_path):
        """An inner intersects-join attaches right-hand columns to matches."""
        out = str(tmp_path / "sj.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "sjoin",
                _COELLO_POLYGONS,
                out,
                "--right",
                _G_POLYGONS,
                "--predicate",
                "intersects",
            ]
        )
        assert rc == 0
        assert len(FeatureCollection.read_file(out)) > 0

    def test_bad_predicate_rejected(self, tmp_path):
        """An out-of-choices --predicate is rejected by argparse."""
        with pytest.raises(SystemExit):
            run(
                [
                    "vector",
                    "overlay",
                    "sjoin",
                    _COELLO_POLYGONS,
                    str(tmp_path / "o.geojson"),
                    "--right",
                    _G_POLYGONS,
                    "--predicate",
                    "bogus",
                ]
            )

    def test_missing_right_exits_one(self, tmp_path, capsys):
        """A missing --right file exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "sjoin",
                _COELLO_POLYGONS,
                str(tmp_path / "o.geojson"),
                "--right",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestSjoinNearest:
    """`pyramids vector sjoin-nearest`."""

    def test_nearest_join(self, tmp_path):
        """A nearest join matches every left feature to its closest right one."""
        out = str(tmp_path / "sn.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "sjoin-nearest",
                _COELLO_POLYGONS,
                out,
                "--right",
                _G_POLYGONS,
            ]
        )
        assert rc == 0
        assert len(FeatureCollection.read_file(out)) >= 4

    def test_max_distance_limits(self, tmp_path):
        """A tiny --max-distance drops matches beyond it (fewer rows)."""
        wide = str(tmp_path / "wide.geojson")
        narrow = str(tmp_path / "narrow.geojson")
        run(
            [
                "vector",
                "overlay",
                "sjoin-nearest",
                _COELLO_POLYGONS,
                wide,
                "--right",
                _G_POLYGONS,
            ]
        )
        rc = run(
            [
                "vector",
                "overlay",
                "sjoin-nearest",
                _COELLO_POLYGONS,
                narrow,
                "--right",
                _G_POLYGONS,
                "--max-distance",
                "1.0",
            ]
        )
        assert rc == 0
        assert len(FeatureCollection.read_file(narrow)) < len(
            FeatureCollection.read_file(wide)
        )

    def test_missing_right_exits_one(self, tmp_path, capsys):
        """A missing --right file exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "sjoin-nearest",
                _COELLO_POLYGONS,
                str(tmp_path / "o.geojson"),
                "--right",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestClipByRect:
    """`pyramids vector clip-by-rect`."""

    def test_clip_window(self, pair_a, tmp_path):
        """clip-by-rect trims the layer to the requested window."""
        out = str(tmp_path / "cr.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "clip-by-rect",
                pair_a,
                out,
                "--rect",
                "0",
                "0",
                "2",
                "2",
            ]
        )
        assert rc == 0
        back = FeatureCollection.read_file(out)
        assert back.total_bounds[2] <= 2.0, "nothing survives beyond x=2"

    def test_bad_rect_arity_rejected(self, pair_a, tmp_path):
        """A --rect with the wrong number of values is rejected by argparse."""
        with pytest.raises(SystemExit):
            run(
                [
                    "vector",
                    "overlay",
                    "clip-by-rect",
                    pair_a,
                    str(tmp_path / "o.geojson"),
                    "--rect",
                    "0",
                    "0",
                    "2",
                ]
            )


class TestSharedPaths:
    """`pyramids vector shared-paths`."""

    def test_shared_segment(self, lines_a, lines_b, tmp_path):
        """shared-paths writes the common segment of two lines."""
        out = str(tmp_path / "sp.geojson")
        assert (
            run(
                ["vector", "advanced", "shared-paths", lines_a, out, "--other", lines_b]
            )
            == 0
        )
        assert len(FeatureCollection.read_file(out)) == 1

    def test_missing_other_exits_one(self, lines_a, tmp_path, capsys):
        """A missing --other file exits 1 with a clean error."""
        rc = run(
            [
                "vector",
                "advanced",
                "shared-paths",
                lines_a,
                str(tmp_path / "o.geojson"),
                "--other",
                str(tmp_path / "nope.geojson"),
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "Traceback" not in err


class TestShortestLine:
    """`pyramids vector shortest-line`."""

    def test_writes_connectors(self, lines_a, lines_b, tmp_path):
        """shortest-line writes the nearest-point connector line per pair."""
        out = str(tmp_path / "sl.geojson")
        assert (
            run(
                [
                    "vector",
                    "advanced",
                    "shortest-line",
                    lines_a,
                    out,
                    "--other",
                    lines_b,
                ]
            )
            == 0
        )
        back = FeatureCollection.read_file(out)
        assert len(back) == 1
        geom_types = {str(t) for t in back.geometry.geom_type.dropna().unique()}
        assert "LineString" in geom_types

    def test_missing_other_exits_one(self, lines_a, tmp_path, capsys):
        """A missing --other file exits 1."""
        rc = run(
            [
                "vector",
                "advanced",
                "shortest-line",
                lines_a,
                str(tmp_path / "o.geojson"),
                "--other",
                str(tmp_path / "nope.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestExplode:
    """`pyramids vector explode`."""

    def test_splits_multipart(self, multipart, tmp_path):
        """explode turns a single two-part MultiPolygon into two single-part rows."""
        out = str(tmp_path / "ex.geojson")
        assert run(["vector", "overlay", "explode", multipart, out]) == 0
        back = FeatureCollection.read_file(out)
        assert len(back) == 2

    def test_missing_src_exits_one(self, tmp_path, capsys):
        """A missing source exits 1."""
        rc = run(
            [
                "vector",
                "overlay",
                "explode",
                str(tmp_path / "nope.geojson"),
                str(tmp_path / "o.geojson"),
            ]
        )
        assert rc == 1
        assert "error: " in capsys.readouterr().err


class TestFilter:
    """`pyramids vector filter`."""

    def test_intersects_reduces_count(self, tmp_path):
        """filter intersects keeps only the source polygons touching the mask."""
        out = str(tmp_path / "fi.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "filter",
                _COELLO_POLYGONS,
                out,
                "--predicate",
                "intersects",
                "--mask",
                _G_POLYGONS,
            ]
        )
        assert rc == 0
        source_count = len(FeatureCollection.read_file(_COELLO_POLYGONS))
        kept = len(FeatureCollection.read_file(out))
        assert kept == 2
        assert kept < source_count, "the predicate reduced the feature count"

    def test_dwithin_requires_distance(self, tmp_path, capsys):
        """filter dwithin without --distance exits 1 with a clear message."""
        rc = run(
            [
                "vector",
                "overlay",
                "filter",
                _COELLO_POLYGONS,
                str(tmp_path / "o.geojson"),
                "--predicate",
                "dwithin",
                "--mask",
                _GAUGES,
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "--distance" in err

    def test_dwithin_with_distance(self, tmp_path):
        """filter dwithin with a distance keeps the within-range features."""
        out = str(tmp_path / "fd.geojson")
        rc = run(
            [
                "vector",
                "overlay",
                "filter",
                _COELLO_POLYGONS,
                out,
                "--predicate",
                "dwithin",
                "--mask",
                _GAUGES,
                "--distance",
                "5000",
            ]
        )
        assert rc == 0
        assert len(FeatureCollection.read_file(out)) == 3

    def test_bad_predicate_rejected(self, tmp_path):
        """An out-of-choices --predicate is rejected by argparse."""
        with pytest.raises(SystemExit):
            run(
                [
                    "vector",
                    "overlay",
                    "filter",
                    _COELLO_POLYGONS,
                    str(tmp_path / "o.geojson"),
                    "--predicate",
                    "bogus",
                    "--mask",
                    _G_POLYGONS,
                ]
            )


class TestMainErrorContract:
    """The two leak paths must exit 1 cleanly through the real `main` dispatch."""

    def test_shared_paths_non_lineal_errors_cleanly(self, tmp_path, capsys):
        """shared-paths on polygons raises GEOSException, mapped to a clean error."""
        out = str(tmp_path / "sp.geojson")
        rc = main(
            [
                "vector",
                "advanced",
                "shared-paths",
                _G_POLYGONS,
                out,
                "--other",
                _G_POLYGONS,
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "unexpected failure" not in err
        assert "Traceback" not in err

    def test_dissolve_bad_by_errors_cleanly(self, tmp_path, capsys):
        """dissolve with an unknown --by column exits 1 with a clean error."""
        out = str(tmp_path / "ds.geojson")
        rc = main(
            ["vector", "overlay", "dissolve", _G_POLYGONS, out, "--by", "nosuchcol"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "unexpected failure" not in err
        assert "Traceback" not in err

    def test_overlay_writes_requested_format(self, tmp_path):
        """intersection-all to .shp writes a real shapefile (sidecars present), not GeoJSON (H1)."""
        out = tmp_path / "inter.shp"
        rc = main(["vector", "overlay", "intersection-all", _G_POLYGONS, str(out)])
        assert rc == 0
        assert (tmp_path / "inter.shx").exists()
        assert (tmp_path / "inter.dbf").exists()

    def test_mixed_crs_errors_cleanly(self, tmp_path, capsys):
        """Two layers in different CRS exit 1 with a clear 'different CRS' error (M1)."""
        geog = str(_DATA / "geometries" / "multi-polygons.geojson")  # EPSG:4326
        out = str(tmp_path / "x.geojson")
        rc = main(
            [
                "vector",
                "overlay",
                "intersection",
                _COELLO_POLYGONS,
                out,
                "--other",
                geog,
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1  # _COELLO_POLYGONS is EPSG:32618
        assert "different CRS" in err
        assert "unexpected failure" not in err
