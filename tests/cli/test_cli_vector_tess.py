"""Tests for the ``pyramids vector`` tessellation, sampling and measurement commands.

These commands are registered by ``add_vector_tess_commands`` and are driven here
through a local parser built from that registrar, mirroring the pattern used for the
other not-yet-wired command groups. Each shipped command gets one happy path (a
tessellation writes an output that re-reads with the expected geometry / count, a
measurement prints or emits the expected value) and one error path (a bad argument or
missing input surfaces as an exception that ``pyramids.cli.main`` would catch).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pyramids.base._errors import _PyramidsError
from pyramids.cli_vector_tess import add_vector_tess_commands
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data"
# Six points in a projected CRS (EPSG:32618): voronoi / delaunay / h3 / sampling source.
_POINTS = str(_DATA / "coello-gauges.geojson")
# Four polygons in the same projected CRS: triangulation / sampling / polygonize source.
_POLYGONS = str(_DATA / "coello_polygons.geojson")
# One polygon in a geographic CRS (EPSG:4326): the geodesic-area source.
_POLY_GEOG = str(_DATA / "geometries" / "multi-polygons.geojson")

# The exception types ``pyramids.cli.main`` maps to a clean ``error:`` exit-1.
_CAUGHT = (ValueError, TypeError, OSError, RuntimeError, _PyramidsError)

_PARSER = argparse.ArgumentParser()
_SUB = _PARSER.add_subparsers(dest="command", required=True)
add_vector_tess_commands(_SUB)


def run(argv: list[str]) -> int:
    """Parse ``argv`` with the local parser and invoke the selected handler.

    Args:
        argv: The command and its flags (no ``pyramids vector`` prefix).

    Returns:
        int: The handler's return code (``0`` on success).
    """
    ns = _PARSER.parse_args(argv)
    return ns.func(ns)


def _geom_types(path: str) -> set[str]:
    """Return the distinct geometry-type names of the vector at ``path``."""
    fc = FeatureCollection.read_file(path)
    return {str(t) for t in fc.geometry.geom_type.dropna().unique()}


def _touch(path: Path) -> str:
    """Create an empty file at ``path`` and return it as a string."""
    path.write_text("placeholder", encoding="utf-8")
    return str(path)


class TestVoronoiPolygons:
    """`pyramids vector voronoi-polygons`."""

    def test_writes_polygons(self, tmp_path):
        """The Voronoi cells re-read as six polygons, one per input point."""
        out = str(tmp_path / "vor.geojson")
        assert run(["voronoi-polygons", _POINTS, out]) == 0
        assert FeatureCollection.read_file(out).__len__() == 6
        assert _geom_types(out) == {"Polygon"}

    def test_refuses_existing(self, tmp_path):
        """An existing output without --overwrite is refused."""
        out = _touch(tmp_path / "vor.geojson")
        with pytest.raises(_CAUGHT):
            run(["voronoi-polygons", _POINTS, out])


class TestDelaunayTriangles:
    """`pyramids vector delaunay-triangles`."""

    def test_writes_triangles(self, tmp_path):
        """The triangulation re-reads as polygons."""
        out = str(tmp_path / "del.geojson")
        assert run(["delaunay-triangles", _POINTS, out]) == 0
        assert _geom_types(out) == {"Polygon"}

    def test_refuses_existing(self, tmp_path):
        """An existing output without --overwrite is refused."""
        out = _touch(tmp_path / "del.geojson")
        with pytest.raises(_CAUGHT):
            run(["delaunay-triangles", _POINTS, out])


class TestConstrainedDelaunayTriangles:
    """`pyramids vector constrained-delaunay-triangles`."""

    def test_writes_one_collection_per_polygon(self, tmp_path):
        """Each of the four polygons yields its own triangle geometry collection."""
        out = str(tmp_path / "cdt.geojson")
        assert run(["constrained-delaunay-triangles", _POLYGONS, out]) == 0
        assert FeatureCollection.read_file(out).__len__() == 4

    def test_refuses_existing(self, tmp_path):
        """An existing output without --overwrite is refused."""
        out = _touch(tmp_path / "cdt.geojson")
        with pytest.raises(_CAUGHT):
            run(["constrained-delaunay-triangles", _POLYGONS, out])


class TestToH3:
    """`pyramids vector to-h3`."""

    def test_adds_h3_column(self, tmp_path):
        """The output keeps its points and gains an `h3` index column."""
        out = str(tmp_path / "h3.geojson")
        assert run(["to-h3", _POINTS, out, "--resolution", "7"]) == 0
        written = FeatureCollection.read_file(out)
        assert "h3" in written.columns
        assert _geom_types(out) == {"Point"}

    def test_bad_resolution_raises(self, tmp_path):
        """A resolution outside 0-15 surfaces a caught error."""
        out = str(tmp_path / "h3.geojson")
        with pytest.raises(_CAUGHT):
            run(["to-h3", _POINTS, out, "--resolution", "99"])


class TestH3Bin:
    """`pyramids vector h3-bin`."""

    def test_bins_points_into_hexagons(self, tmp_path):
        """Each occupied cell is one polygon and the counts total the input points."""
        out = str(tmp_path / "bin.geojson")
        assert run(["h3-bin", _POINTS, out, "--resolution", "7"]) == 0
        written = FeatureCollection.read_file(out)
        assert int(written["count"].sum()) == 6
        assert _geom_types(out) == {"Polygon"}

    def test_bad_resolution_raises(self, tmp_path):
        """A resolution outside 0-15 surfaces a caught error."""
        out = str(tmp_path / "bin.geojson")
        with pytest.raises(_CAUGHT):
            run(["h3-bin", _POINTS, out, "--resolution", "99"])


class TestSamplePoints:
    """`pyramids vector sample-points`."""

    def test_samples_within_polygons(self, tmp_path):
        """One multipoint sample is written per input polygon."""
        out = str(tmp_path / "samp.geojson")
        assert run(["sample-points", _POLYGONS, out, "--size", "5", "--seed", "1"]) == 0
        assert FeatureCollection.read_file(out).__len__() == 4
        assert _geom_types(out) == {"MultiPoint"}

    def test_refuses_existing(self, tmp_path):
        """An existing output without --overwrite is refused."""
        out = _touch(tmp_path / "samp.geojson")
        with pytest.raises(_CAUGHT):
            run(["sample-points", _POLYGONS, out, "--size", "5"])


class TestPolygonize:
    """`pyramids vector polygonize`."""

    def test_builds_polygons(self, tmp_path):
        """Polygonizing the four polygons' edges rebuilds four polygons."""
        out = str(tmp_path / "pz.geojson")
        assert run(["polygonize", _POLYGONS, out]) == 0
        assert FeatureCollection.read_file(out).__len__() == 4
        assert _geom_types(out) == {"Polygon"}

    def test_full_reports_leftovers(self, tmp_path, capsys):
        """--full still writes the polygons and reports the leftover counts."""
        out = str(tmp_path / "pzf.geojson")
        assert run(["polygonize", _POLYGONS, out, "--full"]) == 0
        assert "cuts=" in capsys.readouterr().out
        assert FeatureCollection.read_file(out).__len__() == 4

    def test_refuses_existing(self, tmp_path):
        """An existing output without --overwrite is refused."""
        out = _touch(tmp_path / "pz.geojson")
        with pytest.raises(_CAUGHT):
            run(["polygonize", _POLYGONS, out])


class TestFishnet:
    """`pyramids vector fishnet`."""

    def test_builds_grid(self, tmp_path):
        """A 10x10 extent at cell-size 2 yields a 5x5 = 25-cell grid of polygons."""
        out = str(tmp_path / "net.geojson")
        argv = ["fishnet", out, "--bounds", "0", "0", "10", "10", "--cell-size", "2"]
        assert run([*argv, "--epsg", "4326"]) == 0
        assert FeatureCollection.read_file(out).__len__() == 25
        assert _geom_types(out) == {"Polygon"}

    def test_bad_cell_size_raises(self, tmp_path):
        """A non-positive --cell-size surfaces a caught error."""
        out = str(tmp_path / "net.geojson")
        argv = ["fishnet", out, "--bounds", "0", "0", "10", "10", "--cell-size", "0"]
        with pytest.raises(_CAUGHT):
            run([*argv, "--epsg", "4326"])


class TestGeodesicArea:
    """`pyramids vector geodesic-area`."""

    def test_json_area_positive(self, capsys):
        """--json emits one positive ground area for the single geographic polygon."""
        assert run(["geodesic-area", _POLY_GEOG, "--json"]) == 0
        values = json.loads(capsys.readouterr().out)
        assert len(values) == 1
        assert values[0] > 1_000_000

    def test_bad_unit_raises(self):
        """An unrecognised --unit surfaces a caught error."""
        with pytest.raises(_CAUGHT):
            run(["geodesic-area", _POLY_GEOG, "--unit", "bananas"])


class TestGeodesicLength:
    """`pyramids vector geodesic-length`."""

    def test_plain_summary(self, capsys):
        """The plain form prints a labelled four-feature summary."""
        assert run(["geodesic-length", _POLYGONS, "--unit", "km"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("geodesic_length:")
        assert "count=4" in out

    def test_bad_unit_raises(self):
        """An unrecognised --unit surfaces a caught error."""
        with pytest.raises(_CAUGHT):
            run(["geodesic-length", _POLYGONS, "--unit", "zz"])


class TestGeodesicDistance:
    """`pyramids vector geodesic-distance`."""

    def test_distance_to_self_is_zero(self, capsys):
        """Each point's geodesic distance to itself is zero."""
        argv = ["geodesic-distance", _POINTS, "--other", _POINTS, "--json"]
        assert run(argv) == 0
        values = json.loads(capsys.readouterr().out)
        assert len(values) == 6
        assert max(values) == 0.0

    def test_non_point_raises(self):
        """A non-point source has no geodesic distance and surfaces a caught error."""
        with pytest.raises(_CAUGHT):
            run(["geodesic-distance", _POLYGONS, "--other", _POLYGONS])


class TestDistance:
    """`pyramids vector distance`."""

    def test_distance_to_self_is_zero(self, capsys):
        """Each geometry's planar distance to itself is zero."""
        assert run(["distance", _POINTS, "--other", _POINTS, "--json"]) == 0
        values = json.loads(capsys.readouterr().out)
        assert len(values) == 6
        assert max(values) == 0.0

    def test_missing_other_raises(self, tmp_path):
        """A missing --other file surfaces a caught error."""
        with pytest.raises(_CAUGHT):
            run(["distance", _POINTS, "--other", str(tmp_path / "nope.geojson")])


class TestHausdorffDistance:
    """`pyramids vector hausdorff-distance`."""

    def test_distance_to_self_is_zero(self, capsys):
        """The Hausdorff distance of each geometry to itself is zero."""
        assert run(["hausdorff-distance", _POINTS, "--other", _POINTS, "--json"]) == 0
        values = json.loads(capsys.readouterr().out)
        assert max(values) == 0.0

    def test_missing_other_raises(self, tmp_path):
        """A missing --other file surfaces a caught error."""
        argv = ["hausdorff-distance", _POINTS, "--other", str(tmp_path / "x.geojson")]
        with pytest.raises(_CAUGHT):
            run(argv)


class TestFrechetDistance:
    """`pyramids vector frechet-distance`."""

    def test_distance_to_self_is_zero(self, capsys):
        """The Frechet distance of each geometry to itself is zero."""
        assert run(["frechet-distance", _POINTS, "--other", _POINTS, "--json"]) == 0
        values = json.loads(capsys.readouterr().out)
        assert max(values) == 0.0

    def test_missing_other_raises(self, tmp_path):
        """A missing --other file surfaces a caught error."""
        argv = ["frechet-distance", _POINTS, "--other", str(tmp_path / "x.geojson")]
        with pytest.raises(_CAUGHT):
            run(argv)


class TestMinimumClearance:
    """`pyramids vector minimum-clearance`."""

    def test_json_clearance_positive(self, capsys):
        """--json emits one positive clearance per polygon."""
        assert run(["minimum-clearance", _POLYGONS, "--json"]) == 0
        values = json.loads(capsys.readouterr().out)
        assert len(values) == 4
        assert min(values) > 0.0

    def test_missing_file_raises(self, tmp_path):
        """A missing source file surfaces a caught error."""
        with pytest.raises(_CAUGHT):
            run(["minimum-clearance", str(tmp_path / "nope.geojson")])


class TestHilbertDistance:
    """`pyramids vector hilbert-distance`."""

    def test_json_indices(self, capsys):
        """--json emits one Hilbert-curve index per feature."""
        assert run(["hilbert-distance", _POINTS, "--level", "10", "--json"]) == 0
        values = json.loads(capsys.readouterr().out)
        assert len(values) == 6
        assert all(v >= 0 for v in values)

    def test_missing_file_raises(self, tmp_path):
        """A missing source file surfaces a caught error."""
        with pytest.raises(_CAUGHT):
            run(["hilbert-distance", str(tmp_path / "nope.geojson")])


class TestGetCoordinates:
    """`pyramids vector get-coordinates`."""

    def test_writes_csv(self, tmp_path):
        """The vertex table is written as a CSV with x/y columns and six rows."""
        out = str(tmp_path / "coords.csv")
        assert run(["get-coordinates", _POINTS, out]) == 0
        text = Path(out).read_text(encoding="utf-8")
        assert text.splitlines()[0] == "x,y"
        assert len(text.splitlines()) == 7

    def test_json_dict(self, capsys):
        """--json emits the coordinate table as a column dict."""
        assert run(["get-coordinates", _POINTS, "ignored.csv", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"x", "y"}
        assert len(payload["x"]) == 6

    def test_refuses_existing_csv(self, tmp_path):
        """A CSV that already exists is refused when not emitting JSON."""
        out = _touch(tmp_path / "coords.csv")
        with pytest.raises(_CAUGHT):
            run(["get-coordinates", _POINTS, out])
