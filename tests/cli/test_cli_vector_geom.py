"""Tests for the unary geometry-operation commands of the ``pyramids vector`` group.

The commands are exercised through a local parser built from
:func:`pyramids.cli_vector_geom.add_vector_geom_commands`, so each handler is
called directly (``ns.func(ns)``) rather than through :func:`pyramids.cli.main`.
Handlers therefore raise on error; the happy-path tests write a fresh output and
re-read it to assert an observable effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pyramids.base._errors import _PyramidsError
from pyramids.cli_vector_geom import add_vector_geom_commands
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

_DATA = Path(__file__).resolve().parents[1] / "data"
_GEOM = _DATA / "geometries"
POINTS = str(_GEOM / "points.geojson")
POLYGONS = str(_GEOM / "polygons.geojson")
MULTIPOINTS = str(_GEOM / "multi-points.geojson")
MULTIPOLYGONS = str(_GEOM / "multi-polygons.geojson")

CAUGHT = (ValueError, TypeError, OSError, RuntimeError, _PyramidsError)

_parser = argparse.ArgumentParser()
_vsub = _parser.add_subparsers(dest="command", required=True)
add_vector_geom_commands(_vsub)


def run(argv: list[str]) -> int:
    """Parse ``argv`` with the local parser and invoke the resolved handler.

    Args:
        argv: The command-line tokens (without the program name).

    Returns:
        int: The handler's exit code.
    """
    namespace = _parser.parse_args(argv)
    return namespace.func(namespace)


def _reread(path: str) -> FeatureCollection:
    """Read a written output back for assertions.

    Args:
        path: The output vector path.

    Returns:
        FeatureCollection: The re-read collection.
    """
    return FeatureCollection.read_file(path)


def _types(collection: FeatureCollection) -> list[str]:
    """Return the sorted distinct geometry-type names of ``collection``.

    Args:
        collection: The collection to inspect.

    Returns:
        list[str]: Unique geometry-type names.
    """
    return sorted({str(t) for t in collection.geometry.geom_type.dropna().unique()})


def _missing(tmp_path: Path) -> str:
    """Return a path to a source file that does not exist.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        str: A non-existent ``.geojson`` path.
    """
    return str(tmp_path / "does_not_exist.geojson")


class TestBuffer:
    """`pyramids vector buffer`."""

    def test_points_become_polygons(self, tmp_path):
        """Buffering points yields polygons, feature count preserved."""
        out = str(tmp_path / "buf.geojson")
        assert run(["buffer", POINTS, out, "--distance", "100"]) == 0
        result = _reread(out)
        assert _types(result) == ["Polygon"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "buffer",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--distance",
                    "100",
                ]
            )


class TestSimplify:
    """`pyramids vector simplify`."""

    def test_preserves_features(self, tmp_path):
        """Simplifying keeps every polygon feature."""
        out = str(tmp_path / "simp.geojson")
        assert run(["simplify", POLYGONS, out, "--tolerance", "10"]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "simplify",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--tolerance",
                    "10",
                ]
            )


class TestSimplifyCoverage:
    """`pyramids vector simplify-coverage`."""

    def test_preserves_features(self, tmp_path):
        """Coverage simplification keeps every polygon feature."""
        out = str(tmp_path / "scov.geojson")
        assert run(["simplify-coverage", POLYGONS, out, "--tolerance", "10"]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "simplify-coverage",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--tolerance",
                    "10",
                ]
            )


class TestMakeValid:
    """`pyramids vector make-valid`."""

    def test_output_is_valid(self, tmp_path):
        """The output geometries are all valid."""
        out = str(tmp_path / "valid.geojson")
        assert run(["make-valid", POLYGONS, out]) == 0
        result = _reread(out)
        assert bool(result.geometry.is_valid.all())
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["make-valid", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestNormalize:
    """`pyramids vector normalize`."""

    def test_preserves_features(self, tmp_path):
        """Normalizing keeps every polygon feature."""
        out = str(tmp_path / "norm.geojson")
        assert run(["normalize", POLYGONS, out]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["normalize", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestReverse:
    """`pyramids vector reverse`."""

    def test_preserves_features(self, tmp_path):
        """Reversing winding keeps every polygon feature."""
        out = str(tmp_path / "rev.geojson")
        assert run(["reverse", POLYGONS, out]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["reverse", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestForce2d:
    """`pyramids vector force-2d`."""

    def test_drops_z(self, tmp_path):
        """A polygon layer stays 2-D (no Z present)."""
        out = str(tmp_path / "f2d.geojson")
        assert run(["force-2d", POLYGONS, out]) == 0
        result = _reread(out)
        assert not bool(result.geometry.has_z.any())
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["force-2d", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestForce3d:
    """`pyramids vector force-3d`."""

    def test_adds_z(self, tmp_path):
        """Forcing 3-D marks every geometry as having a Z coordinate."""
        out = str(tmp_path / "f3d.geojson")
        assert run(["force-3d", POINTS, out, "--z", "5"]) == 0
        result = _reread(out)
        assert bool(result.geometry.has_z.all())
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["force-3d", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestSegmentize:
    """`pyramids vector segmentize`."""

    def test_preserves_features(self, tmp_path):
        """Densifying edges keeps every polygon feature."""
        out = str(tmp_path / "seg.geojson")
        assert run(["segmentize", POLYGONS, out, "--max-length", "10"]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "segmentize",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--max-length",
                    "10",
                ]
            )


class TestRemoveRepeatedPoints:
    """`pyramids vector remove-repeated-points`."""

    def test_preserves_features(self, tmp_path):
        """Removing repeated points keeps every polygon feature."""
        out = str(tmp_path / "rrp.geojson")
        assert run(["remove-repeated-points", POLYGONS, out]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "remove-repeated-points",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                ]
            )


class TestOffsetCurve:
    """`pyramids vector offset-curve`."""

    def test_polygon_boundary_becomes_line(self, tmp_path):
        """Offsetting produces line geometries."""
        out = str(tmp_path / "off.geojson")
        assert run(["offset-curve", POLYGONS, out, "--distance", "5"]) == 0
        result = _reread(out)
        assert _types(result) == ["LineString"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "offset-curve",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--distance",
                    "5",
                ]
            )


class TestRepresentativePoint:
    """`pyramids vector representative-point`."""

    def test_polygons_become_points(self, tmp_path):
        """Each polygon reduces to a representative point."""
        out = str(tmp_path / "rep.geojson")
        assert run(["representative-point", POLYGONS, out]) == 0
        result = _reread(out)
        assert _types(result) == ["Point"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "representative-point",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                ]
            )


class TestConcaveHull:
    """`pyramids vector concave-hull`."""

    def test_preserves_features(self, tmp_path):
        """The per-geometry concave hull keeps every feature."""
        out = str(tmp_path / "hull.geojson")
        assert run(["concave-hull", MULTIPOINTS, out, "--ratio", "0.3"]) == 0
        result = _reread(out)
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["concave-hull", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestMinimumBoundingCircle:
    """`pyramids vector minimum-bounding-circle`."""

    def test_produces_polygons(self, tmp_path):
        """Each enclosing circle is a polygon."""
        out = str(tmp_path / "mbc.geojson")
        assert run(["minimum-bounding-circle", POLYGONS, out]) == 0
        result = _reread(out)
        assert _types(result) == ["Polygon"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "minimum-bounding-circle",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                ]
            )


class TestMinimumRotatedRectangle:
    """`pyramids vector minimum-rotated-rectangle`."""

    def test_produces_polygons(self, tmp_path):
        """Each enclosing rectangle is a polygon."""
        out = str(tmp_path / "mrr.geojson")
        assert run(["minimum-rotated-rectangle", POLYGONS, out]) == 0
        result = _reread(out)
        assert _types(result) == ["Polygon"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "minimum-rotated-rectangle",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                ]
            )


class TestMaximumInscribedCircle:
    """`pyramids vector maximum-inscribed-circle`."""

    def test_produces_radius_lines(self, tmp_path):
        """The inscribed-circle result is a center-to-boundary line per geometry."""
        out = str(tmp_path / "mic.geojson")
        assert run(["maximum-inscribed-circle", POLYGONS, out]) == 0
        result = _reread(out)
        assert _types(result) == ["LineString"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "maximum-inscribed-circle",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                ]
            )


class TestAffineTransform:
    """`pyramids vector affine-transform`."""

    def test_translation_shifts_bounds(self, tmp_path):
        """A pure-translation matrix shifts the total bounds by the offsets."""
        out = str(tmp_path / "aff.geojson")
        before = FeatureCollection.read_file(POLYGONS).total_bounds
        argv = [
            "affine-transform",
            POLYGONS,
            out,
            "--matrix",
            "1",
            "0",
            "0",
            "1",
            "100",
            "200",
        ]
        assert run(argv) == 0
        after = _reread(out).total_bounds
        assert after[0] == pytest.approx(before[0] + 100, abs=1e-3)
        assert after[1] == pytest.approx(before[1] + 200, abs=1e-3)

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "affine-transform",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--matrix",
                    "1",
                    "0",
                    "0",
                    "1",
                    "0",
                    "0",
                ]
            )


class TestRotate:
    """`pyramids vector rotate`."""

    def test_preserves_features(self, tmp_path):
        """Rotating keeps every polygon feature."""
        out = str(tmp_path / "rot.geojson")
        assert run(["rotate", POLYGONS, out, "--angle", "90"]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "rotate",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--angle",
                    "90",
                ]
            )


class TestScale:
    """`pyramids vector scale`."""

    def test_enlarges_area(self, tmp_path):
        """Scaling up grows the total polygon area."""
        out = str(tmp_path / "sca.geojson")
        before = float(FeatureCollection.read_file(POLYGONS).geometry.area.sum())
        assert run(["scale", POLYGONS, out, "--xfact", "2", "--yfact", "2"]) == 0
        after = float(_reread(out).geometry.area.sum())
        assert after > before

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["scale", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestTranslate:
    """`pyramids vector translate`."""

    def test_shifts_bounds(self, tmp_path):
        """Translating shifts the total bounds by the offsets."""
        out = str(tmp_path / "tra.geojson")
        before = FeatureCollection.read_file(POLYGONS).total_bounds
        assert run(["translate", POLYGONS, out, "--xoff", "100", "--yoff", "200"]) == 0
        after = _reread(out).total_bounds
        assert after[0] == pytest.approx(before[0] + 100, abs=1e-3)
        assert after[1] == pytest.approx(before[1] + 200, abs=1e-3)

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["translate", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestSnap:
    """`pyramids vector snap`."""

    def test_preserves_features(self, tmp_path):
        """Snapping a layer to itself keeps every feature."""
        out = str(tmp_path / "snap.geojson")
        assert (
            run(["snap", POLYGONS, out, "--other", POLYGONS, "--tolerance", "1"]) == 0
        )
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_other_raises(self, tmp_path):
        """A missing ``--other`` layer surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "snap",
                    POLYGONS,
                    str(tmp_path / "o.geojson"),
                    "--other",
                    _missing(tmp_path),
                    "--tolerance",
                    "1",
                ]
            )


class TestBuildArea:
    """`pyramids vector build-area`."""

    def test_produces_polygons(self, tmp_path):
        """Building areas from a multipolygon yields polygon geometries."""
        out = str(tmp_path / "area.geojson")
        assert run(["build-area", MULTIPOLYGONS, out]) == 0
        result = _reread(out)
        assert _types(result) == ["Polygon"]
        assert len(result) >= 1

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(["build-area", _missing(tmp_path), str(tmp_path / "o.geojson")])


class TestExtractUniquePoints:
    """`pyramids vector extract-unique-points`."""

    def test_polygons_become_multipoints(self, tmp_path):
        """Each polygon's vertices become a multipoint."""
        out = str(tmp_path / "eup.geojson")
        assert run(["extract-unique-points", POLYGONS, out]) == 0
        result = _reread(out)
        assert _types(result) == ["MultiPoint"]
        assert len(result) == 10

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "extract-unique-points",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                ]
            )


class TestSetPrecision:
    """`pyramids vector set-precision`."""

    def test_preserves_features(self, tmp_path):
        """Snapping coordinates to a grid keeps every polygon feature."""
        out = str(tmp_path / "prec.geojson")
        assert run(["set-precision", POLYGONS, out, "--grid-size", "1"]) == 0
        result = _reread(out)
        assert len(result) == 10
        assert _types(result) == ["Polygon"]

    def test_missing_source_raises(self, tmp_path):
        """A missing source surfaces as a caught error type."""
        with pytest.raises(CAUGHT):
            run(
                [
                    "set-precision",
                    _missing(tmp_path),
                    str(tmp_path / "o.geojson"),
                    "--grid-size",
                    "1",
                ]
            )
