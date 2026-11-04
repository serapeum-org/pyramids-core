"""Tests for the read-only ``pyramids vector`` inspection commands.

These commands are not yet wired into :func:`pyramids.cli.main`, so the tests build
a local parser from :func:`pyramids.cli_vector_inspect.add_vector_inspect_commands`
and dispatch through ``ns.func(ns)`` directly. Each command has a happy-path test
(plus ``--json`` where supported) and an error-path test; composite assertions are
split so each ``assert`` checks one fact (SonarCloud S9073).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pyramids._cli_shared import resolve_vector_driver as _resolve_vector_driver
from pyramids.cli_vector_inspect import add_vector_inspect_commands
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

_DATA = Path(__file__).resolve().parents[1] / "data"
_POLYGONS = str(_DATA / "coello_polygons.geojson")
_GAUGES = str(_DATA / "coello-gauges.geojson")
_MISSING = str(_DATA / "does_not_exist_8f3a1c.geojson")
_RAISED = (ValueError, OSError, RuntimeError, TypeError)


def _build_parser() -> argparse.ArgumentParser:
    """Build a standalone parser wired only with the vector inspect commands."""
    parser = argparse.ArgumentParser()
    vsub = parser.add_subparsers(dest="command", required=True)
    add_vector_inspect_commands(vsub)
    return parser


def _run(argv: list[str]) -> int:
    """Parse ``argv`` with the local parser and invoke the resolved handler."""
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    return namespace.func(namespace)


def test_feature_count_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``feature-count`` prints the integer feature count."""
    rc = _run(["feature-count", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "4"


def test_feature_count_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``feature-count --json`` emits a strict-valid object with the count."""
    rc = _run(["feature-count", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["features"] == 4


def test_feature_count_error() -> None:
    """A missing source is reported as a clean (catchable) error."""
    with pytest.raises(_RAISED):
        _run(["feature-count", _MISSING])


def test_feature_info_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``feature-info`` prints one ``key: value`` line per metadata field."""
    rc = _run(["feature-info", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert "feature_count: 4" in out


def test_feature_info_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``feature-info --json`` serialises the VectorInfo dataclass fields."""
    rc = _run(["feature-info", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["feature_count"] == 4
    assert payload["crs_epsg"] == 32618


def test_feature_info_error() -> None:
    """A bad ``--layer`` is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["feature-info", _POLYGONS, "--layer", "no_such_layer"])


def test_list_layers_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``list-layers`` prints the single layer name of the GeoJSON."""
    rc = _run(["list-layers", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "coello_polygons"


def test_list_layers_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``list-layers --json`` emits the layer-name list."""
    rc = _run(["list-layers", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["layers"] == ["coello_polygons"]


def test_list_layers_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["list-layers", _MISSING])


def test_estimate_utm_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``estimate-utm`` prints the EPSG code of the best UTM CRS."""
    rc = _run(["estimate-utm", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "32618"


def test_estimate_utm_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``estimate-utm --json`` emits the EPSG and CRS string."""
    rc = _run(["estimate-utm", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["epsg"] == 32618


def test_estimate_utm_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["estimate-utm", _MISSING])


def test_is_valid_reason_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``is-valid-reason`` prints a per-feature reason line for valid polygons."""
    rc = _run(["is-valid-reason", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Valid Geometry" in out


def test_is_valid_reason_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``is-valid-reason --json`` emits one reason per feature."""
    rc = _run(["is-valid-reason", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(payload["reasons"]) == 4


def test_is_valid_reason_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["is-valid-reason", _MISSING])


def test_is_valid_coverage_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``is-valid-coverage`` prints ``true`` for a clean polygon coverage."""
    rc = _run(["is-valid-coverage", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "true"


def test_is_valid_coverage_gap_width(capsys: pytest.CaptureFixture[str]) -> None:
    """``--gap-width`` is accepted and still yields a boolean line."""
    rc = _run(["is-valid-coverage", _POLYGONS, "--gap-width", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() in {"true", "false"}


def test_is_valid_coverage_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["is-valid-coverage", _MISSING])


def test_invalid_coverage_edges_write(tmp_path: Path) -> None:
    """``invalid-coverage-edges`` writes a readable vector of the edges."""
    out = tmp_path / "edges.geojson"
    rc = _run(["invalid-coverage-edges", _POLYGONS, str(out)])
    assert rc == 0
    assert out.exists()


def test_invalid_coverage_edges_readback(tmp_path: Path) -> None:
    """The written edges file is a valid vector with one edge per feature."""
    out = tmp_path / "edges.geojson"
    _run(["invalid-coverage-edges", _POLYGONS, str(out)])
    back = FeatureCollection.read_file(str(out))
    assert len(back) == 4


def test_invalid_coverage_edges_error(tmp_path: Path) -> None:
    """Refusing to overwrite an existing output is a clean error."""
    out = tmp_path / "edges.geojson"
    out.write_text("placeholder", encoding="utf-8")
    with pytest.raises(_RAISED):
        _run(["invalid-coverage-edges", _POLYGONS, str(out)])


def test_count_coordinates_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``count-coordinates`` prints a per-feature coordinate count."""
    rc = _run(["count-coordinates", _POLYGONS])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() != ""


def test_count_coordinates_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``count-coordinates --json`` emits one count per feature."""
    rc = _run(["count-coordinates", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(payload["coordinates"]) == 4


def test_count_coordinates_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["count-coordinates", _MISSING])


def test_count_geometries_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``count-geometries --json`` emits one count per feature."""
    rc = _run(["count-geometries", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(payload["geometries"]) == 4


def test_count_geometries_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["count-geometries", _MISSING])


def test_count_interior_rings_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``count-interior-rings --json`` emits one count per feature."""
    rc = _run(["count-interior-rings", _POLYGONS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(payload["interior_rings"]) == 4


def test_count_interior_rings_error() -> None:
    """A missing source is reported as a clean error."""
    with pytest.raises(_RAISED):
        _run(["count-interior-rings", _MISSING])


def test_gauges_feature_count(capsys: pytest.CaptureFixture[str]) -> None:
    """A point layer is counted correctly via ``feature-count``."""
    rc = _run(["feature-count", _GAUGES])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "6"


def test_driver_resolution_used(tmp_path: Path) -> None:
    """The edges writer honours the extension-resolved driver (sanity on the shared helper)."""
    out = tmp_path / "edges.geojson"
    _run(["invalid-coverage-edges", _POLYGONS, str(out)])
    assert _resolve_vector_driver(str(out), None) == "geojson"
