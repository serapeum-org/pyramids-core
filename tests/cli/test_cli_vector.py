"""Tests for the ``pyramids vector`` command group (M-1)."""

from __future__ import annotations

import json

import pytest

from pyramids.cli import main
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


class TestVectorInfo:
    """`pyramids vector info`."""

    def test_json_payload(self, cli_points, capsys):
        """--json emits feature count, geometry type(s), CRS, columns, bounds."""
        assert main(["vector", "info", cli_points, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["features"] == 5
        assert payload["epsg"] == 4326
        assert "Point" in payload["geometry_types"]
        assert "value" in payload["columns"]

    def test_plain_lists_keys(self, cli_points, capsys):
        """The human form prints key: value lines."""
        assert main(["vector", "info", cli_points]) == 0
        assert "features: 5" in capsys.readouterr().out

    def test_missing_file_one_line_error(self, tmp_path, capsys):
        """A missing path exits 1 with a one-line error."""
        rc = main(["vector", "info", str(tmp_path / "nope.geojson")])
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "Traceback" not in err


class TestVectorBounds:
    """`pyramids vector bounds`."""

    def test_native_bounds(self, cli_points, capsys):
        """The native bounds print space-separated."""
        assert main(["vector", "bounds", cli_points]) == 0
        assert capsys.readouterr().out.split() == ["0.0", "0.0", "4.0", "4.0"]

    def test_reprojected_json(self, cli_points, capsys):
        """--crs reprojects the bounds; --json wraps them."""
        assert main(["vector", "bounds", cli_points, "--crs", "3857", "--json"]) == 0
        bounds = json.loads(capsys.readouterr().out)["bounds"]
        assert bounds[2] > 400_000, "4 deg of longitude is ~445 km east in EPSG:3857"


class TestVectorWarp:
    """`pyramids vector warp`."""

    def test_reproject(self, cli_points, tmp_path):
        """--crs reprojects and writes the output."""
        out = str(tmp_path / "w.geojson")
        assert main(["vector", "warp", cli_points, out, "--crs", "3857"]) == 0
        assert FeatureCollection.read_file(out).epsg == 3857

    def test_crsless_source_errors(self, cli_crsless_vector, tmp_path, capsys):
        """A CRS-less source is refused with a clear message."""
        rc = main(
            [
                "vector",
                "warp",
                cli_crsless_vector,
                str(tmp_path / "o.geojson"),
                "--crs",
                "3857",
            ]
        )
        assert rc == 1
        assert "no CRS" in capsys.readouterr().err

    def test_refuses_existing(self, cli_points, tmp_path):
        """A second write without --overwrite is refused."""
        out = str(tmp_path / "e.geojson")
        assert main(["vector", "warp", cli_points, out, "--crs", "3857"]) == 0
        assert main(["vector", "warp", cli_points, out, "--crs", "3857"]) == 1


class TestVectorConvert:
    """`pyramids vector convert`."""

    def test_geojson_to_gpkg(self, cli_points, tmp_path):
        """A GeoJSON is re-saved into a GeoPackage (driver inferred from extension)."""
        out = str(tmp_path / "v.gpkg")
        assert main(["vector", "convert", cli_points, out]) == 0
        assert len(FeatureCollection.read_file(out)) == 5

    def test_unknown_extension_errors(self, cli_points, tmp_path, capsys):
        """An unknown extension with no --driver exits 1 with a clear error."""
        rc = main(["vector", "convert", cli_points, str(tmp_path / "o.weird")])
        assert rc == 1
        assert "cannot infer a vector driver" in capsys.readouterr().err


class TestVectorClip:
    """`pyramids vector clip`."""

    def test_bbox_subset(self, cli_points, tmp_path):
        """--bbox keeps only the features intersecting the window."""
        out = str(tmp_path / "c.geojson")
        assert (
            main(["vector", "clip", cli_points, out, "--bbox", "-1", "-1", "1", "1"])
            == 0
        )
        assert len(FeatureCollection.read_file(out)) == 1, (
            "only the (0,0) point is inside"
        )

    def test_mask_clip(self, cli_points, tmp_path):
        """--mask clips to the mask polygon."""
        mask = str(tmp_path / "mask.geojson")
        FeatureCollection.from_bbox((-1.0, -1.0, 1.0, 1.0), epsg=4326).to_file(mask)
        out = str(tmp_path / "cm.geojson")
        assert main(["vector", "clip", cli_points, out, "--mask", mask]) == 0
        assert len(FeatureCollection.read_file(out)) == 1

    def test_bbox_and_mask_mutually_exclusive(self, cli_points, tmp_path):
        """Passing both --bbox and --mask is rejected by argparse."""
        with pytest.raises(SystemExit):
            main(
                [
                    "vector",
                    "clip",
                    cli_points,
                    str(tmp_path / "x.geojson"),
                    "--bbox",
                    "0",
                    "0",
                    "1",
                    "1",
                    "--mask",
                    "m.geojson",
                ]
            )
