"""Tests for the overview / windowed-read `Dataset` CLI verbs.

Covers `recreate-overviews`, `read-overview-array`, `read-part`, `preview`,
`point`, and `read-tile` (wired in via ``add_dataset_commands``). Each command
is dispatched through :func:`pyramids.cli.main` so the real error contract
(clean exit 1 for the mapped exceptions) is exercised.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture
def raster_with_overviews(tmp_path) -> str:
    """A 16x16 GeoTIFF carrying overviews at decimation factors 2 and 4.

    Returns:
        str: Path to the raster written with internal overviews.
    """
    path = str(tmp_path / "ovr.tif")
    Dataset.from_array(
        np.arange(256, dtype="float32").reshape(16, 16),
        geo_ref=GeoReference(top_left_corner=(0, 16), cell_size=1.0, epsg=4326),
    ).to_file(path)
    ds = Dataset.read_file(path, read_only=False)
    ds.create_overviews(overview_levels=[2, 4])
    ds.raster.FlushCache()
    del ds
    return path


class TestRecreateOverviews:
    """`pyramids recreate-overviews` regenerates existing overviews in place."""

    def test_happy_path(self, raster_with_overviews):
        """Regenerating succeeds and the overviews survive the rewrite."""
        assert main(["raster", "recreate-overviews", raster_with_overviews]) == 0
        assert all(
            count > 0
            for count in Dataset.read_file(raster_with_overviews).overview_count
        )

    def test_missing_input_errors(self, tmp_path):
        """A missing input exits 1."""
        assert main(["raster", "recreate-overviews", str(tmp_path / "nope.tif")]) == 1


class TestReadOverviewArray:
    """`pyramids read-overview-array` reads one overview level."""

    def test_writes_decimated_raster(self, raster_with_overviews, tmp_path):
        """Writing level 0 (factor 2) produces an 8x8 raster from the 16x16 source."""
        out = str(tmp_path / "level0.tif")
        assert main(["raster", "read-overview-array", raster_with_overviews, out]) == 0
        result = Dataset.read_file(out)
        assert (result.rows, result.columns) == (8, 8)

    def test_json_stats(self, raster_with_overviews, capsys):
        """--json prints the level's shape and stats without writing a file."""
        assert (
            main(["raster", "read-overview-array", raster_with_overviews, "--json"])
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["shape"] == [8, 8]
        assert payload["max"] is not None

    def test_no_overviews_errors(self, cli_raster):
        """A raster with no overviews exits 1 (not an 'unexpected failure')."""
        assert main(["raster", "read-overview-array", cli_raster, "--json"]) == 1

    def test_requires_output_or_json(self, raster_with_overviews, capsys):
        """Giving neither an output path nor --json exits 1 with a clear message."""
        assert main(["raster", "read-overview-array", raster_with_overviews]) == 1
        assert "OUTPUT path" in capsys.readouterr().err

    def test_output_and_json_together_errors(
        self, raster_with_overviews, tmp_path, capsys
    ):
        """Passing both an OUTPUT path and --json exits 1 instead of silently skipping the write."""
        out = str(tmp_path / "both.tif")
        rc = main(
            ["raster", "read-overview-array", raster_with_overviews, out, "--json"]
        )
        err = capsys.readouterr().err
        assert rc == 1, "output + --json is contradictory and must exit 1"
        assert "not both" in err, f"error should explain the conflict, got: {err!r}"
        assert not os.path.exists(out), (
            "no file should be written when the combination is rejected"
        )


class TestReadPart:
    """`pyramids read-part` reads a geographic window to a raster."""

    def test_writes_window_raster(self, cli_raster, tmp_path):
        """A 4x4 map window of the 8x8 source writes a smaller raster."""
        out = str(tmp_path / "window.tif")
        rc = main(
            ["raster", "read-part", cli_raster, out, "--bbox", "2", "2", "6", "6"]
        )
        assert rc == 0
        result = Dataset.read_file(out)
        assert result.rows < 8, "window should have fewer rows than the source"
        assert result.columns < 8, "window should have fewer columns than the source"

    def test_out_of_bounds_errors(self, cli_raster, tmp_path):
        """A window disjoint from the raster exits 1."""
        out = str(tmp_path / "oob.tif")
        rc = main(
            [
                "raster",
                "read-part",
                cli_raster,
                out,
                "--bbox",
                "100",
                "100",
                "110",
                "110",
            ]
        )
        assert rc == 1

    def test_overwrite_guard(self, cli_raster, tmp_path):
        """Writing over an existing output without --overwrite exits 1."""
        out = str(tmp_path / "w.tif")
        assert (
            main(["raster", "read-part", cli_raster, out, "--bbox", "2", "2", "6", "6"])
            == 0
        )
        rc = main(
            ["raster", "read-part", cli_raster, out, "--bbox", "2", "2", "6", "6"]
        )
        assert rc == 1

    def test_gsd_decimates_window(self, cli_raster, tmp_path):
        """--gsd sets the target cell size for the window read."""
        out = str(tmp_path / "gsd.tif")
        rc = main(
            [
                "raster",
                "read-part",
                cli_raster,
                out,
                "--bbox",
                "0",
                "0",
                "8",
                "8",
                "--gsd",
                "2",
            ]
        )
        assert rc == 0
        result = Dataset.read_file(out)
        assert max(result.rows, result.columns) <= 8

    def test_dst_width_height(self, cli_raster, tmp_path):
        """--dst-width and --dst-height force the output pixel dimensions."""
        out = str(tmp_path / "dst.tif")
        rc = main(
            [
                "raster",
                "read-part",
                cli_raster,
                out,
                "--bbox",
                "0",
                "0",
                "8",
                "8",
                "--dst-width",
                "4",
                "--dst-height",
                "4",
            ]
        )
        assert rc == 0
        result = Dataset.read_file(out)
        assert (result.rows, result.columns) == (4, 4)

    def test_out_of_range_band_errors(self, cli_raster, tmp_path):
        """A band index past the raster's band count exits 1."""
        out = str(tmp_path / "band.tif")
        rc = main(
            [
                "raster",
                "read-part",
                cli_raster,
                out,
                "--bbox",
                "2",
                "2",
                "6",
                "6",
                "--band",
                "5",
            ]
        )
        assert rc == 1


class TestPreview:
    """`pyramids preview` writes a whole-image decimated thumbnail."""

    def test_writes_thumbnail(self, cli_raster, tmp_path):
        """--max-size 4 decimates the 8x8 source to a 4x4 thumbnail."""
        out = str(tmp_path / "thumb.tif")
        assert main(["raster", "preview", cli_raster, out, "--max-size", "4"]) == 0
        result = Dataset.read_file(out)
        assert max(result.rows, result.columns) <= 4

    def test_missing_input_errors(self, tmp_path):
        """A missing input exits 1."""
        out = str(tmp_path / "thumb.tif")
        assert main(["raster", "preview", str(tmp_path / "nope.tif"), out]) == 1


class TestPoint:
    """`pyramids point` samples band value(s) at a map coordinate."""

    def test_samples_value(self, cli_raster, capsys):
        """Sampling the top-left cell centre returns the ramp's first value (0)."""
        rc = main(
            ["raster", "point", cli_raster, "0.5", "7.5", "--band", "0", "--json"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["values"] == [0.0]

    def test_out_of_bounds_errors(self, cli_raster):
        """A point outside the raster extent exits 1."""
        assert main(["raster", "point", cli_raster, "100", "100"]) == 1

    def test_samples_value_plain(self, cli_raster, capsys):
        """Without --json the sampled value is printed as a plain coordinate line."""
        rc = main(["raster", "point", cli_raster, "0.5", "7.5", "--band", "0"])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert "0.5,7.5:" in captured.out, "plain branch must print the coordinate"


class TestReadTile:
    """`pyramids read-tile` reads a Web-Mercator XYZ tile to an EPSG:3857 raster."""

    def test_writes_tile_raster(self, cli_raster, tmp_path):
        """The zoom-0 world tile intersects the raster and writes a 256x256 raster."""
        out = str(tmp_path / "tile.tif")
        assert main(["raster", "read-tile", cli_raster, out, "0", "0", "0"]) == 0
        result = Dataset.read_file(out)
        assert (result.rows, result.columns) == (256, 256)
        assert result.epsg == 3857

    def test_missing_input_errors(self, tmp_path):
        """A missing input exits 1."""
        out = str(tmp_path / "tile.tif")
        assert (
            main(
                ["raster", "read-tile", str(tmp_path / "nope.tif"), out, "0", "0", "0"]
            )
            == 1
        )
