"""Tests for the raster inspection CLI commands (`pyramids.cli_dataset_inspect`).

These commands are registered on the top-level parser by
:func:`add_dataset_inspect_commands` (wired in via ``add_dataset_commands``), so
every test dispatches through the real :func:`pyramids.cli.main` — exercising its
error contract (expected errors exit ``1`` with a one-line message, never the
``unexpected failure`` branch). Each command has a happy-path test (asserting the
text / ``--json`` shape) and an error-path test (asserting exit code ``1``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def raster(tmp_path) -> str:
    """An 8x8 single-band float32 ramp on EPSG:4326 with nodata -9999."""
    path = str(tmp_path / "r.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def multiband(tmp_path) -> str:
    """A 2-band 4x4 float32 raster on EPSG:4326."""
    path = str(tmp_path / "mb.tif")
    Dataset.from_array(
        np.arange(32, dtype="float32").reshape(2, 4, 4),
        geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def other_grid(tmp_path) -> str:
    """A raster on a different grid (half the cell size) from ``raster``."""
    path = str(tmp_path / "other.tif")
    Dataset.from_array(
        np.ones((8, 8), dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=0.5, epsg=4326),
    ).to_file(path)
    return path


_MISSING = "does_not_exist_8f3a1c.tif"


class TestStats:
    """`pyramids stats` reports per-band min/max/mean/std."""

    def test_json(self, multiband, capsys):
        """--json emits one record per band with the four statistics."""
        assert main(["stats", multiband, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["stats"]) == 2
        first = payload["stats"][0]
        assert set(first) == {"band", "min", "max", "mean", "std"}
        assert isinstance(first["mean"], float)

    def test_text_single_band(self, raster, capsys):
        """A --band selection prints just that band's line."""
        assert main(["stats", raster, "--band", "0"]) == 0
        out = capsys.readouterr().out
        assert "min=" in out
        assert "std=" in out

    def test_bad_band_errors(self, raster, capsys):
        """An out-of-range band exits 1 without the 'unexpected failure' branch."""
        assert main(["stats", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestHistogram:
    """`pyramids histogram` buckets a band's values."""

    def test_json(self, raster, capsys):
        """--json emits counts and (low, high) ranges of the requested bin count."""
        assert main(["histogram", raster, "--bins", "4", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["counts"]) == 4
        assert len(payload["ranges"]) == 4
        assert all(len(edge) == 2 for edge in payload["ranges"])

    def test_text(self, raster, capsys):
        """The text form prints one line per bucket."""
        assert main(["histogram", raster, "--bins", "3"]) == 0
        assert capsys.readouterr().out.count("\n") == 3

    def test_bad_band_errors(self, raster, capsys):
        """A bad band is a clean error (IndexError remapped), not 'unexpected failure'."""
        assert main(["histogram", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestGetMetaData:
    """`pyramids get-meta-data` reads a GDAL metadata domain."""

    def test_json(self, raster, capsys):
        """--json emits the domain and its metadata mapping."""
        assert main(["get-meta-data", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["domain"] == ""
        assert isinstance(payload["metadata"], dict)

    def test_missing_file_errors(self, capsys):
        """A missing file exits 1."""
        assert main(["get-meta-data", _MISSING]) == 1


class TestAttributeTable:
    """`pyramids get/set-attribute-table` round-trips a raster attribute table."""

    def test_get_empty_json(self, raster, capsys):
        """A band with no table emits a null attribute_table."""
        assert main(["get-attribute-table", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["attribute_table"] is None

    def test_set_then_get_roundtrip(self, raster, tmp_path, capsys):
        """`set-attribute-table` persists a CSV table that `get` reads back."""
        csv = str(tmp_path / "rat.csv")
        pd.DataFrame({"Value": [1, 2], "Class": ["a", "b"]}).to_csv(csv, index=False)
        assert main(["set-attribute-table", raster, csv]) == 0
        capsys.readouterr()
        assert main(["get-attribute-table", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["attribute_table"] == [
            {"Value": 1, "Class": "a"},
            {"Value": 2, "Class": "b"},
        ]

    def test_get_bad_band_errors(self, raster, capsys):
        """A bad band on `get-attribute-table` exits 1, not 'unexpected failure'."""
        assert main(["get-attribute-table", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err

    def test_set_bad_table_extension_errors(self, raster, tmp_path, capsys):
        """An unsupported table extension is a clean error (exit 1)."""
        bad = str(tmp_path / "rat.txt")
        with open(bad, "w", encoding="utf8") as handle:
            handle.write("nope")
        assert main(["set-attribute-table", raster, bad]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestMaskFlags:
    """`pyramids mask-flags` decodes a band's GDAL mask flags."""

    def test_json(self, raster, capsys):
        """A no-data raster reports the nodata flag true and the others false."""
        assert main(["mask-flags", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["nodata"] is True
        assert payload["all_valid"] is False

    def test_bad_band_errors(self, raster, capsys):
        """A bad band exits 1 without the 'unexpected failure' branch."""
        assert main(["mask-flags", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestFootprint:
    """`pyramids footprint` extracts a band's covered extent as polygons."""

    def test_json(self, raster, capsys):
        """--json emits a GeoJSON FeatureCollection for the covered cells."""
        assert main(["footprint", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) >= 1

    def test_write_vector(self, raster, tmp_path, capsys):
        """An output path writes a readable vector file."""
        out = str(tmp_path / "fp.geojson")
        assert main(["footprint", raster, out]) == 0
        assert FeatureCollection.feature_count(out) >= 1

    def test_missing_file_errors(self, capsys):
        """A missing input exits 1."""
        assert main(["footprint", _MISSING, "--json"]) == 1


class TestCellArea:
    """`pyramids cell-area` summarises the per-cell ground area."""

    def test_json(self, raster, capsys):
        """--json emits the unit and the min/max/mean/total summary."""
        assert main(["cell-area", raster, "--unit", "km2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["unit"] == "km2"
        assert payload["total"] > 0

    def test_missing_file_errors(self, capsys):
        """A missing input exits 1 with a clean message."""
        assert main(["cell-area", _MISSING]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestDomainArea:
    """`pyramids domain-area` reports the ground area of a band's valid cells."""

    def test_json(self, raster, capsys):
        """--json emits the band, unit and area."""
        assert main(["domain-area", raster, "--unit", "km2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["unit"] == "km2"
        assert payload["area"] > 0

    def test_bad_band_errors(self, raster, capsys):
        """A bad band exits 1 with a clean message."""
        assert main(["domain-area", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestCountDomainCells:
    """`pyramids count-domain-cells` counts a band's valid cells."""

    def test_json(self, raster, capsys):
        """--json emits the integer count (64 for a gap-free 8x8 ramp)."""
        assert main(["count-domain-cells", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 64

    def test_bad_band_errors(self, raster, capsys):
        """A bad band exits 1 with a clean message."""
        assert main(["count-domain-cells", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestSameGrid:
    """`pyramids same-grid` compares two rasters' grids."""

    def test_identical_true(self, raster, capsys):
        """Two reads of the same raster share a grid."""
        assert main(["same-grid", raster, raster, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["same_grid"] is True

    def test_different_false(self, raster, other_grid, capsys):
        """A raster on a different grid is reported false (still exit 0)."""
        assert main(["same-grid", raster, other_grid, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["same_grid"] is False

    def test_missing_file_errors(self, raster, capsys):
        """A missing second raster exits 1."""
        assert main(["same-grid", raster, _MISSING]) == 1
