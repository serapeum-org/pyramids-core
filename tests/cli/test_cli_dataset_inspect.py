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
from osgeo import gdal

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
def nodata_only(tmp_path) -> str:
    """An 8x8 single-band raster whose every cell is the nodata sentinel -9999."""
    path = str(tmp_path / "nd.tif")
    Dataset.from_array(
        np.full((8, 8), -9999.0, dtype="float32"),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
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
        assert main(["raster", "stats", multiband, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["stats"]) == 2
        first = payload["stats"][0]
        assert set(first) == {"band", "min", "max", "mean", "std"}
        assert isinstance(first["mean"], float)

    def test_text_single_band(self, raster, capsys):
        """A --band selection prints just that band's line."""
        assert main(["raster", "stats", raster, "--band", "0"]) == 0
        out = capsys.readouterr().out
        assert "min=" in out
        assert "std=" in out

    def test_bad_band_errors(self, raster, capsys):
        """An out-of-range band exits 1 without the 'unexpected failure' branch."""
        assert main(["raster", "stats", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestHistogram:
    """`pyramids histogram` buckets a band's values."""

    def test_json(self, raster, capsys):
        """--json emits counts and (low, high) ranges of the requested bin count."""
        assert main(["raster", "histogram", raster, "--bins", "4", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["counts"]) == 4
        assert len(payload["ranges"]) == 4
        assert all(len(edge) == 2 for edge in payload["ranges"])

    def test_text(self, raster, capsys):
        """The text form prints one line per bucket."""
        assert main(["raster", "histogram", raster, "--bins", "3"]) == 0
        assert capsys.readouterr().out.count("\n") == 3

    def test_bad_band_errors(self, raster, capsys):
        """A bad band is a clean error (IndexError remapped), not 'unexpected failure'."""
        assert main(["raster", "histogram", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestGetMetaData:
    """`pyramids get-metadata` reads a GDAL metadata domain."""

    def test_json(self, raster, capsys):
        """--json emits the domain and its metadata mapping."""
        assert main(["raster", "get-metadata", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["domain"] == ""
        assert isinstance(payload["metadata"], dict)

    def test_text_dict(self, raster, capsys):
        """Without --json a mapping domain prints one ``key: value`` line per entry.

        Args:
            raster: Single-band raster fixture whose default domain carries
                ``AREA_OR_POINT``.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "get-metadata", raster]) == 0, (
            "default domain should succeed"
        )
        out = capsys.readouterr().out
        assert "AREA_OR_POINT: Area" in out, (
            "the dict domain should print key: value lines"
        )

    def test_text_xml_list(self, raster, capsys):
        """Without --json an ``xml:*`` domain prints its list items one per line.

        GDAL returns a list (not a mapping) for an ``xml:`` metadata domain, which
        exercises the ``else`` branch that iterates the sequence.

        Args:
            raster: Single-band raster fixture; an ``xml:TEST`` domain is injected.
            capsys: Pytest stdout/stderr capture fixture.
        """
        handle = gdal.Open(raster, gdal.GA_Update)
        handle.SetMetadata(["<x>hi</x>"], "xml:TEST")
        handle.FlushCache()
        handle = None
        assert main(["raster", "get-metadata", raster, "--domain", "xml:TEST"]) == 0, (
            "reading an xml domain should succeed"
        )
        assert capsys.readouterr().out.strip() == "<x>hi</x>", (
            "the list item should print verbatim"
        )

    def test_missing_file_errors(self, capsys):
        """A missing file exits 1."""
        assert main(["raster", "get-metadata", _MISSING]) == 1


class TestAttributeTable:
    """`pyramids get/set-attribute-table` round-trips a raster attribute table."""

    def test_get_empty_json(self, raster, capsys):
        """A band with no table emits a null attribute_table."""
        assert (
            main(["raster", "advanced", "get-attribute-table", raster, "--json"]) == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["attribute_table"] is None

    def test_set_then_get_roundtrip(self, raster, tmp_path, capsys):
        """`set-attribute-table` persists a CSV table that `get` reads back."""
        csv = str(tmp_path / "rat.csv")
        pd.DataFrame({"Value": [1, 2], "Class": ["a", "b"]}).to_csv(csv, index=False)
        assert main(["raster", "advanced", "set-attribute-table", raster, csv]) == 0
        capsys.readouterr()
        assert (
            main(["raster", "advanced", "get-attribute-table", raster, "--json"]) == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["attribute_table"] == [
            {"Value": 1, "Class": "a"},
            {"Value": 2, "Class": "b"},
        ]

    def test_get_empty_text(self, raster, capsys):
        """Without --json a band with no table prints a one-line notice.

        Args:
            raster: Single-band raster fixture that carries no attribute table.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "get-attribute-table", raster]) == 0, (
            "a missing table is not an error"
        )
        assert capsys.readouterr().out.strip() == "band 0 has no attribute table", (
            "the empty-table notice should name the band"
        )

    def test_get_text_with_records(self, raster, tmp_path, capsys):
        """Without --json each attribute-table row is printed as a record.

        Args:
            raster: Single-band raster fixture edited in place by ``set``.
            tmp_path: Pytest temporary directory for the CSV source.
            capsys: Pytest stdout/stderr capture fixture.
        """
        csv = str(tmp_path / "rat.csv")
        pd.DataFrame({"Value": [1, 2], "Class": ["a", "b"]}).to_csv(csv, index=False)
        assert main(["raster", "advanced", "set-attribute-table", raster, csv]) == 0, (
            "writing the table should succeed"
        )
        capsys.readouterr()
        assert main(["raster", "advanced", "get-attribute-table", raster]) == 0, (
            "reading the table should succeed"
        )
        out = capsys.readouterr().out
        assert "'Class': 'a'" in out, "each row should print as a record dict"
        assert out.count("\n") == 2, "one line per record"

    def test_set_from_json_table(self, raster, tmp_path, capsys):
        """`set-attribute-table` accepts a ``.json`` table source and persists it.

        Args:
            raster: Single-band raster fixture edited in place.
            tmp_path: Pytest temporary directory for the JSON source.
            capsys: Pytest stdout/stderr capture fixture.
        """
        table = str(tmp_path / "rat.json")
        pd.DataFrame({"Value": [1, 2], "Class": ["a", "b"]}).to_json(table)
        assert (
            main(["raster", "advanced", "set-attribute-table", raster, table]) == 0
        ), "a .json table should be accepted"
        capsys.readouterr()
        assert (
            main(["raster", "advanced", "get-attribute-table", raster, "--json"]) == 0
        ), "the table should read back"
        payload = json.loads(capsys.readouterr().out)
        assert payload["attribute_table"] == [
            {"Value": 1, "Class": "a"},
            {"Value": 2, "Class": "b"},
        ], "the JSON-sourced table should round-trip"

    def test_get_bad_band_errors(self, raster, capsys):
        """A bad band on `get-attribute-table` exits 1, not 'unexpected failure'."""
        assert (
            main(["raster", "advanced", "get-attribute-table", raster, "--band", "9"])
            == 1
        )
        assert "unexpected failure" not in capsys.readouterr().err

    def test_set_bad_table_extension_errors(self, raster, tmp_path, capsys):
        """An unsupported table extension is a clean error (exit 1)."""
        bad = str(tmp_path / "rat.txt")
        with open(bad, "w", encoding="utf8") as handle:
            handle.write("nope")
        assert main(["raster", "advanced", "set-attribute-table", raster, bad]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestMaskFlags:
    """`pyramids mask-flags` decodes a band's GDAL mask flags."""

    def test_json(self, raster, capsys):
        """A no-data raster reports the nodata flag true and the others false."""
        assert main(["raster", "advanced", "mask-flags", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["nodata"] is True
        assert payload["all_valid"] is False

    def test_text(self, raster, capsys):
        """Without --json each decoded flag prints as a ``key: value`` line.

        Args:
            raster: Single-band raster fixture with a nodata sentinel.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "mask-flags", raster]) == 0, (
            "decoding mask flags should succeed"
        )
        out = capsys.readouterr().out
        assert "nodata: True" in out, "the nodata flag line should be present"
        assert "all_valid: False" in out, "the all_valid flag line should be present"

    def test_bad_band_errors(self, raster, capsys):
        """A bad band exits 1 without the 'unexpected failure' branch."""
        assert main(["raster", "advanced", "mask-flags", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestFootprint:
    """`pyramids footprint` extracts a band's covered extent as polygons."""

    def test_json(self, raster, capsys):
        """--json emits a GeoJSON FeatureCollection for the covered cells."""
        assert main(["raster", "footprint", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) >= 1

    def test_write_vector(self, raster, tmp_path, capsys):
        """An output path writes a readable vector file."""
        out = str(tmp_path / "fp.geojson")
        assert main(["raster", "footprint", raster, out]) == 0
        assert FeatureCollection.feature_count(out) >= 1

    def test_text_summary(self, raster, capsys):
        """A geographic-CRS summary reports the polygon count and omits a bogus area.

        Args:
            raster: Single-band EPSG:4326 (geographic) raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "footprint", raster]) == 0, (
            "summarising the footprint should succeed"
        )
        captured = capsys.readouterr()
        assert captured.out.startswith("footprint:"), (
            "the summary line should be prefixed"
        )
        assert "polygon(s)" in captured.out, (
            "the summary should report the polygon count"
        )
        assert "total area" not in captured.out, (
            "a geographic CRS must not print a degrees-squared area"
        )
        assert "Geometry is in a geographic CRS" not in captured.err, (
            "the geopandas area warning must not fire for a geographic footprint"
        )

    def test_text_summary_projected(self, tmp_path, capsys):
        """A projected-CRS summary reports a total area labelled as native units."""
        path = str(tmp_path / "proj.tif")
        Dataset.from_array(
            np.arange(64, dtype="float32").reshape(8, 8),
            geo_ref=GeoReference(
                top_left_corner=(0, 8000), cell_size=1000.0, epsg=3857
            ),
        ).to_file(path)
        assert main(["raster", "footprint", path]) == 0, (
            "projected footprint summary must succeed"
        )
        out = capsys.readouterr().out
        assert "total area" in out, "a projected CRS should report a total area"
        assert "native CRS units" in out, (
            "the area must be labelled as native CRS units"
        )

    def test_all_nodata_json_empty(self, nodata_only, capsys):
        """--json on a fully no-data band emits an empty GeoJSON FeatureCollection.

        Args:
            nodata_only: Raster fixture whose single band is entirely no-data.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "footprint", nodata_only, "--json"]) == 0, (
            "an empty footprint is not an error"
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["type"] == "FeatureCollection", (
            "an empty collection is still emitted"
        )
        assert payload["features"] == [], "a no-data band yields no features"

    def test_all_nodata_text_notice(self, nodata_only, capsys):
        """Without --json a fully no-data band prints a no-covered-cells notice.

        Args:
            nodata_only: Raster fixture whose single band is entirely no-data.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "footprint", nodata_only]) == 0, (
            "an empty footprint is not an error"
        )
        assert "no covered cells" in capsys.readouterr().out, (
            "the notice should explain the empty result"
        )

    def test_missing_file_errors(self, capsys):
        """A missing input exits 1."""
        assert main(["raster", "footprint", _MISSING, "--json"]) == 1


class TestCellArea:
    """`pyramids cell-area` summarises the per-cell ground area."""

    def test_json(self, raster, capsys):
        """--json emits the unit and the min/max/mean/total summary."""
        assert main(["raster", "cell-area", raster, "--unit", "km2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["unit"] == "km2"
        assert payload["total"] > 0

    def test_text(self, raster, capsys):
        """Without --json each summary field prints as a ``key: value`` line.

        Args:
            raster: Single-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "cell-area", raster]) == 0, (
            "summarising cell area should succeed"
        )
        out = capsys.readouterr().out
        assert "unit: m2" in out, "the default unit should be reported"
        assert "total:" in out, "the full-grid total should be reported"

    def test_missing_file_errors(self, capsys):
        """A missing input exits 1 with a clean message."""
        assert main(["raster", "cell-area", _MISSING]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestDomainArea:
    """`pyramids domain-area` reports the ground area of a band's valid cells."""

    def test_json(self, raster, capsys):
        """--json emits the band, unit and area."""
        assert main(["raster", "domain-area", raster, "--unit", "km2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["unit"] == "km2"
        assert payload["area"] > 0

    def test_text(self, raster, capsys):
        """Without --json the area is printed with its unit.

        Args:
            raster: Single-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "domain-area", raster]) == 0, (
            "computing the domain area should succeed"
        )
        assert capsys.readouterr().out.strip().endswith("m2"), (
            "the unit should trail the area"
        )

    def test_bad_band_errors(self, raster, capsys):
        """A bad band exits 1 with a clean message."""
        assert main(["raster", "domain-area", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestCountDomainCells:
    """`pyramids count-domain-cells` counts a band's valid cells."""

    def test_json(self, raster, capsys):
        """--json emits the integer count (64 for a gap-free 8x8 ramp)."""
        assert main(["raster", "count-domain-cells", raster, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 64

    def test_text(self, raster, capsys):
        """Without --json the bare integer count is printed.

        Args:
            raster: Single-band gap-free 8x8 ramp fixture (64 valid cells).
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "count-domain-cells", raster]) == 0, (
            "counting valid cells should succeed"
        )
        assert capsys.readouterr().out.strip() == "64", (
            "every cell of the ramp is valid"
        )

    def test_bad_band_errors(self, raster, capsys):
        """A bad band exits 1 with a clean message."""
        assert main(["raster", "count-domain-cells", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err


class TestSameGrid:
    """`pyramids same-grid` compares two rasters' grids."""

    def test_identical_true(self, raster, capsys):
        """Two reads of the same raster share a grid."""
        assert main(["raster", "advanced", "same-grid", raster, raster, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["same_grid"] is True

    def test_different_false(self, raster, other_grid, capsys):
        """A raster on a different grid is reported false (still exit 0)."""
        assert (
            main(["raster", "advanced", "same-grid", raster, other_grid, "--json"]) == 0
        )
        assert json.loads(capsys.readouterr().out)["same_grid"] is False

    def test_text_true(self, raster, capsys):
        """Without --json two reads of the same raster print ``true``.

        Args:
            raster: Single-band raster fixture compared against itself.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "same-grid", raster, raster]) == 0, (
            "comparing identical grids should succeed"
        )
        assert capsys.readouterr().out.strip() == "true", "identical grids print true"

    def test_text_false(self, raster, other_grid, capsys):
        """Without --json mismatched grids print ``false``.

        Args:
            raster: Single-band raster fixture.
            other_grid: Raster fixture on a different grid.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "same-grid", raster, other_grid]) == 0, (
            "comparing different grids should succeed"
        )
        assert capsys.readouterr().out.strip() == "false", "different grids print false"

    def test_missing_file_errors(self, raster, capsys):
        """A missing second raster exits 1."""
        assert main(["raster", "advanced", "same-grid", raster, _MISSING]) == 1


class TestChecksum:
    """`pyramids checksum` reports per-band GDAL checksums."""

    def test_json_all_bands(self, multiband, capsys):
        """--json with no --band emits one checksum per band.

        Args:
            multiband: Two-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "checksum", multiband, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"checksums"}, "all-band JSON has only 'checksums'"
        assert len(payload["checksums"]) == 2, "one checksum per band"
        assert all(isinstance(value, int) for value in payload["checksums"])

    def test_json_single_band(self, multiband, capsys):
        """--json with --band emits that band's single checksum.

        Args:
            multiband: Two-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert (
            main(["raster", "advanced", "checksum", multiband, "--band", "0", "--json"])
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["band"] == 0
        assert isinstance(payload["checksum"], int)

    def test_text_all_bands(self, multiband, capsys):
        """The text form prints one ``band N: <checksum>`` line per band.

        Args:
            multiband: Two-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "checksum", multiband]) == 0
        out = capsys.readouterr().out
        assert out.count("\n") == 2, "one line per band"
        assert out.startswith("band 0: "), "text lists each band index"

    def test_text_single_band(self, raster, capsys):
        """A --band selection prints just that band's checksum as a bare integer.

        Args:
            raster: Single-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "checksum", raster, "--band", "0"]) == 0
        assert capsys.readouterr().out.strip().isdigit(), "a single band prints an int"

    def test_bad_band_errors(self, raster, capsys):
        """An out-of-range band is a clean error (IndexError remapped), not a crash.

        Args:
            raster: Single-band raster fixture.
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "checksum", raster, "--band", "9"]) == 1
        assert "unexpected failure" not in capsys.readouterr().err

    def test_missing_file_errors(self, capsys):
        """A missing input exits 1 without a traceback.

        Args:
            capsys: Pytest stdout/stderr capture fixture.
        """
        assert main(["raster", "advanced", "checksum", _MISSING]) == 1
        assert "Traceback" not in capsys.readouterr().err
