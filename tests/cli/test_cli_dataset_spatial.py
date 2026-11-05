"""Tests for the raster spatial / analysis CLI group (`cli_dataset_spatial`).

Each command is a thin wrapper over a `Dataset` call; these pin the wiring:
exit codes, the observable effect (an output raster on the right grid/CRS, or a
correct profile / table), and the one-line-error contract for expected user
mistakes (missing input, overwrite without `--overwrite`, a CRS mismatch on a
two-raster op). Commands are driven through `pyramids.cli.main` exactly as the
console script runs them.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture
def ramp(tmp_path) -> str:
    """An 8x8 float32 ramp (values 0..63) on EPSG:4326, nodata -9999.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "ramp.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def fine_grid(tmp_path) -> str:
    """A 16x16 reference grid on EPSG:4326 (half the ramp's cell size).

    Returns:
        str: Path to the written reference raster.
    """
    path = str(tmp_path / "fine.tif")
    Dataset.from_array(
        np.zeros((16, 16), dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=0.5, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def twos(tmp_path) -> str:
    """An 8x8 raster of 2.0 aligned with ``ramp`` (same grid/CRS).

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "twos.tif")
    Dataset.from_array(
        np.full((8, 8), 2.0, dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def classes(tmp_path) -> str:
    """An 8x8 class raster aligned with ``ramp``: rows 0-3 -> 0, rows 4-7 -> 1.

    Returns:
        str: Path to the written raster.
    """
    arr = np.array([[row // 4] * 8 for row in range(8)], dtype="int32")
    path = str(tmp_path / "classes.tif")
    Dataset.from_array(
        arr,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def ramp_3857(tmp_path) -> str:
    """An 8x8 raster on EPSG:3857 — a deliberate CRS mismatch against ``ramp``.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "ramp_3857.tif")
    Dataset.from_array(
        np.ones((8, 8), dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=3857),
    ).to_file(path)
    return path


@pytest.fixture
def global_360(tmp_path) -> str:
    """A 1x360 global raster in the 0..360 longitude frame.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "global.tif")
    Dataset.from_array(
        np.arange(360, dtype="float32").reshape(1, 360),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0.0, 0.5), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


def _assert_one_line_error(rc: int, err: str) -> None:
    """Assert a clean one-line CLI error: exit 1, ``error:`` prefix, no traceback."""
    assert rc == 1, "expected exit code 1"
    assert err.startswith("error: "), f"unexpected stderr: {err}"
    assert "Traceback" not in err, "tracebacks must not leak to users"


class TestAlign:
    """`pyramids align`."""

    def test_resamples_onto_reference_grid(self, ramp, fine_grid, tmp_path, capsys):
        """The output adopts the reference's dimensions, cell size, and CRS."""
        out = str(tmp_path / "aligned.tif")
        assert main(["align", ramp, fine_grid, out]) == 0, "align must exit 0"
        aligned = Dataset.read_file(out)
        assert (aligned.rows, aligned.columns) == (16, 16), "wrong dimensions"
        assert aligned.epsg == 4326, "wrong CRS"
        assert aligned.geotransform[1] == pytest.approx(0.5), "wrong cell size"

    def test_missing_input_one_line_error(self, fine_grid, tmp_path, capsys):
        """A missing input raster exits 1 with a one-line error."""
        rc = main(
            ["align", str(tmp_path / "nope.tif"), fine_grid, str(tmp_path / "o.tif")]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestOverlay:
    """`pyramids overlay`."""

    def test_groups_values_by_class(self, ramp, classes, capsys):
        """--json maps each class to the base values that fall in it."""
        assert main(["overlay", ramp, classes, "--json"]) == 0, "overlay must exit 0"
        payload = json.loads(capsys.readouterr().out)
        assert sorted(payload) == ["0", "1"], f"wrong class keys: {list(payload)}"
        assert len(payload["0"]) == 32, "class 0 should hold the top four rows"
        assert len(payload["1"]) == 32, "class 1 should hold the bottom four rows"
        assert payload["0"] == pytest.approx(list(range(32))), "class 0 values wrong"

    def test_crs_mismatch_one_line_error(self, ramp, ramp_3857, capsys):
        """Two rasters on different CRSes exit 1 before any grouping."""
        rc = main(["overlay", ramp, ramp_3857])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestCombine:
    """`pyramids combine`."""

    def test_add_two_rasters(self, ramp, twos, tmp_path, capsys):
        """--op add sums the matching cells onto the shared grid."""
        out = str(tmp_path / "sum.tif")
        assert main(["combine", ramp, twos, out, "--op", "add"]) == 0, "combine exit 0"
        result = Dataset.read_file(out)
        assert (result.rows, result.columns) == (8, 8), "grid must be preserved"
        arr = result.read_array()
        assert float(arr.min()) == pytest.approx(2.0), "0 + 2 wrong"
        assert float(arr.max()) == pytest.approx(65.0), "63 + 2 wrong"

    def test_crs_mismatch_one_line_error(self, ramp, ramp_3857, tmp_path, capsys):
        """Combining rasters on different CRSes exits 1 with a clean message."""
        rc = main(["combine", ramp, ramp_3857, str(tmp_path / "o.tif"), "--op", "add"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestWarpedView:
    """`pyramids warped-view`."""

    def test_writes_reprojected_raster(self, ramp, tmp_path, capsys):
        """The materialised view is in the requested target CRS."""
        out = str(tmp_path / "wv.tif")
        assert main(["warped-view", ramp, out, "--crs", "EPSG:3857"]) == 0, "exit 0"
        assert Dataset.read_file(out).epsg == 3857, "output CRS wrong"

    def test_overwrite_without_flag_one_line_error(self, ramp, tmp_path, capsys):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "exists.tif")
        assert main(["warped-view", ramp, out, "--crs", "EPSG:3857"]) == 0
        capsys.readouterr()
        rc = main(["warped-view", ramp, out, "--crs", "EPSG:3857"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestWrapLongitude:
    """`pyramids wrap-longitude`."""

    def test_shifts_to_minus180_frame(self, global_360, tmp_path, capsys):
        """A global 0..360 grid lands on -180 after the column roll."""
        out = str(tmp_path / "wrapped.tif")
        assert main(["wrap-longitude", global_360, out]) == 0, "exit 0"
        wrapped = Dataset.read_file(out)
        assert wrapped.top_left_corner[0] == pytest.approx(-180.0), "wrong origin"
        assert (wrapped.rows, wrapped.columns) == (1, 360), "shape must be preserved"

    def test_regional_grid_one_line_error(self, ramp, tmp_path, capsys):
        """A non-global (regional) grid is refused with a one-line error."""
        rc = main(["wrap-longitude", ramp, str(tmp_path / "o.tif")])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestTranslate:
    """`pyramids translate`."""

    def test_srcwin_subsets(self, ramp, tmp_path, capsys):
        """--src-win extracts the pixel subwindow."""
        out = str(tmp_path / "sub.tif")
        assert main(["translate", ramp, out, "--src-win", "0", "0", "4", "4"]) == 0
        sub = Dataset.read_file(out)
        assert (sub.rows, sub.columns) == (4, 4), "subwindow size wrong"

    def test_missing_input_one_line_error(self, tmp_path, capsys):
        """A missing input raster exits 1 with a one-line error."""
        rc = main(["translate", str(tmp_path / "nope.tif"), str(tmp_path / "o.tif")])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestTransect:
    """`pyramids transect`."""

    def test_csv_profile_along_top_row(self, ramp, tmp_path, capsys):
        """Sampling the top row yields the row's values with rising distance."""
        out = str(tmp_path / "profile.csv")
        rc = main(["transect", ramp, out, "--line", "0.5,7.5;7.5,7.5", "--n", "8"])
        assert rc == 0, "transect must exit 0"
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["value"] for row in rows] == [str(float(v)) for v in range(8)]
        assert float(rows[0]["distance"]) == pytest.approx(0.0), "first distance wrong"
        assert float(rows[-1]["distance"]) > 0.0, "distance must rise"

    def test_missing_input_one_line_error(self, tmp_path, capsys):
        """A missing input raster exits 1 with a one-line error."""
        rc = main(
            [
                "transect",
                str(tmp_path / "nope.tif"),
                str(tmp_path / "o.csv"),
                "--line",
                "0,0;1,1",
                "--n",
                "2",
            ]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestExtract:
    """`pyramids extract`."""

    def test_csv_values(self, ramp, tmp_path, capsys):
        """Every cell value is written on one band row (none excluded here)."""
        out = str(tmp_path / "values.csv")
        assert main(["extract", ramp, out]) == 0, "extract must exit 0"
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 1, "a single-band raster writes one row"
        assert rows[0][0] == "band_1", "row must be labelled by band"
        assert len(rows[0]) == 1 + 64, "all 64 cell values must be present"

    def test_exclude_value_json(self, ramp, capsys):
        """--exclude-value drops the matching cells; --json prints the result."""
        assert (
            main(["extract", ramp, "ignored.csv", "--exclude-value", "0", "--json"])
            == 0
        )
        values = json.loads(capsys.readouterr().out)["values"]
        assert len(values) == 1, "single band -> one row"
        assert 0.0 not in values[0], "excluded value must be gone"

    def test_overwrite_without_flag_one_line_error(self, ramp, tmp_path, capsys):
        """An existing CSV output without --overwrite exits 1."""
        out = str(tmp_path / "exists.csv")
        assert main(["extract", ramp, out]) == 0
        capsys.readouterr()
        rc = main(["extract", ramp, out])
        _assert_one_line_error(rc, capsys.readouterr().err)
