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

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

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


@pytest.fixture
def ramp_points(tmp_path) -> str:
    """Two EPSG:4326 points over the first two top-row cells of ``ramp``.

    ``Point(0.5, 7.5)`` falls in cell ``(row 0, col 0)`` (value ``0``) and
    ``Point(1.5, 7.5)`` in cell ``(row 0, col 1)`` (value ``1``).

    Returns:
        str: Path to the written point GeoJSON.
    """
    path = str(tmp_path / "points.geojson")
    gpd.GeoDataFrame(geometry=[Point(0.5, 7.5), Point(1.5, 7.5)], crs=4326).to_file(
        path, driver="GeoJSON"
    )
    return path


def _assert_one_line_error(rc: int, err: str) -> None:
    """Assert a clean one-line CLI error: exit 1, ``error:`` prefix, no traceback."""
    assert rc == 1, "expected exit code 1"
    assert err.startswith("error: "), f"unexpected stderr: {err}"
    assert "Traceback" not in err, "tracebacks must not leak to users"
    assert "unexpected failure" not in err, (
        f"expected-error path leaked into the internal-bug branch: {err!r}"
    )


class TestAlign:
    """`pyramids align`."""

    def test_resamples_onto_reference_grid(self, ramp, fine_grid, tmp_path, capsys):
        """The output adopts the reference's dimensions, cell size, and CRS."""
        out = str(tmp_path / "aligned.tif")
        assert main(["raster", "align", ramp, fine_grid, out]) == 0, "align must exit 0"
        aligned = Dataset.read_file(out)
        assert (aligned.rows, aligned.columns) == (16, 16), "wrong dimensions"
        assert aligned.epsg == 4326, "wrong CRS"
        assert aligned.geotransform[1] == pytest.approx(0.5), "wrong cell size"

    def test_missing_input_one_line_error(self, fine_grid, tmp_path, capsys):
        """A missing input raster exits 1 with a one-line error."""
        rc = main(
            [
                "raster",
                "align",
                str(tmp_path / "nope.tif"),
                fine_grid,
                str(tmp_path / "o.tif"),
            ]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestOverlay:
    """`pyramids overlay`."""

    def test_groups_values_by_class(self, ramp, classes, capsys):
        """--json maps each class to the base values that fall in it."""
        assert main(["raster", "overlay", ramp, classes, "--json"]) == 0, (
            "overlay must exit 0"
        )
        payload = json.loads(capsys.readouterr().out)
        assert sorted(payload) == ["0", "1"], f"wrong class keys: {list(payload)}"
        assert len(payload["0"]) == 32, "class 0 should hold the top four rows"
        assert len(payload["1"]) == 32, "class 1 should hold the bottom four rows"
        assert payload["0"] == pytest.approx(list(range(32))), "class 0 values wrong"

    def test_text_output_prints_per_class_counts(self, ramp, classes, capsys):
        """Without --json, each class prints a human-readable value count."""
        assert main(["raster", "overlay", ramp, classes]) == 0, "overlay must exit 0"
        out = capsys.readouterr().out
        assert "class 0: 32 value(s)" in out, f"missing class-0 count line: {out!r}"
        assert "class 1: 32 value(s)" in out, f"missing class-1 count line: {out!r}"

    def test_crs_mismatch_one_line_error(self, ramp, ramp_3857, capsys):
        """Two rasters on different CRSes exit 1 before any grouping."""
        rc = main(["raster", "overlay", ramp, ramp_3857])
        _assert_one_line_error(rc, capsys.readouterr().err)

    def test_out_of_range_band_clean_error(self, ramp, classes, capsys):
        """An out-of-range --band is a clean user error, not 'unexpected failure'."""
        rc = main(["raster", "overlay", ramp, classes, "--band", "99"])
        err = capsys.readouterr().err
        _assert_one_line_error(rc, err)
        assert "unexpected failure" not in err, (
            f"band error must be mapped cleanly: {err!r}"
        )


class TestCombine:
    """`pyramids combine`."""

    def test_add_two_rasters(self, ramp, twos, tmp_path, capsys):
        """--op add sums the matching cells onto the shared grid."""
        out = str(tmp_path / "sum.tif")
        assert main(["raster", "combine", ramp, twos, out, "--op", "add"]) == 0, (
            "combine exit 0"
        )
        result = Dataset.read_file(out)
        assert (result.rows, result.columns) == (8, 8), "grid must be preserved"
        arr = result.read_array()
        assert float(arr.min()) == pytest.approx(2.0), "0 + 2 wrong"
        assert float(arr.max()) == pytest.approx(65.0), "63 + 2 wrong"

    def test_crs_mismatch_one_line_error(self, ramp, ramp_3857, tmp_path, capsys):
        """Combining rasters on different CRSes exits 1 with a clean message."""
        rc = main(
            [
                "raster",
                "combine",
                ramp,
                ramp_3857,
                str(tmp_path / "o.tif"),
                "--op",
                "add",
            ]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestWarpedView:
    """`pyramids warped-view`."""

    def test_writes_reprojected_raster(self, ramp, tmp_path, capsys):
        """The materialised view is in the requested target CRS."""
        out = str(tmp_path / "wv.tif")
        assert main(["raster", "warped-view", ramp, out, "--crs", "EPSG:3857"]) == 0, (
            "exit 0"
        )
        assert Dataset.read_file(out).epsg == 3857, "output CRS wrong"

    def test_overwrite_without_flag_one_line_error(self, ramp, tmp_path, capsys):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "exists.tif")
        assert main(["raster", "warped-view", ramp, out, "--crs", "EPSG:3857"]) == 0
        capsys.readouterr()
        rc = main(["raster", "warped-view", ramp, out, "--crs", "EPSG:3857"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestWrapLongitude:
    """`pyramids wrap-longitude`."""

    def test_shifts_to_minus180_frame(self, global_360, tmp_path, capsys):
        """A global 0..360 grid lands on -180 after the column roll."""
        out = str(tmp_path / "wrapped.tif")
        assert main(["raster", "wrap-longitude", global_360, out]) == 0, "exit 0"
        wrapped = Dataset.read_file(out)
        assert wrapped.top_left_corner[0] == pytest.approx(-180.0), "wrong origin"
        assert (wrapped.rows, wrapped.columns) == (1, 360), "shape must be preserved"

    def test_regional_grid_one_line_error(self, ramp, tmp_path, capsys):
        """A non-global (regional) grid is refused with a one-line error."""
        rc = main(["raster", "wrap-longitude", ramp, str(tmp_path / "o.tif")])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestTranslate:
    """`pyramids translate`."""

    def test_srcwin_subsets(self, ramp, tmp_path, capsys):
        """--src-win extracts the pixel subwindow."""
        out = str(tmp_path / "sub.tif")
        assert (
            main(["raster", "translate", ramp, out, "--src-win", "0", "0", "4", "4"])
            == 0
        )
        sub = Dataset.read_file(out)
        assert (sub.rows, sub.columns) == (4, 4), "subwindow size wrong"

    def test_proj_win_subsets(self, ramp, tmp_path, capsys):
        """--proj-win extracts the [ulx, uly, lrx, lry] projected subwindow."""
        out = str(tmp_path / "pw.tif")
        rc = main(["raster", "translate", ramp, out, "--proj-win", "0", "8", "4", "4"])
        assert rc == 0, "translate --proj-win must exit 0"
        sub = Dataset.read_file(out)
        assert (sub.rows, sub.columns) == (4, 4), "projected subwindow size wrong"

    def test_output_type_retypes(self, ramp, tmp_path, capsys):
        """--output-type casts the stored band to the requested numpy dtype."""
        out = str(tmp_path / "typed.tif")
        rc = main(["raster", "translate", ramp, out, "--output-type", "int16"])
        assert rc == 0, "translate --output-type must exit 0"
        typed = Dataset.read_file(out)
        assert typed.read_array().dtype == np.int16, "output dtype not int16"

    def test_nodata_sets_sentinel(self, ramp, tmp_path, capsys):
        """--nodata sets the output no-data value."""
        out = str(tmp_path / "nd.tif")
        rc = main(["raster", "translate", ramp, out, "--nodata", "-1"])
        assert rc == 0, "translate --nodata must exit 0"
        result = Dataset.read_file(out)
        assert float(result.no_data_value[0]) == pytest.approx(-1.0), "nodata unset"

    def test_unscale_runs(self, ramp, tmp_path, capsys):
        """--unscale forwards unscale=True and still writes an output raster."""
        out = str(tmp_path / "unscaled.tif")
        rc = main(["raster", "translate", ramp, out, "--unscale"])
        assert rc == 0, "translate --unscale must exit 0"
        unscaled = Dataset.read_file(out)
        assert (unscaled.rows, unscaled.columns) == (8, 8), "grid must be preserved"

    def test_bands_selects(self, ramp, tmp_path, capsys):
        """--bands re-selects the (1-based) bands to keep."""
        out = str(tmp_path / "band1.tif")
        rc = main(["raster", "translate", ramp, out, "--bands", "1"])
        assert rc == 0, "translate --bands must exit 0"
        assert Dataset.read_file(out).band_count == 1, "wrong band count"

    def test_resampling_forwarded(self, ramp, tmp_path, capsys):
        """--resampling forwards a resampleAlg and still writes the output."""
        out = str(tmp_path / "resampled.tif")
        rc = main(["raster", "translate", ramp, out, "--resampling", "nearest"])
        assert rc == 0, "translate --resampling must exit 0"
        resampled = Dataset.read_file(out)
        assert (resampled.rows, resampled.columns) == (8, 8), "grid must be preserved"

    def test_scale_rescales(self, ramp, tmp_path, capsys):
        """--scale with four values rescales the band from src to dst range."""
        out = str(tmp_path / "scaled.tif")
        rc = main(["raster", "translate", ramp, out, "--scale", "0", "63", "0", "100"])
        assert rc == 0, "translate --scale must exit 0"
        arr = Dataset.read_file(out).read_array()
        assert float(arr.max()) == pytest.approx(100.0), "63 -> 100 rescale wrong"
        assert float(arr.min()) == pytest.approx(0.0), "0 -> 0 rescale wrong"

    def test_scale_wrong_length_one_line_error(self, ramp, tmp_path, capsys):
        """--scale with 3 values is rejected with a clean one-line error."""
        out = str(tmp_path / "bad_scale.tif")
        rc = main(["raster", "translate", ramp, out, "--scale", "0", "1", "2"])
        _assert_one_line_error(rc, capsys.readouterr().err)

    def test_missing_input_one_line_error(self, tmp_path, capsys):
        """A missing input raster exits 1 with a one-line error."""
        rc = main(
            ["raster", "translate", str(tmp_path / "nope.tif"), str(tmp_path / "o.tif")]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestTransect:
    """`pyramids transect`."""

    def test_csv_profile_along_top_row(self, ramp, tmp_path, capsys):
        """Sampling the top row yields the row's values with rising distance."""
        out = str(tmp_path / "profile.csv")
        rc = main(
            ["raster", "transect", ramp, out, "--line", "0.5,7.5;7.5,7.5", "--n", "8"]
        )
        assert rc == 0, "transect must exit 0"
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["value"] for row in rows] == [str(float(v)) for v in range(8)]
        # The first densified point coincides with the line's start, so its
        # cumulative distance is 0. macOS PROJ returns tiny geodesic noise
        # (~1e-11) rather than a hard 0.0, and pytest.approx against an exact-0
        # target only honours its absolute tolerance (default 1e-12 — too tight
        # for that noise), so pin an explicit, generous abs tolerance.
        assert float(rows[0]["distance"]) == pytest.approx(0.0, abs=1e-6), (
            "first distance wrong"
        )
        assert float(rows[-1]["distance"]) > 0.0, "distance must rise"

    def test_json_prints_profile(self, ramp, capsys):
        """--json prints the distance/value/x/y profile and skips the output guard."""
        rc = main(
            [
                "raster",
                "transect",
                ramp,
                "ignored.geojson",
                "--line",
                "0.5,7.5;7.5,7.5",
                "--n",
                "8",
                "--json",
            ]
        )
        assert rc == 0, "transect --json must exit 0"
        records = json.loads(capsys.readouterr().out)
        assert len(records) == 8, "one record per densified position"
        assert [rec["value"] for rec in records] == [float(v) for v in range(8)]
        assert set(records[0]) == {"distance", "value", "x", "y"}, "wrong keys"

    def test_vector_output_writes_points(self, ramp, tmp_path, capsys):
        """A non-CSV output extension writes a vector file via the resolved driver."""
        out = tmp_path / "profile.geojson"
        rc = main(
            [
                "raster",
                "transect",
                ramp,
                str(out),
                "--line",
                "0.5,7.5;7.5,7.5",
                "--n",
                "8",
            ]
        )
        assert rc == 0, "transect vector output must exit 0"
        assert out.exists(), "vector profile file was not written"
        written = gpd.read_file(str(out))
        assert len(written) == 8, "one feature per densified position"

    def test_line_single_vertex_one_line_error(self, ramp, tmp_path, capsys):
        """A --line with fewer than two vertices exits 1 with a clean error."""
        rc = main(
            [
                "raster",
                "transect",
                ramp,
                str(tmp_path / "o.csv"),
                "--line",
                "0,0",
                "--n",
                "2",
            ]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)

    def test_line_bad_vertex_shape_one_line_error(self, ramp, tmp_path, capsys):
        """A vertex that is not an 'x,y' pair exits 1 with a clean error."""
        rc = main(
            [
                "raster",
                "transect",
                ramp,
                str(tmp_path / "o.csv"),
                "--line",
                "0,0;1",
                "--n",
                "2",
            ]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)

    def test_line_non_numeric_one_line_error(self, ramp, tmp_path, capsys):
        """A vertex with non-numeric coordinates exits 1 with a clean error."""
        rc = main(
            [
                "raster",
                "transect",
                ramp,
                str(tmp_path / "o.csv"),
                "--line",
                "0,0;a,b",
                "--n",
                "2",
            ]
        )
        _assert_one_line_error(rc, capsys.readouterr().err)

    def test_missing_input_one_line_error(self, tmp_path, capsys):
        """A missing input raster exits 1 with a one-line error."""
        rc = main(
            [
                "raster",
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
        assert main(["raster", "extract", ramp, out]) == 0, "extract must exit 0"
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 1, "a single-band raster writes one row"
        assert rows[0][0] == "band_1", "row must be labelled by band"
        assert len(rows[0]) == 1 + 64, "all 64 cell values must be present"

    def test_exclude_value_json(self, ramp, capsys):
        """--exclude-value drops the matching cells; --json prints the result."""
        assert (
            main(
                [
                    "raster",
                    "extract",
                    ramp,
                    "ignored.csv",
                    "--exclude-value",
                    "0",
                    "--json",
                ]
            )
            == 0
        )
        values = json.loads(capsys.readouterr().out)["values"]
        assert len(values) == 1, "single band -> one row"
        assert 0.0 not in values[0], "excluded value must be gone"

    def test_mask_reads_values_at_points(self, ramp, ramp_points, tmp_path, capsys):
        """--mask reads one value per point instead of every cell."""
        out = str(tmp_path / "at_points.csv")
        assert main(["raster", "extract", ramp, out, "--mask", ramp_points]) == 0
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 1, "single band -> one row"
        assert rows[0][0] == "band_1", "row must be labelled by band"
        assert [float(value) for value in rows[0][1:]] == pytest.approx([0.0, 1.0])

    def test_overwrite_without_flag_one_line_error(self, ramp, tmp_path, capsys):
        """An existing CSV output without --overwrite exits 1."""
        out = str(tmp_path / "exists.csv")
        assert main(["raster", "extract", ramp, out]) == 0
        capsys.readouterr()
        rc = main(["raster", "extract", ramp, out])
        _assert_one_line_error(rc, capsys.readouterr().err)

    def test_out_of_range_band_clean_error(self, ramp, tmp_path, capsys):
        """An out-of-range --band is a clean user error, not 'unexpected failure'."""
        rc = main(["raster", "extract", ramp, str(tmp_path / "o.csv"), "--band", "99"])
        err = capsys.readouterr().err
        _assert_one_line_error(rc, err)
        assert "unexpected failure" not in err, (
            f"band error must be mapped cleanly: {err!r}"
        )
