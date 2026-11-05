"""Smoke tests for the `pyramids` raster-analysis CLI commands.

Each command in :mod:`pyramids.cli_dataset_analysis` is a thin wrapper over a
`Dataset` analysis method; these tests pin the wiring: exit codes, the written
artifact (raster / vector / CSV) or the JSON shape, and the one-line-error
contract for expected user mistakes.
"""

from __future__ import annotations

import json
import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

_GEO = GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326)


def _assert_one_line_error(rc: int, err: str) -> None:
    """Assert a handler exited 1 with a clean one-line error (no traceback).

    Args:
        rc: The return code from :func:`pyramids.cli.main`.
        err: The captured stderr text.
    """
    assert rc == 1, f"expected exit 1, got {rc}"
    assert err.startswith("error: "), f"stderr not a clean error line: {err!r}"
    assert "Traceback" not in err, "a traceback leaked to the user"


@pytest.fixture
def ramp(tmp_path) -> str:
    """An 8x8 float32 ramp (0..63) GeoTIFF on EPSG:4326 with nodata -9999.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "ramp.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        no_data_value=-9999.0,
        geo_ref=_GEO,
    ).to_file(path)
    return path


@pytest.fixture
def gaps(tmp_path) -> str:
    """An 8x8 ramp with one no-data gap at cell (0, 0).

    Returns:
        str: Path to the written raster.
    """
    arr = np.arange(64, dtype="float32").reshape(8, 8)
    arr[0, 0] = -9999.0
    path = str(tmp_path / "gaps.tif")
    Dataset.from_array(arr, no_data_value=-9999.0, geo_ref=_GEO).to_file(path)
    return path


@pytest.fixture
def target(tmp_path) -> str:
    """An 8x8 int raster whose only non-zero (target) cell is (4, 4).

    Returns:
        str: Path to the written raster.
    """
    arr = np.zeros((8, 8), dtype="int32")
    arr[4, 4] = 1
    path = str(tmp_path / "target.tif")
    Dataset.from_array(arr, geo_ref=_GEO).to_file(path)
    return path


@pytest.fixture
def zones(tmp_path) -> str:
    """One polygon covering the raster's lower-left quadrant, on EPSG:4326.

    Returns:
        str: Path to the written GeoJSON.
    """
    fc = FeatureCollection(
        gpd.GeoDataFrame({"zone": ["a"]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:4326")
    )
    path = str(tmp_path / "zones.geojson")
    fc.to_file(path)
    return path


@pytest.fixture
def kelvin(tmp_path) -> str:
    """A 2x2 Kelvin raster with its band unit labelled ``K``.

    Returns:
        str: Path to the written raster.
    """
    ds = Dataset.from_array(
        np.array([[273.15, 283.15], [293.15, 303.15]]),
        geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
    )
    ds.band_units = ["K"]
    path = str(tmp_path / "kelvin.tif")
    ds.to_file(path)
    return path


@pytest.fixture
def pan_pair(tmp_path) -> tuple[str, str]:
    """A 3-band multispectral raster and a finer 1-band pan raster, co-located.

    Returns:
        tuple[str, str]: ``(multispectral_path, pan_path)``.
    """
    rng = np.random.default_rng(0)
    geo_ms = GeoReference(top_left_corner=(31.0, 30.0), cell_size=0.01, epsg=4326)
    geo_pan = GeoReference(top_left_corner=(31.0, 30.0), cell_size=0.0025, epsg=4326)
    ms_path = str(tmp_path / "ms.tif")
    pan_path = str(tmp_path / "pan.tif")
    Dataset.from_array(
        rng.integers(0, 4000, size=(3, 32, 32)).astype("uint16"),
        geo_ref=geo_ms,
        no_data_value=None,
    ).to_file(ms_path)
    Dataset.from_array(
        rng.integers(0, 4000, size=(128, 128)).astype("uint16"),
        geo_ref=geo_pan,
        no_data_value=None,
    ).to_file(pan_path)
    return ms_path, pan_path


class TestProximity:
    """`pyramids proximity`."""

    def test_distance_to_target(self, target, tmp_path, capsys):
        """The target cell reads 0 and a two-cell offset reads 2 in PIXEL units."""
        out = str(tmp_path / "prox.tif")
        rc = main(["proximity", target, out, "--distance-units", "PIXEL"])
        assert rc == 0, capsys.readouterr().err
        dist = Dataset.read_file(out).read_array()
        assert float(dist[4, 4]) == 0.0, "target cell must be distance 0"
        assert float(dist[4, 6]) == pytest.approx(2.0), "two cells away must be 2"

    def test_negative_max_distance_errors(self, target, tmp_path, capsys):
        """A negative --max-distance is a clean one-line error."""
        out = str(tmp_path / "prox.tif")
        rc = main(["proximity", target, out, "--max-distance", "-1"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestCluster:
    """`pyramids cluster`."""

    def test_labels_and_count_json(self, ramp, tmp_path, capsys):
        """A label raster is written and --json reports a positive cluster count."""
        out = str(tmp_path / "clust.tif")
        rc = main(["cluster", ramp, out, "--lower", "2", "--upper", "40", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0, "cluster must exit 0"
        assert payload["output"] == out, "JSON must name the output"
        assert payload["count"] >= 1, "there must be at least one cluster"
        assert Dataset.read_file(out).shape[-2:] == (8, 8), "labels keep the grid"

    def test_overwrite_without_flag_errors(self, ramp, tmp_path, capsys):
        """Writing over an existing file without --overwrite is refused."""
        out = str(tmp_path / "clust.tif")
        open(out, "w").close()
        rc = main(["cluster", ramp, out, "--lower", "2", "--upper", "40"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestZonalStats:
    """`pyramids zonal-stats`."""

    def test_json_records(self, ramp, zones, capsys):
        """--json emits one record per zone with the requested stats."""
        rc = main(["zonal-stats", ramp, zones, "--stats", "mean", "count", "--json"])
        records = json.loads(capsys.readouterr().out)
        assert rc == 0, "zonal-stats must exit 0"
        assert len(records) == 1, "one zone, one record"
        assert "mean" in records[0] and "count" in records[0], "stats missing"
        assert records[0]["count"] == pytest.approx(16.0), "4x4 quadrant = 16 cells"

    def test_csv_output(self, ramp, zones, tmp_path, capsys):
        """--output writes a CSV carrying the stat column."""
        out = str(tmp_path / "z.csv")
        rc = main(["zonal-stats", ramp, zones, "--stats", "mean", "--output", out])
        assert rc == 0, capsys.readouterr().err
        assert os.path.exists(out), "CSV not written"
        assert "mean" in open(out).read(), "CSV missing the mean column"

    def test_unknown_stat_errors(self, ramp, zones, capsys):
        """An unknown stat name is a clean one-line error."""
        rc = main(["zonal-stats", ramp, zones, "--stats", "median"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestMatchHistogram:
    """`pyramids match-histogram`."""

    def test_match_writes_raster(self, ramp, tmp_path, capsys):
        """The matched raster shares the source grid and exists."""
        reference = str(tmp_path / "ref.tif")
        Dataset.from_array(
            np.arange(64, dtype="float32").reshape(8, 8) * 0.5, geo_ref=_GEO
        ).to_file(reference)
        out = str(tmp_path / "mh.tif")
        rc = main(["match-histogram", ramp, reference, out])
        assert rc == 0, capsys.readouterr().err
        assert Dataset.read_file(out).shape[-2:] == (8, 8), "grid must be preserved"

    def test_zero_bins_errors(self, ramp, tmp_path, capsys):
        """--bins 0 is a clean one-line error."""
        reference = str(tmp_path / "ref.tif")
        Dataset.from_array(
            np.arange(64, dtype="float32").reshape(8, 8) * 0.5, geo_ref=_GEO
        ).to_file(reference)
        out = str(tmp_path / "mh.tif")
        rc = main(["match-histogram", ramp, reference, out, "--bins", "0"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestPercentileStretch:
    """`pyramids percentile-stretch`."""

    def test_stretch_into_unit_range(self, ramp, tmp_path, capsys):
        """The stretched output lies within [0, 1]."""
        out = str(tmp_path / "ps.tif")
        rc = main(["percentile-stretch", ramp, out, "--lower", "0", "--upper", "100"])
        assert rc == 0, capsys.readouterr().err
        values = Dataset.read_file(out).read_array()
        assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0, "out of [0,1]"

    def test_equal_percentiles_error(self, ramp, tmp_path, capsys):
        """Equal lower/upper percentiles are a clean one-line error."""
        out = str(tmp_path / "ps.tif")
        rc = main(["percentile-stretch", ramp, out, "--lower", "50", "--upper", "50"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestPanSharpen:
    """`pyramids pan-sharpen`."""

    def test_fuse_to_pan_resolution(self, pan_pair, tmp_path, capsys):
        """The fused raster is at the pan resolution with one band per spectral band."""
        ms_path, pan_path = pan_pair
        out = str(tmp_path / "fused.tif")
        rc = main(["pan-sharpen", ms_path, pan_path, out])
        assert rc == 0, capsys.readouterr().err
        fused = Dataset.read_file(out)
        assert (fused.rows, fused.columns, fused.band_count) == (128, 128, 3)

    def test_unknown_algorithm_errors(self, pan_pair, tmp_path, capsys):
        """An unsupported --algorithm is a clean one-line error."""
        ms_path, pan_path = pan_pair
        out = str(tmp_path / "fused.tif")
        rc = main(["pan-sharpen", ms_path, pan_path, out, "--algorithm", "nope"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestFillna:
    """`pyramids fillna`."""

    def test_fills_gap(self, gaps, tmp_path, capsys):
        """The gap cell takes the fill value in the output."""
        out = str(tmp_path / "filled.tif")
        rc = main(["fillna", gaps, out, "--value", "0"])
        assert rc == 0, capsys.readouterr().err
        assert float(Dataset.read_file(out).read_array()[0, 0]) == 0.0, "gap not filled"

    def test_overwrite_without_flag_errors(self, gaps, tmp_path, capsys):
        """Writing over an existing file without --overwrite is refused."""
        out = str(tmp_path / "filled.tif")
        open(out, "w").close()
        rc = main(["fillna", gaps, out, "--value", "0"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestWhere:
    """`pyramids where`."""

    def test_keeps_selected_masks_rest(self, ramp, tmp_path, capsys):
        """Cells failing the condition become the raster's no-data value."""
        out = str(tmp_path / "where.tif")
        rc = main(["where", ramp, out, "--cond", "A > 30"])
        assert rc == 0, capsys.readouterr().err
        values = Dataset.read_file(out).read_array()
        assert float(values[0, 0]) == -9999.0, "unselected cell must be masked"
        assert float(values[7, 7]) == pytest.approx(63.0), "selected cell kept"

    def test_disallowed_condition_errors(self, ramp, tmp_path, capsys):
        """A condition outside the arithmetic whitelist is a clean one-line error."""
        out = str(tmp_path / "where.tif")
        rc = main(["where", ramp, out, "--cond", "__import__('os')"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestConvertUnits:
    """`pyramids convert-units`."""

    def test_kelvin_to_celsius(self, kelvin, tmp_path, capsys):
        """A Kelvin raster converts to the expected Celsius values."""
        out = str(tmp_path / "celsius.tif")
        rc = main(["convert-units", kelvin, out, "--target", "celsius"])
        assert rc == 0, capsys.readouterr().err
        values = Dataset.read_file(out).read_array()
        assert values.tolist() == [[0.0, 10.0], [20.0, 30.0]], "bad conversion"

    def test_unsupported_target_errors(self, kelvin, tmp_path, capsys):
        """An unsupported --target unit is a clean one-line error."""
        out = str(tmp_path / "celsius.tif")
        rc = main(["convert-units", kelvin, out, "--target", "furlongs"])
        _assert_one_line_error(rc, capsys.readouterr().err)


class TestContour:
    """`pyramids contour`."""

    def test_interval_lines(self, ramp, tmp_path, capsys):
        """An interval traces LineString contours into the vector output."""
        out = str(tmp_path / "contours.geojson")
        rc = main(["contour", ramp, out, "--interval", "10"])
        assert rc == 0, capsys.readouterr().err
        lines = FeatureCollection.read_file(out)
        assert len(lines) > 0, "no contours produced"
        assert set(lines.geometry.geom_type.unique()) == {"LineString"}, "not lines"

    def test_non_positive_interval_errors(self, ramp, tmp_path, capsys):
        """A non-positive --interval is a clean one-line error."""
        out = str(tmp_path / "contours.geojson")
        rc = main(["contour", ramp, out, "--interval", "-1"])
        _assert_one_line_error(rc, capsys.readouterr().err)
