"""Tests for the raster band / mask / colour / sensor-model CLI commands.

Each command in :mod:`pyramids.cli_dataset_bands` is a thin wrapper over a
:class:`~pyramids.dataset.Dataset` call; these tests pin the wiring: the
observable effect of the happy path (a written raster with the right bands /
colour table / metadata, or the right stdout) and the one-line-error contract
for expected user mistakes (missing input, out-of-range band, refusing to
overwrite, bad arguments).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from osgeo import osr

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

_RPC_KEYS = (
    "LINE_OFF",
    "SAMP_OFF",
    "LAT_OFF",
    "LONG_OFF",
    "HEIGHT_OFF",
    "LINE_SCALE",
    "SAMP_SCALE",
    "LAT_SCALE",
    "LONG_SCALE",
    "HEIGHT_SCALE",
    "LINE_NUM_COEFF",
    "LINE_DEN_COEFF",
    "SAMP_NUM_COEFF",
    "SAMP_DEN_COEFF",
)


def _assert_clean_error(rc: int, err: str) -> None:
    """Assert a non-zero exit carried a one-line ``error:`` message, no traceback."""
    assert rc == 1, "expected a non-zero exit"
    assert err.startswith("error: "), f"unexpected stderr: {err}"
    assert "Traceback" not in err, "tracebacks must not leak to users"
    assert "unexpected failure" not in err, (
        f"expected-error path leaked into the internal-bug branch: {err!r}"
    )


@pytest.fixture
def single_band(tmp_path) -> str:
    """An 8x8 float32 ramp GeoTIFF on EPSG:4326 (nodata -9999)."""
    path = str(tmp_path / "single.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def three_band(tmp_path) -> str:
    """A 3-band 8x8 float32 GeoTIFF on EPSG:4326."""
    path = str(tmp_path / "three.tif")
    Dataset.from_array(
        np.arange(3 * 64, dtype="float32").reshape(3, 8, 8),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def rgb_band(tmp_path) -> str:
    """A 3-band GeoTIFF with red/green/blue colour interpretations set."""
    path = str(tmp_path / "rgb.tif")
    ds = Dataset.from_array(
        np.arange(3 * 64, dtype="float32").reshape(3, 8, 8),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    )
    ds.band_color = {0: "red", 1: "green", 2: "blue"}
    ds.to_file(path)
    return path


@pytest.fixture
def nodata_raster(tmp_path) -> str:
    """An 8x8 raster whose second column holds the nodata sentinel."""
    path = str(tmp_path / "nodata.tif")
    arr = np.array([[1.0, -9999.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * 8, dtype="float32")
    Dataset.from_array(
        arr,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def geoloc_raster(tmp_path) -> str:
    """A 4x4 raster carrying a persisted GDAL GEOLOCATION domain of lon/lat arrays."""
    rows, cols = 4, 4
    lon = np.tile(np.linspace(-10.0, -7.0, cols), (rows, 1)).astype("float64")
    lat = np.tile(np.linspace(40.0, 37.0, rows).reshape(rows, 1), (1, cols))
    lon_path = str(tmp_path / "lon.tif")
    lat_path = str(tmp_path / "lat.tif")
    for arr, arr_path in [(lon, lon_path), (lat.astype("float64"), lat_path)]:
        Dataset.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        ).to_file(arr_path)
    path = str(tmp_path / "geoloc.tif")
    Dataset.from_array(
        np.arange(rows * cols).reshape(rows, cols).astype("float32"),
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    ).to_file(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    domain = {
        "X_DATASET": lon_path,
        "Y_DATASET": lat_path,
        "X_BAND": "1",
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "PIXEL_STEP": "1",
        "LINE_OFFSET": "0",
        "LINE_STEP": "1",
        "SRS": srs.ExportToWkt(),
    }
    # Set the domain in update mode and flush so GDAL persists it (a GTiff
    # CreateCopy drops the GEOLOCATION domain; an update-mode write keeps it).
    writable = Dataset.read_file(path, read_only=False)
    writable.set_meta_data(domain, domain="GEOLOCATION")
    writable.raster.FlushCache()
    writable = None
    return path


class TestAddBand:
    """`pyramids add-band`."""

    def test_appends_a_band(self, single_band, three_band, tmp_path):
        """The output has one more band than the input."""
        out = str(tmp_path / "added.tif")
        rc = main(["raster", "add-band", single_band, out, "--source", three_band])
        assert rc == 0, "add-band must exit 0"
        assert Dataset.read_file(out).band_count == 2, "a band must be appended"

    def test_mismatched_source_dims_error(self, single_band, geoloc_raster, capsys):
        """A source band of different dimensions exits 1 with a clean error."""
        out = str(single_band).replace("single", "mismatch")
        rc = main(["raster", "add-band", single_band, out, "--source", geoloc_raster])
        _assert_clean_error(rc, capsys.readouterr().err)


class TestSelectBands:
    """`pyramids select-bands`."""

    def test_subsets_and_reorders(self, three_band, tmp_path):
        """Selecting two 1-based bands yields a two-band raster."""
        out = str(tmp_path / "subset.tif")
        rc = main(["raster", "select-bands", three_band, out, "--bands", "3", "1"])
        assert rc == 0, "select-bands must exit 0"
        assert Dataset.read_file(out).band_count == 2, "two bands must be kept"

    def test_out_of_range_band_error(self, three_band, tmp_path, capsys):
        """An out-of-range band index exits 1 with a clean error."""
        out = str(tmp_path / "bad.tif")
        rc = main(["raster", "select-bands", three_band, out, "--bands", "9"])
        _assert_clean_error(rc, capsys.readouterr().err)


class TestGetBandByColor:
    """`pyramids get-band-by-color`."""

    def test_prints_band_index(self, rgb_band, capsys):
        """The matching band's 0-based index is printed."""
        assert main(["raster", "get-band-by-color", rgb_band, "--color", "red"]) == 0
        assert capsys.readouterr().out.strip() == "0", "red is band 0"

    def test_json_payload(self, rgb_band, capsys):
        """--json wraps the colour and band index."""
        assert (
            main(["raster", "get-band-by-color", rgb_band, "--color", "blue", "--json"])
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload == {"color": "blue", "band": 2}, f"wrong payload: {payload}"

    def test_absent_color_is_none(self, rgb_band, capsys):
        """A colour no band carries prints `none` and still exits 0."""
        assert main(["raster", "get-band-by-color", rgb_band, "--color", "alpha"]) == 0
        assert capsys.readouterr().out.strip() == "none", "no alpha band"

    def test_missing_file_error(self, tmp_path, capsys):
        """A missing input exits 1 with a clean error."""
        rc = main(
            [
                "raster",
                "get-band-by-color",
                str(tmp_path / "nope.tif"),
                "--color",
                "red",
            ]
        )
        _assert_clean_error(rc, capsys.readouterr().err)


class TestCreateMaskBand:
    """`pyramids create-mask-band`."""

    def test_writes_per_dataset_mask(self, single_band, tmp_path):
        """The output carries a per-dataset mask band."""
        out = str(tmp_path / "masked.tif")
        rc = main(["raster", "create-mask-band", single_band, out])
        assert rc == 0, "create-mask-band must exit 0"
        assert Dataset.read_file(out).mask_flags().per_dataset, "mask must persist"

    def test_overwrite_without_flag_error(self, single_band, three_band, capsys):
        """Writing over an existing output without --overwrite exits 1."""
        rc = main(["raster", "create-mask-band", single_band, three_band])
        _assert_clean_error(rc, capsys.readouterr().err)


class TestReadMasks:
    """`pyramids read-masks`."""

    def test_writes_mask_raster(self, nodata_raster, tmp_path):
        """The mask raster marks the nodata column as 0 and valid cells as 255."""
        out = str(tmp_path / "masks.tif")
        rc = main(["raster", "read-masks", nodata_raster, out, "--band", "0"])
        assert rc == 0, "read-masks must exit 0"
        mask = np.asarray(Dataset.read_file(out).read_array())
        assert bool((mask[:, 1] == 0).all()), "nodata column must be 0"
        assert bool((mask[:, 0] == 255).all()), "valid column must be 255"

    def test_bad_band_error(self, nodata_raster, tmp_path, capsys):
        """An out-of-range band exits 1 with a clean error (no traceback)."""
        out = str(tmp_path / "masks_bad.tif")
        rc = main(["raster", "read-masks", nodata_raster, out, "--band", "5"])
        _assert_clean_error(rc, capsys.readouterr().err)


class TestGetMask:
    """`pyramids get-mask`."""

    def test_writes_single_band_mask(self, nodata_raster, tmp_path):
        """A single-band mask raster is written for the requested band."""
        out = str(tmp_path / "mask0.tif")
        rc = main(["raster", "get-mask", nodata_raster, out, "--band", "0"])
        assert rc == 0, "get-mask must exit 0"
        assert Dataset.read_file(out).band_count == 1, "one mask band expected"

    def test_overwrite_without_flag_error(self, nodata_raster, single_band, capsys):
        """Writing over an existing output without --overwrite exits 1."""
        rc = main(["raster", "get-mask", nodata_raster, single_band, "--band", "0"])
        _assert_clean_error(rc, capsys.readouterr().err)


class TestSetColorRamp:
    """`pyramids set-color-ramp`."""

    def test_writes_color_table(self, single_band, tmp_path):
        """A two-colour ramp fills one colour entry per value in the range."""
        # set-color-ramp builds the colour table through Dataset._set_color_table, which
        # requires cleopatra (the [viz] extra). Skip where it is absent (core wheel env) so
        # this core-marked module's other band tests still run there. The error-path test
        # below does not reach cleopatra, so it stays core.
        pytest.importorskip("cleopatra")
        out = str(tmp_path / "ramped.tif")
        rc = main(
            [
                "raster",
                "set-color-ramp",
                single_band,
                out,
                "--band",
                "1",
                "--start-value",
                "1",
                "--end-value",
                "5",
                "--start-color",
                "#709959",
                "--end-color",
                "#F2CE85",
            ]
        )
        assert rc == 0, "set-color-ramp must exit 0"
        # The GDAL palette is dense from index 0, so a 1..5 ramp spans 0..5
        # (value 0 padded transparent) — six rows, with every ramp value present.
        table = Dataset.read_file(out).color_table
        assert set(range(1, 6)) <= set(table["values"]), "values 1..5 must be coloured"

    def test_bad_range_error(self, single_band, tmp_path, capsys):
        """An --end-value not greater than --start-value exits 1 cleanly."""
        out = str(tmp_path / "ramped_bad.tif")
        rc = main(
            [
                "raster",
                "set-color-ramp",
                single_band,
                out,
                "--start-value",
                "5",
                "--end-value",
                "1",
                "--colormap",
                "viridis",
            ]
        )
        _assert_clean_error(rc, capsys.readouterr().err)


class TestSetRpcs:
    """`pyramids set-rpcs`."""

    def test_attaches_rpc_metadata(self, single_band, tmp_path):
        """A complete coefficient object attaches RPC metadata to the output."""
        rpc_path = str(tmp_path / "rpc.json")
        with open(rpc_path, "w", encoding="utf-8") as handle:
            json.dump({key: "0" for key in _RPC_KEYS}, handle)
        out = str(tmp_path / "with_rpc.tif")
        rc = main(["raster", "set-rpcs", single_band, out, "--rpc-json", rpc_path])
        assert rc == 0, "set-rpcs must exit 0"
        assert Dataset.read_file(out).has_rpcs, "RPC metadata must persist"

    def test_missing_keys_error(self, single_band, tmp_path, capsys):
        """An incomplete coefficient object exits 1 with a clean error."""
        rpc_path = str(tmp_path / "rpc_bad.json")
        with open(rpc_path, "w", encoding="utf-8") as handle:
            json.dump({"LINE_OFF": "0"}, handle)
        out = str(tmp_path / "with_rpc_bad.tif")
        rc = main(["raster", "set-rpcs", single_band, out, "--rpc-json", rpc_path])
        _assert_clean_error(rc, capsys.readouterr().err)

    def test_non_object_json_error(self, single_band, tmp_path, capsys):
        """A ``--rpc-json`` holding a JSON array (not an object) exits 1 cleanly."""
        rpc_path = str(tmp_path / "rpc_list.json")
        with open(rpc_path, "w", encoding="utf-8") as handle:
            json.dump(["LINE_OFF", "SAMP_OFF"], handle)
        out = str(tmp_path / "with_rpc_list.tif")
        rc = main(["raster", "set-rpcs", single_band, out, "--rpc-json", rpc_path])
        err = capsys.readouterr().err
        _assert_clean_error(rc, err)
        assert "JSON object" in err, (
            f"error should name the JSON-object requirement, got: {err}"
        )


class TestGeolocate:
    """`pyramids geolocate`."""

    def test_grids_from_geolocation_arrays(self, geoloc_raster, tmp_path):
        """A raster with a GEOLOCATION domain warps to a north-up grid."""
        out = str(tmp_path / "geo.tif")
        rc = main(["raster", "geolocate", geoloc_raster, out, "--to-crs", "EPSG:4326"])
        assert rc == 0, "geolocate must exit 0"
        result = Dataset.read_file(out)
        assert result.epsg == 4326, "output must be in the requested CRS"
        assert result.geotransform[5] < 0, "output must be north-up"

    def test_no_geolocation_error(self, single_band, tmp_path, capsys):
        """A plain raster (no GEOLOCATION domain) exits 1 with a clean error."""
        out = str(tmp_path / "geo_bad.tif")
        rc = main(["raster", "geolocate", single_band, out])
        _assert_clean_error(rc, capsys.readouterr().err)
