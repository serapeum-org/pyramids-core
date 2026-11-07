"""Tests for the raster (`Dataset`) export CLI commands.

Each shipped ``pyramids`` export subcommand is a thin wrapper over a single
``Dataset`` method; these tests pin the wiring: a happy path that asserts the
output artifact is created (and parseable where cheap) and an error path
(missing input or overwrite-without-flag) that asserts the one-line-error exit
code ``1``. Optional-dependency commands carry the matching extras marker so they
auto-skip when the extra is absent.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pandas as pd
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset


@pytest.fixture
def dem_raster(tmp_path) -> str:
    """A 16x16 float32 elevation ramp GeoTIFF on EPSG:4326.

    Returns:
        str: Path to the written raster (metres of elevation).
    """
    path = str(tmp_path / "dem.tif")
    Dataset.from_array(
        np.linspace(0.0, 3000.0, 256, dtype="float32").reshape(16, 16),
        geo_ref=GeoReference(top_left_corner=(0, 16), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


class TestToBytes:
    """`pyramids to-bytes`."""

    def test_writes_output_file(self, cli_raster, tmp_path, capsys):
        """A GTiff byte payload is written to the output file and re-reads."""
        out = str(tmp_path / "bytes.tif")
        assert main(["to-bytes", cli_raster, out]) == 0, "to-bytes must exit 0"
        assert os.path.getsize(out) > 0, "payload must be non-empty"
        restored = Dataset.read_file(out)
        assert (restored.rows, restored.columns) == (8, 8), "round-trip shape wrong"

    def test_missing_input_exits_1(self, tmp_path, capsys):
        """A missing input exits 1 with a one-line error."""
        rc = main(["to-bytes", str(tmp_path / "nope.tif"), str(tmp_path / "o.tif")])
        err = capsys.readouterr().err
        assert rc == 1, "missing input must exit 1"
        assert err.startswith("error: "), f"unexpected stderr: {err}"

    def test_writes_to_stdout(self, cli_raster, capsysbinary):
        """With no output path the payload is written to stdout as a TIFF."""
        assert main(["to-bytes", cli_raster]) == 0, "stdout to-bytes must exit 0"
        payload = capsysbinary.readouterr().out
        assert payload[:2] in (b"II", b"MM"), f"not a TIFF payload: {payload[:2]!r}"


class TestToCogBytes:
    """`pyramids to-cog-bytes`."""

    def test_writes_cog_bytes(self, cli_raster, tmp_path):
        """A COG byte payload is written and begins with a TIFF marker."""
        out = str(tmp_path / "scene_cog.tif")
        assert main(["to-cog-bytes", cli_raster, out]) == 0, "must exit 0"
        with open(out, "rb") as handle:
            marker = handle.read(2)
        assert marker in (b"II", b"MM"), f"not a TIFF payload: {marker!r}"

    def test_overwrite_guard(self, cli_raster, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "exists.tif")
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("x")
        rc = main(["to-cog-bytes", cli_raster, out])
        assert rc == 1, "existing output without --overwrite must exit 1"

    def test_compression_and_blocksize(self, cli_raster, tmp_path):
        """--compression and --blocksize are forwarded and a COG is written."""
        out = str(tmp_path / "opts_cog.tif")
        rc = main(
            [
                "to-cog-bytes",
                cli_raster,
                out,
                "--compression",
                "DEFLATE",
                "--blocksize",
                "256",
            ]
        )
        assert rc == 0, "must exit 0 with compression/blocksize set"
        with open(out, "rb") as handle:
            marker = handle.read(2)
        assert marker in (b"II", b"MM"), f"not a TIFF payload: {marker!r}"


class TestToDataframe:
    """`pyramids to-dataframe`."""

    def test_writes_csv(self, cli_raster, tmp_path):
        """The CSV carries the (band, y, x) index columns and a values column."""
        out = str(tmp_path / "cells.csv")
        assert main(["to-dataframe", cli_raster, out]) == 0, "must exit 0"
        with open(out, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert {"band", "y", "x", "values"} <= set(header), f"columns wrong: {header}"

    def test_missing_input_exits_1(self, tmp_path):
        """A missing input exits 1."""
        rc = main(["to-dataframe", str(tmp_path / "nope.tif"), str(tmp_path / "o.csv")])
        assert rc == 1, "missing input must exit 1"

    @pytest.mark.parquet
    def test_writes_parquet(self, cli_raster, tmp_path):
        """A .parquet extension writes a Parquet file (needs the [parquet] extra)."""
        out = str(tmp_path / "cells.parquet")
        assert main(["to-dataframe", cli_raster, out]) == 0, "must exit 0"
        assert os.path.getsize(out) > 0, "parquet file must be non-empty"

    def test_missing_parquet_extra_remapped(
        self, cli_raster, tmp_path, monkeypatch, capsys
    ):
        """A missing pyarrow is remapped to a one-line error naming the [parquet] extra."""

        def _raise(*_args, **_kwargs):
            raise ImportError("no pyarrow")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise, raising=False)
        out = str(tmp_path / "cells.parquet")
        rc = main(["to-dataframe", cli_raster, out])
        err = capsys.readouterr().err
        assert rc == 1, "missing extra must exit 1"
        assert "[parquet]" in err, f"error must name the extra: {err}"


class TestToPolygons:
    """`pyramids to-polygons`."""

    def test_writes_geojson_features(self, cli_raster, tmp_path):
        """The vector output parses as GeoJSON with at least one feature."""
        out = str(tmp_path / "regions.geojson")
        assert main(["to-polygons", cli_raster, out]) == 0, "must exit 0"
        with open(out, encoding="utf-8") as handle:
            doc = json.load(handle)
        assert doc["features"], "expected at least one polygon feature"

    def test_unknown_extension_exits_1(self, cli_raster, tmp_path):
        """No --driver and an unknown extension exits 1."""
        rc = main(["to-polygons", cli_raster, str(tmp_path / "out.weird")])
        assert rc == 1, "unknown vector extension must exit 1"


class TestToRaster:
    """`pyramids to-raster`."""

    def test_writes_raster(self, cli_raster, tmp_path):
        """The alias write produces a readable raster of the same shape."""
        out = str(tmp_path / "copy.tif")
        assert main(["to-raster", cli_raster, out]) == 0, "must exit 0"
        restored = Dataset.read_file(out)
        assert (restored.rows, restored.columns) == (8, 8), "shape wrong"

    def test_bad_creation_option_exits_1(self, cli_raster, tmp_path):
        """A creation option not of KEY=VALUE form exits 1."""
        rc = main(
            [
                "to-raster",
                cli_raster,
                str(tmp_path / "o.tif"),
                "--creation-option",
                "NOPE",
            ]
        )
        assert rc == 1, "malformed creation option must exit 1"

    def test_valid_creation_option(self, cli_raster, tmp_path):
        """A valid KEY=VALUE creation option is parsed and the raster is written."""
        out = str(tmp_path / "copy_opt.tif")
        rc = main(
            ["to-raster", cli_raster, out, "--creation-option", "COMPRESS=DEFLATE"]
        )
        assert rc == 0, "a valid creation option must exit 0"
        restored = Dataset.read_file(out)
        assert (restored.rows, restored.columns) == (8, 8), "shape wrong"

    def test_in_range_band_exits_0(self, cli_raster, tmp_path):
        """An in-range --band is accepted (band 0 exists on every raster)."""
        out = str(tmp_path / "band0.tif")
        assert main(["to-raster", cli_raster, out, "--band", "0"]) == 0, (
            "band 0 must exit 0"
        )

    def test_out_of_range_band_clean_error(self, cli_raster, tmp_path, capsys):
        """An out-of-range --band exits 1 cleanly instead of silently writing."""
        rc = main(["to-raster", cli_raster, str(tmp_path / "o.tif"), "--band", "99"])
        err = capsys.readouterr().err
        assert rc == 1, "an out-of-range band must exit 1"
        assert err.startswith("error: "), f"expected a clean error, got: {err!r}"
        assert "unexpected failure" not in err, (
            f"band error must be mapped cleanly: {err!r}"
        )


class TestToImage:
    """`pyramids to-image`."""

    @pytest.mark.plot
    def test_writes_png(self, cli_raster, tmp_path):
        """A colour-mapped PNG is written and non-empty (needs the [viz] extra)."""
        out = str(tmp_path / "band.png")
        assert main(["to-image", cli_raster, out]) == 0, "must exit 0"
        assert os.path.getsize(out) > 0, "image must be non-empty"

    @pytest.mark.plot
    def test_overwrite_guard(self, cli_raster, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "band.png")
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("x")
        rc = main(["to-image", cli_raster, out])
        assert rc == 1, "existing output without --overwrite must exit 1"


class TestToXyz:
    """`pyramids to-xyz`."""

    def test_writes_xyz(self, cli_raster, tmp_path):
        """An XYZ point table is written and non-empty."""
        out = str(tmp_path / "pts.xyz")
        assert main(["to-xyz", cli_raster, out]) == 0, "must exit 0"
        assert os.path.getsize(out) > 0, "xyz table must be non-empty"

    def test_missing_input_exits_1(self, tmp_path):
        """A missing input exits 1."""
        rc = main(["to-xyz", str(tmp_path / "nope.tif"), str(tmp_path / "o.xyz")])
        assert rc == 1, "missing input must exit 1"


class TestToZarr:
    """`pyramids to-zarr`."""

    @pytest.mark.lazy
    def test_writes_store(self, cli_raster, tmp_path):
        """A Zarr store directory is created (needs the pyramids-lazy backend)."""
        out = str(tmp_path / "store.zarr")
        assert main(["to-zarr", cli_raster, out]) == 0, "must exit 0"
        assert os.path.isdir(out), "expected a Zarr store directory"

    @pytest.mark.lazy
    def test_overwrite_guard(self, cli_raster, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "store.zarr")
        os.mkdir(out)
        rc = main(["to-zarr", cli_raster, out])
        assert rc == 1, "existing output without --overwrite must exit 1"

    def test_missing_lazy_extra_remapped(
        self, cli_raster, tmp_path, monkeypatch, capsys
    ):
        """A missing zarr/dask is remapped to a one-line error naming the pyramids-lazy backend.

        `to-zarr` dispatches to the compute backend, so the actionable install target is
        `pyramids-lazy`, not the `[lazy]` extra (which pulls the stack but not the backend).
        """

        def _raise(*_args, **_kwargs):
            raise ImportError("no zarr")

        monkeypatch.setattr(Dataset, "to_zarr", _raise, raising=False)
        out = str(tmp_path / "store.zarr")
        rc = main(["to-zarr", cli_raster, out])
        err = capsys.readouterr().err
        assert rc == 1, "missing backend must exit 1"
        assert "pyramids-lazy" in err, f"error must name the backend: {err}"


class TestToStacItem:
    """`pyramids to-stac-item`."""

    def test_writes_item_json(self, cli_raster, tmp_path):
        """A STAC item JSON is written with the given id and an asset."""
        out = str(tmp_path / "item.json")
        rc = main(["to-stac-item", cli_raster, out, "--item-id", "scene-1"])
        assert rc == 0, "must exit 0"
        with open(out, encoding="utf-8") as handle:
            item = json.load(handle)
        assert item["id"] == "scene-1", f"id wrong: {item.get('id')}"
        assert item["assets"], "expected at least one asset"

    def test_missing_input_exits_1(self, tmp_path):
        """A missing input exits 1."""
        rc = main(
            [
                "to-stac-item",
                str(tmp_path / "nope.tif"),
                str(tmp_path / "i.json"),
                "--item-id",
                "x",
            ]
        )
        assert rc == 1, "missing input must exit 1"


class TestToTerrainRgb:
    """`pyramids to-terrain-rgb`."""

    def test_writes_single_raster(self, dem_raster, tmp_path):
        """A single terrain-RGB raster is written and readable."""
        out = str(tmp_path / "terrain.tif")
        assert main(["to-terrain-rgb", dem_raster, out]) == 0, "must exit 0"
        restored = Dataset.read_file(out)
        assert restored.band_count >= 3, "terrain-RGB must have >= 3 bands"

    def test_missing_input_exits_1(self, tmp_path):
        """A missing input exits 1."""
        rc = main(
            ["to-terrain-rgb", str(tmp_path / "nope.tif"), str(tmp_path / "o.tif")]
        )
        assert rc == 1, "missing input must exit 1"
