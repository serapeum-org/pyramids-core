"""Tests for the ``pyramids`` raster-constructor CLI commands (``cli_dataset_read``).

Each command gets a happy path (the output raster is written and openable via
``Dataset.read_file``) and an error path (missing placement / overwrite guard)
asserting exit code 1. Commands are driven through ``pyramids.cli.main`` so the
argument wiring is exercised end to end. Inputs are built locally (small rasters,
CSVs, a point layer, a zip) so the tests do not depend on the process CWD.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Point

from pyramids.cli import main
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core

_DATA_TIF = str(Path(__file__).parents[1] / "data" / "acc4000.tif")


def _band_tif(directory: Path, name: str, value: int, shape=(4, 5)) -> str:
    """Write a small single-band GeoTIFF on a 1-degree WGS84 grid and return its path."""
    path = str(directory / name)
    Dataset.from_array(
        np.full(shape, value, dtype="int16"),
        geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
        path=path,
    ).close()
    return path


class _FakeDataset:
    """A stand-in raster whose ``to_file`` just touches the output path."""

    def to_file(self, path: str) -> None:
        with open(path, "wb"):
            pass


class _FakeStdinBuffer:
    """A stand-in ``sys.stdin.buffer`` that yields a fixed byte payload once."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class TestFromBytes:
    """`pyramids from-bytes`."""

    def test_happy_path(self, tmp_path):
        """The bytes of a GeoTIFF file round-trip into an openable raster."""
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "from-bytes", _DATA_TIF, out])
        assert rc == 0
        assert Dataset.read_file(out).band_count == 1

    def test_missing_input_errors(self, tmp_path):
        """Neither an input path nor --stdin exits 1."""
        argv = ["raster", "from-bytes", str(tmp_path / "o.tif")]
        assert main(argv) == 1

    def test_stdin_branch(self, tmp_path, monkeypatch):
        """With --stdin the raster bytes are read from standard input and written."""
        payload = Path(_DATA_TIF).read_bytes()
        monkeypatch.setattr(
            "sys.stdin", SimpleNamespace(buffer=_FakeStdinBuffer(payload))
        )
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "from-bytes", out, "--stdin"])
        assert rc == 0
        assert Dataset.read_file(out).band_count == 1


class TestFromDataframe:
    """`pyramids from-dataframe`."""

    def _csv(self, directory: Path) -> str:
        path = str(directory / "grid.csv")
        rows = []
        for y in (10.0, 9.0):
            for x in (0.0, 1.0):
                rows.append({"band": 0, "y": y, "x": x, "val": x + y})
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_happy_path(self, tmp_path):
        """A (band, y, x) CSV table rebuilds a 2x2 raster."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-dataframe",
                self._csv(tmp_path),
                out,
                "--x",
                "x",
                "--y",
                "y",
                "--band",
                "band",
                "--value",
                "val",
                "--crs",
                "4326",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns) == (2, 2)

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        argv = [
            "raster",
            "from-dataframe",
            self._csv(tmp_path),
            out,
            "--x",
            "x",
            "--y",
            "y",
        ]
        assert main(argv) == 1

    def test_no_value_with_no_data_value(self, tmp_path):
        """Omitting --value keeps the lone value column; --no-data-value is parsed."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-dataframe",
                self._csv(tmp_path),
                out,
                "--x",
                "x",
                "--y",
                "y",
                "--band",
                "band",
                "--crs",
                "4326",
                "--no-data-value",
                "-9999",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns) == (2, 2)


class TestFromPoints:
    """`pyramids from-points`."""

    def _vector(self, directory: Path) -> str:
        path = str(directory / "pts.geojson")
        gdf = GeoDataFrame(
            {"rain": [10.0, 20.0, 30.0, 40.0]},
            geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
            crs="EPSG:4326",
        )
        FeatureCollection(gdf).to_file(path, driver="geojson")
        return path

    def test_happy_path(self, tmp_path):
        """A point layer is interpolated onto a 10x10 grid."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-points",
                self._vector(tmp_path),
                out,
                "--value-column",
                "rain",
                "--cell-size",
                "1.0",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).band_count == 1

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        argv = [
            "raster",
            "from-points",
            self._vector(tmp_path),
            out,
            "--value-column",
            "rain",
            "--cell-size",
            "1.0",
        ]
        assert main(argv) == 1

    def test_epsg_forwarded(self, tmp_path):
        """--epsg is forwarded to Dataset.from_points and stamped on the output."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-points",
                self._vector(tmp_path),
                out,
                "--value-column",
                "rain",
                "--cell-size",
                "1.0",
                "--epsg",
                "4326",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).epsg == 4326


class TestFromPointArrays:
    """`pyramids from-point-arrays`."""

    def _csv(self, directory: Path) -> str:
        path = str(directory / "xyz.csv")
        pd.DataFrame(
            {
                "x": [0.0, 10.0, 0.0, 10.0],
                "y": [0.0, 0.0, 10.0, 10.0],
                "z": [10.0, 20.0, 30.0, 40.0],
            }
        ).to_csv(path, index=False)
        return path

    def test_happy_path(self, tmp_path):
        """Raw x/y/z columns are gridded onto a 10x10 raster with a stamped CRS."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-point-arrays",
                self._csv(tmp_path),
                out,
                "--x",
                "x",
                "--y",
                "y",
                "--z",
                "z",
                "--cell-size",
                "1.0",
                "--epsg",
                "4326",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).epsg == 4326

    def test_width_height_bbox_and_crs(self, tmp_path):
        """--width/--height/--bbox size the grid and --crs labels it (no --cell-size)."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-point-arrays",
                self._csv(tmp_path),
                out,
                "--x",
                "x",
                "--y",
                "y",
                "--z",
                "z",
                "--width",
                "10",
                "--height",
                "10",
                "--bbox",
                "0",
                "0",
                "10",
                "10",
                "--crs",
                "4326",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns) == (10, 10)

    def test_bad_column_errors(self, tmp_path):
        """A column name that is not in the CSV exits 1."""
        out = str(tmp_path / "o.tif")
        argv = [
            "raster",
            "from-point-arrays",
            self._csv(tmp_path),
            out,
            "--x",
            "nope",
            "--y",
            "y",
            "--z",
            "z",
            "--cell-size",
            "1.0",
        ]
        assert main(argv) == 1


class TestFromBandFiles:
    """`pyramids from-band-files`."""

    def test_happy_path(self, tmp_path):
        """Two single-band rasters stack into a two-band raster."""
        b1 = _band_tif(tmp_path, "b1.tif", 2)
        b2 = _band_tif(tmp_path, "b2.tif", 3)
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "from-band-files", b1, b2, out])
        assert rc == 0
        assert Dataset.read_file(out).band_count == 2

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1."""
        b1 = _band_tif(tmp_path, "b1.tif", 2)
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        assert main(["raster", "from-band-files", b1, out]) == 1

    def test_align_resampling_and_no_data_value(self, tmp_path):
        """--align with --resampling and --no-data-value reach Dataset.from_band_files."""
        b1 = _band_tif(tmp_path, "b1.tif", 2)
        b2 = _band_tif(tmp_path, "b2.tif", 3)
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-band-files",
                b1,
                b2,
                out,
                "--align",
                "--resampling",
                "nearest",
                "--no-data-value",
                "-9999",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).band_count == 2


class TestFromArchive:
    """`pyramids from-archive`."""

    def _zip(self, directory: Path) -> str:
        b1 = _band_tif(directory, "scene.B2.tif", 2)
        b2 = _band_tif(directory, "scene.B3.tif", 3)
        path = str(directory / "download.zip")
        with zipfile.ZipFile(path, "w") as handle:
            handle.write(b1, arcname="scene.B2.tif")
            handle.write(b2, arcname="scene.B3.tif")
        return path

    def test_happy_path(self, tmp_path):
        """The raster members of a zip stack into a two-band raster."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-archive",
                self._zip(tmp_path),
                out,
                "--member-glob",
                "*.tif",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).band_count == 2

    def test_missing_archive_errors(self, tmp_path):
        """A non-existent archive exits 1."""
        out = str(tmp_path / "o.tif")
        assert main(["raster", "from-archive", str(tmp_path / "nope.zip"), out]) == 1

    def test_no_data_value_forwarded(self, tmp_path):
        """--no-data-value is parsed and forwarded to Dataset.from_archive."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-archive",
                self._zip(tmp_path),
                out,
                "--member-glob",
                "*.tif",
                "--no-data-value",
                "-9999",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).band_count == 2


class TestFromZarr:
    """`pyramids from-zarr`."""

    @pytest.mark.lazy
    def test_happy_path(self, tmp_path):
        """A pyramids-written Zarr store reads back into an openable raster."""
        store = str(tmp_path / "store.zarr")
        Dataset.read_file(_band_tif(tmp_path, "src.tif", 7)).to_zarr(store)
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "from-zarr", store, out])
        assert rc == 0
        assert Dataset.read_file(out).band_count == 1

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1 before any zarr read."""
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        assert main(["raster", "from-zarr", str(tmp_path / "store.zarr"), out]) == 1

    def test_missing_zarr_extra_is_remapped(self, tmp_path, monkeypatch):
        """A missing-``zarr`` ImportError becomes a clean user error (exit 1)."""

        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'zarr'")

        monkeypatch.setattr(
            "pyramids.cli_dataset_read.Dataset.from_zarr",
            staticmethod(raise_import_error),
        )
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "from-zarr", str(tmp_path / "store.zarr"), out])
        assert rc == 1

    def test_storage_options_parsed(self, tmp_path, monkeypatch, capsys):
        """--storage-options is JSON-decoded and forwarded to Dataset.from_zarr."""
        captured: dict = {}

        def fake(store, **kwargs):
            captured["store"] = store
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr(
            "pyramids.cli_dataset_read.Dataset.from_zarr", staticmethod(fake)
        )
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "from-zarr",
                "store.zarr",
                out,
                "--storage-options",
                '{"anon": true}',
            ]
        )
        assert rc == 0
        assert captured["storage_options"] == {"anon": True}


class TestCreate:
    """`pyramids create`."""

    def test_happy_path(self, tmp_path):
        """A placed, no-data-filled raster is created and openable."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "create",
                out,
                "--rows",
                "3",
                "--columns",
                "4",
                "--dtype",
                "float32",
                "--bands",
                "1",
                "--top-left",
                "0",
                "10",
                "--cell-size",
                "0.5",
                "--epsg",
                "4326",
                "--no-data-value",
                "-9999",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns, ds.band_count) == (3, 4, 1)

    def test_geo_transform_without_no_data(self, tmp_path):
        """A six-term --geo affine places the raster; no --no-data-value is passed."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "create",
                out,
                "--rows",
                "3",
                "--columns",
                "4",
                "--dtype",
                "float32",
                "--bands",
                "1",
                "--geo",
                "0",
                "1",
                "0",
                "10",
                "0",
                "-1",
                "--epsg",
                "4326",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns, ds.epsg) == (3, 4, 4326)

    def test_missing_placement_errors(self, tmp_path):
        """No --geo / --top-left placement exits 1."""
        out = str(tmp_path / "o.tif")
        argv = [
            "raster",
            "create",
            out,
            "--rows",
            "3",
            "--columns",
            "4",
            "--dtype",
            "float32",
            "--bands",
            "1",
        ]
        assert main(argv) == 1

    def test_epsg_only_gives_tailored_placement_message(self, tmp_path, capsys):
        """--epsg without --geo/--top-left gets the tailored 'needs a placement' error."""
        out = str(tmp_path / "o.tif")
        argv = [
            "raster",
            "create",
            out,
            "--rows",
            "4",
            "--columns",
            "4",
            "--dtype",
            "float32",
            "--bands",
            "1",
            "--epsg",
            "4326",
        ]
        rc = main(argv)
        err = capsys.readouterr().err
        assert rc == 1, "an epsg-only create must still exit 1"
        assert "create needs a placement" in err, (
            f"epsg-only create should get the tailored placement message, got: {err!r}"
        )


class TestCreateEmpty:
    """`pyramids create-empty`."""

    def test_happy_path(self, tmp_path):
        """A header-only multi-band raster is created and openable."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "create-empty",
                out,
                "--rows",
                "4",
                "--cols",
                "5",
                "--bands",
                "2",
                "--dtype",
                "int16",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns, ds.band_count) == (4, 5, 2)

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        assert main(["raster", "create-empty", out, "--rows", "4", "--cols", "5"]) == 1

    def test_placement_and_none_no_data_value(self, tmp_path):
        """A --top-left/--cell-size placement is applied; --no-data-value none -> None."""
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "create-empty",
                out,
                "--rows",
                "4",
                "--cols",
                "5",
                "--top-left",
                "0",
                "10",
                "--cell-size",
                "1.0",
                "--epsg",
                "4326",
                "--no-data-value",
                "none",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns, ds.epsg) == (4, 5, 4326)

    def test_bad_no_data_value_errors(self, tmp_path):
        """A --no-data-value that is not a number / nan / none exits 1."""
        out = str(tmp_path / "o.tif")
        argv = [
            "raster",
            "create-empty",
            out,
            "--rows",
            "4",
            "--cols",
            "5",
            "--no-data-value",
            "notanumber",
        ]
        assert main(argv) == 1


class TestEmptyLike:
    """`pyramids empty-like`."""

    def test_happy_path(self, tmp_path):
        """The output copies the template's footprint with a new dtype."""
        template = _band_tif(tmp_path, "tmpl.tif", 1)
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "empty-like", template, out, "--dtype", "int16"])
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns) == (4, 5)

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1."""
        template = _band_tif(tmp_path, "tmpl.tif", 1)
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        assert main(["raster", "empty-like", template, out]) == 1

    def test_no_data_value_forwarded(self, tmp_path):
        """--no-data-value is parsed and forwarded to Dataset.empty_like."""
        template = _band_tif(tmp_path, "tmpl.tif", 1)
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "raster",
                "empty-like",
                template,
                out,
                "--dtype",
                "int16",
                "--no-data-value",
                "-9999",
            ]
        )
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns) == (4, 5)


class TestDatasetLike:
    """`pyramids dataset-like`."""

    def test_happy_path(self, tmp_path):
        """A .npy array is stamped with the template's georeference."""
        template = _band_tif(tmp_path, "tmpl.tif", 1)
        npy = str(tmp_path / "arr.npy")
        np.save(npy, np.arange(20, dtype="float32").reshape(4, 5))
        out = str(tmp_path / "o.tif")
        rc = main(["raster", "dataset-like", template, npy, out])
        assert rc == 0
        ds = Dataset.read_file(out)
        assert (ds.rows, ds.columns) == (4, 5)

    def test_overwrite_guard(self, tmp_path):
        """An existing output without --overwrite exits 1."""
        template = _band_tif(tmp_path, "tmpl.tif", 1)
        npy = str(tmp_path / "arr.npy")
        np.save(npy, np.arange(20, dtype="float32").reshape(4, 5))
        out = str(tmp_path / "o.tif")
        Path(out).write_bytes(b"x")
        assert main(["raster", "dataset-like", template, npy, out]) == 1
