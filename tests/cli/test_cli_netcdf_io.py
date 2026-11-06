"""Tests for the ``pyramids nc`` I/O export, construction, and combine commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyramids import cli_netcdf_io
from pyramids._cli_shared import json_sanitize
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
_NC = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
# A gridded geographic CF file with a plain (lat, lon) MultiIndex to round-trip as a frame.
_NC_BANDS = _DATA / "cf__6v__1d2-2d4__geog__y-asc.nc"


def _cube(path: str, values: list[float], name: str = "t") -> str:
    """Write a clean single-variable HDF5 time cube and return ``path``.

    Args:
        path: Where to write the NetCDF (HDF5/netCDF-4, so kerchunk can read it).
        values: One value per time step (a 1x1 grid per step).
        name: The data variable's name.

    Returns:
        str: ``path``.
    """
    NetCDF.from_array(
        np.asarray(values, dtype="float64").reshape(len(values), 1, 1),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
        variable_name=name,
        dims=ExtraDimensions(
            name="time", values=[float(i) for i in range(len(values))]
        ),
    ).to_file(path)
    return path


def _reject_non_finite(token):
    """A `parse_constant` hook that fails on any NaN/Infinity JSON token."""
    raise AssertionError(f"non-finite JSON token emitted: {token}")


class TestNcConvert:
    """`pyramids nc convert`."""

    def test_rewrites_as_netcdf(self, tmp_path):
        """convert re-writes the file as a readable NetCDF."""
        out = str(tmp_path / "out.nc")
        assert main(["nc", "convert", str(_NC), out]) == 0
        assert Dataset.read_file(out) is not None

    def test_refuses_existing(self, tmp_path):
        """A second convert without --overwrite is refused."""
        out = str(tmp_path / "out.nc")
        assert main(["nc", "convert", str(_NC), out]) == 0
        assert main(["nc", "convert", str(_NC), out]) == 1


class TestNcToDataframe:
    """`pyramids nc to-dataframe`."""

    def test_writes_csv(self, tmp_path):
        """to-dataframe writes a non-empty CSV for the selected variable."""
        out = tmp_path / "out.csv"
        assert main(["nc", "to-dataframe", str(_NC), str(out), "--vars", "tas"]) == 0
        assert out.exists()
        assert out.stat().st_size > 0


class TestNcToDict:
    """`pyramids nc to-dict`."""

    def test_json_structure(self, capsys):
        """--json emits the file's structure as strict-valid JSON with the expected keys."""
        assert main(["nc", "to-dict", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out, parse_constant=_reject_non_finite)
        assert "data_vars" in payload
        assert "coords" in payload

    def test_plain_structure(self, capsys):
        """The plain form prints key: value lines for the structure."""
        assert main(["nc", "to-dict", str(_NC)]) == 0
        assert "data_vars:" in capsys.readouterr().out


class TestNcToZarr:
    """`pyramids nc to-zarr` (needs the [lazy] extra for zarr)."""

    @pytest.mark.lazy
    def test_writes_zarr_store(self, tmp_path):
        """to-zarr writes a per-variable Zarr store."""
        out = tmp_path / "out.zarr"
        assert main(["nc", "to-zarr", str(_NC), str(out), "--var", "tas"]) == 0
        assert out.exists()

    @pytest.mark.lazy
    def test_unknown_variable_errors(self, tmp_path, capsys):
        """An unknown --var exits 1 with a clean error, not a traceback."""
        rc = main(
            ["nc", "to-zarr", str(_NC), str(tmp_path / "o.zarr"), "--var", "nope"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "unexpected failure" not in err
        assert "Traceback" not in err

    @pytest.mark.lazy
    def test_non_raster_variable_errors(self, tmp_path, monkeypatch, capsys):
        """A variable with no .to_zarr (a labeled array) exits 1 with a clear message."""
        monkeypatch.setattr(
            cli_netcdf_io.NetCDF, "get_variable", lambda self, *a, **k: object()
        )
        rc = main(["nc", "to-zarr", str(_NC), str(tmp_path / "o.zarr"), "--var", "tas"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "not exposable as a raster plane" in err
        assert "unexpected failure" not in err


class TestNcFromDataframe:
    """`pyramids nc from-dataframe` (issue #1467)."""

    def test_round_trips_from_csv(self, tmp_path):
        """A CSV written by to-dataframe rebuilds into a readable cube."""
        csv = str(tmp_path / "frame.csv")
        assert main(["nc", "to-dataframe", str(_NC_BANDS), csv]) == 0
        out = str(tmp_path / "rebuilt.nc")
        rc = main(
            [
                "nc",
                "from-dataframe",
                csv,
                out,
                "--index-col",
                "lat",
                "lon",
                "--crs",
                "4326",
            ]
        )
        assert rc == 0
        assert NetCDF.read_file(out).epsg == 4326

    def test_refuses_existing(self, tmp_path):
        """A second from-dataframe without --overwrite is refused."""
        csv = str(tmp_path / "frame.csv")
        assert main(["nc", "to-dataframe", str(_NC_BANDS), csv]) == 0
        out = str(tmp_path / "rebuilt.nc")
        argv = [
            "nc",
            "from-dataframe",
            csv,
            out,
            "--index-col",
            "lat",
            "lon",
            "--crs",
            "4326",
        ]
        assert main(argv) == 0
        assert main(argv) == 1

    def test_bad_index_cols_errors(self, tmp_path, capsys):
        """A tidy (single-index) frame exits 1 cleanly, not with a traceback."""
        csv = str(tmp_path / "frame.csv")
        assert main(["nc", "to-dataframe", str(_NC_BANDS), csv]) == 0
        rc = main(
            ["nc", "from-dataframe", csv, str(tmp_path / "o.nc"), "--index-col", "lat"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "unexpected failure" not in err
        assert "Traceback" not in err


class TestNcFromDict:
    """`pyramids nc from-dict` (issue #1467)."""

    def test_round_trips_from_json(self, tmp_path):
        """A JSON structure dump rebuilds into a cube with the same variables."""
        payload = NetCDF.read_file(str(_NC_BANDS)).to_dict(data=True)
        src = tmp_path / "dump.json"
        src.write_text(json.dumps(json_sanitize(payload)), encoding="utf-8")
        out = str(tmp_path / "rebuilt.nc")
        assert main(["nc", "from-dict", str(src), out]) == 0
        assert "Band1" in list(NetCDF.read_file(out).variable_names)

    def test_refuses_existing(self, tmp_path):
        """A second from-dict without --overwrite is refused."""
        payload = NetCDF.read_file(str(_NC_BANDS)).to_dict(data=True)
        src = tmp_path / "dump.json"
        src.write_text(json.dumps(json_sanitize(payload)), encoding="utf-8")
        out = str(tmp_path / "rebuilt.nc")
        assert main(["nc", "from-dict", str(src), out]) == 0
        assert main(["nc", "from-dict", str(src), out]) == 1


class TestNcConcat:
    """`pyramids nc concat` (issue #1467)."""

    def test_joins_along_time(self, tmp_path):
        """Two 2-step cubes join into one 4-step cube."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0])
        b = _cube(str(tmp_path / "b.nc"), [3.0, 4.0])
        out = str(tmp_path / "cat.nc")
        rc = main(["nc", "concat", out, "--inputs", a, b, "--dim", "time"])
        assert rc == 0
        assert NetCDF.read_file(out).dimension_sizes["time"] == 4

    def test_refuses_existing(self, tmp_path):
        """A second concat without --overwrite is refused."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0])
        b = _cube(str(tmp_path / "b.nc"), [3.0, 4.0])
        out = str(tmp_path / "cat.nc")
        assert main(["nc", "concat", out, "--inputs", a, b, "--dim", "time"]) == 0
        assert main(["nc", "concat", out, "--inputs", a, b, "--dim", "time"]) == 1


class TestNcMerge:
    """`pyramids nc merge` (issue #1467)."""

    def test_unions_variables(self, tmp_path):
        """Two single-variable cubes on one grid merge into both variables."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0], name="rain")
        b = _cube(str(tmp_path / "b.nc"), [3.0, 4.0], name="temp")
        out = str(tmp_path / "mg.nc")
        rc = main(["nc", "merge", out, "--inputs", a, b])
        assert rc == 0
        names = list(NetCDF.read_file(out).variable_names)
        assert "rain" in names
        assert "temp" in names

    def test_refuses_existing(self, tmp_path):
        """A second merge without --overwrite is refused."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0], name="rain")
        out = str(tmp_path / "mg.nc")
        assert main(["nc", "merge", out, "--inputs", a]) == 0
        assert main(["nc", "merge", out, "--inputs", a]) == 1


class TestNcToKerchunk:
    """`pyramids nc to-kerchunk` (needs kerchunk + h5py; issue #1467)."""

    @pytest.mark.netcdf_lazy
    def test_writes_manifest(self, tmp_path):
        """to-kerchunk writes a non-empty JSON reference manifest for an HDF5 cube."""
        src = _cube(str(tmp_path / "cube.nc"), [1.0, 2.0, 3.0])
        out = tmp_path / "ref.json"
        assert main(["nc", "to-kerchunk", src, str(out)]) == 0
        assert out.exists()
        assert out.stat().st_size > 0

    @pytest.mark.netcdf_lazy
    def test_refuses_existing(self, tmp_path):
        """A second to-kerchunk without --overwrite is refused."""
        src = _cube(str(tmp_path / "cube.nc"), [1.0, 2.0, 3.0])
        out = str(tmp_path / "ref.json")
        assert main(["nc", "to-kerchunk", src, out]) == 0
        assert main(["nc", "to-kerchunk", src, out]) == 1


class TestNcCombineKerchunk:
    """`pyramids nc combine-kerchunk` (needs kerchunk + h5py; issue #1467)."""

    @pytest.mark.netcdf_lazy
    def test_writes_combined_manifest(self, tmp_path):
        """combine-kerchunk writes one manifest spanning several HDF5 cubes."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0])
        b = _cube(str(tmp_path / "b.nc"), [3.0, 4.0])
        out = tmp_path / "combined.json"
        rc = main(["nc", "combine-kerchunk", str(out), "--inputs", a, b])
        assert rc == 0
        assert out.exists()
        assert out.stat().st_size > 0


class TestNcOpenMf:
    """`pyramids nc open-mf` (needs dask; issue #1467)."""

    @pytest.mark.lazy
    def test_reports_stack_shape(self, tmp_path, capsys):
        """open-mf reports the lazy stack's shape/dtype over several files."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0])
        b = _cube(str(tmp_path / "b.nc"), [3.0, 4.0])
        assert main(["nc", "open-mf", "--paths", a, b, "--var", "t", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["files"] == 2
        assert payload["shape"][0] == 2
        assert payload["variable"] == "t"

    @pytest.mark.lazy
    def test_text_output_reports_stack(self, tmp_path, capsys):
        """Without --json the stack report prints as 'key: value' lines."""
        a = _cube(str(tmp_path / "a.nc"), [1.0, 2.0])
        b = _cube(str(tmp_path / "b.nc"), [3.0, 4.0])
        assert main(["nc", "open-mf", "--paths", a, b, "--var", "t"]) == 0
        out = capsys.readouterr().out
        assert "variable: t" in out
        assert "files: 2" in out
