"""Tests for the ``pyramids nc`` I/O export commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids import cli_netcdf_io
from pyramids.cli import main
from pyramids.dataset import Dataset

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
_NC = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


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
        """--json emits the file's structure with the expected top-level keys."""
        assert main(["nc", "to-dict", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
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
