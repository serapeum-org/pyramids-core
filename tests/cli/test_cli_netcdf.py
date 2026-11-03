"""Tests for the ``pyramids nc`` command group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
# A gridded geographic CF file (Band1..; crop-friendly; no time axis).
_NC = _DATA / "cf__6v__1d2-2d4__geog__y-asc.nc"
# A CF file that carries a time axis (one step: 2000-05-16).
_NC_TIME = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


class TestNcListVars:
    """`pyramids nc list-vars`."""

    def test_json_lists_variables(self, capsys):
        """--json lists the raster-exposable variables."""
        assert main(["nc", "list-vars", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "Band1" in payload["variables"]

    def test_plain_output(self, capsys):
        """The human form prints one variable per line."""
        assert main(["nc", "list-vars", str(_NC)]) == 0
        assert "Band1" in capsys.readouterr().out


class TestNcExtract:
    """`pyramids nc extract`."""

    def test_extract_band_writes_raster(self, tmp_path):
        """--var writes the variable as a georeferenced raster."""
        out = str(tmp_path / "b1.tif")
        assert main(["nc", "extract", str(_NC), out, "--var", "Band1"]) == 0
        assert Dataset.read_file(out).band_count >= 1

    def test_refuses_existing(self, tmp_path):
        """A second extract without --overwrite is refused."""
        out = str(tmp_path / "b1.tif")
        assert main(["nc", "extract", str(_NC), out, "--var", "Band1"]) == 0
        assert main(["nc", "extract", str(_NC), out, "--var", "Band1"]) == 1

    def test_unknown_variable_errors(self, tmp_path, capsys):
        """An unknown variable exits 1 with a one-line error."""
        rc = main(["nc", "extract", str(_NC), str(tmp_path / "o.tif"), "--var", "Nope"])
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "Traceback" not in err


class TestNcInfo:
    """`pyramids nc info`."""

    def test_json_summary(self, capsys):
        """--json reports variables, dimensions, CRS, bounds, and time-step count."""
        assert main(["nc", "info", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "Band1" in payload["variables"]
        assert isinstance(payload["dimensions"], dict) and payload["dimensions"]
        assert payload["epsg"] == 4326
        assert len(payload["bounds"]) == 4

    def test_plain_lists_keys(self, capsys):
        """The human form prints key: value lines."""
        assert main(["nc", "info", str(_NC)]) == 0
        assert "variables:" in capsys.readouterr().out

    def test_time_axis_reported(self, capsys):
        """A file with a time axis reports its step count and endpoints."""
        assert main(["nc", "info", str(_NC_TIME), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["time_steps"] >= 1
        assert payload["time_start"] is not None


class TestNcSel:
    """`pyramids nc sel`."""

    def test_sel_by_time_writes_netcdf(self, tmp_path):
        """--time selects the step and writes a readable NetCDF."""
        out = str(tmp_path / "sel.nc")
        assert main(["nc", "sel", str(_NC_TIME), out, "--time", "2000-05-16"]) == 0
        assert Dataset.read_file(out) is not None

    def test_refuses_existing(self, tmp_path):
        """A second sel without --overwrite is refused."""
        out = str(tmp_path / "sel.nc")
        assert main(["nc", "sel", str(_NC_TIME), out, "--time", "2000-05-16"]) == 0
        assert main(["nc", "sel", str(_NC_TIME), out, "--time", "2000-05-16"]) == 1

    def test_unknown_var_errors(self, tmp_path, capsys):
        """--var naming an unknown variable exits 1 listing the available ones."""
        rc = main(
            [
                "nc",
                "sel",
                str(_NC_TIME),
                str(tmp_path / "o.nc"),
                "--time",
                "2000-05-16",
                "--var",
                "nope",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "unknown variable 'nope'" in err
        assert "Traceback" not in err


class TestNcCrop:
    """`pyramids nc crop`."""

    def _centered_bbox(self):
        """A centered quarter-extent bbox inside the file's valid data."""
        minx, miny, maxx, maxy = (float(v) for v in NetCDF.read_file(str(_NC)).bbox)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        dx, dy = (maxx - minx) / 4, (maxy - miny) / 4
        return [cx - dx, cy - dy, cx + dx, cy + dy]

    def test_crop_bbox_writes_netcdf(self, tmp_path):
        """--bbox subsets the file and writes a readable output."""
        out = str(tmp_path / "crop.nc")
        bbox = [str(v) for v in self._centered_bbox()]
        assert main(["nc", "crop", str(_NC), out, "--bbox", *bbox]) == 0
        assert Dataset.read_file(out) is not None

    def test_disjoint_bbox_errors(self, tmp_path, capsys):
        """A bbox outside the data exits 1 with a clean error, not a traceback."""
        rc = main(
            [
                "nc",
                "crop",
                str(_NC),
                str(tmp_path / "o.nc"),
                "--bbox",
                "99990",
                "99990",
                "99999",
                "99999",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "Traceback" not in err
