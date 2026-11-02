"""Tests for the ``pyramids nc`` command group (M-2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

# A small gridded CF NetCDF shipped with the test suite (Band1..Band4).
_NC = Path(__file__).parents[1] / "data" / "netcdf" / "cf__6v__1d2-2d4__geog__y-asc.nc"


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
