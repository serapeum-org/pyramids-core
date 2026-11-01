"""Tests for the ``pyramids collection`` command group (M-5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestCollectionInfo:
    """`pyramids collection info`."""

    def test_json_summary(self, cli_stack, capsys):
        """--json summarises the stack (count, CRS, grid)."""
        assert main(["collection", "info", cli_stack, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["rasters"] == 3
        assert payload["epsg"] == 4326
        assert (payload["rows"], payload["columns"]) == (4, 4)

    def test_empty_glob_errors(self, tmp_path, capsys):
        """A glob matching nothing exits 1 with a clear message."""
        rc = main(["collection", "info", str(tmp_path / "none*.tif")])
        assert rc == 1
        assert "no rasters matched" in capsys.readouterr().err


class TestCollectionStats:
    """`pyramids collection stats`."""

    def test_mean_reduction(self, cli_stack, tmp_path):
        """--reduce mean writes the element-wise mean across the stack."""
        out = str(tmp_path / "mean.tif")
        assert main(["collection", "stats", cli_stack, out, "--reduce", "mean"]) == 0
        # mean of constant rasters 0, 1, 2 is 1.0 everywhere.
        assert np.allclose(np.asarray(Dataset.read_file(out).read_array()), 1.0)

    def test_refuses_existing(self, cli_stack, tmp_path):
        """A second stats write without --overwrite is refused."""
        out = str(tmp_path / "m.tif")
        assert main(["collection", "stats", cli_stack, out, "--reduce", "max"]) == 0
        assert main(["collection", "stats", cli_stack, out, "--reduce", "max"]) == 1
