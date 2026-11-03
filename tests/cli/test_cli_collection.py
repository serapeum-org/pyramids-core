"""Tests for the ``pyramids collection`` command group (M-5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset


def _stack_with(tmp_path, *, epsg: int, no_data: float) -> str:
    """Write three co-registered 3x3 rasters and return a glob matching them.

    Args:
        tmp_path: pytest temp dir.
        epsg: CRS of the stack.
        no_data: no-data sentinel stamped on each raster.

    Returns:
        str: A glob matching the three written rasters.
    """
    for index in range(3):
        Dataset.from_array(
            np.full((3, 3), float(index), dtype="float32"),
            no_data_value=no_data,
            geo_ref=GeoReference(top_left_corner=(0, 3), cell_size=1.0, epsg=epsg),
        ).to_file(str(tmp_path / f"s{index}.tif"))
    return str(tmp_path / "s*.tif")


class TestCollectionInfo:
    """`pyramids collection info` — metadata only, so it runs in the core wheel env."""

    pytestmark = pytest.mark.core

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
    """`pyramids collection stats`.

    Marked ``lazy``, not ``core``: ``stats --reduce`` calls the DatasetCollection reducers, which
    route through ``pyramids.dataset._reduce_ops.resolve_dask_op`` and require the optional ``dask``
    dependency (the ``[lazy]`` extra). The core wheel has no dask, so these must not run in the
    ``-m "core ..."`` wheel jobs — only in the dask-equipped envs (dev / lazy), where the
    Tests - Suite selection ``-m 'not plot ...'`` still picks them up.
    """

    pytestmark = pytest.mark.lazy

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

    def test_carries_base_no_data_value(self, tmp_path):
        """The reduction output carries the base stack's no-data, not -9999 (L2)."""
        glob = _stack_with(tmp_path, epsg=4326, no_data=-1.0)
        out = str(tmp_path / "nd.tif")
        assert main(["collection", "stats", glob, out, "--reduce", "mean"]) == 0
        assert Dataset.read_file(out).no_data_value[0] == -1.0, (
            "stats must carry the base no-data sentinel, not from_array's -9999 default"
        )

    def test_preserves_non_4326_crs(self, tmp_path):
        """A non-4326 stack keeps its CRS on the reduction output (M2 via crs_spec)."""
        glob = _stack_with(tmp_path, epsg=32633, no_data=-9999.0)
        out = str(tmp_path / "utm.tif")
        assert main(["collection", "stats", glob, out, "--reduce", "sum"]) == 0
        assert Dataset.read_file(out).epsg == 32633, "output CRS must match the stack"
