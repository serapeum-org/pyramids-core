"""Tests for the `pyramids blocks` CLI verb (list a raster's native block windows)."""

import json

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture
def src_raster(tmp_path) -> str:
    """A small single-band GeoTIFF on disk; returns its path."""
    path = str(tmp_path / "r.tif")
    Dataset.from_array(
        np.ones((1, 8, 8), np.float32),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


class TestBlocksCommand:
    """`pyramids blocks` reports the native block (tile) windows of a raster."""

    def test_json_lists_blocks(self, src_raster, capsys):
        """--json emits a list of block records with window + bounds."""
        assert main(["blocks", src_raster, "--json"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert isinstance(out, list)
        assert len(out) >= 1
        first = out[0]
        assert set(first["window"]) == {"col_off", "row_off", "cols", "rows"}
        assert len(first["bounds"]) == 4
        assert first["index"] == 0

    def test_text_output(self, src_raster, capsys):
        """The default text form prints one readable line per block."""
        assert main(["blocks", src_raster]) == 0
        assert "block 0:" in capsys.readouterr().out

    def test_bad_band_errors(self, src_raster, capsys):
        """An out-of-range band exits 1 with a one-line error."""
        rc = main(["blocks", src_raster, "--band", "9"])
        assert rc == 1
        assert capsys.readouterr().err.strip() != ""

    def test_missing_file_errors(self, tmp_path, capsys):
        """A missing input exits 1."""
        assert main(["blocks", str(tmp_path / "nope.tif")]) == 1
