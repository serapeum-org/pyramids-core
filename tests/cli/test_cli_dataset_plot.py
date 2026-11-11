"""Tests for the raster (`Dataset`) plotting CLI commands.

Each shipped ``pyramids`` plot subcommand is a thin wrapper over a single
``Dataset`` plotting method that delegates to cleopatra; these tests pin the
wiring headlessly (Agg backend forced by the handler): a happy path that asserts
a non-empty image/animation file and an error path (missing input or
overwrite-without-flag) that asserts exit code ``1``. All are tagged ``plot``
(the ``[viz]`` extra) so they auto-skip when cleopatra is absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.plot


@pytest.fixture
def uv_raster(tmp_path) -> str:
    """A two-band (u, v) float32 raster on EPSG:4326 for vector-field plots.

    Returns:
        str: Path to the written raster.
    """
    rng = np.random.default_rng(0)
    path = str(tmp_path / "uv.tif")
    Dataset.from_array(
        rng.standard_normal((2, 8, 8)).astype("float32"),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


class TestPlotHistogram:
    """`pyramids plot-histogram`."""

    def test_writes_png(self, cli_raster, tmp_path):
        """A histogram PNG is written and non-empty."""
        out = str(tmp_path / "hist.png")
        assert main(["raster", "plot-histogram", cli_raster, out]) == 0, "must exit 0"
        assert os.path.getsize(out) > 0, "histogram image must be non-empty"

    def test_missing_input_exits_1(self, tmp_path, capsys):
        """A missing input exits 1 with a one-line error."""
        rc = main(
            [
                "raster",
                "plot-histogram",
                str(tmp_path / "nope.tif"),
                str(tmp_path / "h.png"),
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, "missing input must exit 1"
        assert err.startswith("error: "), f"unexpected stderr: {err}"


class TestPlotVectorField:
    """`pyramids plot-vector-field`."""

    def test_writes_png(self, uv_raster, tmp_path):
        """A vector-field PNG is written and non-empty."""
        out = str(tmp_path / "field.png")
        assert main(["raster", "plot-vector-field", uv_raster, out]) == 0, "must exit 0"
        assert os.path.getsize(out) > 0, "vector-field image must be non-empty"

    def test_overwrite_guard(self, uv_raster, tmp_path):
        """An existing output without --overwrite exits 1."""
        out = str(tmp_path / "field.png")
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("x")
        rc = main(["raster", "plot-vector-field", uv_raster, out])
        assert rc == 1, "existing output without --overwrite must exit 1"


class TestAnimate:
    """`pyramids animate`."""

    def test_writes_gif(self, uv_raster, tmp_path):
        """A multi-band GIF animation is written and non-empty."""
        out = str(tmp_path / "anim.gif")
        assert main(["raster", "animate", uv_raster, out]) == 0, "must exit 0"
        assert os.path.getsize(out) > 0, "animation must be non-empty"

    def test_missing_input_exits_1(self, tmp_path):
        """A missing input exits 1."""
        rc = main(
            ["raster", "animate", str(tmp_path / "nope.tif"), str(tmp_path / "a.gif")]
        )
        assert rc == 1, "missing input must exit 1"
