"""Tests for the ``pyramids nc`` plotting commands (headless, via cleopatra)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.cli import main

pytestmark = pytest.mark.plot

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
_NC = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
_NC_TIME = _DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"


class TestNcPlot:
    """`pyramids nc plot`."""

    def test_writes_png(self, tmp_path):
        """plot saves a non-empty PNG for a variable."""
        out = tmp_path / "map.png"
        assert main(["nc", "plot", str(_NC), str(out), "--var", "tas"]) == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_contour_kind(self, tmp_path):
        """--kind contour renders a contour map."""
        out = tmp_path / "contour.png"
        assert (
            main(
                ["nc", "plot", str(_NC), str(out), "--var", "tas", "--kind", "contour"]
            )
            == 0
        )
        assert out.stat().st_size > 0

    def test_refuses_existing(self, tmp_path):
        """A second plot without --overwrite is refused."""
        out = str(tmp_path / "map.png")
        assert main(["nc", "plot", str(_NC), out, "--var", "tas"]) == 0
        assert main(["nc", "plot", str(_NC), out, "--var", "tas"]) == 1


class TestNcPlotHistogram:
    """`pyramids nc plot-histogram`."""

    def test_writes_png(self, tmp_path):
        """plot-histogram saves a non-empty PNG of a variable's distribution."""
        out = tmp_path / "hist.png"
        assert main(["nc", "plot-histogram", str(_NC), str(out), "--var", "tas"]) == 0
        assert out.stat().st_size > 0


class TestNcAnimate:
    """`pyramids nc animate`."""

    def test_writes_gif(self, tmp_path):
        """animate saves a non-empty animation over the time dimension."""
        out = tmp_path / "anim.gif"
        assert (
            main(
                [
                    "nc",
                    "animate",
                    str(_NC_TIME),
                    str(out),
                    "--var",
                    "tos",
                    "--dim",
                    "time",
                ]
            )
            == 0
        )
        assert out.exists()
        assert out.stat().st_size > 0
