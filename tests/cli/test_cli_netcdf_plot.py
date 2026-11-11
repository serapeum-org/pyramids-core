"""Tests for the ``pyramids nc`` plotting commands (headless, via cleopatra)."""

from __future__ import annotations

import os
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
        assert main(["nc", "plot", "render", str(_NC), str(out), "--var", "tas"]) == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_contour_kind(self, tmp_path):
        """--kind contour renders a contour map."""
        out = tmp_path / "contour.png"
        assert (
            main(
                [
                    "nc",
                    "plot",
                    "render",
                    str(_NC),
                    str(out),
                    "--var",
                    "tas",
                    "--kind",
                    "contour",
                ]
            )
            == 0
        )
        assert out.stat().st_size > 0

    def test_refuses_existing(self, tmp_path):
        """A second plot without --overwrite is refused."""
        out = str(tmp_path / "map.png")
        assert main(["nc", "plot", "render", str(_NC), out, "--var", "tas"]) == 0
        assert main(["nc", "plot", "render", str(_NC), out, "--var", "tas"]) == 1

    def test_unknown_variable_errors_cleanly(self, tmp_path, capsys):
        """An unknown --var exits 1 with a clean error, not the catch-all."""
        rc = main(
            ["nc", "plot", "render", str(_NC), str(tmp_path / "o.png"), "--var", "nope"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "unexpected failure" not in err
        assert "Traceback" not in err

    def test_forces_agg_backend(self, tmp_path, monkeypatch):
        """A pre-set interactive backend is overridden to Agg and the PNG still renders."""
        monkeypatch.setenv("MPLBACKEND", "TkAgg")
        out = str(tmp_path / "forced.png")
        assert main(["nc", "plot", "render", str(_NC), out, "--var", "tas"]) == 0
        assert os.environ["MPLBACKEND"] == "Agg"
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


class TestNcPlotHistogram:
    """`pyramids nc plot-histogram`."""

    def test_writes_png(self, tmp_path):
        """plot-histogram saves a non-empty PNG of a variable's distribution."""
        out = tmp_path / "hist.png"
        assert (
            main(["nc", "plot", "plot-histogram", str(_NC), str(out), "--var", "tas"])
            == 0
        )
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
                    "plot",
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

    def test_unknown_dim_errors_cleanly(self, tmp_path, capsys):
        """An unknown --dim exits 1 with a clean error, not the catch-all (H2)."""
        out = str(tmp_path / "o.gif")
        rc = main(
            [
                "nc",
                "plot",
                "animate",
                str(_NC_TIME),
                out,
                "--var",
                "tos",
                "--dim",
                "nope",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "not a band dim" in err
        assert "unexpected failure" not in err
        assert "Traceback" not in err
