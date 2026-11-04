"""Tests for the ``pyramids nc`` variable- and dimension-management commands.

The commands are registered by :func:`pyramids.cli_netcdf_vars.add_nc_vars_commands`
but are not yet wired into :func:`pyramids.cli.main`, so each test drives a local
parser built from that registrar directly. The ``run`` helper parses an argv and
calls the resolved handler, returning its exit code; handlers raise on bad input
(``pyramids.cli.main`` is what turns those into exit code 1 in production).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pyramids.cli_netcdf_vars import _parse_pairs, add_nc_vars_commands
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
# Gridded geographic CF file: vars Band1..Band4; dims lon/lat (no band dims).
_NC_BANDS = _DATA / "cf__6v__1d2-2d4__geog__y-asc.nc"
# CF file with a length-one time axis and several variables (area/msk_rgn/pr/tas/ua).
_NC_WIDE = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
# CF file with a 24-step time band dimension and one data variable (tos).
_NC_TIME = _DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"


def _parser() -> argparse.ArgumentParser:
    """Build a parser whose subcommands are only the nc vars commands.

    Returns:
        argparse.ArgumentParser: A parser with ``add_nc_vars_commands`` registered.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    add_nc_vars_commands(sub)
    return parser


def run(argv: list[str]) -> int:
    """Parse ``argv`` with the local parser and invoke the resolved handler.

    Args:
        argv: The command-line tokens, e.g. ``["rename-var", file, out, ...]``.

    Returns:
        int: The handler's exit code.
    """
    namespace = _parser().parse_args(argv)
    return namespace.func(namespace)


class TestParsePairs:
    """The local ``_parse_pairs`` helper."""

    def test_splits_on_first_equals(self):
        """A value containing ``=`` is kept whole; only the first ``=`` splits."""
        parsed = _parse_pairs(["a=b=c", "x=y"])
        assert parsed["a"] == "b=c"
        assert parsed["x"] == "y"

    def test_bad_token_raises(self):
        """A token with no ``=`` is refused with a clear message."""
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            _parse_pairs(["noequals"])


class TestRenameVar:
    """`pyramids nc rename-var`."""

    def test_renames_variable(self, tmp_path):
        """The output carries the new name and not the old one."""
        out = str(tmp_path / "renamed.nc")
        rc = run(["rename-var", str(_NC_BANDS), out, "--old", "Band1", "--new", "B1x"])
        names = list(NetCDF.read_file(out).variable_names)
        assert rc == 0
        assert "B1x" in names
        assert "Band1" not in names

    def test_unknown_old_raises(self, tmp_path):
        """An unknown ``--old`` surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["rename-var", str(_NC_BANDS), out, "--old", "Nope", "--new", "X"])


class TestRemoveVar:
    """`pyramids nc remove-var`."""

    def test_removes_variable(self, tmp_path):
        """The removed variable is absent and the others remain."""
        out = str(tmp_path / "removed.nc")
        rc = run(["remove-var", str(_NC_BANDS), out, "--var", "Band1"])
        names = list(NetCDF.read_file(out).variable_names)
        assert rc == 0
        assert "Band1" not in names
        assert "Band2" in names

    def test_unknown_var_raises(self, tmp_path):
        """An unknown ``--var`` surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["remove-var", str(_NC_BANDS), out, "--var", "Nope"])


class TestReprojectVar:
    """`pyramids nc reproject-var`."""

    def test_reprojects_variable(self, tmp_path):
        """The variable is written in the requested EPSG."""
        out = str(tmp_path / "reproj.nc")
        rc = run(
            ["reproject-var", str(_NC_BANDS), out, "--var", "Band1", "--epsg", "3857"]
        )
        assert rc == 0
        assert NetCDF.read_file(out).get_variable("Band1").epsg == 3857

    def test_unknown_var_raises(self, tmp_path):
        """An unknown ``--var`` surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(
                [
                    "reproject-var",
                    str(_NC_BANDS),
                    out,
                    "--var",
                    "Nope",
                    "--epsg",
                    "3857",
                ]
            )


class TestResampleVar:
    """`pyramids nc resample-var`."""

    def test_resamples_variable(self, tmp_path):
        """The written variable carries the requested cell size."""
        out = str(tmp_path / "resample.nc")
        rc = run(
            [
                "resample-var",
                str(_NC_BANDS),
                out,
                "--var",
                "Band1",
                "--cell-size",
                "1.0",
            ]
        )
        assert rc == 0
        assert NetCDF.read_file(out).get_variable("Band1").cell_size == 1.0

    def test_unknown_var_raises(self, tmp_path):
        """An unknown ``--var`` surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(
                [
                    "resample-var",
                    str(_NC_BANDS),
                    out,
                    "--var",
                    "Nope",
                    "--cell-size",
                    "1.0",
                ]
            )


class TestRenameDims:
    """`pyramids nc rename-dims`."""

    def test_renames_band_dimension(self, tmp_path):
        """The old band dimension name is replaced by the new one."""
        out = str(tmp_path / "rndims.nc")
        rc = run(["rename-dims", str(_NC_TIME), out, "--map", "time=t"])
        dims = list(NetCDF.read_file(out).dimension_names)
        assert rc == 0
        assert "t" in dims
        assert "time" not in dims

    def test_unknown_dimension_raises(self, tmp_path):
        """Renaming a non-band dimension surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["rename-dims", str(_NC_TIME), out, "--map", "nosuch=x"])


class TestDropDims:
    """`pyramids nc drop-dims`."""

    def test_drops_dimension(self, tmp_path):
        """The dropped dimension is absent from the output."""
        out = str(tmp_path / "dropped.nc")
        rc = run(["drop-dims", str(_NC_WIDE), out, "--dims", "plev"])
        dims = list(NetCDF.read_file(out).dimension_names)
        assert rc == 0
        assert "plev" not in dims

    def test_ignore_missing_is_noop(self, tmp_path):
        """``--errors ignore`` makes an unknown dimension a successful no-op."""
        out = str(tmp_path / "ignored.nc")
        rc = run(
            ["drop-dims", str(_NC_WIDE), out, "--dims", "nosuch", "--errors", "ignore"]
        )
        assert rc == 0
        assert "plev" in list(NetCDF.read_file(out).dimension_names)

    def test_missing_dimension_raises(self, tmp_path):
        """An unknown dimension with the default ``raise`` surfaces a ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["drop-dims", str(_NC_WIDE), out, "--dims", "nosuch"])


class TestSetCoords:
    """`pyramids nc set-coords`."""

    def test_promotes_variable(self, tmp_path):
        """A promoted variable leaves ``variable_names`` (now a CF aux coordinate)."""
        out = str(tmp_path / "setcoords.nc")
        rc = run(["set-coords", str(_NC_WIDE), out, "--names", "area"])
        names = list(NetCDF.read_file(out).variable_names)
        assert rc == 0
        assert "area" not in names
        assert "pr" in names

    def test_unknown_name_raises(self, tmp_path):
        """Promoting an unknown variable surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["set-coords", str(_NC_WIDE), out, "--names", "nosuch"])


class TestResetCoords:
    """`pyramids nc reset-coords`."""

    def test_reset_all_is_noop_when_none(self, tmp_path):
        """With no aux coordinate to demote, the output keeps its data variables."""
        out = str(tmp_path / "resetcoords.nc")
        rc = run(["reset-coords", str(_NC_WIDE), out])
        names = list(NetCDF.read_file(out).variable_names)
        assert rc == 0
        assert "area" in names
        assert "msk_rgn" in names

    def test_unknown_name_raises(self, tmp_path):
        """Demoting a name that is not an aux coordinate surfaces a ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["reset-coords", str(_NC_WIDE), out, "--names", "nosuch"])


class TestSqueeze:
    """`pyramids nc squeeze`."""

    def test_drops_length_one_dimension(self, tmp_path):
        """The length-one ``time`` dimension is dropped from the output."""
        out = str(tmp_path / "squeezed.nc")
        rc = run(["squeeze", str(_NC_WIDE), out])
        dims = list(NetCDF.read_file(out).dimension_names)
        assert rc == 0
        assert "time" not in dims

    def test_non_unit_dimension_raises(self, tmp_path):
        """Squeezing a dimension longer than one surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["squeeze", str(_NC_WIDE), out, "--dim", "plev"])


class TestOverwriteGuard:
    """The ``--overwrite`` guard shared by every mutating command."""

    def test_refuses_existing_without_overwrite(self, tmp_path):
        """A second write without ``--overwrite`` is refused with a ValueError."""
        out = str(tmp_path / "guard.nc")
        first = run(["remove-var", str(_NC_BANDS), out, "--var", "Band1"])
        assert first == 0
        with pytest.raises(ValueError):
            run(["remove-var", str(_NC_BANDS), out, "--var", "Band2"])
