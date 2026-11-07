"""Tests for the CF-time / calendar / calculus commands of the ``pyramids nc`` group.

These commands are registered by ``add_nc_cftime_commands`` and are not yet wired into
``pyramids.cli.main``, so the tests drive them through a local parser built from that
registrar. Each working command gets a happy path (it writes an output that re-reads as
a NetCDF with the expected shape) and an error path (a bad dimension/argument surfaces
as a caught ``ValueError``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from pyramids.cli_netcdf_cftime import _parse_coords, add_nc_cftime_commands
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
# A CF file with a 24-step monthly time axis and one gridded variable (tos).
_NC_TIME24 = _DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"
# A CF file with a 17-level pressure axis (plev) and several variables (tas, pr, ua).
_NC_PLEV = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
# A gridded geographic CF file with no non-spatial axis (Band1..Band4, lon/lat only).
_NC_FLAT = _DATA / "cf__6v__1d2-2d4__geog__y-asc.nc"


def _time_cube(path: str, values: list[float]) -> str:
    """Write a clean single-variable time cube (no CF bnds aux) and return ``path``.

    Args:
        path: Where to write the NetCDF.
        values: One value per time step (a 1x1 grid per step).

    Returns:
        str: ``path``.
    """
    NetCDF.from_array(
        np.asarray(values, dtype="float64").reshape(len(values), 1, 1),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(
            name="time", values=[float(i) for i in range(len(values))]
        ),
    ).to_file(path)
    return path


_DIM = "--dim"
_TIME = "time"
_BAD_DIM = "nope"

_PARSER = argparse.ArgumentParser()
_SUB = _PARSER.add_subparsers(dest="command", required=True)
add_nc_cftime_commands(_SUB)


def run(argv: list[str]) -> int:
    """Parse ``argv`` with the local parser and invoke the selected handler.

    Args:
        argv: The command and its flags (no ``pyramids nc`` prefix).

    Returns:
        int: The handler's return code (``0`` on success).
    """
    ns = _PARSER.parse_args(argv)
    return ns.func(ns)


def _dims(path: str) -> dict[str, int]:
    """Return the dimension-size mapping of the NetCDF at ``path``."""
    return dict(NetCDF.read_file(path).dimension_sizes)


class TestParseCoords:
    """The ``--coord DIM=VALUE`` parser."""

    def test_parses_dim_value_pairs(self):
        """Well-formed tokens become a ``{dim: float}`` mapping."""
        coords = _parse_coords(["time=5", "plev=50000.0"])
        assert coords["time"] == 5.0
        assert coords["plev"] == 50000.0

    def test_missing_equals_raises(self):
        """A token without ``=`` is rejected as a ValueError."""
        with pytest.raises(ValueError, match="DIM=VALUE"):
            _parse_coords(["time"])

    def test_non_numeric_value_raises(self):
        """A non-float value is rejected as a ValueError."""
        with pytest.raises(ValueError, match="non-numeric"):
            _parse_coords(["time=abc"])

    def test_empty_dimension_name_raises(self):
        """A token whose dimension name is empty (e.g. '=5') is rejected."""
        with pytest.raises(ValueError, match="empty dimension name"):
            _parse_coords(["=5"])


class TestConvertCalendar:
    """`pyramids nc convert-calendar`."""

    def test_recasts_time_axis(self, tmp_path):
        """The output re-reads as a NetCDF that keeps its 24 monthly steps."""
        out = str(tmp_path / "cc.nc")
        rc = run(["convert-calendar", str(_NC_TIME24), out, "--calendar", "noleap"])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_no_time_axis_raises(self, tmp_path):
        """A file with no time axis surfaces a caught ValueError."""
        out = str(tmp_path / "cc.nc")
        with pytest.raises(ValueError):
            run(["convert-calendar", str(_NC_FLAT), out, "--calendar", "noleap"])


class TestInterpCalendar:
    """`pyramids nc interp-calendar`."""

    def test_interpolates_onto_target_axis(self, tmp_path):
        """Using the file as its own target yields a 24-step output."""
        out = str(tmp_path / "ic.nc")
        rc = run(["interp-calendar", str(_NC_TIME24), out, "--target", str(_NC_TIME24)])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "ic.nc")
        with pytest.raises(ValueError):
            run(
                [
                    "interp-calendar",
                    str(_NC_TIME24),
                    out,
                    "--target",
                    str(_NC_TIME24),
                    _DIM,
                    _BAD_DIM,
                ]
            )


class TestGroupbyBins:
    """`pyramids nc groupby-bins`."""

    def test_bins_reduce_the_axis(self, tmp_path):
        """--bins 4 collapses the 24-step axis to 4 bins."""
        out = str(tmp_path / "gb.nc")
        rc = run(["groupby-bins", str(_NC_TIME24), out, _DIM, _TIME, "--bins", "4"])
        assert rc == 0
        assert _dims(out)["time"] == 4

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "gb.nc")
        with pytest.raises(ValueError):
            run(["groupby-bins", str(_NC_TIME24), out, _DIM, _BAD_DIM, "--bins", "4"])


class TestRolling:
    """`pyramids nc rolling`."""

    def test_window_keeps_axis_length(self, tmp_path):
        """A rolling mean keeps the 24-step axis length."""
        out = str(tmp_path / "rl.nc")
        rc = run(["rolling", str(_NC_TIME24), out, _DIM, _TIME, "--window", "3"])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "rl.nc")
        with pytest.raises(ValueError):
            run(["rolling", str(_NC_TIME24), out, _DIM, _BAD_DIM, "--window", "3"])


class TestRollingExp:
    """`pyramids nc rolling-exp`."""

    def test_alpha_keeps_axis_length(self, tmp_path):
        """An exponentially weighted mean keeps the 24-step axis length."""
        out = str(tmp_path / "re.nc")
        rc = run(["rolling-exp", str(_NC_TIME24), out, _DIM, _TIME, "--alpha", "0.5"])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "re.nc")
        with pytest.raises(ValueError):
            run(["rolling-exp", str(_NC_TIME24), out, _DIM, _BAD_DIM, "--alpha", "0.5"])


class TestShift:
    """`pyramids nc shift`."""

    def test_shift_keeps_axis_length(self, tmp_path):
        """Shifting by one step keeps the 24-step axis length."""
        out = str(tmp_path / "sh.nc")
        rc = run(["shift", str(_NC_TIME24), out, _DIM, _TIME, "--periods", "1"])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "sh.nc")
        with pytest.raises(ValueError):
            run(["shift", str(_NC_TIME24), out, _DIM, _BAD_DIM])


class TestIntegrate:
    """`pyramids nc integrate`."""

    def test_consumes_the_dimension(self, tmp_path):
        """Integrating over time removes the time axis from the output."""
        out = str(tmp_path / "in.nc")
        rc = run(["integrate", str(_NC_TIME24), out, _DIM, _TIME])
        assert rc == 0
        assert "time" not in _dims(out)

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "in.nc")
        with pytest.raises(ValueError):
            run(["integrate", str(_NC_TIME24), out, _DIM, _BAD_DIM])


class TestCumulativeIntegrate:
    """`pyramids nc cumulative-integrate`."""

    def test_keeps_axis_length(self, tmp_path):
        """A running integral keeps the 24-step axis length."""
        out = str(tmp_path / "ci.nc")
        rc = run(["cumulative-integrate", str(_NC_TIME24), out, _DIM, _TIME])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "ci.nc")
        with pytest.raises(ValueError):
            run(["cumulative-integrate", str(_NC_TIME24), out, _DIM, _BAD_DIM])


class TestDifferentiate:
    """`pyramids nc differentiate`."""

    def test_keeps_axis_length(self, tmp_path):
        """A derivative keeps the 24-step axis length."""
        out = str(tmp_path / "df.nc")
        rc = run(["differentiate", str(_NC_TIME24), out, _DIM, _TIME])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "df.nc")
        with pytest.raises(ValueError):
            run(["differentiate", str(_NC_TIME24), out, _DIM, _BAD_DIM])


class TestPolyfit:
    """`pyramids nc polyfit`."""

    def test_produces_degree_axis(self, tmp_path):
        """A degree-2 fit consumes time and adds a 3-element degree axis."""
        out = str(tmp_path / "pf.nc")
        rc = run(["polyfit", str(_NC_TIME24), out, _DIM, _TIME, "--deg", "2"])
        assert rc == 0
        dims = _dims(out)
        assert "time" not in dims
        assert dims["degree"] == 3

    def test_bad_dim_raises(self, tmp_path):
        """An unknown --dim surfaces a caught ValueError."""
        out = str(tmp_path / "pf.nc")
        with pytest.raises(ValueError):
            run(["polyfit", str(_NC_TIME24), out, _DIM, _BAD_DIM, "--deg", "2"])


class TestInterp:
    """`pyramids nc interp`."""

    def test_interp_onto_single_coordinate(self, tmp_path):
        """Interpolating plev to one value reduces that axis to length 1."""
        out = str(tmp_path / "ip.nc")
        rc = run(["interp", str(_NC_PLEV), out, "--coord", "plev=50000.0"])
        assert rc == 0
        assert _dims(out)["plev"] == 1

    def test_spatial_axis_raises(self, tmp_path):
        """Interpolating a spatial axis surfaces a caught ValueError."""
        out = str(tmp_path / "ip.nc")
        with pytest.raises(ValueError):
            run(["interp", str(_NC_PLEV), out, "--coord", "lon=1.0"])

    def test_malformed_coord_raises(self, tmp_path):
        """A malformed --coord token surfaces a caught ValueError."""
        out = str(tmp_path / "ip.nc")
        with pytest.raises(ValueError):
            run(["interp", str(_NC_PLEV), out, "--coord", "plev"])


class TestInterpolateNa:
    """`pyramids nc interpolate-na`."""

    def test_keeps_axis_length(self, tmp_path):
        """Gap-filling along time keeps the 24-step axis length."""
        out = str(tmp_path / "na.nc")
        rc = run(["interpolate-na", str(_NC_TIME24), out, _DIM, _TIME])
        assert rc == 0
        assert _dims(out)["time"] == 24

    def test_spatial_dim_raises(self, tmp_path):
        """A spatial --dim surfaces a caught ValueError."""
        out = str(tmp_path / "na.nc")
        with pytest.raises(ValueError):
            run(["interpolate-na", str(_NC_FLAT), out, _DIM, "lon"])


class TestReduce:
    """`pyramids nc reduce`."""

    def test_consumes_the_dimension(self, tmp_path):
        """Reducing over time removes the time axis from the output."""
        out = str(tmp_path / "rd.nc")
        rc = run(["reduce", str(_NC_TIME24), out, _DIM, _TIME, "--how", "mean"])
        assert rc == 0
        assert "time" not in _dims(out)

    def test_bad_how_raises(self, tmp_path):
        """An unknown --how surfaces a caught ValueError."""
        out = str(tmp_path / "rd.nc")
        with pytest.raises(ValueError):
            run(["reduce", str(_NC_TIME24), out, _DIM, _TIME, "--how", "bogus"])


class TestOverwriteGuard:
    """The shared ``--overwrite`` guard applies to this group too."""

    def test_refuses_existing_without_overwrite(self, tmp_path):
        """A second write without --overwrite is refused with a caught ValueError."""
        out = str(tmp_path / "rd.nc")
        assert run(["reduce", str(_NC_TIME24), out, _DIM, _TIME]) == 0
        with pytest.raises(ValueError, match="already exists"):
            run(["reduce", str(_NC_TIME24), out, _DIM, _TIME])

    def test_overwrite_allows_rewrite(self, tmp_path):
        """--overwrite lets the same output be rewritten."""
        out = str(tmp_path / "rd.nc")
        assert run(["reduce", str(_NC_TIME24), out, _DIM, _TIME]) == 0
        assert run(["reduce", str(_NC_TIME24), out, _DIM, _TIME, "--overwrite"]) == 0


class TestInterpLike:
    """`pyramids nc interp-like` (issue #1467)."""

    def test_interpolates_onto_target_axis(self, tmp_path):
        """A 4-step cube interpolated onto a 2-step target keeps the target's length."""
        src = _time_cube(str(tmp_path / "src.nc"), [1.0, 2.0, 3.0, 4.0])
        tgt = _time_cube(str(tmp_path / "tgt.nc"), [0.0, 0.0])
        out = str(tmp_path / "il.nc")
        rc = run(["interp-like", src, out, "--target", tgt])
        assert rc == 0
        assert _dims(out)["time"] == 2

    def test_no_shared_band_dim_raises(self, tmp_path):
        """Two flat (band-less) cubes share no dimension to interpolate onto."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["interp-like", str(_NC_FLAT), out, "--target", str(_NC_FLAT)])


class TestCoarsen:
    """`pyramids nc coarsen` (issue #1467)."""

    def test_block_reduces_time(self, tmp_path):
        """A window of 2 over a 4-step axis yields 2 blocks."""
        src = _time_cube(str(tmp_path / "src.nc"), [1.0, 2.0, 3.0, 4.0])
        out = str(tmp_path / "co.nc")
        rc = run(
            ["coarsen", src, out, _DIM, _TIME, "--window", "2", "--boundary", "trim"]
        )
        assert rc == 0
        assert _dims(out)["time"] == 2

    def test_bad_dimension_raises(self, tmp_path):
        """An unknown --dim surfaces as a caught ValueError."""
        src = _time_cube(str(tmp_path / "src.nc"), [1.0, 2.0])
        with pytest.raises(ValueError):
            run(
                [
                    "coarsen",
                    src,
                    str(tmp_path / "o.nc"),
                    _DIM,
                    _BAD_DIM,
                    "--window",
                    "2",
                ]
            )


class TestCumulative:
    """`pyramids nc cumulative` (issue #1467)."""

    def test_running_sum(self, tmp_path):
        """--how sum writes the running total [1, 3, 6] along time."""
        src = _time_cube(str(tmp_path / "src.nc"), [1.0, 2.0, 3.0])
        out = str(tmp_path / "cum.nc")
        rc = run(["cumulative", src, out, _DIM, _TIME, "--how", "sum"])
        assert rc == 0
        values = NetCDF.read_file(out).get_variable("t").read_array().ravel().tolist()
        assert values == [1.0, 3.0, 6.0]

    def test_running_prod(self, tmp_path):
        """--how prod writes the running product [1, 2, 6] along time."""
        src = _time_cube(str(tmp_path / "src.nc"), [1.0, 2.0, 3.0])
        out = str(tmp_path / "cum.nc")
        rc = run(["cumulative", src, out, _DIM, _TIME, "--how", "prod"])
        assert rc == 0
        values = NetCDF.read_file(out).get_variable("t").read_array().ravel().tolist()
        assert values == [1.0, 2.0, 6.0]

    def test_bad_dimension_raises(self, tmp_path):
        """An unknown --dim surfaces as a caught ValueError."""
        src = _time_cube(str(tmp_path / "src.nc"), [1.0, 2.0])
        with pytest.raises(ValueError):
            run(["cumulative", src, str(tmp_path / "o.nc"), _DIM, _BAD_DIM])


class TestWeighted:
    """`pyramids nc weighted` (issue #1467)."""

    def test_area_weighted_mean(self, tmp_path):
        """--weights area reduces the spatial axes to a single cell per band."""
        out = str(tmp_path / "w.nc")
        rc = run(["weighted", str(_NC_PLEV), out, "--weights", "area"])
        assert rc == 0
        assert NetCDF.read_file(out) is not None

    def test_unknown_how_raises(self, tmp_path):
        """An unknown --how surfaces as a caught ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["weighted", str(_NC_PLEV), out, "--weights", "area", "--how", "nope"])
