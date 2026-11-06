"""Tests for the ``pyramids nc`` selection / subset commands.

These commands are not yet wired into ``pyramids.cli.main``, so the suite drives a local
parser built from :func:`add_nc_select_commands` and calls the handler directly. Because
the handlers raise (rather than returning ``1`` the way ``main`` would after catching the
exception), the error-path tests assert the raised ``ValueError`` — which is exactly the
caught type ``main`` maps to a clean ``error:`` exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pyramids.cli_netcdf_select import add_nc_select_commands
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
# Single variable `tos`, 24 monthly time steps — ideal for band windowing on `time`.
_NC_MULTI = _DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"
# Four data variables (Band1..Band4), no time axis — to exercise the single-variable guard.
_NC_MULTIVAR = _DATA / "cf__6v__1d2-2d4__geog__y-asc.nc"
# CF file whose `ua` is (time, plev, lat, lon) — a non-spatial `plev` to select in `subset`.
_NC_WIDE = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"

MULTI = str(_NC_MULTI)
OVERWRITE = "--overwrite"
SOURCE_STEPS = 24

_PARSER = argparse.ArgumentParser()
_SUB = _PARSER.add_subparsers(dest="command", required=True)
add_nc_select_commands(_SUB)


def run(argv: list[str]) -> int:
    """Parse ``argv`` against the local parser and run the selected handler."""
    namespace = _PARSER.parse_args(argv)
    return namespace.func(namespace)


def _readback_variable_count(path: str) -> int:
    """How many data variables (i.e. kept bands) the written file exposes."""
    return len(list(NetCDF.read_file(path).variable_names))


def _readback_time(path: str) -> int:
    """The length of the written file's ``time`` dimension."""
    return NetCDF.read_file(path).dimension_sizes["time"]


class TestNcIsel:
    """`pyramids nc isel`."""

    def test_scalar_index_reduces_time_to_one_step(self, tmp_path):
        """--index time=0 collapses the 24-step axis to a single step."""
        out = str(tmp_path / "isel.nc")
        rc = run(["isel", MULTI, out, "--index", "time=0", OVERWRITE])
        assert rc == 0
        assert _readback_time(out) == 1

    def test_bad_index_raises_value_error(self, tmp_path):
        """An out-of-range index surfaces as a caught ValueError (mapped from IndexError)."""
        out = str(tmp_path / "isel.nc")
        with pytest.raises(ValueError):
            run(["isel", MULTI, out, "--index", "time=999", OVERWRITE])

    def test_malformed_token_raises_value_error(self, tmp_path):
        """A token without '=' is rejected by the DIM=VALUE parser."""
        out = str(tmp_path / "isel.nc")
        with pytest.raises(ValueError, match="DIM=VALUE"):
            run(["isel", MULTI, out, "--index", "time", OVERWRITE])


class TestNcThin:
    """`pyramids nc thin`."""

    def test_stride_halves_the_band_count(self, tmp_path):
        """--step time=2 keeps every second band: 24 -> 12."""
        out = str(tmp_path / "thin.nc")
        rc = run(["thin", MULTI, out, "--step", "time=2", OVERWRITE])
        assert rc == 0
        kept = _readback_variable_count(out)
        assert kept == 12
        assert kept < SOURCE_STEPS

    def test_bad_dimension_raises_value_error(self, tmp_path):
        """A dimension that is not a band dimension raises ValueError."""
        out = str(tmp_path / "thin.nc")
        with pytest.raises(ValueError):
            run(["thin", MULTI, out, "--step", "nope=2", OVERWRITE])


class TestNcHead:
    """`pyramids nc head`."""

    def test_keeps_the_first_n_bands(self, tmp_path):
        """--count time=3 keeps the first three bands."""
        out = str(tmp_path / "head.nc")
        rc = run(["head", MULTI, out, "--count", "time=3", OVERWRITE])
        assert rc == 0
        kept = _readback_variable_count(out)
        assert kept == 3
        assert kept < SOURCE_STEPS

    def test_bad_dimension_raises_value_error(self, tmp_path):
        """A dimension that is not a band dimension raises ValueError."""
        out = str(tmp_path / "head.nc")
        with pytest.raises(ValueError):
            run(["head", MULTI, out, "--count", "nope=3", OVERWRITE])


class TestNcTail:
    """`pyramids nc tail`."""

    def test_keeps_the_last_n_bands(self, tmp_path):
        """--count time=3 keeps the last three bands."""
        out = str(tmp_path / "tail.nc")
        rc = run(["tail", MULTI, out, "--count", "time=3", OVERWRITE])
        assert rc == 0
        kept = _readback_variable_count(out)
        assert kept == 3
        assert kept < SOURCE_STEPS

    def test_bad_dimension_raises_value_error(self, tmp_path):
        """A dimension that is not a band dimension raises ValueError."""
        out = str(tmp_path / "tail.nc")
        with pytest.raises(ValueError):
            run(["tail", MULTI, out, "--count", "nope=3", OVERWRITE])


class TestNcDropIsel:
    """`pyramids nc drop-isel`."""

    def test_dropping_one_position_reduces_the_band_count(self, tmp_path):
        """--index time=0 drops one band: 24 -> 23."""
        out = str(tmp_path / "dropisel.nc")
        rc = run(["drop-isel", MULTI, out, "--index", "time=0", OVERWRITE])
        assert rc == 0
        kept = _readback_variable_count(out)
        assert kept == 23
        assert kept < SOURCE_STEPS

    def test_bad_index_raises_value_error(self, tmp_path):
        """An out-of-range index surfaces as a caught ValueError (mapped from IndexError)."""
        out = str(tmp_path / "dropisel.nc")
        with pytest.raises(ValueError):
            run(["drop-isel", MULTI, out, "--index", "time=999", OVERWRITE])


class TestNcDropSel:
    """`pyramids nc drop-sel`."""

    def test_dropping_one_label_reduces_the_band_count(self, tmp_path):
        """A CF date label drops the matching step: 24 -> 23."""
        out = str(tmp_path / "dropsel.nc")
        rc = run(["drop-sel", MULTI, out, "--label", "time=2001-02-16", OVERWRITE])
        assert rc == 0
        kept = _readback_variable_count(out)
        assert kept == 23
        assert kept < SOURCE_STEPS

    def test_absent_label_raises_value_error(self, tmp_path):
        """An absent label surfaces as a caught ValueError (mapped from KeyError)."""
        out = str(tmp_path / "dropsel.nc")
        with pytest.raises(ValueError):
            run(["drop-sel", MULTI, out, "--label", "time=1999-01-01", OVERWRITE])

    def test_absent_label_ignored_keeps_everything(self, tmp_path):
        """--errors ignore skips an absent label instead of raising."""
        out = str(tmp_path / "dropsel.nc")
        rc = run(
            [
                "drop-sel",
                MULTI,
                out,
                "--label",
                "time=1999-01-01",
                "--errors",
                "ignore",
                OVERWRITE,
            ]
        )
        assert rc == 0
        assert _readback_variable_count(out) == SOURCE_STEPS


class TestNcDropDuplicates:
    """`pyramids nc drop-duplicates`."""

    def test_no_duplicates_preserves_every_band(self, tmp_path):
        """With no repeated stamp the axis is preserved: all 24 bands survive."""
        out = str(tmp_path / "dropdup.nc")
        rc = run(["drop-duplicates", MULTI, out, "--dim", "time", OVERWRITE])
        assert rc == 0
        assert _readback_variable_count(out) == SOURCE_STEPS

    def test_bad_dimension_raises_value_error(self, tmp_path):
        """A dimension that is not a band dimension raises ValueError."""
        out = str(tmp_path / "dropdup.nc")
        with pytest.raises(ValueError):
            run(["drop-duplicates", MULTI, out, "--dim", "nope", OVERWRITE])


class TestNcDropna:
    """`pyramids nc dropna`."""

    def test_how_all_preserves_the_time_axis(self, tmp_path):
        """how=all drops only fully empty steps; `tos` has none, so all 24 survive."""
        out = str(tmp_path / "dropna.nc")
        rc = run(["dropna", MULTI, out, "--dim", "time", "--how", "all", OVERWRITE])
        assert rc == 0
        assert _readback_time(out) == SOURCE_STEPS

    def test_bad_dimension_raises_value_error(self, tmp_path):
        """A dimension present in no variable raises ValueError."""
        out = str(tmp_path / "dropna.nc")
        with pytest.raises(ValueError):
            run(["dropna", MULTI, out, "--dim", "nope", OVERWRITE])


class TestNcPad:
    """`pyramids nc pad`."""

    def test_padding_grows_the_time_axis(self, tmp_path):
        """--width time=1 adds one step on each side: 24 -> 26."""
        out = str(tmp_path / "pad.nc")
        rc = run(["pad", MULTI, out, "--width", "time=1", OVERWRITE])
        assert rc == 0
        grown = _readback_time(out)
        assert grown == 26
        assert grown > SOURCE_STEPS

    def test_padding_with_an_explicit_fill_value(self, tmp_path):
        """--value supplies the constant fill and the command still grows the axis."""
        out = str(tmp_path / "pad.nc")
        rc = run(["pad", MULTI, out, "--width", "time=1", "--value", "0", OVERWRITE])
        assert rc == 0
        assert _readback_time(out) == 26

    def test_bad_dimension_raises_value_error(self, tmp_path):
        """A dimension present in no variable raises ValueError."""
        out = str(tmp_path / "pad.nc")
        with pytest.raises(ValueError):
            run(["pad", MULTI, out, "--width", "nope=1", OVERWRITE])


class TestSingleVariableGuard:
    """The band-windowing commands refuse a file with more than one data variable."""

    def test_thin_refuses_a_multi_variable_file(self, tmp_path):
        """A four-variable file cannot be windowed without first extracting one variable."""
        out = str(tmp_path / "thin.nc")
        with pytest.raises(ValueError, match="single-variable"):
            run(["thin", str(_NC_MULTIVAR), out, "--step", "lat=2", OVERWRITE])


class TestNcSubset:
    """`pyramids nc subset` (issue #1467)."""

    def test_selects_variable_and_extra_dim(self, tmp_path):
        """--var ua --dim plev=0 writes a single-variable georeferenced raster."""
        out = str(tmp_path / "subset.nc")
        rc = run(
            ["subset", str(_NC_WIDE), out, "--var", "ua", "--dim", "plev=0", OVERWRITE]
        )
        assert rc == 0
        assert NetCDF.read_file(out).epsg is not None

    def test_bbox_and_crs_window(self, tmp_path):
        """A --bbox in --crs crops the selected plane without error."""
        out = str(tmp_path / "subset_bbox.nc")
        rc = run(
            [
                "subset",
                str(_NC_WIDE),
                out,
                "--var",
                "ua",
                "--dim",
                "plev=0",
                "--bbox",
                "-50",
                "-50",
                "50",
                "50",
                "--crs",
                "4326",
                OVERWRITE,
            ]
        )
        assert rc == 0

    def test_bbox_string_crs_window(self, tmp_path):
        """--crs accepts a CRS string (e.g. 'EPSG:4326'), not only an EPSG int (L1)."""
        out = str(tmp_path / "subset_strcrs.nc")
        rc = run(
            [
                "subset",
                str(_NC_WIDE),
                out,
                "--var",
                "ua",
                "--dim",
                "plev=0",
                "--bbox",
                "-50",
                "-50",
                "50",
                "50",
                "--crs",
                "EPSG:4326",
                OVERWRITE,
            ]
        )
        assert rc == 0, "a string CRS must be accepted, not rejected by argparse"

    def test_unselected_dimension_raises(self, tmp_path):
        """Leaving the length>1 plev dimension unselected surfaces as a ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(["subset", str(_NC_WIDE), out, "--var", "ua", OVERWRITE])

    def test_bad_dim_token_raises(self, tmp_path):
        """A malformed --dim token is refused with a ValueError."""
        out = str(tmp_path / "o.nc")
        with pytest.raises(ValueError):
            run(
                [
                    "subset",
                    str(_NC_WIDE),
                    out,
                    "--var",
                    "ua",
                    "--dim",
                    "plev",
                    OVERWRITE,
                ]
            )
