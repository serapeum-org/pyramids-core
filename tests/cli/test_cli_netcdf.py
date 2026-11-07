"""Tests for the ``pyramids nc`` command group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids import cli_netcdf
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
# A gridded geographic CF file (Band1..; crop-friendly; no time axis).
_NC = _DATA / "cf__6v__1d2-2d4__geog__y-asc.nc"
# A CF file that carries a time axis (one step: 2000-05-16) and several variables.
_NC_TIME = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
# A CF file with a multi-step time axis (24 monthly steps) for selection tests.
_NC_MULTI = _DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"
# A netCDF-4 file carrying several named sub-groups (per-flight profiles).
_NC_GROUPS = _DATA / "none__35v__1d35__groups-nc4.nc"
_UNEXPECTED = "unexpected failure"


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
        """An unknown variable exits 1 with a clean, mapped error (not the catch-all)."""
        rc = main(["nc", "extract", str(_NC), str(tmp_path / "o.tif"), "--var", "Nope"])
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "not a valid variable name" in err
        assert _UNEXPECTED not in err
        assert "Traceback" not in err


class TestNcNonRasterGuards:
    """The defensive guards for variables GDAL exposes as labeled (non-raster) arrays."""

    def test_extract_non_raster_variable(self, tmp_path, monkeypatch, capsys):
        """extract reports a clear error when get_variable yields a non-raster array."""
        monkeypatch.setattr(
            cli_netcdf.NetCDF, "get_variable", lambda self, *a, **k: object()
        )
        rc = main(
            ["nc", "extract", str(_NC), str(tmp_path / "o.tif"), "--var", "Band1"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "not exposable as a raster plane" in err
        assert "Traceback" not in err

    def test_sel_var_not_selectable(self, tmp_path, monkeypatch, capsys):
        """sel --var reports a clear error when the variable has no .sel method."""
        monkeypatch.setattr(
            cli_netcdf.NetCDF, "get_variable", lambda self, *a, **k: object()
        )
        rc = main(
            [
                "nc",
                "sel",
                str(_NC_TIME),
                str(tmp_path / "o.nc"),
                "--time",
                "2000-05-16",
                "--var",
                "tas",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "not exposable as a raster plane" in err
        assert "Traceback" not in err

    def test_sel_selection_not_writable(self, tmp_path, monkeypatch, capsys):
        """sel reports a clear error when the selection cannot be written to a file."""
        monkeypatch.setattr(cli_netcdf.NetCDF, "sel", lambda self, **k: object())
        rc = main(
            ["nc", "sel", str(_NC_TIME), str(tmp_path / "o.nc"), "--time", "2000-05-16"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "cannot be written to" in err
        assert "Traceback" not in err


class TestNcInfo:
    """`pyramids nc info`."""

    def test_json_summary(self, capsys):
        """--json reports variables, dimensions, CRS, bounds, and time-step count."""
        assert main(["nc", "info", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "Band1" in payload["variables"]
        assert isinstance(payload["dimensions"], dict)
        assert payload["dimensions"]
        assert payload["epsg"] == 4326
        assert len(payload["bounds"]) == 4
        assert payload["time_steps"] == 0, "a file without a time axis reports 0 steps"
        assert payload["time_start"] is None
        assert payload["time_end"] is None

    def test_plain_lists_keys(self, capsys):
        """The human form prints key: value lines."""
        assert main(["nc", "info", str(_NC)]) == 0
        assert "variables:" in capsys.readouterr().out

    def test_time_axis_reported(self, capsys):
        """A file with a time axis reports its step count and endpoints."""
        assert main(["nc", "info", str(_NC_TIME), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["time_steps"] >= 1
        assert payload["time_start"] is not None


class TestNcSel:
    """`pyramids nc sel`."""

    def test_sel_reduces_time_axis_to_one_step(self, tmp_path):
        """--time selects a single step: the output's time axis shrinks to it."""
        out = str(tmp_path / "sel.nc")
        source_steps = len(NetCDF.read_file(str(_NC_MULTI)).get_time_variable())
        assert source_steps > 1, "fixture must have multiple time steps"
        assert main(["nc", "sel", str(_NC_MULTI), out, "--time", "2001-02-16"]) == 0
        out_steps = len(NetCDF.read_file(out).get_time_variable())
        assert out_steps == 1, f"sel should keep one step, got {out_steps}"
        assert out_steps < source_steps

    def test_var_narrows_to_one_variable(self, tmp_path):
        """--var narrows the output to a single variable (the source has several)."""
        out = str(tmp_path / "tas.nc")
        source_vars = len(list(NetCDF.read_file(str(_NC_TIME)).variable_names))
        assert source_vars > 1, "fixture must have several variables"
        rc = main(
            ["nc", "sel", str(_NC_TIME), out, "--time", "2000-05-16", "--var", "tas"]
        )
        assert rc == 0
        out_vars = len(list(NetCDF.read_file(out).variable_names))
        assert out_vars == 1, f"--var should leave one variable, got {out_vars}"

    def test_refuses_existing(self, tmp_path):
        """A second sel without --overwrite is refused."""
        out = str(tmp_path / "sel.nc")
        assert main(["nc", "sel", str(_NC_TIME), out, "--time", "2000-05-16"]) == 0
        assert main(["nc", "sel", str(_NC_TIME), out, "--time", "2000-05-16"]) == 1

    def test_unknown_var_errors(self, tmp_path, capsys):
        """--var naming an unknown variable exits 1 with get_variable's clean error."""
        rc = main(
            [
                "nc",
                "sel",
                str(_NC_TIME),
                str(tmp_path / "o.nc"),
                "--time",
                "2000-05-16",
                "--var",
                "nope",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "nope" in err
        assert "not a valid variable name" in err
        assert _UNEXPECTED not in err
        assert "Traceback" not in err

    def test_group_qualified_var_reaches_get_variable(self, tmp_path, capsys):
        """A group-qualified --var is forwarded to get_variable, not pre-rejected.

        The flat-name pre-check used to refuse any name containing '/'; now sel
        accepts 'group/var' like extract does. This file has no groups, so the error
        is get_variable's own 'Group not found', proving the name reached it.
        """
        rc = main(
            [
                "nc",
                "sel",
                str(_NC_TIME),
                str(tmp_path / "o.nc"),
                "--time",
                "2000-05-16",
                "--var",
                "forecast/temperature",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "Group" in err
        assert "not found" in err
        assert _UNEXPECTED not in err

    def test_file_without_time_axis_errors_cleanly(self, tmp_path, capsys):
        """sel on a file that has no time axis exits 1 with a clean, mapped error."""
        rc = main(
            ["nc", "sel", str(_NC), str(tmp_path / "o.nc"), "--time", "2000-05-16"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "non-spatial dimension" in err
        assert _UNEXPECTED not in err
        assert "Traceback" not in err

    def test_absent_time_label_errors_cleanly(self, tmp_path, capsys):
        """A well-formed --time label that is not on the axis exits 1 cleanly."""
        rc = main(
            ["nc", "sel", str(_NC_TIME), str(tmp_path / "o.nc"), "--time", "1999-01-01"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "No bands match" in err
        assert _UNEXPECTED not in err
        assert "Traceback" not in err


class TestNcCrop:
    """`pyramids nc crop`."""

    def _centered_bbox(self):
        """A centered quarter-extent bbox inside the file's valid data."""
        minx, miny, maxx, maxy = (float(v) for v in NetCDF.read_file(str(_NC)).bbox)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        dx, dy = (maxx - minx) / 4, (maxy - miny) / 4
        return [cx - dx, cy - dy, cx + dx, cy + dy]

    def test_crop_changes_the_spatial_extent(self, tmp_path):
        """--bbox produces an output whose bounds differ from the source (not a copy)."""
        out = str(tmp_path / "crop.nc")
        source_bbox = list(NetCDF.read_file(str(_NC)).bbox)
        bbox = [str(v) for v in self._centered_bbox()]
        assert main(["nc", "crop", str(_NC), out, "--bbox", *bbox]) == 0
        assert list(NetCDF.read_file(out).bbox) != source_bbox, (
            "crop must change the spatial extent, not copy the source"
        )

    def test_refuses_existing(self, tmp_path):
        """A second crop without --overwrite is refused."""
        out = str(tmp_path / "crop.nc")
        bbox = [str(v) for v in self._centered_bbox()]
        assert main(["nc", "crop", str(_NC), out, "--bbox", *bbox]) == 0
        assert main(["nc", "crop", str(_NC), out, "--bbox", *bbox]) == 1

    def test_disjoint_bbox_errors(self, tmp_path, capsys):
        """A bbox outside the data exits 1 with a clean error, not a traceback."""
        rc = main(
            [
                "nc",
                "crop",
                str(_NC),
                str(tmp_path / "o.nc"),
                "--bbox",
                "99990",
                "99990",
                "99999",
                "99999",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "no valid pixels" in err
        assert _UNEXPECTED not in err
        assert "Traceback" not in err


class TestNcInfoFull:
    """`pyramids nc info --full` (issue #1467)."""

    def test_full_prints_ncdump_report(self, capsys):
        """--full prints NetCDF.info()'s ncdump-style report (dims/vars/attrs)."""
        assert main(["nc", "info", str(_NC_TIME), "--full"]) == 0
        out = capsys.readouterr().out
        assert "pyramids.NetCDF {" in out
        assert "dimensions:" in out
        assert "variables:" in out

    def test_default_summary_unchanged(self, capsys):
        """Without --full the compact JSON summary still works."""
        assert main(["nc", "info", str(_NC_TIME), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "variables" in payload
        assert "dimensions" in payload

    def test_full_and_json_are_mutually_exclusive(self):
        """--full and --json together are rejected at argparse, not silently merged (N3)."""
        with pytest.raises(SystemExit):
            main(["nc", "info", str(_NC_TIME), "--full", "--json"])


class TestNcGroups:
    """`pyramids nc groups` (issue #1467)."""

    def test_json_lists_group_names(self, capsys):
        """--json lists the file's sub-group names."""
        assert main(["nc", "groups", str(_NC_GROUPS), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["groups"]) > 0

    def test_flat_file_lists_no_groups(self, capsys):
        """A flat file reports an empty group list and still exits 0."""
        assert main(["nc", "groups", str(_NC), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["groups"] == []

    def test_text_output_lists_names(self, capsys):
        """Without --json the group names print as an indented bullet list."""
        assert main(["nc", "groups", str(_NC_GROUPS)]) == 0
        out = capsys.readouterr().out
        assert out.startswith(f"file: {_NC_GROUPS}")
        assert "  - " in out, "each group name prints as an indented bullet"


class TestNcGetGroup:
    """`pyramids nc get-group` (issue #1467)."""

    def test_writes_subgroup(self, tmp_path):
        """A named sub-group is written out as its own readable NetCDF container."""
        group = NetCDF.read_file(str(_NC_GROUPS)).group_names[0]
        out = str(tmp_path / "grp.nc")
        assert main(["nc", "get-group", str(_NC_GROUPS), out, "--group", group]) == 0
        assert len(list(NetCDF.read_file(out).variable_names)) > 0

    def test_unknown_group_errors(self, tmp_path, capsys):
        """An unknown group exits 1 with a clean error, not a traceback."""
        rc = main(
            [
                "nc",
                "get-group",
                str(_NC_GROUPS),
                str(tmp_path / "o.nc"),
                "--group",
                "nope",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert _UNEXPECTED not in err
        assert "Traceback" not in err

    def test_refuses_existing(self, tmp_path):
        """A second get-group without --overwrite is refused."""
        group = NetCDF.read_file(str(_NC_GROUPS)).group_names[0]
        out = str(tmp_path / "grp.nc")
        assert main(["nc", "get-group", str(_NC_GROUPS), out, "--group", group]) == 0
        assert main(["nc", "get-group", str(_NC_GROUPS), out, "--group", group]) == 1
