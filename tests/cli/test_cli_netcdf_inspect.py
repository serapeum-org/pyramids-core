"""Tests for the ``pyramids nc`` inspection and global-attribute commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids.cli import main
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_DATA = Path(__file__).parents[1] / "data" / "netcdf"
_NC = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


class TestNcSummary:
    """`pyramids nc summary`."""

    def test_json_summary(self, capsys):
        """--json emits a per-variable statistics mapping."""
        assert main(["nc", "summary", str(_NC), "--vars", "tas", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "tas" in payload

    def test_plain_summary(self, capsys):
        """The plain form prints a table that names the variable."""
        assert main(["nc", "summary", str(_NC), "--vars", "tas"]) == 0
        assert "tas" in capsys.readouterr().out

    def test_unknown_variable_errors_cleanly(self, capsys):
        """An unknown --vars exits 1 with a clean error, not the catch-all (H1)."""
        rc = main(["nc", "summary", str(_NC), "--vars", "nosuchvar"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "nosuchvar" in err
        assert "unexpected failure" not in err
        assert "Traceback" not in err


class TestNcMetadata:
    """`pyramids nc metadata`."""

    def test_json_metadata(self, capsys):
        """--json emits a parseable metadata record naming the variables."""
        assert main(["nc", "metadata", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "variables" in payload


class TestNcDimValues:
    """`pyramids nc dim-values`."""

    def test_existing_dimension(self, capsys):
        """--json lists a dimension's coordinate values."""
        assert main(["nc", "dim-values", str(_NC), "--dim", "lat", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["dim"] == "lat"
        assert len(payload["values"]) > 0

    def test_missing_dimension_is_empty(self, capsys):
        """A dimension with no coordinate values reports an empty list, not an error."""
        assert main(["nc", "dim-values", str(_NC), "--dim", "nope", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["values"] == []


class TestNcTimeValues:
    """`pyramids nc time-values`."""

    def test_json_time_values(self, capsys):
        """--json lists the formatted time-axis labels."""
        assert main(["nc", "time-values", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["var"] == "time"
        assert "2000-05-16" in payload["values"]

    def test_non_time_var_is_empty(self, capsys):
        """A --var that is not a time axis yields an empty list and exit 0 (documented)."""
        assert (
            main(["nc", "time-values", str(_NC), "--var", "nosuchtime", "--json"]) == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["values"] == []


class TestNcSetAttr:
    """`pyramids nc set-attr`."""

    def test_sets_attribute_without_touching_source(self, tmp_path):
        """set-attr writes the attribute into OUT and leaves the source unchanged."""
        out = str(tmp_path / "out.nc")
        assert (
            main(["nc", "set-attr", str(_NC), out, "--name", "foo", "--value", "bar"])
            == 0
        )
        written = NetCDF.read_file(out).get_all_metadata().global_attributes
        assert written.get("foo") == "bar"
        source = NetCDF.read_file(str(_NC)).get_all_metadata().global_attributes
        assert "foo" not in source

    def test_refuses_existing(self, tmp_path):
        """A second set-attr without --overwrite is refused."""
        out = str(tmp_path / "out.nc")
        assert (
            main(["nc", "set-attr", str(_NC), out, "--name", "a", "--value", "b"]) == 0
        )
        assert (
            main(["nc", "set-attr", str(_NC), out, "--name", "a", "--value", "b"]) == 1
        )


class TestNcDelAttr:
    """`pyramids nc del-attr`."""

    def test_deletes_existing_attribute(self, tmp_path):
        """del-attr removes an existing global attribute in the written copy."""
        out = str(tmp_path / "out.nc")
        assert main(["nc", "del-attr", str(_NC), out, "--name", "title"]) == 0
        written = NetCDF.read_file(out).get_all_metadata().global_attributes
        assert "title" not in written

    def test_missing_attribute_is_noop(self, tmp_path):
        """Deleting a missing attribute still succeeds (idempotent)."""
        out = str(tmp_path / "out.nc")
        assert main(["nc", "del-attr", str(_NC), out, "--name", "not_present"]) == 0
