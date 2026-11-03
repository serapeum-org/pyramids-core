"""Tests for the ``pyramids nc`` inspection and global-attribute commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyramids._cli_shared import json_sanitize, map_lookup_errors
from pyramids.cli import main
from pyramids.cli_netcdf_inspect import _metadata_payload
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def _reject_non_finite(token):
    """A `parse_constant` hook that fails on any NaN/Infinity JSON token."""
    raise AssertionError(f"non-finite JSON token emitted: {token}")


def _raise_through_map_lookup(exc):
    """Raise `exc` inside `map_lookup_errors` (the single call under test)."""
    with map_lookup_errors():
        raise exc


class TestMetadataPayload:
    """`_metadata_payload` fallbacks for non-dataclass inputs."""

    def test_object_with_dict_uses_vars(self):
        """An object exposing __dict__ is converted via vars()."""

        class Record:
            def __init__(self):
                self.a = 1

        assert _metadata_payload(Record()) == {"a": 1}

    def test_plain_value_falls_back_to_str(self):
        """A value with no __dict__ falls back to its string form."""
        assert _metadata_payload(42) == "42"


class TestMapLookupErrors:
    """`map_lookup_errors` remaps KeyError/IndexError to a caught ValueError."""

    def test_keyerror_message_preserved(self):
        """A KeyError's argument is carried into the ValueError message."""
        with pytest.raises(ValueError, match="missing"):
            _raise_through_map_lookup(KeyError("missing"))

    def test_argless_error_falls_back_to_str(self):
        """An argless IndexError still remaps (the str(exc) fallback branch)."""
        with pytest.raises(ValueError):
            _raise_through_map_lookup(IndexError())


class TestJsonSanitize:
    """`json_sanitize` makes non-finite / numpy values strict-JSON-safe (M1)."""

    def test_non_finite_floats_become_null(self):
        """NaN and +/-Infinity (nested) are replaced with None; bools are kept."""
        result = json_sanitize(
            {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")], "d": True}
        )
        assert result == {"a": None, "b": None, "c": [1.0, None], "d": True}
        json.loads(json.dumps(result), parse_constant=_reject_non_finite)

    def test_numpy_scalars_and_arrays(self):
        """numpy ints become ints and numpy NaN (scalar or in an array) becomes None."""
        result = json_sanitize(
            {"i": np.int64(5), "f": np.float64("nan"), "a": np.array([1.0, np.nan])}
        )
        assert result == {"i": 5, "f": None, "a": [1.0, None]}
        json.loads(json.dumps(result), parse_constant=_reject_non_finite)


_DATA = Path(__file__).parents[1] / "data" / "netcdf"
_NC = _DATA / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


class TestNcSummary:
    """`pyramids nc summary`."""

    def test_json_summary(self, capsys):
        """--json emits a per-variable statistics mapping, strict-valid JSON."""
        assert main(["nc", "summary", str(_NC), "--vars", "tas", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out, parse_constant=_reject_non_finite)
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
        """--json emits a strict-valid metadata record naming the variables."""
        assert main(["nc", "metadata", str(_NC), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out, parse_constant=_reject_non_finite)
        assert "variables" in payload

    def test_plain_metadata(self, capsys):
        """The plain form prints the metadata record to stdout."""
        assert main(["nc", "metadata", str(_NC)]) == 0
        assert capsys.readouterr().out.strip() != ""


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

    def test_plain_existing(self, capsys):
        """The plain form prints the dimension name and its values."""
        assert main(["nc", "dim-values", str(_NC), "--dim", "lat"]) == 0
        assert "lat:" in capsys.readouterr().out

    def test_plain_missing(self, capsys):
        """The plain form reports a dimension with no coordinate values."""
        assert main(["nc", "dim-values", str(_NC), "--dim", "nope"]) == 0
        assert "no coordinate values" in capsys.readouterr().out


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

    def test_plain_time_values(self, capsys):
        """The plain form prints the variable name and its labels."""
        assert main(["nc", "time-values", str(_NC)]) == 0
        assert "time:" in capsys.readouterr().out


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
