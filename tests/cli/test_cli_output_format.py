"""Tests for the global ``--format {text,json,csv,table}`` output surface (E-1 / #1489)."""

from __future__ import annotations

import argparse
import json

import pytest

from pyramids._cli_shared import (
    emit,
    resolved_format,
    structured_requested,
)
from pyramids.cli import main

pytestmark = pytest.mark.core


class TestResolvedFormat:
    """`resolved_format` / `structured_requested` precedence."""

    def test_json_flag_is_an_alias(self):
        """The legacy ``--json`` flag resolves to the ``json`` format."""
        args = argparse.Namespace(json=True, format=None)
        assert resolved_format(args) == "json", "legacy --json should map to json"

    def test_explicit_format_wins_when_no_json(self):
        """An explicit ``--format csv`` is returned unchanged."""
        args = argparse.Namespace(json=False, format="csv")
        assert resolved_format(args) == "csv", "explicit --format should be honored"

    def test_explicit_format_beats_json_alias(self):
        """An explicit ``--format csv`` wins over the legacy ``--json`` alias (L1)."""
        args = argparse.Namespace(json=True, format="csv")
        assert resolved_format(args) == "csv", (
            "explicit --format must beat the --json alias"
        )

    def test_default_is_text(self):
        """Nothing set falls back to ``text`` and is not 'structured'."""
        args = argparse.Namespace()
        assert resolved_format(args) == "text", "default should be text"
        assert structured_requested(args) is False, "text is not a structured format"


class TestEmit:
    """`emit` rendering across formats from a payload."""

    def test_json_is_sanitised_compact(self, capsys):
        """JSON output is the compact, non-finite-scrubbed form."""
        emit(argparse.Namespace(format="json"), {"x": float("nan"), "y": 2})
        assert json.loads(capsys.readouterr().out) == {"x": None, "y": 2}

    def test_csv_dict_is_header_and_row(self, capsys):
        """A flat dict renders as a CSV header line plus one value row."""
        emit(argparse.Namespace(format="csv"), {"a": 1, "b": 2})
        assert capsys.readouterr().out.splitlines() == ["a,b", "1,2"]

    def test_csv_single_key_list_is_one_value_per_row(self, capsys):
        """A dict wrapping one list unwraps to a ``value`` column, one row per item."""
        emit(argparse.Namespace(format="csv"), {"mesh_names": ["m1", "m2"]})
        assert capsys.readouterr().out.splitlines() == ["value", "m1", "m2"]

    def test_table_list_of_dicts_is_aligned(self, capsys):
        """A list of dicts renders as an aligned table with a header row."""
        emit(
            argparse.Namespace(format="table"),
            [{"name": "m1", "n": 3}, {"name": "m2", "n": 10}],
        )
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].split() == ["name", "n"], (
            "header columns should be the union of keys"
        )
        assert lines[1].startswith("m1"), "first data row should start with its name"

    def test_text_uses_supplied_rendering(self, capsys):
        """``text`` mode prints the caller's human string, not JSON."""
        emit(argparse.Namespace(format="text"), {"a": 1}, text="a: 1")
        assert capsys.readouterr().out.strip() == "a: 1"

    def test_text_without_rendering_falls_back_to_json(self, capsys):
        """``text`` mode with no ``text=`` prints the JSON form."""
        emit(argparse.Namespace(format=None), {"a": 1})
        assert json.loads(capsys.readouterr().out) == {"a": 1}

    def test_table_renders_none_cell_as_blank(self, capsys):
        """A ``None`` field renders as an empty cell, not the string 'None'."""
        emit(argparse.Namespace(format="table"), [{"a": 1, "b": None}])
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].split() == ["a", "b"], (
            "both columns should appear in the header"
        )
        assert lines[1].strip() == "1", (
            "the None cell should be blank, leaving just the a value"
        )

    def test_csv_ragged_rows_fill_missing_with_blank(self, capsys):
        """Rows with different keys share the union of columns, missing cells blank."""
        emit(argparse.Namespace(format="csv"), [{"a": 1}, {"b": 2}])
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "a,b", "columns should be the first-seen union of keys"
        assert lines[1:] == ["1,", ",2"], (
            "each row fills only its own key, blanks elsewhere"
        )

    def test_csv_nested_value_is_json_encoded_cell(self, capsys):
        """A nested list value is JSON-encoded so it stays one CSV cell."""
        emit(argparse.Namespace(format="csv"), [{"a": [1, 2]}])
        assert capsys.readouterr().out.splitlines()[1] == '"[1, 2]"', (
            "list cell is JSON in one field"
        )

    def test_csv_defuses_formula_injection(self, capsys):
        """A string cell starting with =/+/-/@ is prefixed with a quote (L3)."""
        emit(argparse.Namespace(format="csv"), [{"name": "=SUM(A1:A9)"}])
        row = capsys.readouterr().out.splitlines()[1]
        assert row == "'=SUM(A1:A9)", (
            "a formula-like string cell must be quoted so it is literal"
        )

    def test_csv_leaves_numeric_values_untouched(self, capsys):
        """A negative numeric value is not a string, so it is not quoted."""
        emit(argparse.Namespace(format="csv"), [{"v": -9999.0}])
        assert capsys.readouterr().out.splitlines()[1] == "-9999.0", (
            "numbers are not formula-quoted"
        )


class TestGlobalFormatThroughCLI:
    """The global ``--format`` flag, placed before the command, reaches the verb."""

    def test_format_json_bridges_to_json_output(self, cli_points, capsys):
        """`--format json` makes a --json-capable verb emit JSON without passing --json."""
        assert main(["--format", "json", "vector", "info", cli_points]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["features"] == 5, "features should come through the JSON payload"

    def test_legacy_json_flag_still_works(self, cli_points, capsys):
        """The per-command ``--json`` flag keeps working unchanged (back-compat)."""
        assert main(["vector", "info", cli_points, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["epsg"] == 4326

    def test_format_csv_on_info(self, cli_points, capsys):
        """`--format csv` renders the info payload as CSV (header + one row)."""
        assert main(["--format", "csv", "vector", "info", cli_points]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("path,"), "first CSV line should be the header"
        assert len(lines) == 2, "a single-record payload is one header + one row"

    def test_format_table_on_bounds(self, cli_points, capsys):
        """`--format table` renders the bounds payload as a one-value-per-row table.

        ``{"bounds": [...]}`` is a single-key list, so it unwraps to a ``value`` column
        (one row per bound) — the same clean shape used for name lists.
        """
        assert main(["--format", "table", "vector", "bounds", cli_points]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "value", "single-key list unwraps to a value column"
        assert len(lines) == 5, "header plus the four bound values"

    def test_default_text_is_unchanged(self, cli_points, capsys):
        """With no format flag the verb prints its human text form."""
        assert main(["vector", "info", cli_points]) == 0
        assert "features: 5" in capsys.readouterr().out, "text output must be unchanged"

    def test_csv_on_unsupported_verb_is_a_clean_error(self, cli_raster, capsys):
        """`--format csv` on a verb that doesn't support it errors, not silent text (M3)."""
        rc = main(["--format", "csv", "info", cli_raster])
        assert rc == 1, "csv on an unsupported verb must fail, not silently print text"
        assert "does not support --format csv" in capsys.readouterr().err, (
            "error must name the format"
        )

    def test_json_still_works_on_unsupported_verb(self, cli_raster, capsys):
        """`--format json` still bridges to every --json-capable verb (not gated by M3)."""
        assert main(["--format", "json", "info", cli_raster]) == 0, (
            "json bridge is unaffected"
        )
        assert capsys.readouterr().out.strip().startswith("{"), (
            "info should emit a JSON object"
        )
