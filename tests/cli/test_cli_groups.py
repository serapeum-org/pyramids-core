"""Tests for the CLI grouping foundations (Phase 1 of the restructure)."""

from __future__ import annotations

import argparse
import io

import pytest

from pyramids import _cli_groups as g
from pyramids._cli_groups import (
    SuggestingArgumentParser,
    _suggestion,
    _verb_index,
    add_group,
    rewrite_deprecated,
)

pytestmark = pytest.mark.core


class TestSuggestion:
    """`_suggestion` builds a did-you-mean line from an invalid-choice message."""

    def test_close_match_is_suggested(self):
        """An old-format invalid-choice message yields the nearest valid choice."""
        msg = (
            "argument cmd: invalid choice: 'wrp' (choose from 'warp', 'info', 'merge')"
        )
        assert _suggestion(msg) == "did you mean: warp?", (
            "the close verb should be suggested"
        )

    def test_native_suggestion_is_not_doubled(self):
        """A message that already carries a native 'maybe you meant' is left alone."""
        msg = "invalid choice: 'wrp', maybe you meant 'warp'? (choose from 'warp', 'info')"
        assert _suggestion(msg) is None, (
            "must not add a second suggestion on top of the native one"
        )

    def test_non_choice_message_is_ignored(self):
        """A message that is not an invalid-choice error yields no suggestion."""
        assert _suggestion("argument --foo: expected one argument") is None, (
            "only invalid-choice errors suggest"
        )

    def test_nothing_close_yields_none(self):
        """A wildly different token yields no suggestion."""
        assert (
            _suggestion("invalid choice: 'zzzzzz' (choose from 'warp', 'info')") is None
        )


class TestSuggestingParser:
    """`SuggestingArgumentParser` surfaces a suggestion on a misspelled subcommand."""

    def test_misspelled_subcommand_suggests(self, capsys):
        """Parsing a near-miss subcommand exits 2 and prints a suggestion (native or backport)."""
        parser = SuggestingArgumentParser(prog="t")
        sub = parser.add_subparsers(
            dest="cmd", required=True, parser_class=SuggestingArgumentParser
        )
        sub.add_parser("warp")
        sub.add_parser("info")
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["wrp"])
        assert exc.value.code == 2, "an invalid choice exits with code 2"
        err = capsys.readouterr().err
        assert "warp" in err and "mean" in err.lower(), (
            "stderr should point the user at `warp`"
        )


class TestAddGroup:
    """`add_group` mounts a nested group and returns its subparsers action."""

    def test_group_registers_and_dispatches(self):
        """A verb registered under a group is reachable at the nested path."""
        parser = SuggestingArgumentParser(prog="t")
        top = parser.add_subparsers(
            dest="group", required=True, parser_class=SuggestingArgumentParser
        )
        bands = add_group(top, "bands", "band ops")
        add = bands.add_parser("add")
        add.set_defaults(func=lambda a: 7)
        ns = parser.parse_args(["bands", "add"])
        assert ns.func(ns) == 7, "the nested verb should dispatch"
        assert ns.bands_command == "add", "the group's dest records the chosen verb"

    def test_group_without_verb_is_an_error(self):
        """`<group>` with no verb is a usage error (required subcommand), not a silent no-op."""
        parser = SuggestingArgumentParser(prog="t")
        top = parser.add_subparsers(
            dest="group", required=True, parser_class=SuggestingArgumentParser
        )
        add_group(top, "bands", "band ops")
        with pytest.raises(SystemExit):
            parser.parse_args(["bands"])


class TestVerbIndex:
    """`_verb_index` locates the subcommand token past global options."""

    @pytest.mark.parametrize(
        "argv, expected",
        [
            (["info", "x"], 0),
            (["--quiet", "info", "x"], 1),
            (["--format", "json", "info", "x"], 2),
            (["--quiet", "--progress", "info"], 2),
            (["--config", "c.yaml", "--quiet", "info"], 3),
            (["--quiet"], None),
        ],
    )
    def test_index(self, argv, expected):
        """The first non-option token is found, skipping value-taking globals."""
        assert _verb_index(argv) == expected, (
            f"verb index of {argv} should be {expected}"
        )


class TestRewriteDeprecated:
    """`rewrite_deprecated` maps an old flat verb to its new nested path, warning once."""

    def test_empty_table_is_noop(self):
        """With no deprecations registered the argv is returned unchanged."""
        assert rewrite_deprecated(["info", "x"]) == ["info", "x"], (
            "no-op while the table is empty"
        )

    def test_rewrites_and_warns(self, monkeypatch):
        """A deprecated verb is rewritten to its new path and a warning is printed."""
        monkeypatch.setattr(g, "DEPRECATED", {"oldinfo": ("raster", "info")})
        err = io.StringIO()
        out = rewrite_deprecated(["--quiet", "oldinfo", "x.tif"], stderr=err)
        assert out == ["--quiet", "raster", "info", "x.tif"], (
            "the verb token is replaced by the new path"
        )
        text = err.getvalue()
        assert "deprecated" in text and "raster info" in text and "1.0.0" in text, (
            "a one-line sunset warning"
        )

    def test_non_deprecated_verb_untouched(self, monkeypatch):
        """A verb not in the table passes through unchanged and silently."""
        monkeypatch.setattr(g, "DEPRECATED", {"oldinfo": ("raster", "info")})
        err = io.StringIO()
        assert rewrite_deprecated(["warp", "a", "b"], stderr=err) == ["warp", "a", "b"]
        assert err.getvalue() == "", "a non-deprecated verb prints nothing"
