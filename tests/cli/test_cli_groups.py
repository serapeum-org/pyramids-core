"""Tests for the CLI grouping foundations (Phase 1 of the restructure)."""

from __future__ import annotations

import pytest

from pyramids._cli_groups import (
    SuggestingArgumentParser,
    _suggestion,
    add_group,
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
