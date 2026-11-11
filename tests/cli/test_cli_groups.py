"""Tests for the CLI grouping foundations (Phase 1 of the restructure)."""

from __future__ import annotations

import argparse

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

    def test_close_match_unquoted_choices(self):
        """The 3.12+ unquoted ``(choose from a, b)`` format is parsed too (not just quoted)."""
        msg = "argument cmd: invalid choice: 'wrp' (choose from warp, info, merge)"
        assert _suggestion(msg) == "did you mean: warp?", (
            "the backport must handle the unquoted choice listing argparse emits on Python 3.12+"
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
        assert "warp" in err, "stderr should name the nearest valid verb `warp`"
        assert "mean" in err.lower(), (
            "stderr should phrase it as a did-you-mean suggestion"
        )

    def test_error_appends_backport_hint(self, capsys):
        """`error()` appends the difflib hint when given an old-format (no-native) message.

        Calling ``error`` directly with a pre-3.14-style invalid-choice message exercises the
        backport branch deterministically, regardless of the running interpreter's native
        argparse suggestions.
        """
        parser = SuggestingArgumentParser(prog="t")
        with pytest.raises(SystemExit) as exc:
            parser.error(
                "argument cmd: invalid choice: 'wrp' (choose from 'warp', 'info')"
            )
        assert exc.value.code == 2, "error() exits 2"
        assert "did you mean: warp?" in capsys.readouterr().err, (
            "the backport hint should be appended to the message"
        )

    def test_non_choice_error_has_no_suggestion(self, capsys):
        """A non-invalid-choice error passes through verbatim (no 'did you mean')."""
        parser = SuggestingArgumentParser(prog="t")
        parser.add_argument("--n", type=int, required=True)
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([])
        assert exc.value.code == 2, "a missing-required error still exits 2"
        err = capsys.readouterr().err
        assert "mean" not in err.lower(), (
            "a non-choice error must not gain a did-you-mean hint"
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

    def test_explicit_dest_overrides_default(self):
        """A caller-supplied ``dest`` names the attribute instead of ``<name>_command``."""
        parser = SuggestingArgumentParser(prog="t")
        top = parser.add_subparsers(
            dest="group", required=True, parser_class=SuggestingArgumentParser
        )
        grp = add_group(top, "bands", "band ops", dest="chosen")
        grp.add_parser("add").set_defaults(func=lambda a: None)
        ns = parser.parse_args(["bands", "add"])
        assert ns.chosen == "add", "the explicit dest should record the verb"
        assert not hasattr(ns, "bands_command"), "the default dest must not also appear"


def _subparsers(parser):
    """Return a parser's first subparsers action (or ``None``)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


class TestGroupedCliStructure:
    """Epic invariants: the real `pyramids` parser follows the grouped convention.

    These lock in the restructure so a regression (a verb slipping back to the top level, a
    primitive leaving `advanced`, or a core verb vanishing) fails loudly here rather than only
    in a far-away help-surface test.
    """

    @pytest.fixture(scope="class")
    def root(self):
        """The top-level `pyramids` subparsers action, built from the real CLI."""
        from pyramids.cli import _build_parser

        return _subparsers(_build_parser())

    def _group(self, root, name):
        """Return the ``name`` group's subparsers action."""
        return _subparsers(root.choices[name])

    def test_top_level_lists_the_domain_groups(self, root):
        """`pyramids` exposes exactly the domain groups — no flat verbs remain at the top."""
        top = set(root.choices)
        expected = {
            "raster",
            "vector",
            "nc",
            "collection",
            "stac",
            "fetch",
            "mesh",
            "process",
            "completion",
            "plugins",
        }
        assert expected <= top, f"missing top-level groups: {expected - top}"
        for flat in (
            "info",
            "warp",
            "clip",
            "crop",
            "to-cog",
            "from-bytes",
            "apply",
            "cog",
        ):
            assert flat not in top, f"flat verb {flat!r} must not sit at the top level"

    @pytest.mark.parametrize(
        "domain, advanced_verbs, core_verbs",
        [
            (
                "raster",
                {"read-part", "read-tile", "preview", "point", "blocks", "checksum"},
                {"info", "bounds", "crop", "warp", "merge", "calc"},
            ),
            (
                "nc",
                {"squeeze", "expand-dims", "assign-coords", "weighted", "rolling"},
                {"info", "groups", "get-group", "list-vars", "extract", "sel", "crop"},
            ),
            (
                "vector",
                {"force-2d", "normalize", "shared-paths", "hilbert-distance"},
                {"info", "bounds", "convert", "warp", "clip"},
            ),
            (
                "mesh",
                {"get", "keys", "squeeze", "connected-components", "set-crs"},
                {"info", "extract", "convert"},
            ),
        ],
    )
    def test_advanced_tier_and_core_verbs(
        self, root, domain, advanced_verbs, core_verbs
    ):
        """Each domain hides its primitives under `advanced` and keeps its task verbs at root.

        Args:
            root: The top-level subparsers action.
            domain: The domain group under test.
            advanced_verbs: A sample of primitives that must live under ``<domain> advanced``.
            core_verbs: Task verbs that must remain directly under ``<domain>``.

        Test scenario:
            The sampled primitives resolve only under ``advanced`` (never at the domain root),
            and every core verb is a direct child of the domain group.
        """
        group = self._group(root, domain)
        root_verbs = set(group.choices)
        advanced = set(_subparsers(group.choices["advanced"]).choices)
        assert advanced_verbs <= advanced, (
            f"{domain}: primitives missing from advanced: {advanced_verbs - advanced}"
        )
        assert not (advanced_verbs & root_verbs), (
            f"{domain}: primitives must not also sit at the root: {advanced_verbs & root_verbs}"
        )
        assert core_verbs <= root_verbs, (
            f"{domain}: core verbs missing from the root: {core_verbs - root_verbs}"
        )

    def test_process_tool_pair_is_a_subgroup(self, root):
        """`process tool` resolves the old `tools`/`tool` ambiguity into `list`/`show`."""
        process = self._group(root, "process")
        assert {"run", "apply", "tool"} <= set(process.choices), "process verbs present"
        tool = _subparsers(process.choices["tool"])
        assert {"list", "show"} <= set(tool.choices), "`process tool list|show` present"

    @pytest.mark.parametrize(
        "domain, subsystem", [("vector", "geom"), ("nc", "select"), ("mesh", "read")]
    )
    def test_subgroup_parsers_suggest(self, root, domain, subsystem):
        """A domain's subsystem parser is a `SuggestingArgumentParser`.

        Args:
            root: The top-level subparsers action.
            domain: The domain group whose subsystem is checked.
            subsystem: A subsystem that must be mounted as a suggesting parser.

        Test scenario:
            The subsystem parser carries the did-you-mean backport so a misspelled third-level
            verb (``pyramids vector geom buffr``) is suggested on Python < 3.14 too — not only
            under ``raster``/``process`` (which were always built via ``add_group``).
        """
        subgroup = self._group(root, domain).choices[subsystem]
        assert isinstance(subgroup, SuggestingArgumentParser), (
            f"{domain} {subsystem} parser must suggest near-matches for its verbs"
        )

    def test_nc_io_split_into_export_import(self, root):
        """`nc io` is gone, replaced by `nc export` (writers) / `nc import` (readers)."""
        nc = self._group(root, "nc")
        assert "io" not in nc.choices, "the combined `nc io` subgroup must be removed"
        exp = set(_subparsers(nc.choices["export"]).choices)
        imp = set(_subparsers(nc.choices["import"]).choices)
        assert {
            "dataframe",
            "zarr",
            "kerchunk",
            "convert",
            "combine-kerchunk",
        } <= exp, f"nc export writers missing: {exp}"
        assert {"dataframe", "open-mf", "concat", "merge"} <= imp, (
            f"nc import readers missing: {imp}"
        )
        assert not ({"open-mf", "concat", "merge"} & exp), (
            f"a reader leaked into nc export: {exp}"
        )
        assert not ({"convert", "zarr", "kerchunk", "combine-kerchunk"} & imp), (
            f"a writer leaked into nc import: {imp}"
        )

    def test_vector_io_split_into_export_import(self, root):
        """`vector io` is gone, replaced by `vector export` (writers) / `vector import` (readers)."""
        vector = self._group(root, "vector")
        assert "io" not in vector.choices, (
            "the combined `vector io` subgroup must be removed"
        )
        exp = set(_subparsers(vector.choices["export"]).choices)
        imp = set(_subparsers(vector.choices["import"]).choices)
        assert {"parquet", "feather", "json", "wkt", "wkb", "mvt", "pmtiles"} <= exp, (
            f"vector export writers missing: {exp}"
        )
        assert {"bbox", "xyz", "features", "featureserver", "parquet", "gpx"} <= imp, (
            f"vector import readers missing: {imp}"
        )
        assert not ({"bbox", "xyz", "features", "gpx"} & exp), (
            f"a reader leaked into vector export: {exp}"
        )
        assert not ({"feather", "wkt", "mvt", "pmtiles"} & imp), (
            f"a writer leaked into vector import: {imp}"
        )

    @pytest.mark.parametrize("subsystem", ["export", "read"])
    def test_mesh_io_verbs_have_no_prefix(self, root, subsystem):
        """`mesh export`/`mesh read` verbs carry no leftover `to-`/`from-` prefix.

        Args:
            root: The top-level subparsers action.
            subsystem: The mesh I/O subsystem under test.

        Test scenario:
            Every verb under the subsystem is a plain name (``zarr``, not ``to-zarr``), matching
            the prefix-free `raster export`/`import` surface.
        """
        verbs = _subparsers(self._group(root, "mesh").choices[subsystem]).choices
        leaked = [v for v in verbs if v.startswith(("to-", "from-"))]
        assert not leaked, f"mesh {subsystem} still has prefixed verbs: {leaked}"

    @pytest.mark.parametrize(
        "domain, subsystem, old, new",
        [("nc", "plot", "plot", "render"), ("vector", "overlay", "overlay", "apply")],
    )
    def test_doubled_verbs_renamed(self, root, domain, subsystem, old, new):
        """The verbs that doubled their subsystem's name were renamed.

        Args:
            root: The top-level subparsers action.
            domain: The domain group.
            subsystem: The subsystem whose doubled verb was renamed.
            old: The removed doubled verb name (equal to the subsystem).
            new: The replacement verb name.

        Test scenario:
            ``<domain> <subsystem> <subsystem>`` no longer resolves; the renamed verb does.
        """
        verbs = _subparsers(self._group(root, domain).choices[subsystem]).choices
        assert new in verbs, f"{domain} {subsystem} {new} must exist"
        assert old not in verbs, (
            f"the doubled `{domain} {subsystem} {old}` must be gone"
        )
