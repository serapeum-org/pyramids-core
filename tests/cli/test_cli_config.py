"""Tests for the CLI config-file support (E-2 / #1490)."""

from __future__ import annotations

import argparse

import pytest

from pyramids._cli_config import (
    apply_config_defaults,
    find_config_file,
    load_config,
)
from pyramids.cli import _build_parser, main

pytestmark = pytest.mark.core


class TestFindConfigFile:
    """`find_config_file` discovery and precedence."""

    def test_explicit_path_wins(self, tmp_path):
        """An explicit path is returned when it exists."""
        cfg = tmp_path / "custom.toml"
        cfg.write_text("[defaults]\n")
        assert find_config_file(str(cfg)) == cfg, (
            "explicit --config path should be used"
        )

    def test_missing_explicit_raises(self, tmp_path):
        """A non-existent explicit path is a clean error."""
        with pytest.raises(ValueError, match="does not exist"):
            find_config_file(str(tmp_path / "nope.toml"))

    def test_env_var_is_used(self, tmp_path, monkeypatch):
        """`PYRAMIDS_CONFIG` is honored when no explicit path is passed."""
        cfg = tmp_path / "env.toml"
        cfg.write_text("[defaults]\n")
        monkeypatch.setenv("PYRAMIDS_CONFIG", str(cfg))
        assert find_config_file() == cfg, "PYRAMIDS_CONFIG should resolve the file"

    def test_upward_search_finds_project_file(self, tmp_path, monkeypatch):
        """A ``pyramids.toml`` is found by searching upward from the start dir."""
        monkeypatch.delenv("PYRAMIDS_CONFIG", raising=False)
        (tmp_path / "pyramids.toml").write_text("[defaults]\n")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_config_file(start=str(nested)) == tmp_path / "pyramids.toml"

    def test_missing_env_path_raises(self, tmp_path, monkeypatch):
        """A ``PYRAMIDS_CONFIG`` pointing at a missing file is a clean error."""
        monkeypatch.setenv("PYRAMIDS_CONFIG", str(tmp_path / "ghost.toml"))
        with pytest.raises(ValueError, match="does not exist"):
            find_config_file()

    def test_no_config_anywhere_returns_none(self, tmp_path, monkeypatch):
        """With no env, no project file, and no user file, discovery returns ``None``."""
        monkeypatch.delenv("PYRAMIDS_CONFIG", raising=False)
        monkeypatch.setattr(
            "pyramids._cli_config.Path.home", lambda: tmp_path / "nohome"
        )
        empty = tmp_path / "empty"
        empty.mkdir()
        assert find_config_file(start=str(empty)) is None, "no config means None"


class TestLoadConfig:
    """`load_config` parsing."""

    def test_none_is_empty(self):
        """``None`` path loads to an empty dict."""
        assert load_config(None) == {}, "no config file means no defaults"

    def test_valid_toml(self, tmp_path):
        """A valid file parses into nested tables."""
        cfg = tmp_path / "pyramids.toml"
        cfg.write_text('[vector.warp]\ndriver = "gpkg"\n')
        assert load_config(cfg) == {"vector": {"warp": {"driver": "gpkg"}}}

    def test_invalid_toml_raises(self, tmp_path):
        """A malformed file is a clean ValueError, not a traceback."""
        cfg = tmp_path / "pyramids.toml"
        cfg.write_text("x = = =")
        with pytest.raises(ValueError, match="invalid config file"):
            load_config(cfg)


class TestApplyConfigDefaults:
    """`apply_config_defaults` onto the real parser."""

    def test_defaults_section_applies_to_matching_verbs(self):
        """A ``[defaults]`` value option is applied where that option exists."""
        parser = _build_parser()
        apply_config_defaults(parser, {"defaults": {"driver": "gpkg"}})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.driver == "gpkg", "[defaults] driver should seed the verb default"

    def test_only_plain_store_options_are_configurable(self):
        """Config seeds plain value options but not booleans or append options (round-2 M1).

        An ``append`` default would MERGE with the CLI value instead of being overridden, the
        same sticky failure as a boolean — so only plain ``store`` options are configurable.
        """
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        verb = sub.add_parser("x")
        verb.add_argument("--store")
        verb.add_argument("--tag", action="append")
        verb.add_argument("--flag", action="store_true")
        verb.set_defaults(func=lambda a: 0)
        apply_config_defaults(parser, {"x": {"store": "s", "tag": ["t"], "flag": True}})
        ns = parser.parse_args(["x"])
        assert ns.store == "s", "a plain store option should be seeded"
        assert ns.tag is None, (
            "an append option must NOT be seeded (it would merge, not override)"
        )
        assert ns.flag is False, "a boolean flag must NOT be seeded"

    def test_boolean_flag_is_not_configurable(self):
        """A ``store_true`` flag (``overwrite``) is NOT seeded from config (M1).

        Booleans are excluded because there is no ``--no-overwrite`` token to override a
        config ``true`` back to false, which would make the flag sticky.
        """
        parser = _build_parser()
        apply_config_defaults(parser, {"defaults": {"overwrite": True}})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.overwrite is False, (
            "a boolean flag must stay at its built-in False default"
        )

    def test_command_section_overrides_defaults(self):
        """A ``[group.command]`` value overrides the global ``[defaults]`` one."""
        parser = _build_parser()
        apply_config_defaults(
            parser,
            {"defaults": {"driver": "geojson"}, "vector": {"warp": {"driver": "gpkg"}}},
        )
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.driver == "gpkg", (
            "command-specific default should win over [defaults]"
        )

    def test_unknown_key_is_ignored(self):
        """A key no verb defines is silently ignored (not injected as a stray attr)."""
        parser = _build_parser()
        apply_config_defaults(parser, {"vector": {"warp": {"not_an_option": 1}}})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert not hasattr(ns, "not_an_option"), (
            "unknown config keys must not leak onto args"
        )

    def test_empty_config_is_a_noop(self):
        """An empty config applies nothing and does not raise."""
        parser = _build_parser()
        apply_config_defaults(parser, {})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.driver is None, "no config should leave the built-in default"

    def test_non_dict_intermediate_section_is_ignored(self):
        """A malformed section (``[vector]`` set to a scalar) is skipped, not fatal."""
        parser = _build_parser()
        apply_config_defaults(parser, {"vector": 5})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.driver is None, "a non-dict section must not crash or set anything"

    def test_non_dict_defaults_section_is_ignored(self):
        """A ``[defaults]`` set to a non-table is tolerated and applies nothing."""
        parser = _build_parser()
        apply_config_defaults(parser, {"defaults": "oops"})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.overwrite is False, "a malformed [defaults] must not change defaults"


class TestConfigThroughCLI:
    """End-to-end: a config file changes a verb's default, the CLI overrides it."""

    def test_config_boolean_overwrite_is_ignored_end_to_end(self, cli_points, tmp_path):
        """`[defaults] overwrite=true` does NOT lift the guard — the write is still refused (M1).

        Booleans are not configurable, so the overwrite guard cannot be silently disabled
        from a config file; the user must pass ``--overwrite`` explicitly.
        """
        out = tmp_path / "out.gpkg"
        out.write_text("sentinel")
        cfg = tmp_path / "pyramids.toml"
        cfg.write_text("[defaults]\noverwrite = true\n")
        rc = main(
            [
                "--config",
                str(cfg),
                "vector",
                "warp",
                cli_points,
                str(out),
                "--crs",
                "EPSG:3857",
            ]
        )
        assert rc == 1, "config overwrite=true must not disable the overwrite guard"

    def test_no_config_skips_a_bad_config_file(self, tmp_path):
        """`--no-config` short-circuits before config resolution, so a bad --config is ignored."""
        rc = main(
            [
                "--no-config",
                "--config",
                str(tmp_path / "missing.toml"),
                "completion",
                "bash",
            ]
        )
        assert rc == 0, "--no-config must skip resolving the (missing) config entirely"

    def test_bad_config_does_not_block_help(self, tmp_path, capsys):
        """`--help` still works even when `--config` points at a missing file (round-2 L2)."""
        with pytest.raises(SystemExit) as exc:
            main(["--config", str(tmp_path / "nope.toml"), "--help"])
        assert exc.value.code == 0, "--help should exit 0 despite a broken config"
        assert "usage:" in capsys.readouterr().out.lower(), (
            "help text should still be shown"
        )

    def test_missing_config_is_clean_error_not_traceback(self, tmp_path, capsys):
        """A bad ``--config`` exits 1 with an ``error:`` line, not an uncaught traceback (M2)."""
        rc = main(
            ["--config", str(tmp_path / "nope.toml"), "vector", "info", "x.geojson"]
        )
        assert rc == 1, "a missing --config should map to a clean exit 1"
        assert capsys.readouterr().err.startswith("error:"), (
            "it should print a one-line error:"
        )

    def test_invalid_toml_through_main_is_clean_error(self, tmp_path, capsys):
        """An invalid config file surfaces as ``error:`` + exit 1 through ``main`` (M2)."""
        cfg = tmp_path / "pyramids.toml"
        cfg.write_text("x = = =")
        rc = main(["--config", str(cfg), "vector", "info", "x.geojson"])
        assert rc == 1, "a malformed config should map to a clean exit 1"
        assert "invalid config file" in capsys.readouterr().err, (
            "the TOML error should be reported"
        )
