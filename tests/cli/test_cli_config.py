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
        """A ``[defaults]`` key is applied only where that option exists."""
        parser = _build_parser()
        apply_config_defaults(parser, {"defaults": {"overwrite": True}})
        ns = parser.parse_args(
            ["vector", "warp", "a.shp", "b.gpkg", "--crs", "EPSG:4326"]
        )
        assert ns.overwrite is True, "[defaults] overwrite should seed the verb default"

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


class TestConfigThroughCLI:
    """End-to-end: a config file changes a verb's default, the CLI overrides it."""

    def test_config_overwrite_default_lets_write_replace(self, cli_points, tmp_path):
        """`[defaults] overwrite=true` lets a write verb replace an existing output."""
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
        assert rc == 0, "overwrite default from config should allow replacing the file"

    def test_no_config_ignores_the_file(self, cli_points, tmp_path):
        """`--no-config` ignores a present config, so the write is refused."""
        out = tmp_path / "out.gpkg"
        out.write_text("sentinel")
        cfg = tmp_path / "pyramids.toml"
        cfg.write_text("[defaults]\noverwrite = true\n")
        rc = main(
            [
                "--no-config",
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
        assert rc == 1, "--no-config must ignore the config's overwrite default"
