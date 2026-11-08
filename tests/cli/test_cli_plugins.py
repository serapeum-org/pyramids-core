"""Tests for CLI plugin discovery (E-4 / #1492)."""

from __future__ import annotations

import json

import pytest

from pyramids import cli_plugins
from pyramids.cli import main
from pyramids.cli_plugins import discover_plugins, load_plugin_commands

pytestmark = pytest.mark.core


class _FakeDist:
    """A stand-in for ``importlib.metadata`` ``Distribution`` with name/version."""

    def __init__(self, name, version):
        self.name = name
        self.version = version


class _FakeEP:
    """A stand-in entry point whose ``load`` returns an injected callable or raises."""

    def __init__(self, name, loader, value="mod:attr", dist=None):
        self.name = name
        self.value = value
        self.dist = dist
        self._loader = loader

    def load(self):
        """Return the registration callable, or raise the injected error."""
        return self._loader()


def _good_loader():
    """Return an ``add_<domain>_commands`` callable that registers one verb."""

    def add(sub):
        parser = sub.add_parser("myplugcmd", help="a plugin verb")
        parser.set_defaults(func=lambda args: 0)

    return add


def _broken_loader():
    """Raise as if the plugin module failed to import."""
    raise ImportError("no module named 'ghost'")


class TestDiscoverPlugins:
    """`discover_plugins` records load status without registering anything."""

    def test_good_plugin_has_no_error(self):
        """A plugin that imports cleanly is recorded with ``error=None``."""
        ep = _FakeEP("myplug", _good_loader, dist=_FakeDist("pyramids-myplug", "1.2.3"))
        (record,) = discover_plugins([ep])
        assert record.error is None, "a clean plugin should have no error"
        assert record.version == "1.2.3", "the distribution version should be captured"

    def test_broken_plugin_is_recorded_not_raised(self):
        """A plugin that fails to import is recorded with its error, not raised."""
        ep = _FakeEP("ghostplug", _broken_loader)
        (record,) = discover_plugins([ep])
        assert record.error is not None, "a broken plugin must record an error"
        assert "ImportError" in record.error, "the error should name the failure type"


class TestLoadPluginCommands:
    """`load_plugin_commands` registers good plugins and isolates broken ones."""

    def test_good_plugin_registers_its_verb(self):
        """A good plugin's registration callable is invoked on the subparsers."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        errors = load_plugin_commands(sub, [_FakeEP("myplug", _good_loader)])
        assert errors == [], "a good plugin should load without error"
        ns = parser.parse_args(["myplugcmd"])
        assert ns.func(ns) == 0, "the plugin verb should be registered and callable"

    def test_broken_plugin_is_skipped_with_error(self):
        """A broken plugin does not abort registration; its error is returned."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        errors = load_plugin_commands(
            sub, [_FakeEP("ghostplug", _broken_loader), _FakeEP("myplug", _good_loader)]
        )
        assert len(errors) == 1, "only the broken plugin should produce an error"
        assert "ghostplug" in errors[0], "the error should name the broken plugin"
        assert parser.parse_args(["myplugcmd"]).func is not None, (
            "the good plugin still registered"
        )


class TestPluginsListCLI:
    """`pyramids plugins list` end-to-end."""

    def test_list_empty_when_none_installed(self, monkeypatch, capsys):
        """With no plugins installed the verb reports that and exits 0."""
        monkeypatch.setattr(cli_plugins, "entry_points", lambda group=None: [])
        assert main(["plugins", "list"]) == 0
        assert "no pyramids CLI plugins" in capsys.readouterr().out

    def test_list_reports_installed_plugin_json(self, monkeypatch, capsys):
        """`plugins list --json` reports an installed plugin's name/version/status."""
        ep = _FakeEP("myplug", _good_loader, dist=_FakeDist("pyramids-myplug", "1.2.3"))
        monkeypatch.setattr(cli_plugins, "entry_points", lambda group=None: [ep])
        assert main(["plugins", "list", "--json"]) == 0
        (record,) = json.loads(capsys.readouterr().out)
        assert record["name"] == "myplug", "the plugin name should be listed"
        assert record["status"] == "ok", "a clean plugin's status should be ok"

    def test_list_supports_format_csv(self, monkeypatch, capsys):
        """`--format csv plugins list` renders CSV (it is emit-routed and marked, round-2 L1)."""
        ep = _FakeEP("myplug", _good_loader, dist=_FakeDist("pyramids-myplug", "1.2.3"))
        monkeypatch.setattr(cli_plugins, "entry_points", lambda group=None: [ep])
        assert main(["--format", "csv", "plugins", "list"]) == 0, (
            "csv must not be rejected on plugins list"
        )
        header = capsys.readouterr().out.splitlines()[0]
        assert header.startswith("name,"), (
            "csv output should start with the record header"
        )

    def test_installed_plugin_verb_is_dispatchable(self, monkeypatch):
        """A discovered plugin's verb is registered on the real top-level parser."""
        ep = _FakeEP("myplug", _good_loader)
        monkeypatch.setattr(cli_plugins, "entry_points", lambda group=None: [ep])
        assert main(["myplugcmd"]) == 0, "the plugin-contributed verb should dispatch"
