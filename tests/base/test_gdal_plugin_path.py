"""Composition of ``GDAL_DRIVER_PATH`` from the bundled, discovered and companion directories.

Three independent pieces of code used to learn about a GDAL plugin directory and each simply
overwrote ``GDAL_DRIVER_PATH`` with its own. The last writer won, which made the outcome depend on
import order: `_bootstrap` forced the wheel's version-locked directory (the #465 fix), then
`Config.dynamic_env_variables` replaced it with a conda directory whenever ``CONDA_PREFIX`` was set.

`compose_driver_path` turns all of them into one ordered list and `Config.initialize_gdal` writes it
once. These tests pin the ordering contract, the companion entry-point scan, and the regression that
a discovered conda directory can no longer displace the bundled one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyramids.base import _gdal_plugins
from pyramids.base import config as config_module
from pyramids.base._gdal_plugins import (
    DEFAULT_PLUGIN_SUBDIR,
    ENTRY_POINT_GROUP,
    clear_companion_plugin_dir_cache,
    companion_plugin_dirs,
    compose_driver_path,
)
from pyramids.base.config import Config

pytestmark = pytest.mark.core


class _FakeEntryPoint:
    """The two attributes the scan reads off an `importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value


@pytest.fixture
def plugin_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three distinct, existing directories to stand in for the three kinds of input."""
    bundled = tmp_path / "bundled"
    discovered = tmp_path / "discovered"
    companion = tmp_path / "companion"
    for path in (bundled, discovered, companion):
        path.mkdir()
    return bundled, discovered, companion


class TestComposeOrdering:
    """The authoritative directory first, companions always last."""

    def test_bundled_precedes_discovered(self, plugin_dirs):
        """The wheel's ABI-locked dir outranks a conda dir — this is the #465 invariant."""
        bundled, discovered, _ = plugin_dirs

        composed = compose_driver_path(bundled=bundled, discovered=discovered)

        assert composed == os.pathsep.join([str(bundled), str(discovered)])

    def test_companions_come_last(self, plugin_dirs):
        """A companion may add a driver; it may never shadow the version-locked ones."""
        bundled, discovered, companion = plugin_dirs

        composed = compose_driver_path(
            bundled=bundled, discovered=discovered, companions=[companion]
        )

        assert composed.split(os.pathsep) == [
            str(bundled),
            str(discovered),
            str(companion),
        ]

    def test_discovered_leads_when_no_wheel_supplied_one(self, plugin_dirs):
        """On a conda/source install there is no bundled dir, so the active GDAL's own wins."""
        _, discovered, companion = plugin_dirs

        composed = compose_driver_path(
            bundled=None, discovered=discovered, companions=[companion]
        )

        assert composed.split(os.pathsep) == [str(discovered), str(companion)]


class TestComposeHygiene:
    """What goes in is filtered, so GDAL is never handed a path that is not there."""

    def test_nothing_to_point_at_yields_none(self):
        """`None` means "leave GDAL_DRIVER_PATH alone", which an empty string would not."""
        assert compose_driver_path(bundled=None) is None

    def test_a_missing_directory_is_dropped(self, plugin_dirs):
        """A discovered path that has since vanished must not reach GDAL."""
        bundled, _, _ = plugin_dirs

        composed = compose_driver_path(bundled=bundled, discovered=bundled / "gone")

        assert composed == str(bundled)

    def test_duplicates_collapse(self, plugin_dirs):
        """The same directory found twice is listed once."""
        bundled, _, _ = plugin_dirs

        composed = compose_driver_path(
            bundled=bundled, discovered=bundled, companions=[bundled]
        )

        assert composed == str(bundled)

    def test_a_file_is_not_a_plugin_directory(self, tmp_path: Path):
        """`is_dir()` is the filter, so a regular file never lands in the path."""
        regular = tmp_path / "not-a-dir"
        regular.write_text("", encoding="utf-8")

        assert compose_driver_path(bundled=regular) is None


class TestKeepEnvEscapeHatch:
    """`PYRAMIDS_KEEP_GDAL_ENV=1` restores "the inherited value wins"."""

    def test_inherited_value_leads(self, plugin_dirs):
        """With the hatch open the user's own path outranks even the bundled dir."""
        bundled, discovered, _ = plugin_dirs

        composed = compose_driver_path(
            bundled=bundled,
            discovered=discovered,
            inherited=str(discovered),
            keep_env=True,
        )

        assert composed == str(discovered)

    def test_companions_are_still_appended(self, plugin_dirs):
        """The hatch is about precedence, not about dropping registered plugins."""
        bundled, discovered, companion = plugin_dirs

        composed = compose_driver_path(
            bundled=bundled,
            discovered=discovered,
            companions=[companion],
            inherited=str(discovered),
            keep_env=True,
        )

        assert composed.split(os.pathsep) == [str(discovered), str(companion)]

    def test_a_multi_entry_inherited_value_is_preserved(self, plugin_dirs):
        """An inherited path list is split on os.pathsep, not treated as one path."""
        bundled, discovered, companion = plugin_dirs
        inherited = os.pathsep.join([str(discovered), str(companion)])

        composed = compose_driver_path(
            bundled=bundled, inherited=inherited, keep_env=True
        )

        assert composed == inherited

    def test_an_empty_inherited_value_falls_back_to_the_normal_order(self, plugin_dirs):
        """The hatch cannot leave GDAL with nothing when there was nothing to keep."""
        bundled, _, _ = plugin_dirs

        composed = compose_driver_path(bundled=bundled, inherited=None, keep_env=True)

        assert composed == str(bundled)


class TestCompanionEntryPointScan:
    """Companion directories are discovered by metadata, never by importing the companion."""

    def test_the_group_name_is_the_published_contract(self):
        """Renaming it would silently stop finding every installed companion."""
        assert ENTRY_POINT_GROUP == "pyramids.gdal_plugin_path"
        assert DEFAULT_PLUGIN_SUBDIR == "gdalplugins"

    def test_a_module_value_resolves_to_its_default_subdir(self, monkeypatch, tmp_path):
        """`parquet = "pkg"` means `<pkg>/gdalplugins/`."""
        package = tmp_path / "fake_companion"
        (package / DEFAULT_PLUGIN_SUBDIR).mkdir(parents=True)
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("parquet", "fake_companion")],
        )

        assert companion_plugin_dirs(refresh=True) == (package / DEFAULT_PLUGIN_SUBDIR,)

    def test_an_explicit_subdir_is_honoured(self, monkeypatch, tmp_path):
        """`other = "pkg:native/plugins"` points at that relative directory."""
        package = tmp_path / "fake_companion"
        (package / "native" / "plugins").mkdir(parents=True)
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("other", "fake_companion:native/plugins")],
        )

        assert companion_plugin_dirs(refresh=True) == (package / "native" / "plugins",)

    def test_a_registration_cannot_escape_its_package(self, monkeypatch, tmp_path):
        """An absolute or `..` subdir is refused, so GDAL is never aimed off-package."""
        package = tmp_path / "fake_companion"
        package.mkdir()
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [
                _FakeEntryPoint("up", "fake_companion:../elsewhere"),
                _FakeEntryPoint("abs", f"fake_companion:{tmp_path / 'elsewhere'}"),
            ],
        )

        assert companion_plugin_dirs(refresh=True) == ()

    def test_a_missing_directory_is_skipped(self, monkeypatch, tmp_path):
        """A companion that declares the group but ships no directory is simply ignored."""
        package = tmp_path / "fake_companion"
        package.mkdir()
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("parquet", "fake_companion")],
        )

        assert companion_plugin_dirs(refresh=True) == ()

    def test_an_uninstallable_module_is_skipped(self, monkeypatch):
        """`find_spec` returning nothing must not raise out of `import pyramids`."""
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("ghost", "no_such_companion_xyz")],
        )

        assert companion_plugin_dirs(refresh=True) == ()

    def test_a_broken_dist_info_cannot_break_the_import(self, monkeypatch):
        """Metadata on disk is not under our control; a raising scan degrades to empty."""

        def boom(group):
            raise ValueError("corrupt dist-info")

        monkeypatch.setattr(_gdal_plugins, "entry_points", boom)

        assert companion_plugin_dirs(refresh=True) == ()

    def test_the_scan_is_cached_until_cleared(self, monkeypatch):
        """It runs at import time; re-reading the metadata on every call would be waste."""
        calls = []

        def counted(group):
            calls.append(group)
            return []

        monkeypatch.setattr(_gdal_plugins, "entry_points", counted)
        clear_companion_plugin_dir_cache()

        companion_plugin_dirs()
        companion_plugin_dirs()
        assert len(calls) == 1

        clear_companion_plugin_dir_cache()
        companion_plugin_dirs()
        assert len(calls) == 2


class TestConfigIsTheSingleWriter:
    """`Config.initialize_gdal` composes and writes; nothing downstream overwrites it."""

    def test_a_source_checkout_has_no_bundled_directory(self):
        """`_data/gdalplugins` ships only in a wheel, so a dev install must report None."""
        assert Config._bundled_plugin_dir() is None

    def test_a_discovered_conda_dir_no_longer_displaces_the_bundled_one(
        self, monkeypatch, plugin_dirs
    ):
        """The regression: `set_env_conda` used to win simply because it ran last.

        GDAL's own calls are stubbed out — the point is which value `initialize_gdal`
        decides on, and a real `AllRegister()` against a temporary directory would
        leak a bogus driver path into the rest of the session through
        `gdal.SetConfigOption`, which `monkeypatch.setenv` cannot undo.
        """
        bundled, discovered, companion = plugin_dirs
        recorded: dict[str, str] = {}
        monkeypatch.setattr(
            Config, "_bundled_plugin_dir", staticmethod(lambda: bundled)
        )
        monkeypatch.setattr(Config, "dynamic_env_variables", lambda self: discovered)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("parquet", "companion")],
        )
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: companion.parent)
        monkeypatch.setattr(
            _gdal_plugins, "DEFAULT_PLUGIN_SUBDIR", companion.name, raising=True
        )
        clear_companion_plugin_dir_cache()
        monkeypatch.setattr(config_module.gdal, "SetConfigOption", recorded.__setitem__)
        monkeypatch.setattr(config_module.gdal, "AllRegister", lambda: None)
        monkeypatch.setenv("GDAL_DRIVER_PATH", str(discovered))

        Config().initialize_gdal()

        expected = os.pathsep.join([str(bundled), str(discovered), str(companion)])
        assert os.environ["GDAL_DRIVER_PATH"] == expected
        assert recorded["GDAL_DRIVER_PATH"] == expected
