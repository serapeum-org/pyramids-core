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
import sys
from pathlib import Path

import pytest

from pyramids.base import _gdal_plugins
from pyramids.base import config as config_module
from pyramids.base._gdal_plugins import (
    DEFAULT_PLUGIN_SUBDIR,
    ENTRY_POINT_GROUP,
    _entry_point_dir,
    _module_dir,
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


@pytest.fixture(autouse=True)
def isolate_companion_cache():
    """Never let a faked companion scan survive the test that faked it.

    The cache is a module global, so `monkeypatch` restoring `entry_points` does not undo it.
    A leaked tuple of still-existing tmp_path directories made every later
    `Config.initialize_gdal()` in the session compose them into `GDAL_DRIVER_PATH` -- and on
    Linux, where neither a bundled nor a discovered directory exists, they became the WHOLE
    path, unloading the conda environment's netCDF/HDF5 plugins. The cross-process tests in
    tests/netcdf/io/test_pickle.py then failed in their spawned children with
    "plugin gdal_HDF5.so is not available".
    """
    clear_companion_plugin_dir_cache()
    yield
    clear_companion_plugin_dir_cache()


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


class TestInheritedIsTheLastResort:
    """When nothing else knows where the plugins are, the inherited value must survive.

    On a conda Linux environment `set_env_conda` finds nothing — it probes the Windows
    `Library/lib/gdalplugins` layout — and there is no bundled directory either, so the
    activation script's `GDAL_DRIVER_PATH` is the only pointer to the netCDF/HDF5 plugins.
    Dropping it handed GDAL a companions-only path and unloaded those drivers.
    """

    def test_it_is_kept_when_nothing_else_exists(self, plugin_dirs):
        """The conda-Linux case with no companion installed."""
        _, discovered, _ = plugin_dirs

        composed = compose_driver_path(
            bundled=None, discovered=None, inherited=str(discovered)
        )

        assert composed == str(discovered)

    def test_a_companion_does_not_displace_it(self, plugin_dirs):
        """The regression: a companion wheel must add to the path, not become it."""
        _, discovered, companion = plugin_dirs

        composed = compose_driver_path(
            bundled=None,
            discovered=None,
            companions=[companion],
            inherited=str(discovered),
        )

        assert composed.split(os.pathsep) == [str(discovered), str(companion)]

    def test_a_bundled_directory_still_outranks_it(self, plugin_dirs):
        """#465 is unchanged: a foreign inherited path never wins over the wheel's own."""
        bundled, discovered, _ = plugin_dirs

        composed = compose_driver_path(
            bundled=bundled, discovered=None, inherited=str(discovered)
        )

        assert composed == str(bundled)

    def test_a_discovered_directory_still_outranks_it(self, plugin_dirs):
        """Same for the conda/system directory that matches the active GDAL."""
        _, discovered, companion = plugin_dirs

        composed = compose_driver_path(
            bundled=None, discovered=discovered, inherited=str(companion)
        )

        assert composed == str(discovered)


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

    @pytest.mark.parametrize(
        "subdir",
        [
            "../elsewhere",
            "/elsewhere",
            "\elsewhere",
            "C:/elsewhere",
            "sub/../../elsewhere",
        ],
    )
    def test_a_registration_cannot_escape_its_package_by_any_spelling(
        self, monkeypatch, tmp_path, subdir
    ):
        """Every way of writing a path out of the package is refused.

        Test scenario:
            The previous guard used `Path.is_absolute()`, which on Windows reports False for a
            drive-less rooted path like `/elsewhere`. `base / "/elsewhere"` then discards the
            package directory and keeps only the drive, so a registration could aim GDAL at
            `C:\elsewhere` and have it `LoadLibrary` every DLL there. Each spelling below must
            resolve to nothing, including the ones containing no `..` at all.
        """
        package = tmp_path / "fake_companion"
        package.mkdir()
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("escape", f"fake_companion:{subdir}")],
        )

        assert companion_plugin_dirs(refresh=True) == (), (
            f"subdir {subdir!r} escaped the package instead of being refused"
        )

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

    def test_the_companion_package_is_never_imported(self, monkeypatch, tmp_path):
        """The scan resolves a real, importable package without importing it.

        This is the load-bearing contract of the whole group: discovery runs inside
        `import pyramids`, before GDAL registers drivers, so importing a companion here would
        drag that distribution's entire dependency tree into every `import pyramids`. The other
        scan tests stub `_module_dir`, so only this one proves the real resolution path is
        import-free.

        Test scenario:
            A genuine package is placed on `sys.path` with a `gdalplugins/` directory and
            registered under the group. The scan must find the directory, and the package must
            still be absent from `sys.modules` afterwards.
        """
        package = tmp_path / "pyramids_fake_companion_pkg"
        (package / DEFAULT_PLUGIN_SUBDIR).mkdir(parents=True)
        (package / "__init__.py").write_text(
            "raise AssertionError('imported!')", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "pyramids_fake_companion_pkg", raising=False)
        monkeypatch.setattr(
            _gdal_plugins,
            "entry_points",
            lambda group: [_FakeEntryPoint("fake", "pyramids_fake_companion_pkg")],
        )

        found = companion_plugin_dirs(refresh=True)

        assert found == (package / DEFAULT_PLUGIN_SUBDIR,), (
            f"the real resolution path must locate the directory, got {found}"
        )
        assert "pyramids_fake_companion_pkg" not in sys.modules, (
            "the companion package was imported; its __init__ would have raised, so discovery "
            "is no longer a pure metadata scan"
        )

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


class TestModuleDir:
    """Locating a companion's directory without importing it.

    Every test in `TestCompanionEntryPointScan` monkeypatches this helper, so its real body is
    exercised only here. It is the one place a companion package name is turned into a path, and
    it must never raise out of `import pyramids` whatever the installed metadata looks like.
    """

    def test_a_package_resolves_to_its_own_directory(self):
        """A regular package is located through its submodule search locations.

        Test scenario:
            `json` is a package, so `submodule_search_locations` is populated and the directory
            holding its `__init__.py` is returned.
        """
        found = _module_dir("json")

        assert found is not None, "a stdlib package must be locatable"
        assert found.name == "json", (
            f"expected the 'json' package directory, got {found}"
        )
        assert (found / "__init__.py").is_file(), f"not a package directory: {found}"

    def test_a_single_file_module_falls_back_to_its_parent(self):
        """A module that is not a package has no search locations, only an origin.

        Test scenario:
            `ast` is a single `ast.py`, so `submodule_search_locations` is empty and the parent of
            `spec.origin` is used instead. This is the `elif spec.origin` branch.
        """
        found = _module_dir("ast")

        assert found is not None, "a stdlib module must be locatable"
        assert (found / "ast.py").is_file(), (
            f"expected the directory containing ast.py, got {found}"
        )

    def test_a_missing_module_is_not_located(self):
        """A name nothing provides yields None rather than raising.

        Test scenario:
            `find_spec` returns None for an unknown top-level name, so the helper reports nothing
            found and the companion is skipped.
        """
        assert _module_dir("no_such_companion_xyz") is None, (
            "an unknown module must not resolve"
        )

    def test_a_missing_parent_package_is_swallowed(self):
        """A dotted name whose parent does not exist raises inside `find_spec`.

        Test scenario:
            `find_spec("no_such_pkg_xyz.sub")` raises `ModuleNotFoundError` (an `ImportError`).
            That must be caught, because a broken registration cannot be allowed to break
            `import pyramids`.
        """
        assert _module_dir("no_such_pkg_xyz.sub") is None, (
            "a missing parent package must not raise"
        )

    def test_a_value_error_from_find_spec_is_swallowed(self):
        """`find_spec` raises ValueError for a module whose `__spec__` is None.

        Test scenario:
            The probe is forced to raise `ValueError`; the helper must report nothing found rather
            than propagate. Patched rather than provoked, because the real trigger depends on
            interpreter state a test should not rely on.
        """
        import pyramids.base._gdal_plugins as module

        def boom(name):
            raise ValueError("__spec__ is None")

        original = module.find_spec
        module.find_spec = boom
        try:
            found = _module_dir("anything")
        finally:
            module.find_spec = original

        assert found is None, "a ValueError from find_spec must be swallowed"

    def test_a_namespace_package_without_a_location_is_not_located(self, monkeypatch):
        """A spec carrying neither search locations nor an origin yields nothing.

        Test scenario:
            Namespace packages can present an empty `submodule_search_locations` and a null
            `origin`. Neither branch can produce a path, so the helper returns None instead of
            constructing `Path(None)`.
        """

        class _Spec:
            submodule_search_locations = []
            origin = None

        monkeypatch.setattr(_gdal_plugins, "find_spec", lambda name: _Spec())

        assert _module_dir("ghost") is None, "a location-less spec must not resolve"


class TestEntryPointDir:
    """Turning one entry-point value into a plugin directory.

    The scan covers the common shapes; these are the edges it cannot reach.
    """

    def test_a_value_naming_no_module_resolves_to_nothing(self):
        """A value that is only a subdir has no package to anchor it.

        Test scenario:
            `":gdalplugins"` leaves the module name empty, so there is nothing to resolve against
            and the registration is ignored.
        """
        assert _entry_point_dir(":gdalplugins") is None, (
            "an anchorless value must not resolve"
        )

    def test_surrounding_whitespace_is_tolerated(self, monkeypatch, tmp_path):
        """Entry-point values are hand-written in TOML and may carry stray spaces.

        Test scenario:
            `"  pkg  :  sub  "` resolves exactly as `"pkg:sub"` would.
        """
        package = tmp_path / "fake_companion"
        (package / "sub").mkdir(parents=True)
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)

        assert _entry_point_dir("  fake_companion  :  sub  ") == package / "sub", (
            "whitespace around the module name or subdir must be ignored"
        )

    def test_an_empty_subdir_falls_back_to_the_default(self, monkeypatch, tmp_path):
        """A trailing colon with nothing after it means "use the default".

        Test scenario:
            `"pkg:"` is treated as `"pkg"`, resolving to `<pkg>/gdalplugins/`.
        """
        package = tmp_path / "fake_companion"
        (package / DEFAULT_PLUGIN_SUBDIR).mkdir(parents=True)
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)

        assert _entry_point_dir("fake_companion:") == package / DEFAULT_PLUGIN_SUBDIR, (
            "an empty subdir must fall back to the default"
        )

    def test_a_file_where_the_directory_should_be_is_refused(
        self, monkeypatch, tmp_path
    ):
        """The target must be a directory; GDAL cannot scan a regular file.

        Test scenario:
            The companion ships `gdalplugins` as a file, so the registration is ignored.
        """
        package = tmp_path / "fake_companion"
        package.mkdir()
        (package / DEFAULT_PLUGIN_SUBDIR).write_text("", encoding="utf-8")
        monkeypatch.setattr(_gdal_plugins, "_module_dir", lambda name: package)

        assert _entry_point_dir("fake_companion") is None, (
            "a file is not a plugin directory"
        )
