"""Guard the shared GDAL pin and its wiring against silent breakage.

The conda GDAL stack is declared once, in ``[tool.pixi.feature.gdal.dependencies]``, and
shared into every pixi environment by listing the ``gdal`` feature. The SWIG ``osgeo``
bindings are ABI-locked to ``libgdal``, so that single pin must reach every env — dev / CI
and the lean ``wheel-build`` container alike (the latter is ``no-default-feature``, so it
cannot inherit the pin from ``[tool.pixi.dependencies]``).

Single-sourcing removes version drift, but introduces one new way to break it: an
environment that forgets to list the ``gdal`` feature resolves with no GDAL at all. These
tests enforce the invariants that keep the wiring sound — every environment references the
feature, the feature pins all four packages, and ``ci/gdal-pin.py`` (which CI uses to
install gdal in the conda-forge-shape wheel test) reads that same table.
"""

import importlib.util
import re
import runpy
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GDAL_PIN_SCRIPT = REPO_ROOT / "ci" / "gdal-pin.py"

# The conda packages the shared gdal feature must pin. swig is build-only and lives in
# the wheel-build feature, so it is intentionally excluded here.
GDAL_PACKAGES = [
    "gdal",
    "libgdal-netcdf",
    "libgdal-hdf4",
    "libgdal-grib",
    "libgdal-jp2openjpeg",
]


def _load_pixi() -> dict:
    """Return the ``[tool.pixi]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["pixi"]


def _env_features(env_spec) -> list:
    """Return an environment's feature list (pixi allows a bare list or a table)."""
    return env_spec["features"] if isinstance(env_spec, dict) else env_spec


# The packages ci/setup-gdal-micromamba.sh asks ci/gdal-pin.py to resolve — the shared
# gdal libs plus build-only swig. gdal-pin.py must serve every one of them.
MICROMAMBA_PACKAGES = [
    "gdal",
    "libgdal-netcdf",
    "libgdal-hdf4",
    "libgdal-jp2openjpeg",
    "swig",
]


def _load_gdal_pin():
    """Import ``ci/gdal-pin.py`` as a module (its hyphenated path is not importable)."""
    spec = importlib.util.spec_from_file_location("gdal_pin", GDAL_PIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gdal_feature_pins_every_package():
    """The shared gdal feature must pin exactly the four conda GDAL packages."""
    deps = _load_pixi()["feature"]["gdal"]["dependencies"]
    assert sorted(deps) == sorted(GDAL_PACKAGES)


@pytest.mark.parametrize("env", sorted(_load_pixi()["environments"]))
def test_every_environment_references_gdal_feature(env: str):
    """Every pixi env must list the ``gdal`` feature, or it resolves with no GDAL at all."""
    features = _env_features(_load_pixi()["environments"][env])
    assert "gdal" in features, (
        f"environment {env!r} does not list the 'gdal' feature; it would resolve without "
        "the shared GDAL pin. Add 'gdal' to its features list in [tool.pixi.environments]."
    )


def test_gdal_pin_script_matches_pyproject():
    """``ci/gdal-pin.py`` must emit the shared gdal pin CI installs the wheel against."""
    assert (
        _load_gdal_pin().gdal_spec(PYPROJECT)
        == _load_pixi()["feature"]["gdal"]["dependencies"]["gdal"]
    )


def test_gdal_pin_script_resolves_micromamba_packages():
    """``ci/gdal-pin.py`` must resolve every pin ci/setup-gdal-micromamba.sh requests."""
    pins = _load_gdal_pin().feature_pins()
    missing = [pkg for pkg in MICROMAMBA_PACKAGES if pkg not in pins]
    assert not missing, (
        f"ci/gdal-pin.py cannot resolve {missing} that setup-gdal-micromamba.sh requests"
    )


LOCK = REPO_ROOT / "pixi.lock"
MICROMAMBA_SCRIPT = REPO_ROOT / "ci" / "setup-gdal-micromamba.sh"

# The libraries whose versions ci/native-stack-baseline.toml asserts in a built wheel. The
# cross-compile branch must pin these from the lock, or it re-solves them to whatever is newest.
LOCKED_FROM_PIXI_LOCK = ["proj", "geos", "hdf5"]


def _locked_platforms() -> list[str]:
    """The conda subdirs `pixi.lock` carries entries for in the wheel-build environment."""
    module = _load_gdal_pin()
    candidates = ("linux-64", "linux-aarch64", "osx-64", "osx-arm64", "win-64")
    present = []
    for platform in candidates:
        try:
            module.locked_versions(platform)
        except KeyError:
            continue
        present.append(platform)
    return present


class TestTheLockedPinsFeedTheCrossCompileSolve:
    """`--locked` must give the cross-compile branch the versions the lock already resolved.

    A range spec pins a *version*, not a *solve*. Re-resolving `gdal >=3.13.3,<3.13.4` for
    osx-64 picked the newest matching build, which links proj 9.9.0 and geos 3.15.0, while every
    platform installed from `pixi.lock` got the older build with proj 9.8.1 and geos 3.14.1. One
    release shipped two different native stacks under one version number, and only the osx-64
    wheel -- the single cross-compiled target -- was affected (#1627).
    """

    def test_the_locked_native_stack_agrees_across_every_platform(self):
        """The parity property itself: one native stack per release, whatever the platform."""
        module = _load_gdal_pin()
        platforms = _locked_platforms()
        resolved = {
            platform: module.locked_pins(LOCKED_FROM_PIXI_LOCK, platform)
            for platform in platforms
        }

        assert len(platforms) >= 2, (
            f"expected several locked platforms, found {platforms}"
        )
        distinct = {tuple(sorted(pins.items())) for pins in resolved.values()}
        assert len(distinct) == 1, (
            "pixi.lock resolves a different native stack per platform, so no single pin can make "
            f"the wheels agree: {resolved}"
        )

    def test_locked_pins_are_exact_equality_specs(self):
        """A floor would let the solver drift upward again, so the pin must be `==`."""
        pins = _load_gdal_pin().locked_pins(LOCKED_FROM_PIXI_LOCK, "osx-64")

        assert sorted(pins) == sorted(LOCKED_FROM_PIXI_LOCK), (
            f"every name must resolve: {pins}"
        )
        for name, spec in pins.items():
            assert spec.startswith("=="), f"{name} must be pinned exactly, got {spec!r}"

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (
                {"names": ["no-such-package"], "platform": "osx-64"},
                "pins no no-such-package",
            ),
            ({"names": ["proj"], "platform": "osx-99"}, "no platform"),
            (
                {"names": ["proj"], "platform": "osx-64", "environment": "nope"},
                "no environment",
            ),
        ],
        ids=["unknown-package", "unknown-platform", "unknown-environment"],
    )
    def test_an_unresolvable_pin_raises_rather_than_being_dropped(
        self, kwargs, expected
    ):
        """A pin that resolves to nothing must fail loudly, not hand the choice back to the solver.

        Args:
            kwargs: Arguments naming something absent from the lock.
            expected: Text the raised message must contain.
        """
        module = _load_gdal_pin()

        with pytest.raises(KeyError, match=expected):
            module.locked_pins(**kwargs)

    def test_the_range_mode_still_reads_pyproject(self):
        """Adding `--locked` must not change what the original mode emits."""
        module = _load_gdal_pin()

        assert module.gdal_spec(PYPROJECT).startswith(">="), (
            "the range mode must keep emitting the pyproject spec, not an exact pin"
        )


class TestTheCrossCompileScriptUsesTheLockedPins:
    """The wiring: `setup-gdal-micromamba.sh` must actually consume `--locked`."""

    def test_the_script_reads_the_locked_pins(self):
        """Without this call the branch re-solves and the divergence returns."""
        script = MICROMAMBA_SCRIPT.read_text(encoding="utf-8")

        assert "--locked" in script, (
            "ci/setup-gdal-micromamba.sh must call gdal-pin.py --locked, or it re-resolves the "
            "native stack to whatever conda-forge currently ships newest (#1627)"
        )
        assert '--platform "${TARGET_PLATFORM}"' in script, (
            "the locked lookup must use the target platform, not the host's"
        )

    @pytest.mark.parametrize("library", LOCKED_FROM_PIXI_LOCK)
    def test_every_baseline_library_is_passed_to_micromamba(self, library):
        """Each library the baseline asserts a version for must reach the solve as a pin.

        Args:
            library: The conda package name that must appear in the create command.
        """
        script = MICROMAMBA_SCRIPT.read_text(encoding="utf-8")
        variable = f"${{{library.upper()}_SPEC}}"

        assert f'"{library}{variable}"' in script, (
            f"{library} must be passed to micromamba create as {library}{variable}, or its "
            "version is left to the solver on the cross-compile branch"
        )


# The reader matches an entry only on a line ending in a newline, and the environments section's
# very last line does not end in one -- it is where the top-level `packages:` section was split off.
# Every synthetic lock therefore ends with this sentinel, whose file name carries no parseable
# version, so these tests describe how a name and version are read rather than which entry lands
# last. It contributes no pin whether or not the reader reaches it.
_TRAILING_ENTRY = "unparseable-trailing-entry.conda"


def _lock_entries(
    platform: str, filenames: list[str], environment: str = "wheel-build"
) -> str:
    """One environment's platform block, in the shape `pixi.lock` writes it.

    Args:
        platform: The conda subdir the entries belong to.
        filenames: Package file names, e.g. `proj-9.8.1-h856bcef_1.conda`. `_TRAILING_ENTRY` is
            appended after them, so none of these is the block's final line.
        environment: The environment name to declare them under.

    Returns:
        The indented YAML lines, ready to hand to `_synthetic_lock`.
    """
    head = (
        f"  {environment}:\n"
        "    channels:\n"
        "    - url: https://conda.anaconda.org/conda-forge/\n"
        "    packages:\n"
        f"      {platform}:\n"
    )
    channel = "https://conda.anaconda.org/conda-forge"
    entries = "".join(
        f"      - conda: {channel}/{platform}/{name}\n"
        for name in [*filenames, _TRAILING_ENTRY]
    )
    return head + entries


def _synthetic_lock(tmp_path: Path, body: str) -> Path:
    """Write a lock carrying only the `environments:` section the reader parses.

    Args:
        tmp_path: Directory to write the lock into.
        body: The lines under `environments:`, already indented, from `_lock_entries`.

    Returns:
        The path to the written lock.
    """
    lock = tmp_path / "pixi.lock"
    lock.write_text(
        f"version: 6\nenvironments:\n{body}packages:\n- conda: unused\n",
        encoding="utf-8",
    )
    return lock


class TestTheLockReaderTrustsOnlyTheCondaUrl:
    """Versions come out of each entry's URL, so what the reader will and will not accept matters.

    It parses `pixi.lock` with a regular expression rather than YAML, because it runs on a bare CI
    runner before any environment exists. That is sound only as long as it reads the package name
    and version from the conda URL itself and ignores everything it cannot parse that way -- a pin
    quietly resolving to nothing is the failure this whole mode exists to prevent (#1627).
    """

    def test_a_dashed_package_name_keeps_its_dashes(self, tmp_path):
        """`libgdal-netcdf-3.13.3-h1_0.conda` is one package, not `libgdal` at version `netcdf`."""
        lock = _synthetic_lock(
            tmp_path,
            _lock_entries(
                "osx-64", ["libgdal-netcdf-3.13.3-h1_0.conda", "proj-9.8.1-h1_0.conda"]
            ),
        )

        versions = _load_gdal_pin().locked_versions("osx-64", lock=lock)

        assert versions == {"libgdal-netcdf": "3.13.3", "proj": "9.8.1"}, (
            f"the name must be split at the version, not at the first dash: {versions}"
        )

    @pytest.mark.parametrize(
        ("label", "filename"),
        [
            ("non-numeric-version", "weird-abc-h1_0.conda"),
            ("no-version-at-all", "openssl.conda"),
            ("version-with-a-dash", "proj-9.8.1-rc1-h1_0.conda"),
        ],
    )
    def test_an_entry_whose_url_carries_no_version_is_skipped(
        self, tmp_path, label, filename
    ):
        """An unparseable file name contributes nothing rather than a wrong pin.

        Test scenario:
            The version is read positionally out of the file name, so anything that does not look
            like `<name>-<version>-<build>.conda` must drop out silently and leave the well-formed
            entry beside it untouched -- a wrong version here would be installed as an exact `==`.

        Args:
            label: Which malformed file name is under test.
            filename: The entry to write beside a well-formed `proj`.
        """
        lock = _synthetic_lock(
            tmp_path, _lock_entries("osx-64", ["proj-9.8.1-h1_0.conda", filename])
        )

        versions = _load_gdal_pin().locked_versions("osx-64", lock=lock)

        assert versions == {"proj": "9.8.1"}, (
            f"{label}: {filename!r} must contribute no pin, got {versions}"
        )

    def test_a_platform_of_only_tar_bz2_entries_pins_nothing(self, tmp_path):
        """The legacy `.tar.bz2` artefact is not read, and the absent pin must then be loud.

        Test scenario:
            Only `.conda` URLs are matched, so a platform locked to the older artefact format
            resolves to an empty mapping -- which `locked_pins` must turn into a refusal rather
            than hand the version choice back to the solver.
        """
        module = _load_gdal_pin()
        lock = _synthetic_lock(
            tmp_path,
            _lock_entries(
                "osx-64", ["proj-9.8.1-h1_0.tar.bz2", "geos-3.14.1-h1_0.tar.bz2"]
            ),
        )

        assert module.locked_versions("osx-64", lock=lock) == {}, (
            "a .tar.bz2 entry is not a conda URL this reader parses"
        )
        with pytest.raises(KeyError, match="pins no proj"):
            module.locked_pins(["proj"], "osx-64", lock=lock)

    def test_an_environment_that_locks_another_platform_only_raises(self, tmp_path):
        """A declared environment with no block for the asked platform is a KeyError, not `{}`.

        Test scenario:
            The cross-compile branch asks for its *target* platform, which the host never has an
            environment for; returning an empty mapping there would install the whole stack
            unpinned, which is exactly the divergence of #1627.
        """
        lock = _synthetic_lock(
            tmp_path, _lock_entries("linux-64", ["proj-9.8.1-h1_0.conda"])
        )

        module = _load_gdal_pin()

        with pytest.raises(KeyError, match="has no platform 'osx-64'"):
            module.locked_versions("osx-64", lock=lock)

    def test_an_undeclared_environment_raises(self, tmp_path):
        """Asking for an environment the lock never declares must name it in the refusal."""
        lock = _synthetic_lock(
            tmp_path,
            _lock_entries("osx-64", ["proj-9.8.1-h1_0.conda"], environment="bare"),
        )

        module = _load_gdal_pin()

        with pytest.raises(KeyError, match="declares no environment 'wheel-build'"):
            module.locked_versions("osx-64", lock=lock)

    def test_a_missing_lock_file_raises(self, tmp_path):
        """No lock at all is a FileNotFoundError, not an empty set of pins."""
        module = _load_gdal_pin()

        with pytest.raises(FileNotFoundError):
            module.locked_versions("osx-64", lock=tmp_path / "absent.lock")


class TestTheCommandLineRoutesToTheRightMode:
    """The two modes are selected by `--locked`, and each refuses an invocation it cannot serve.

    `ci/setup-gdal-micromamba.sh` builds this command line from shell variables, so an unset one
    arrives as a missing argument. Printing nothing and exiting 0 there would leave the cross-compile
    branch solving the native stack itself, with a green step to show for it (#1627).
    """

    @pytest.mark.parametrize(
        ("label", "argv"),
        [
            ("no-platform", ["proj"]),
            ("no-packages", ["--platform", "osx-64"]),
            ("platform-flag-without-a-value", ["--platform"]),
            ("environment-but-nothing-else", ["--environment", "wheel-build"]),
        ],
    )
    def test_an_incomplete_locked_invocation_exits_with_the_usage(self, label, argv):
        """Every way of naming no platform or no package must stop with the usage line.

        Args:
            label: Which incomplete invocation is under test.
            argv: The arguments after `--locked`.
        """
        with pytest.raises(SystemExit) as excinfo:
            _load_gdal_pin()._locked_main(argv)

        assert "usage: gdal-pin.py --locked" in str(excinfo.value), (
            f"{label}: the refusal must say how the mode is called, got: {excinfo.value}"
        )

    def test_the_locked_mode_prints_one_pin_per_package_in_order(self, capsys):
        """Each requested name yields one `==` line, in the order the caller asked for them.

        Test scenario:
            The shell reads the lines positionally into `PROJ_SPEC` / `GEOS_SPEC`, so a reordered
            or merged line would pin each library to the other's version.
        """
        module = _load_gdal_pin()
        expected = module.locked_pins(["proj", "geos"], "osx-64")

        status = module._locked_main(["--platform", "osx-64", "proj", "geos"])

        assert status == 0, f"a resolvable request must succeed, got {status}"
        assert capsys.readouterr().out.splitlines() == [
            expected["proj"],
            expected["geos"],
        ], "one pin per package, in the requested order"

    def test_an_explicit_environment_is_honoured(self, capsys):
        """`--environment` selects which locked environment is read."""
        module = _load_gdal_pin()

        status = module._locked_main(
            ["--platform", "osx-64", "--environment", "wheel-build", "proj"]
        )

        assert status == 0, (
            f"the default environment named explicitly must work: {status}"
        )
        assert (
            capsys.readouterr().out.strip()
            == module.locked_pins(["proj"], "osx-64")["proj"]
        ), "naming the default environment must resolve what omitting it does"

    def test_a_trailing_environment_flag_keeps_the_default(self, capsys):
        """An `--environment` with no value falls back rather than consuming the next package.

        Test scenario:
            An unset shell variable arrives as a bare trailing flag; dropping to the default keeps
            the pins resolvable, where consuming a package name would have silently pinned fewer
            libraries than the caller asked for.
        """
        module = _load_gdal_pin()

        status = module._locked_main(["--platform", "osx-64", "proj", "--environment"])

        assert status == 0, f"the default environment must still resolve, got {status}"
        assert (
            capsys.readouterr().out.strip()
            == module.locked_pins(["proj"], "osx-64")["proj"]
        ), "the trailing flag must not change which pin is printed"

    def test_main_routes_locked_to_the_locked_mode(self, capsys):
        """`--locked` anywhere in the arguments selects the lock reader, and is not a package."""
        module = _load_gdal_pin()

        status = module.main(["--locked", "--platform", "osx-64", "proj"])

        assert status == 0, f"the locked mode must be reachable from main, got {status}"
        assert (
            capsys.readouterr().out.strip()
            == module.locked_pins(["proj"], "osx-64")["proj"]
        ), "main must print what the locked mode resolves"

    def test_main_with_no_arguments_prints_the_gdal_range(self, capsys):
        """The default remains the shared gdal spec, which the wheel-test job installs against."""
        module = _load_gdal_pin()

        status = module.main([])

        assert status == 0, f"the default mode must succeed, got {status}"
        assert capsys.readouterr().out.strip() == module.gdal_spec(PYPROJECT), (
            "no arguments must still emit the gdal range from pyproject"
        )

    def test_main_prints_one_range_per_named_package(self, capsys):
        """Named packages are emitted one per line, which is how the shell reads them back."""
        module = _load_gdal_pin()
        pins = module.feature_pins()

        status = module.main(["gdal", "swig"])

        assert status == 0, f"both names resolve, so it must succeed, got {status}"
        assert capsys.readouterr().out.splitlines() == [pins["gdal"], pins["swig"]], (
            "each requested package must get its own line, in order"
        )

    def test_running_the_file_as_a_script_exits_with_the_status(
        self, monkeypatch, capsys
    ):
        """CI calls it as `python ci/gdal-pin.py`, so the script dispatch must reach `main`.

        Test scenario:
            The workflows and `ci/setup-gdal-micromamba.sh` capture this script's stdout into a
            shell variable; were the `__main__` block to stop exiting with `main`'s status, a
            failed lookup would be read as an empty pin and the solver would choose instead.
        """
        monkeypatch.setattr(sys, "argv", [str(GDAL_PIN_SCRIPT)])

        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(GDAL_PIN_SCRIPT), run_name="__main__")

        assert excinfo.value.code == 0, (
            f"the default invocation must exit 0, got {excinfo.value.code}"
        )
        assert capsys.readouterr().out.strip() == _load_gdal_pin().gdal_spec(
            PYPROJECT
        ), "running it as a script must print the same spec the function returns"


class TestTheLockIsReadToItsLastEntry:
    """Every locked entry must be read, including the last line of the last block.

    `locked_versions` splits the lock on its `packages:` header, which consumes the newline that
    terminated the previous line, while the entry pattern needs one per line. The final entry of the
    last platform block of the last environment was therefore dropped silently. It could not produce a
    *wrong* pin -- `locked_pins` raises for any requested name it cannot find -- but the `(found N)`
    count in that error was off, and a lock re-order or a new trailing environment could have made the
    dropped entry one of proj/geos/hdf5 and broken the cross-compile step with a misleading message.
    """

    @pytest.mark.parametrize("platform", ["linux-64", "osx-64", "osx-arm64", "win-64"])
    def test_every_conda_entry_in_the_block_is_parsed(self, platform):
        """The parsed count must equal the entries the lock actually carries.

        Args:
            platform: The conda subdir whose block must be read completely.
        """
        module = _load_gdal_pin()
        environments = LOCK.read_text(encoding="utf-8", errors="replace").split(
            "\npackages:", 1
        )[0]
        block = re.search(
            r"\n  wheel-build:\n(.*?)(?=\n  [A-Za-z][\w-]*:\n|\Z)", environments, re.S
        )
        section = re.search(
            rf"\n      {platform}:\n((?:      - .*\n?)+)", block.group(1)
        )
        if section is None:
            pytest.skip(f"the lock carries no {platform} block for wheel-build")
        declared = re.findall(rf"/{platform}/([^/\s]+)\.conda", section.group(1))
        parsed = module.locked_versions(platform)
        unparsed = [name for name in declared if name.rsplit("-", 2)[0] not in parsed]

        assert not unparsed, (
            f"these locked entries were not read, so a pin naming one would wrongly look absent: "
            f"{unparsed}"
        )

    def test_the_last_entry_of_a_block_is_not_dropped(self, tmp_path):
        """A lock whose final line is an entry with no trailing newline must still be read."""
        lock = tmp_path / "pixi.lock"
        lock.write_text(
            "version: 6\n"
            "environments:\n"
            "  wheel-build:\n"
            "    channels:\n"
            "    - url: https://conda.anaconda.org/conda-forge/\n"
            "    packages:\n"
            "      osx-64:\n"
            "      - conda: https://conda.anaconda.org/conda-forge/osx-64/proj-9.8.1-h856bcef_1.conda\n"
            "      - conda: https://conda.anaconda.org/conda-forge/osx-64/zstd-1.5.7-h534d264_7.conda",
            encoding="utf-8",
        )

        parsed = _load_gdal_pin().locked_versions("osx-64", lock=lock)

        assert parsed == {"proj": "9.8.1", "zstd": "1.5.7"}, (
            f"the final entry must be read even with no trailing newline: {parsed}"
        )
