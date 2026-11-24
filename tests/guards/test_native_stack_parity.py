"""Guard the bundled native stack against per-route drift.

`pyramids-core` obtains its native stack three ways — compiled from source for the Linux wheels, built
through vcpkg for `win_arm64`, and taken as conda-forge binaries for `win_amd64`/macOS — and each route
silently inherits its upstream's defaults for anything not explicitly requested. That is how one release
came to ship HDF5 with thread-safety on three platforms and off on the other two (#1620), and how
`win_arm64` still ships a GDAL a minor version behind everyone else (#1627).

`ci/check-bundled-stack.py` reads what a built wheel actually contains and compares it against
`ci/native-stack-baseline.toml`. These tests drive that checker over synthetic wheels: a zip carrying a
member named like a bundled library whose bytes contain the build-settings strings HDF5, PROJ and GEOS
embed in themselves. Synthetic rather than real wheels because the point is the comparison logic, and a
real 33 MB wheel per case would make the suite network-bound.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "ci" / "check-bundled-stack.py"
_INSTALLED_CHECKER = _REPO_ROOT / "ci" / "check-installed-stack.py"
_BASELINE = _REPO_ROOT / "ci" / "native-stack-baseline.toml"


def _library_bytes(hdf5: str = "2.2.0", threadsafety: str = "ON") -> bytes:
    """Bytes imitating the build-settings summary HDF5 embeds in its own binary.

    Args:
        hdf5: The version string to embed.
        threadsafety: The thread-safety setting to embed.

    Returns:
        A payload the checker's extractors can read, padded so it looks like a real object file
        rather than a bare string.
    """
    head = b"\x7fELF" + b"\x00" * 60
    body = f"    HDF5 Version: {hdf5}\n    Threadsafety: {threadsafety}\n".encode()
    return head + body + b"\x00" * 32


def _wheel(
    tmp_path: Path,
    platform_tag: str = "manylinux_2_28_x86_64",
    hdf5: str = "2.2.0",
    threadsafety: str = "ON",
    proj: str = "9.8.1",
    geos: str = "3.14.1",
    include_hdf5: bool = True,
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    """Build a synthetic wheel whose bundled libraries report the given versions.

    Args:
        tmp_path: Directory to write the wheel into.
        platform_tag: The wheel's platform tag, which is how the checker resolves exceptions.
        hdf5: HDF5 version to embed.
        threadsafety: HDF5 thread-safety setting to embed.
        proj: PROJ version to embed.
        geos: GEOS version to embed.
        include_hdf5: When False, omit the HDF5 library entirely.
        extra_members: Further `{archive member: bytes}` entries to write, for the companion
            libraries and duplicate copies a repair tool can land beside the real ones.

    Returns:
        The path to the written wheel.
    """
    path = tmp_path / f"pyramids_core-9.9.9-cp312-cp312-{platform_tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        if include_hdf5:
            archive.writestr(
                "pyramids_core.libs/libhdf5-abcdef12.so.320.2.0",
                _library_bytes(hdf5=hdf5, threadsafety=threadsafety),
            )
        archive.writestr(
            "pyramids_core.libs/libproj-abcdef12.so.25", f"PROJ {proj}\x00".encode()
        )
        archive.writestr(
            "pyramids_core.libs/libgeos_c-abcdef12.so.1",
            f"{geos}-CAPI-1.20.5\x00".encode(),
        )
        for member, payload in (extra_members or {}).items():
            archive.writestr(member, payload)
        archive.writestr("pyramids/__init__.py", b"")
    return path


def _run(*targets: Path) -> subprocess.CompletedProcess:
    """Run the checker over `targets` with the repo's real baseline.

    Args:
        targets: Wheel paths or directories to check.

    Returns:
        The completed process, with stdout and stderr captured as text.
    """
    return subprocess.run(
        [sys.executable, str(_CHECKER), *[str(t) for t in targets]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _checker_module(script: str):
    """Load one of the checker scripts as a module.

    Args:
        script: A filename under `ci/`, e.g. `check-bundled-stack.py`. Their dashed names are
            not importable, so each is loaded from its path.

    Returns:
        The imported module object, freshly executed so a caller may patch it freely.
    """
    path = _REPO_ROOT / "ci" / script
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Driver names the synthetic installed-side baseline requires. Deliberately short: what is under
#: test is the present/missing comparison, while the real lists belong to the repo's own baseline.
_REQUIRED_RASTER = ["GTiff", "COG", "netCDF"]
_REQUIRED_VECTOR = ["GeoJSON", "GPKG"]


class _FakeDriver:
    """A stand-in driver handle, which GDAL and OGR expose under different names."""

    def __init__(self, name: str):
        """Hold the one name both APIs report.

        Args:
            name: The driver's short name.
        """
        self.ShortName = name

    def GetName(self) -> str:
        """The name OGR's API reports.

        Returns:
            The driver's short name.
        """
        return self.ShortName


class _FakeOsgeo:
    """A stand-in for `osgeo.gdal`/`osgeo.ogr`, answering with fixed values.

    Driving the installed-side checker against this exercises its comparisons offline, where the
    real `osgeo` would answer with whatever GDAL the environment running the suite happens to carry.
    """

    def __init__(self, version: str, drivers: list[str]):
        """Record what this fake library reports about itself.

        Args:
            version: The value to serve as `__version__`.
            drivers: The driver names to offer, in order.
        """
        self.__version__ = version
        self._drivers = list(drivers)

    def GetDriverCount(self) -> int:
        """How many drivers are registered.

        Returns:
            The number of drivers.
        """
        return len(self._drivers)

    def GetDriver(self, index: int) -> _FakeDriver:
        """The driver at one index.

        Args:
            index: Position in the registered list.

        Returns:
            The driver handle at `index`.
        """
        return _FakeDriver(self._drivers[index])


def _installed_baseline(
    tmp_path: Path,
    gdal_version: str | None = "3.13.3",
    raster: list[str] | None = None,
    vector: list[str] | None = None,
) -> Path:
    """Write a baseline carrying only the `[installed]` tables the installed side reads.

    Args:
        tmp_path: Directory to write the baseline into.
        gdal_version: Version to pin, or None to leave GDAL unpinned.
        raster: Required raster drivers. Defaults to `_REQUIRED_RASTER`.
        vector: Required vector drivers. Defaults to `_REQUIRED_VECTOR`.

    Returns:
        The path to the written baseline.
    """
    raster = _REQUIRED_RASTER if raster is None else raster
    vector = _REQUIRED_VECTOR if vector is None else vector
    lines = []
    if gdal_version is not None:
        lines += ["[installed.gdal]", f'version = "{gdal_version}"', ""]
    lines += [
        "[installed.drivers]",
        "raster = [" + ", ".join(f'"{name}"' for name in raster) + "]",
        "vector = [" + ", ".join(f'"{name}"' for name in vector) + "]",
        "",
    ]
    path = tmp_path / "native-stack-baseline.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _installed_checker(
    monkeypatch,
    baseline: Path,
    gdal_version: str,
    raster: list[str],
    vector: list[str],
):
    """Load the installed-side checker wired to a fake `osgeo` and a given baseline.

    Args:
        monkeypatch: The fixture that restores every patched attribute afterwards.
        baseline: The baseline file the checker should read.
        gdal_version: The version the fake GDAL reports.
        raster: Raster driver short names the fake GDAL offers.
        vector: Vector driver names the fake OGR offers.

    Returns:
        The patched module, ready to have `main()` called on it.
    """
    module = _checker_module("check-installed-stack.py")
    monkeypatch.setattr(module, "_BASELINE_PATH", baseline)
    monkeypatch.setattr(module, "gdal", _FakeOsgeo(gdal_version, raster))
    monkeypatch.setattr(module, "ogr", _FakeOsgeo(gdal_version, vector))
    return module


class TestTheCheckerShipsWithABaseline:
    """The checker and the baseline it reads are both present and parseable."""

    def test_the_checker_exists(self):
        """`ci/check-bundled-stack.py` is the script the repair steps invoke."""
        assert _CHECKER.is_file(), f"missing checker at {_CHECKER}"

    def test_the_baseline_exists_and_declares_the_byte_readable_libraries(self):
        """The baseline pins exactly the libraries whose versions a wheel can be read for.

        Test scenario:
            GDAL is deliberately absent: its version is not reliably greppable from the binary
            (a manylinux `libgdal` yields only a spurious `3.6.0`, and the Windows DLLs carry no
            version in their name), so GDAL parity is asserted from the installed wheel instead.
        """
        baseline = tomllib.loads(_BASELINE.read_text(encoding="utf-8"))

        assert set(baseline["libraries"]) == {"hdf5", "proj", "geos"}, (
            f"unexpected baseline libraries: {sorted(baseline['libraries'])}"
        )
        assert baseline["libraries"]["hdf5"]["threadsafety"] == "ON", (
            "the baseline must require a thread-safe HDF5 (#1620)"
        )


class TestAConformingWheelPasses:
    """A wheel matching the baseline is accepted."""

    def test_a_wheel_matching_the_baseline_passes(self, tmp_path):
        """Every library reporting its baseline version exits 0."""
        result = _run(_wheel(tmp_path))

        assert result.returncode == 0, (
            f"expected pass, got {result.returncode}\n{result.stdout}\n{result.stderr}"
        )

    def test_a_directory_of_wheels_is_accepted(self, tmp_path):
        """The repair steps pass `{dest_dir}`, so a directory argument must work."""
        _wheel(tmp_path)

        result = _run(tmp_path)

        assert result.returncode == 0, (
            f"expected pass for a directory, got:\n{result.stdout}\n{result.stderr}"
        )

    def test_a_companion_library_contributes_nothing(self, tmp_path):
        """`libhdf5_hl` matches the `hdf5` token but embeds no build-settings string.

        Test scenario:
            The token match is a substring test, so a wheel's companion libraries -- and files
            that merely mention the name -- are scanned alongside the real library. Neither may
            displace what the real one reports, or a conforming wheel would fail.
        """
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(
            tmp_path,
            extra_members={
                "pyramids_core.libs/libhdf5_hl-abcdef12.so.320.2.0": b"\x7fELF"
                + b"\x00" * 64,
                "pyramids_core.libs/hdf5-build-settings.txt": b"HDF5 Version: 1.8.0\n",
            },
        )

        failures = checker.check_wheel(wheel, checker.load_baseline())

        assert failures == [], (
            f"a companion library and a note file must not fail the wheel: {failures}"
        )

    def test_every_bundled_copy_of_a_library_is_read(self, tmp_path):
        """Two copies of a library yield two readings, not one.

        Test scenario:
            A repair tool can land one library twice under different mangled names. Collapsing them
            to a single reading made the verdict depend on archive order, which is what
            `TestABadCopyCannotHideBehindAGoodOne` pins; this is the reading side of that contract.
        """
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(
            tmp_path,
            extra_members={
                "pyramids_core.libs/libhdf5-99999999.so.320.2.0": _library_bytes()
            },
        )

        reported = checker.read_reported(wheel, ["hdf5"])

        assert len(reported["hdf5"]) == 2, (
            f"both bundled copies must be reported, not collapsed to one: {reported}"
        )
        assert {entry["version"] for entry in reported["hdf5"]} == {"2.2.0"}, (
            f"each copy must report its own version: {reported}"
        )

    def test_a_companion_without_a_settings_string_is_not_reported(self, tmp_path):
        """`libhdf5_hl` and friends match the token but carry no settings string, so they drop out."""
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(
            tmp_path,
            extra_members={
                "pyramids_core.libs/libhdf5_hl-abc.so.320.2.0": b"\x7fELF"
                + b"\x00" * 64
            },
        )

        reported = checker.read_reported(wheel, ["hdf5"])

        assert len(reported["hdf5"]) == 1, (
            f"a companion carrying no settings string must not become a reading: {reported}"
        )


class TestABadCopyCannotHideBehindAGoodOne:
    """A non-conforming bundled copy fails the wheel whatever its name sorts next to.

    The checker read every matching member but kept only the last reading, so the verdict depended on
    archive order: a non-threadsafe `aa_libhdf5.so` beside a conforming `zz_libhdf5.so` passed, and the
    same pair renamed failed. Thread-safety is the property #1620 was about, so that ordering
    dependency meant the original defect could ship again in a wheel reporting `ok` (#1627).
    """

    @pytest.mark.parametrize(
        ("first", "second", "ids"),
        [
            ("OFF", "ON", "bad-copy-sorts-first"),
            ("ON", "OFF", "bad-copy-sorts-last"),
        ],
        ids=["bad-first", "bad-last"],
    )
    def test_a_bad_copy_fails_whatever_the_order(self, tmp_path, first, second, ids):
        """Either ordering must fail, since both bundle a non-threadsafe library.

        Args:
            tmp_path: Per-test directory.
            first: Thread-safety of the alphabetically earlier copy.
            second: Thread-safety of the later copy.
            ids: Unused label kept so the parametrisation reads explicitly.
        """
        wheel = _wheel(
            tmp_path,
            extra_members={
                "pyramids_core.libs/aa_libhdf5-x.so.320.2.0": _library_bytes(
                    threadsafety=first
                ),
                "pyramids_core.libs/zz_libhdf5-y.so.320.2.0": _library_bytes(
                    threadsafety=second
                ),
            },
        )

        result = _run(wheel)

        assert result.returncode == 1, (
            f"a wheel bundling a non-threadsafe copy must fail regardless of name order: {result.stdout}"
        )
        assert "threadsafety=OFF" in result.stdout, (
            f"the offending copy must be named in the report: {result.stdout}"
        )

    def test_both_copies_are_reported_when_both_conform(self, tmp_path):
        """A conforming wheel verdicts each copy rather than silently checking one."""
        wheel = _wheel(
            tmp_path,
            extra_members={
                "pyramids_core.libs/aa_libhdf5-x.so.320.2.0": _library_bytes(),
                "pyramids_core.libs/zz_libhdf5-y.so.320.2.0": _library_bytes(),
            },
        )

        result = _run(wheel)

        assert result.returncode == 0, (
            f"both copies conform, so the wheel passes: {result.stdout}"
        )
        assert result.stdout.count("threadsafety=ON") >= 2, (
            f"each bundled copy must get its own verdict line: {result.stdout}"
        )


class TestDriftIsRefused:
    """Each property the baseline pins fails the build when it drifts."""

    def test_a_non_threadsafe_hdf5_is_refused(self, tmp_path):
        """The #1620 regression itself: thread-safety off must fail."""
        result = _run(_wheel(tmp_path, threadsafety="OFF"))

        assert result.returncode == 1, (
            f"a non-threadsafe HDF5 must fail, got {result.returncode}"
        )
        assert "threadsafety" in result.stdout.lower(), (
            f"the report should name the property: {result.stdout}"
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("hdf5", "2.1.0"), ("proj", "9.7.0"), ("geos", "3.13.0")],
        ids=["hdf5-behind", "proj-behind", "geos-behind"],
    )
    def test_a_library_version_off_the_baseline_is_refused(
        self, tmp_path, field, value
    ):
        """A route drifting to a different version of any pinned library fails.

        Args:
            field: Which library drifts.
            value: The off-baseline version it reports.
        """
        result = _run(_wheel(tmp_path, **{field: value}))

        assert result.returncode == 1, (
            f"{field}={value} must fail, got {result.returncode}"
        )
        assert field in result.stdout.lower(), (
            f"the report should name {field}: {result.stdout}"
        )

    def test_a_missing_library_is_refused(self, tmp_path):
        """A binary wheel without HDF5 cannot read NetCDF, so its absence is a regression too."""
        result = _run(_wheel(tmp_path, include_hdf5=False))

        assert result.returncode == 1, "a wheel with no bundled HDF5 must fail"


class TestDeclaredExceptionsAreHonoured:
    """A platform that cannot meet the baseline is pinned to what it can meet."""

    def test_win_arm64_passes_on_its_declared_hdf5_version(self, tmp_path):
        """`win_arm64` is allowed HDF5 2.1.1, because vcpkg's baseline has no 2.2.x port.

        Test scenario:
            A bare equality check would fail this platform on something unachievable, so the
            baseline carries the exception and the reason with it (#1627).
        """
        result = _run(_wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1"))

        assert result.returncode == 0, (
            f"win_arm64 on its declared version must pass:\n{result.stdout}"
        )

    def test_the_exception_does_not_leak_to_other_platforms(self, tmp_path):
        """The same version that `win_arm64` is allowed still fails on manylinux."""
        result = _run(
            _wheel(tmp_path, platform_tag="manylinux_2_28_x86_64", hdf5="2.1.1")
        )

        assert result.returncode == 1, (
            "the win_arm64 exception must not apply to manylinux"
        )

    def test_an_exception_does_not_waive_thread_safety(self, tmp_path):
        """A declared version exception is not a licence to ship a non-threadsafe build."""
        result = _run(
            _wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1", threadsafety="OFF")
        )

        assert result.returncode == 1, (
            "thread-safety must hold even where a version exception applies"
        )

    def test_an_unparseable_filename_says_exceptions_cannot_resolve(self, tmp_path):
        """A wheel whose name carries no platform tag warns rather than silently losing exceptions.

        Test scenario:
            Applying no exceptions quietly would compare a platform against the baseline instead
            of against its own declared values, and the mismatch would read as real drift -- the
            same silent-divergence failure this check exists to catch.
        """
        renamed = tmp_path / "renamed.whl"
        _wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1").rename(renamed)

        result = _run(renamed)

        assert "no platform tag" in result.stderr, (
            f"an unparseable wheel name should warn on stderr, got: {result.stderr}"
        )

    @pytest.mark.parametrize(
        "name",
        ["renamed.whl", "pyramids_core-9.9.9-cp312-win_arm64.whl"],
        ids=["no-dashes", "four-parts"],
    )
    def test_a_name_with_too_few_parts_carries_no_platform_tag(self, tmp_path, name):
        """A filename that does not parse as PEP 427 resolves to no tag at all.

        Test scenario:
            The tag is what selects the exception set, so a short name must yield "" -- which
            the checker then warns about -- rather than mistake a Python or ABI field for a
            platform and apply the wrong exceptions.

        Args:
            name: A wheel filename with fewer than five dash-separated parts.
        """
        checker = _checker_module("check-bundled-stack.py")

        assert checker.platform_tag(tmp_path / name) == "", (
            f"{name!r} carries no platform tag and must not resolve to one"
        )

    def test_an_exception_for_an_undeclared_library_is_ignored(self):
        """An exception naming a library absent from `[libraries]` adds nothing to check.

        Test scenario:
            An exception left behind after its library was dropped from the baseline must not
            resurrect that library as a requirement, which would fail every wheel over
            something nobody declares any more.
        """
        checker = _checker_module("check-bundled-stack.py")
        baseline = {
            "libraries": {"proj": {"version": "9.8.1"}},
            "exceptions": {"win_arm64": {"sqlite3": {"version": "3.50.0"}}},
        }

        expected = checker.expected_for(baseline, "win_arm64")

        assert expected == {"proj": {"version": "9.8.1"}}, (
            f"an exception for an undeclared library must be ignored: {expected}"
        )

    def test_an_exception_cannot_override_a_non_version_property(self):
        """A declared exception may relax a version and nothing else.

        Test scenario:
            Thread-safety is the property that already regressed (#1620), so declaring it in a
            platform's exception table must not opt that platform out of it -- only out of the
            version it genuinely cannot reach.
        """
        checker = _checker_module("check-bundled-stack.py")
        baseline = {
            "libraries": {"hdf5": {"version": "2.2.0", "threadsafety": "ON"}},
            "exceptions": {
                "win_arm64": {"hdf5": {"version": "2.1.1", "threadsafety": "OFF"}}
            },
        }

        expected = checker.expected_for(baseline, "win_arm64")

        assert expected["hdf5"] == {"version": "2.1.1", "threadsafety": "ON"}, (
            f"only a version may be relaxed by an exception: {expected}"
        )

    def test_a_reason_string_is_not_compared_as_a_property(self, tmp_path, capsys):
        """`reason` documents a baseline table; it is prose, not something a binary reports.

        Test scenario:
            Carrying the justification next to the values is the whole point of the key, so
            comparing it as a property would fail every wheel on something no library embeds.
        """
        checker = _checker_module("check-bundled-stack.py")
        baseline = {
            "libraries": {
                "hdf5": {
                    "version": "2.2.0",
                    "threadsafety": "ON",
                    "reason": "every route that compiles its own HDF5 must ask for it",
                }
            }
        }

        failures = checker.check_wheel(_wheel(tmp_path), baseline)

        assert failures == [], (
            f"`reason` must not be compared as a property: {failures}"
        )
        assert "reason" not in capsys.readouterr().out, (
            "the per-property report should not mention the prose key"
        )


class TestTheTargetsAreExpandedToWheels:
    """What the checker accepts on its command line, and how it reports over several wheels.

    The repair steps hand it `{dest_dir}` or paths from a shell loop, so argument handling is part
    of the guard: a target that silently expands to nothing would leave the drift this check exists
    to catch unreported, with a green step to show for it.
    """

    def test_no_arguments_exits_with_the_usage(self):
        """Invoked with nothing to check, the script exits with its own docstring.

        Test scenario:
            A repair step whose wheel variable expanded to nothing must stop and say how it is
            called, rather than report success over zero wheels.
        """
        checker = _checker_module("check-bundled-stack.py")

        with pytest.raises(SystemExit) as excinfo:
            checker.main([])

        assert "Usage:" in str(excinfo.value), (
            f"the usage text should be the exit message, got: {excinfo.value}"
        )

    def test_a_target_that_is_neither_a_wheel_nor_a_directory_is_refused(
        self, tmp_path
    ):
        """A path that can never name wheels is an error, not an empty check.

        Test scenario:
            The release steps also produce sdists and logs; pointing the checker at one of them
            is a wiring mistake, and accepting it quietly would assert nothing at all.
        """
        checker = _checker_module("check-bundled-stack.py")
        target = tmp_path / "pyramids_core-9.9.9.tar.gz"
        target.write_bytes(b"")

        with pytest.raises(SystemExit) as excinfo:
            checker._wheels([str(target)])

        assert "neither a wheel nor a directory" in str(excinfo.value), (
            f"the refusal should name the problem, got: {excinfo.value}"
        )

    def test_a_directory_with_no_wheels_is_refused(self, tmp_path):
        """An empty target set fails rather than reporting a vacuous success.

        Test scenario:
            A repair step is always pointed at the directory holding the wheel it just repaired, so
            an empty one means the wiring broke. Exiting 0 there would be the same
            silently-unchecked outcome this guard exists to prevent, which is why it is a failure
            and not a no-op.
        """
        empty = tmp_path / "nothing"
        empty.mkdir()

        result = _run(empty)

        assert result.returncode == 1, (
            f"an empty directory must fail, got {result.returncode}"
        )
        assert (
            "comparing nothing" in result.stderr or "compared nothing" in result.stderr
        ), f"the refusal should say why: {result.stderr}"

    def test_the_same_wheel_named_twice_is_checked_once(self, tmp_path, capsys):
        """Duplicate targets are deduplicated, so each wheel is reported exactly once.

        Test scenario:
            A shell loop can easily name a wheel twice; a doubled report would make the
            summary's mismatch count wrong, and do twice the work for the same answer.
        """
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(tmp_path)

        assert checker.main([str(wheel), str(wheel)]) == 0, (
            "a conforming wheel named twice must still pass"
        )

        reported = [
            line
            for line in capsys.readouterr().out.splitlines()
            if "threadsafety" in line
        ]
        assert len(reported) == 1, f"the wheel should be reported once, got: {reported}"

    def test_a_wheel_inside_a_named_directory_is_not_checked_twice(self, tmp_path):
        """Naming both a directory and a wheel inside it yields that wheel once.

        Test scenario:
            The repair steps pass `{dest_dir}`, and a caller adding the wheel path alongside it
            must not double the comparisons.
        """
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(tmp_path)

        assert checker._wheels([str(tmp_path), str(wheel)]) == [wheel], (
            "a directory and a wheel inside it must expand to one target"
        )

    def test_failures_across_several_wheels_share_one_exit_code(self, tmp_path, capsys):
        """One call over many wheels aggregates every mismatch into a single exit 1.

        Test scenario:
            The repair step checks a whole directory, so a platform whose stack drifted must not
            be masked by the conforming wheels beside it, and every failure must reach the
            summary -- the per-route divergence of #1620 is exactly what one wheel at a time
            hides.
        """
        checker = _checker_module("check-bundled-stack.py")
        drifted = _wheel(tmp_path, platform_tag="manylinux_2_28_x86_64", hdf5="2.1.0")
        incomplete = _wheel(
            tmp_path, platform_tag="manylinux_2_28_aarch64", include_hdf5=False
        )

        assert checker.main([str(tmp_path)]) == 1, (
            "either wheel failing must fail the whole call"
        )

        captured = capsys.readouterr()
        assert drifted.name in captured.out, (
            f"the drifted wheel should be named: {captured.out}"
        )
        assert incomplete.name in captured.out, (
            f"the wheel missing HDF5 should be named: {captured.out}"
        )
        assert "2 bundled-library mismatch(es)" in captured.err, (
            f"both failures should reach the summary: {captured.err}"
        )


class TestTheInstalledSideResolvesExceptions:
    """`ci/check-installed-stack.py` applies the same declared exceptions as the bundled check.

    Loaded by path because the filename has dashes and cannot be imported as a module. Only the pure
    resolution logic is driven here; the assertions it feeds are driven in
    `TestTheInstalledStackIsAsserted`, against a stand-in for `osgeo`.
    """

    @staticmethod
    def _module():
        """Load the installed-side checker by path.

        Returns:
            The imported module object.
        """
        return _checker_module("check-installed-stack.py")

    def test_a_platform_without_an_exception_gets_the_baseline_version(self):
        """`win_amd64` has no GDAL exception, so it must meet the baseline."""
        module = self._module()
        baseline = {"installed": {"gdal": {"version": "3.13.3"}}, "exceptions": {}}

        assert module._expected_gdal(baseline, "win_amd64") == "3.13.3"

    def test_a_declared_exception_overrides_the_baseline(self):
        """`win_arm64` is held to 3.12.4, the newest GDAL its vcpkg baseline offers."""
        module = self._module()
        baseline = {
            "installed": {"gdal": {"version": "3.13.3"}},
            "exceptions": {"win_arm64": {"gdal": {"version": "3.12.4"}}},
        }

        assert module._expected_gdal(baseline, "win_arm64") == "3.12.4"
        assert module._expected_gdal(baseline, "manylinux_2_28_x86_64") == "3.13.3", (
            "the exception must not leak to other platforms"
        )

    def test_the_repo_baseline_holds_win_arm64_to_its_vcpkg_version(self):
        """The shipped baseline encodes the real, measured exception rather than a placeholder."""
        module = self._module()
        baseline = tomllib.loads(_BASELINE.read_text(encoding="utf-8"))

        assert module._expected_gdal(baseline, "win_arm64") == "3.12.4"
        assert module._expected_gdal(baseline, "win_amd64") == "3.13.3"

    def test_the_platform_tag_is_resolvable_on_this_interpreter(self):
        """A tag must always resolve, since it is what selects the exception set."""
        module = self._module()

        tag = module._platform_tag()

        assert tag, "a platform tag must never be empty"
        assert tag.replace("_", "").isalnum(), f"unexpected tag shape: {tag!r}"

    @pytest.mark.parametrize(
        ("platform", "tag"),
        [
            ("win-amd64", "win_amd64"),
            ("win-arm64", "win_arm64"),
            ("linux-x86_64", "linux_x86_64"),
            ("macosx-11.0-arm64", "macosx_11_0_arm64"),
        ],
        ids=["win-amd64", "win-arm64", "linux", "macos"],
    )
    def test_each_platform_string_maps_to_the_tag_the_baseline_keys_on(
        self, monkeypatch, platform, tag
    ):
        """Every shape of platform string derives the tag its exceptions are filed under.

        Test scenario:
            Driven over all four shapes rather than only the host's, because `win_arm64` is the
            one platform carrying exceptions and no single machine reports every string.

        Args:
            platform: What `sysconfig.get_platform()` reports.
            tag: The wheel platform tag that must come out of it.
        """
        module = self._module()
        monkeypatch.setattr(module.sysconfig, "get_platform", lambda: platform)

        assert module._platform_tag() == tag, (
            f"{platform!r} must resolve to {tag!r}, got {module._platform_tag()!r}"
        )


class TestTheInstalledStackIsAsserted:
    """`ci/check-installed-stack.py` checks the installed GDAL's version and driver set.

    Driven against a stand-in for `osgeo` and a synthetic baseline, so the comparisons run offline
    and deterministically. In CI the same code runs in the wheel smoke jobs, where the real
    installed GDAL answers instead.
    """

    @pytest.mark.parametrize(
        "reported", ["3.13.3", "3.13.3-dev"], ids=["exact", "build-suffix"]
    )
    def test_the_baseline_version_and_a_full_driver_set_pass(
        self, tmp_path, monkeypatch, capsys, reported
    ):
        """An installed stack meeting the baseline exits 0.

        Test scenario:
            GDAL reports a trailing build suffix on some routes, which is not drift, so only the
            part before the first dash is compared.

        Args:
            reported: What the installed GDAL reports as its version.
        """
        module = _installed_checker(
            monkeypatch,
            _installed_baseline(tmp_path),
            reported,
            _REQUIRED_RASTER,
            _REQUIRED_VECTOR,
        )

        assert module.main() == 0, "a conforming installed stack must pass"
        assert "gdal version=3.13.3" in capsys.readouterr().out, (
            "the report should name the compared version"
        )

    def test_a_drifted_gdal_version_fails(self, tmp_path, monkeypatch, capsys):
        """A GDAL a minor version off the baseline fails the smoke job.

        Test scenario:
            This is the property the bundled check deliberately leaves alone, because a wheel's
            bytes cannot be trusted for it -- so it has to fail here or nowhere.
        """
        module = _installed_checker(
            monkeypatch,
            _installed_baseline(tmp_path),
            "3.12.4",
            _REQUIRED_RASTER,
            _REQUIRED_VECTOR,
        )

        assert module.main() == 1, "a GDAL off the baseline must fail"
        assert "gdal version=3.12.4 (want 3.13.3)" in capsys.readouterr().out, (
            "the report should name both versions"
        )

    @pytest.mark.parametrize("kind", ["raster", "vector"], ids=["raster", "vector"])
    def test_a_missing_required_driver_fails(self, tmp_path, monkeypatch, capsys, kind):
        """A route that built GDAL without a required driver fails, on either side.

        Test scenario:
            The source build already refuses a missing driver, but that check runs inside the
            source build, so it says nothing about the vcpkg and conda-forge wheels (#1627).

        Args:
            kind: Whether the raster or the vector driver list is short.
        """
        dropped = (_REQUIRED_RASTER if kind == "raster" else _REQUIRED_VECTOR)[0]
        module = _installed_checker(
            monkeypatch,
            _installed_baseline(tmp_path),
            "3.13.3",
            _REQUIRED_RASTER[1:] if kind == "raster" else _REQUIRED_RASTER,
            _REQUIRED_VECTOR[1:] if kind == "vector" else _REQUIRED_VECTOR,
        )

        assert module.main() == 1, f"a missing {kind} driver must fail"

        out = capsys.readouterr().out
        assert f"missing {kind} drivers" in out, (
            f"the report should name the side that is short: {out}"
        )
        assert dropped in out, f"the report should name the missing driver: {out}"

    def test_a_baseline_pinning_no_gdal_version_accepts_what_is_installed(
        self, tmp_path, monkeypatch, capsys
    ):
        """With no declared GDAL version there is nothing to compare, which is not a failure.

        Test scenario:
            Keeps the driver assertions usable on their own, so dropping the version pin while a
            release is in flight does not turn every smoke job red.
        """
        module = _installed_checker(
            monkeypatch,
            _installed_baseline(tmp_path, gdal_version=None),
            "3.11.0",
            _REQUIRED_RASTER,
            _REQUIRED_VECTOR,
        )

        assert module.main() == 0, "an unpinned GDAL version cannot drift"
        assert "pins no version" in capsys.readouterr().out, (
            "the report should say the version was not compared"
        )


class TestTheInstalledCheckerBootstrapsTheVendoredOsgeo:
    """`import pyramids` must precede `import osgeo`, or the checker cannot run in a wheel job.

    A published wheel ships no top-level `osgeo` package: it lives at `pyramids/_vendor/osgeo`, reachable
    only after `pyramids/__init__.py` injects it onto `sys.path`. A bare `from osgeo import ...` therefore
    raises `ModuleNotFoundError` in exactly the environment this checker is written for, while passing in a
    conda dev environment where `osgeo` is a real top-level package. `ci/verify-wheel.py` pins the same
    ordering with an `# isort: off` block; this test pins it for the checker too (#1627).
    """

    def test_pyramids_is_imported_before_osgeo(self):
        """The vendor bootstrap must come first in module source order."""
        tree = ast.parse(_INSTALLED_CHECKER.read_text(encoding="utf-8"))
        order = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                order.extend(
                    (alias.name.split(".")[0], node.lineno) for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                order.append((node.module.split(".")[0], node.lineno))
        pyramids_at = [line for name, line in order if name == "pyramids"]
        osgeo_at = [line for name, line in order if name == "osgeo"]

        assert pyramids_at, (
            "ci/check-installed-stack.py must `import pyramids` to run the vendor bootstrap; a published "
            "wheel has no top-level osgeo package, so a bare osgeo import raises ModuleNotFoundError"
        )
        assert osgeo_at, "the checker is expected to import osgeo"
        assert min(pyramids_at) < min(osgeo_at), (
            f"`import pyramids` (line {min(pyramids_at)}) must precede the osgeo import "
            f"(line {min(osgeo_at)}) so the vendored package is on sys.path"
        )
