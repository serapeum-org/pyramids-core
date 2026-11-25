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
import json
import re
import runpy
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

# These drive the checkers over synthetic wheels and parsed config, so they need nothing but
# the standard library -- no osgeo, no built wheel -- and belong in the `core` selection like
# the other wiring guards.
pytestmark = pytest.mark.core

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "ci" / "check-bundled-stack.py"
_VERIFY_WHEEL = _REPO_ROOT / "ci" / "verify-wheel.py"
_SHARED = _REPO_ROOT / "ci" / "native_stack_baseline.py"
_BASELINE = _REPO_ROOT / "ci" / "native-stack-baseline.toml"


def _library_bytes(hdf5: str = "2.2.0", threadsafety: str | None = "ON") -> bytes:
    """Bytes imitating the build-settings summary HDF5 embeds in its own binary.

    Args:
        hdf5: The version string to embed.
        threadsafety: The thread-safety setting to embed, or None to omit the line altogether, as a
            stripped or truncated binary carrying only part of its summary does.

    Returns:
        A payload the checker's extractors can read, padded so it looks like a real object file
        rather than a bare string.
    """
    head = b"\x7fELF" + b"\x00" * 60
    body = f"    HDF5 Version: {hdf5}\n"
    if threadsafety is not None:
        body += f"    Threadsafety: {threadsafety}\n"
    return head + body.encode() + b"\x00" * 32


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


def _matched_members(tmp_path: Path, token: str, members: list[str]) -> list[str]:
    """The members `_members_for` picks out of an archive holding exactly `members`.

    Args:
        tmp_path: Directory to write the throwaway archive into.
        token: The library token to match, e.g. `hdf5`.
        members: Archive member names to write, each with a one-byte payload since only the name
            decides whether a member is scanned.

    Returns:
        The member names the checker considers a bundled copy of that library.
    """
    path = tmp_path / "members.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"\x00")
    checker = _checker_module("check-bundled-stack.py")
    with zipfile.ZipFile(path) as archive:
        return checker._members_for(archive, token)


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
        """A member matching the token but carrying no settings string must not break the check.

        Test scenario:
            Two cases, both real in published wheels: `libhdf5_hl`, which has a native suffix and is
            scanned but embeds no settings string, and a `.txt` sidecar, which `_members_for`
            filters out *before* any read because it has no native suffix. The filter is what this
            pins -- an earlier docstring claimed the sidecar was scanned alongside the library,
            which is the opposite of what happens.

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


class TestWhichArchiveMembersLookLikeALibrary:
    """Only the members that could be a bundled native library are read, by basename alone.

    The filter decides what the whole comparison ever sees: too narrow and a repair tool's mangled
    name goes unchecked, too wide and a note file or a directory name becomes a reading. Both arms
    of it are load-bearing, since a repaired wheel carries `geos_c-<hash>.dll` on one platform and
    `libproj.so.25.9.8.1` on another, and nothing normalises the case of either.
    """

    @pytest.mark.parametrize(
        ("label", "member", "matches"),
        [
            ("plain-so", "pyramids_core.libs/libhdf5-abcdef12.so", True),
            ("plain-dll", "pyramids_core/hdf5-abcdef12.dll", True),
            ("plain-dylib", "pyramids_core/.dylibs/libhdf5.320.2.0.dylib", True),
            ("versioned-soname", "pyramids_core.libs/libhdf5-abc.so.320.1.0", True),
            ("upper-case-dll", "PYRAMIDS_CORE/LIBHDF5.DLL", True),
            ("upper-case-soname", "PYRAMIDS_CORE/LIBHDF5.SO.320.1.0", True),
            ("note-file", "pyramids_core.libs/hdf5-build-settings.txt", False),
            ("suffix-with-a-trailing-letter", "pyramids_core.libs/libhdf5.sox", False),
            ("token-only-in-the-directory", "libhdf5.so.1/README.md", False),
            ("another-library", "pyramids_core.libs/libproj-abc.so.25", False),
        ],
    )
    def test_a_member_is_scanned_only_when_its_basename_names_the_library(
        self, tmp_path, label, member, matches
    ):
        """Each shape a published wheel carries must be classified the way the comparison needs.

        Args:
            tmp_path: Per-test directory.
            label: Which shape is under test.
            member: The archive member name.
            matches: Whether the checker must treat it as a copy of HDF5.
        """
        found = _matched_members(tmp_path, "hdf5", [member])

        assert found == ([member] if matches else []), (
            f"{label}: {member!r} was classified the wrong way, got {found}"
        )

    def test_every_matching_copy_is_returned(self, tmp_path):
        """The filter yields all copies, since one bad copy among good ones must still be read.

        Test scenario:
            A repair tool can land the same library twice under different mangled names, which is
            how the ordering defect of #1627 arose; dropping all but one here would reintroduce it
            before any comparison happens.
        """
        found = _matched_members(
            tmp_path,
            "hdf5",
            [
                "pyramids_core.libs/aa_libhdf5-x.so.320.2.0",
                "pyramids_core.libs/hdf5-notes.txt",
                "pyramids_core.libs/zz_libhdf5-y.so.320.2.0",
            ],
        )

        assert sorted(found) == [
            "pyramids_core.libs/aa_libhdf5-x.so.320.2.0",
            "pyramids_core.libs/zz_libhdf5-y.so.320.2.0",
        ], f"both copies and only those must be scanned: {found}"


class TestAPartialSettingsStringIsStillJudged:
    """A copy reporting some of its properties but not all must fail, not drop out of the check.

    `read_reported` keeps any member that yielded at least one property, and the properties it did
    not yield are then absent from the comparison, which makes them unequal to whatever the baseline
    pins. The opposite rule -- requiring every pattern to match before a member counts as a copy --
    would make the #1620 defect invisible exactly where it matters: a library whose `Threadsafety:`
    line is gone would simply stop being a copy of HDF5, and the wheel would report no HDF5 drift.
    """

    def test_a_copy_reporting_only_its_version_still_becomes_a_reading(self, tmp_path):
        """One matched pattern is enough to make a member a reading."""
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(tmp_path, threadsafety=None)

        reported = checker.read_reported(wheel, ["hdf5"])

        assert len(reported["hdf5"]) == 1, (
            f"a partial settings string must still be read: {reported}"
        )
        assert reported["hdf5"][0]["version"] == "2.2.0", (
            f"the property that is present must be read: {reported}"
        )
        assert "threadsafety" not in reported["hdf5"][0], (
            f"the property that is absent must not be invented: {reported}"
        )

    def test_the_property_it_does_not_report_fails_the_comparison(self, tmp_path):
        """An unreadable `threadsafety` is drift, since the baseline pins it unconditionally."""
        checker = _checker_module("check-bundled-stack.py")
        wheel = _wheel(tmp_path, threadsafety=None)

        failures = checker.check_wheel(wheel, checker.load_baseline())

        assert any("threadsafety=None" in failure for failure in failures), (
            f"a copy whose thread-safety cannot be read must fail (#1620): {failures}"
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

    def test_an_exception_relaxing_no_version_at_all_changes_nothing(self):
        """An exception table that pins only another property must leave the baseline untouched.

        Test scenario:
            The validator already refuses such a table at load, so this is the second line of the
            same defence: were one to reach `expected_for` -- from a hand-built mapping, or from a
            future table shape -- relaxing `threadsafety` is still not something it can do.
        """
        checker = _checker_module("check-bundled-stack.py")
        baseline = {
            "libraries": {"hdf5": {"version": "2.2.0", "threadsafety": "ON"}},
            "exceptions": {"win_arm64": {"hdf5": {"threadsafety": "OFF"}}},
        }

        expected = checker.expected_for(baseline, "win_arm64")

        assert expected["hdf5"] == {"version": "2.2.0", "threadsafety": "ON"}, (
            f"an exception pinning no version must relax nothing: {expected}"
        )

    def test_an_empty_tag_resolves_no_exception(self):
        """A wheel whose name yielded no tag is compared against the unrelaxed baseline.

        Test scenario:
            The empty tag must not match an exception key by accident, since that would relax a
            platform nobody declared; `check_wheel` warns about it instead.
        """
        checker = _checker_module("check-bundled-stack.py")
        baseline = {
            "libraries": {"hdf5": {"version": "2.2.0"}},
            "exceptions": {"win_arm64": {"hdf5": {"version": "2.1.1"}}},
        }

        expected = checker.expected_for(baseline, "")

        assert expected == {"hdf5": {"version": "2.2.0"}}, (
            f"no tag must resolve no exception: {expected}"
        )

    def test_the_warning_names_every_exception_it_could_not_resolve(
        self, tmp_path, capsys
    ):
        """The WARN lists the declared tags, so the log says which relaxations went unapplied.

        Test scenario:
            Driven in-process rather than through a subprocess so the message itself is asserted.
            A wheel whose name carries no tag is compared against the unrelaxed baseline, and this
            line is the only trace of that -- a `win_arm64` wheel renamed in a repair step would
            otherwise look as though its stack had genuinely drifted.
        """
        checker = _checker_module("check-bundled-stack.py")
        renamed = tmp_path / "renamed.whl"
        _wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1").rename(renamed)

        failures = checker.check_wheel(renamed, checker.load_baseline())

        captured = capsys.readouterr()
        assert "no platform tag" in captured.err, (
            f"the warning must reach stderr: {captured.err}"
        )
        assert "win_arm64" in captured.err, (
            f"the warning must name the exceptions it could not resolve: {captured.err}"
        )
        assert failures, (
            "compared against the unrelaxed baseline, the win_arm64 HDF5 version must fail"
        )

    def test_a_baseline_without_exceptions_warns_about_nothing(self, tmp_path, capsys):
        """With no relaxations declared, a tagless filename costs nothing and draws no WARN."""
        checker = _checker_module("check-bundled-stack.py")
        renamed = tmp_path / "renamed.whl"
        _wheel(tmp_path).rename(renamed)
        baseline = {"libraries": {"hdf5": {"version": "2.2.0", "threadsafety": "ON"}}}

        failures = checker.check_wheel(renamed, baseline)

        assert failures == [], f"the wheel conforms, so nothing must fail: {failures}"
        assert "WARN" not in capsys.readouterr().err, (
            "there are no exceptions to resolve, so there is nothing to warn about"
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

    def test_a_wheel_path_that_does_not_exist_is_refused(self, tmp_path):
        """A typo'd wheel path must draw the script's own refusal, not a zipfile traceback.

        Test scenario:
            `_wheels` accepted any path whose suffix was `.whl`, so a mistyped name reached
            `zipfile.ZipFile` and surfaced as a FileNotFoundError traceback. Unreachable from the
            repair steps, which pass a directory, but a confusing failure for anyone running it by
            hand.
        """
        result = _run(tmp_path / "pyramids_core-0.81.0-cp312-cp312-win_amd64.whl")

        assert result.returncode != 0, "a non-existent wheel must be refused"
        assert "does not exist" in (result.stdout + result.stderr), (
            f"the refusal should say the path does not exist: {result.stdout} {result.stderr}"
        )
        assert "Traceback" not in result.stderr, (
            f"it must refuse cleanly rather than crash: {result.stderr}"
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

    def test_a_mistyped_wheel_path_is_refused_by_name(self, tmp_path):
        """A path ending in `.whl` that names no file is the script's own refusal, in-process.

        Test scenario:
            The subprocess variant pins the absence of a traceback; this pins the message, since
            `_wheels` must distinguish a mistyped wheel from an argument that could never name one.
        """
        checker = _checker_module("check-bundled-stack.py")
        missing = tmp_path / "pyramids_core-9.9.9-cp312-cp312-win_amd64.whl"

        with pytest.raises(SystemExit) as excinfo:
            checker._wheels([str(missing)])

        assert "does not exist" in str(excinfo.value), (
            f"the refusal must say the path does not exist, got: {excinfo.value}"
        )
        assert missing.name in str(excinfo.value), (
            f"the refusal must name the path, got: {excinfo.value}"
        )

    def test_an_empty_directory_refuses_to_report_success(self, tmp_path, capsys):
        """`main` over a directory holding no wheels exits 1 and says it compared nothing.

        Test scenario:
            This is the vacuous-success path, so the exit code and the explanation are both part of
            the contract: a repair step pointed at the wrong directory must fail the job rather than
            print a green line having checked no wheel at all.
        """
        checker = _checker_module("check-bundled-stack.py")
        empty = tmp_path / "nothing"
        empty.mkdir()

        status = checker.main([str(empty)])

        captured = capsys.readouterr()
        assert status == 1, f"an empty target set must fail, got {status}"
        assert "no wheels found" in captured.err, (
            f"the refusal must say no wheels were found: {captured.err}"
        )
        assert "compared nothing" in captured.err, (
            f"the refusal must say why that is a failure: {captured.err}"
        )
        assert str(empty) in captured.err, (
            f"the refusal must name the target it scanned: {captured.err}"
        )

    def test_one_drifted_wheel_fails_the_whole_directory(self, tmp_path, capsys):
        """A conforming wheel beside a drifted one neither masks it nor is mistaken for it.

        Test scenario:
            The mixed case is the realistic one: the repair step checks a directory of wheels built
            by five different routes, and #1620 was precisely three platforms conforming while two
            did not. The conforming wheel must still report `ok`, and the call must still exit 1.
        """
        checker = _checker_module("check-bundled-stack.py")
        passing = _wheel(tmp_path, platform_tag="manylinux_2_28_x86_64")
        drifted = _wheel(
            tmp_path, platform_tag="manylinux_2_28_aarch64", threadsafety="OFF"
        )

        status = checker.main([str(tmp_path)])

        captured = capsys.readouterr()
        assert status == 1, f"one drifted wheel must fail the call, got {status}"
        assert f"ok   {passing.name}" in captured.out, (
            f"the conforming wheel must still be reported ok: {captured.out}"
        )
        assert f"FAIL {drifted.name}" in captured.out, (
            f"the drifted wheel must be reported as failing: {captured.out}"
        )
        assert "1 bundled-library mismatch(es)" in captured.err, (
            f"exactly the one real mismatch must reach the summary: {captured.err}"
        )

    def test_running_the_file_as_a_script_exits_with_the_status(
        self, tmp_path, monkeypatch
    ):
        """The repair steps invoke it as `python ci/check-bundled-stack.py`, so that path must fail.

        Test scenario:
            A drifted wheel has to make the *process* exit non-zero, not merely return 1 from a
            function nobody checks -- the repair step's only signal is the exit status.
        """
        wheel = _wheel(tmp_path, threadsafety="OFF")
        monkeypatch.setattr(sys, "argv", [str(_CHECKER), str(wheel)])

        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(_CHECKER), run_name="__main__")

        assert excinfo.value.code == 1, (
            f"a drifted wheel must exit 1 through the script dispatch, got {excinfo.value.code}"
        )


def _shared_module():
    """Import `ci/native_stack_baseline.py`, the validated reader both checkers share."""
    spec = importlib.util.spec_from_file_location("native_stack_baseline", _SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestABaselineThatWouldCompareNothingIsRefused:
    """Validation is the point: a guard whose declaration can be mistyped into vacuity reports `ok`.

    Each case below passed silently before `native_stack_baseline.validate` existed -- a library no
    extractor knows, a property nothing reads, an exception with no reason, a platform key that
    matches no wheel tag. A declaration that compares nothing is the failure this whole file exists
    to prevent, so it must be refused by name rather than accepted (#1627).
    """

    @pytest.mark.parametrize(
        ("label", "body", "expected"),
        [
            (
                "unsupported-library",
                '[libraries.sqlite3]\nversion = "3.50"\n',
                "has no extractor",
            ),
            (
                "unknown-property",
                '[libraries.hdf5]\nparallel = "OFF"\n',
                "unknown property",
            ),
            ("empty-libraries", "[libraries]\n", "declares no"),
            (
                "no-libraries",
                '[installed.gdal]\nversion = "3.13.3"\n',
                "declares no",
            ),
            (
                "exception-without-reason",
                '[libraries.hdf5]\nversion = "2.2.0"\n[exceptions.win_arm64.hdf5]\nversion = "2.1.1"\n',
                "carries no reason",
            ),
            (
                "conda-subdir-as-tag",
                '[libraries.hdf5]\nversion = "2.2.0"\n'
                '[exceptions.osx-64.hdf5]\nversion = "2.1.1"\nreason = "x"\n',
                "not a wheel platform tag",
            ),
            (
                "exception-without-version",
                '[libraries.hdf5]\nversion = "2.2.0"\n[exceptions.win_arm64.hdf5]\nreason = "x"\n',
                "relaxes no version",
            ),
            (
                "installed-without-version",
                '[libraries.hdf5]\nversion = "2.2.0"\n[installed.gdal]\nbuild = "1"\n',
                "pins no version",
            ),
            (
                "library-pinning-nothing",
                "[libraries.hdf5]\n",
                "pins no property",
            ),
            (
                "library-pinning-only-prose",
                '[libraries.hdf5]\nreason = "a note is not a comparison"\n',
                "pins no property",
            ),
            (
                "exception-for-an-uncomparable-library",
                '[libraries.hdf5]\nversion = "2.2.0"\n'
                '[exceptions.win_arm64.sqlite3]\nversion = "3.50"\nreason = "x"\n',
                "names a library nothing compares",
            ),
        ],
    )
    def test_an_unusable_baseline_is_refused_by_name(
        self, tmp_path, label, body, expected
    ):
        """Each malformed declaration must raise, naming the table at fault.

        Args:
            tmp_path: Per-test directory.
            label: Which malformation is under test.
            body: The baseline contents.
            expected: Text the error must contain.
        """
        baseline = tmp_path / "native-stack-baseline.toml"
        baseline.write_text(body, encoding="utf-8")

        module = _shared_module()

        with pytest.raises(ValueError, match=expected):
            module.load(baseline)

    def test_the_shipped_baseline_validates(self):
        """The real declaration must satisfy its own rules."""
        module = _shared_module()

        assert module.load(), "the shipped baseline must load and validate"

    def test_the_extractors_cover_exactly_the_supported_libraries(self):
        """A library the shared module allows but the checker cannot read would compare nothing."""
        checker = _checker_module("check-bundled-stack.py")

        assert set(checker._EXTRACTORS) == set(_shared_module().SUPPORTED_LIBRARIES), (
            "the extractor table and the supported-library list must agree"
        )


class TestTheInstalledPlatformTagSharesTheWheelNamespace:
    """Both sides must resolve exceptions from the same key space, or a relaxation matches nothing.

    The installed side previously derived its tag from `sysconfig.get_platform()`, which reports
    `linux_x86_64` for both the manylinux and musllinux wheels and the interpreter's deployment
    target on macOS. A Linux or macOS exception would therefore have needed two different keys, and
    whichever was missing would have silently relaxed nothing (#1627).
    """

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("pyramids_core-0.81.0-cp312-cp312-win_arm64.whl", "win_arm64"),
            (
                "pyramids_core-0.81.0-cp312-cp312-macosx_11_0_x86_64.whl",
                "macosx_11_0_x86_64",
            ),
            (
                "pyramids_core-0.81.0-cp312-cp312-manylinux_2_28_aarch64.whl",
                "manylinux_2_28_aarch64",
            ),
            (
                "pyramids_core-0.81.0-cp312-cp312-musllinux_1_2_x86_64.whl",
                "musllinux_1_2_x86_64",
            ),
            ("not-a-wheel.whl", ""),
        ],
    )
    def test_a_wheel_name_yields_its_platform_tag(self, filename, expected):
        """The filename tag must come out exactly, since exception keys are matched against it.

        Args:
            filename: The wheel name.
            expected: The tag it must yield, empty when the name is not PEP 427 shaped.
        """
        assert _shared_module().wheel_name_platform_tag(filename) == expected

    def test_an_absent_distribution_yields_no_tag_rather_than_raising(self):
        """A missing distribution must degrade to the unrelaxed baseline, with the caller warning."""
        assert (
            _shared_module().installed_platform_tag("no-such-distribution-1627") == ""
        )

    def test_the_declared_exception_tags_are_wheel_tags(self):
        """Every exception key in the shipped baseline must be resolvable from a wheel name."""
        module = _shared_module()
        baseline = module.load()
        wheel_names = {
            tag: f"pyramids_core-0.81.0-cp312-cp312-{tag}.whl"
            for tag in baseline["exceptions"]
        }

        for tag, name in wheel_names.items():
            assert module.wheel_name_platform_tag(name) == tag, (
                f"exception key {tag!r} is not what a wheel filename yields, so it relaxes nothing"
            )


def _fake_distribution(tmp_path: Path, name: str, tags: list[str] | None) -> Path:
    """Write a `.dist-info` an `importlib.metadata` scan of the returned directory will find.

    Args:
        tmp_path: Directory to serve as a synthetic site-packages.
        name: The distribution name to install under, unique per test so no cache is shared.
        tags: The `Tag:` values for `WHEEL`, without the `Tag: ` prefix; None omits `WHEEL`
            entirely, as a distribution not installed from a wheel does.

    Returns:
        The directory to prepend to `sys.path`.
    """
    info = tmp_path / f"{name}-1.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    if tags is not None:
        (info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            + "".join(f"Tag: {tag}\n" for tag in tags),
            encoding="utf-8",
        )
    return tmp_path


class TestTheInstalledTagComesFromTheWheelMetadata:
    """`installed_platform_tag` reads `.dist-info/WHEEL`, and refuses to guess when it cannot.

    The tag is the key an exception is resolved under, so a wrong one is worse than none: it would
    compare one platform against another platform's declared values. Anything short of a single
    concrete platform therefore yields "" and leaves the caller to warn -- a fat or `any`-platform
    install, a distribution carrying no `Tag:` line, one not installed from a wheel at all. Driven
    against a real `.dist-info` rather than a patched reader, so the parsing is what is under test.
    """

    @pytest.mark.parametrize(
        ("label", "tags", "expected"),
        [
            ("one-platform", ["cp312-cp312-win_arm64"], "win_arm64"),
            (
                "one-platform-several-abis",
                ["cp311-cp311-win_amd64", "cp312-cp312-win_amd64"],
                "win_amd64",
            ),
            (
                "several-platforms",
                ["cp312-cp312-macosx_11_0_x86_64", "cp312-cp312-macosx_11_0_arm64"],
                "",
            ),
            ("no-tag-line", [], ""),
            ("pure-python", ["py3-none-any"], ""),
            ("not-installed-from-a-wheel", None, ""),
        ],
    )
    def test_the_tag_is_read_from_the_installed_wheel_metadata(
        self, tmp_path, monkeypatch, label, tags, expected
    ):
        """Each `WHEEL` shape must yield exactly one platform tag, or none at all.

        Args:
            tmp_path: Per-test directory, used as a synthetic site-packages.
            monkeypatch: Restores `sys.path` afterwards.
            label: Which shape is under test; also the distribution name, kept unique so
                `importlib.metadata` cannot serve one case's answer to another.
            tags: The `Tag:` values to write, or None to omit `WHEEL`.
            expected: The tag the reader must return.
        """
        name = f"fakedist{label.replace('-', '')}"
        monkeypatch.syspath_prepend(str(_fake_distribution(tmp_path, name, tags)))
        importlib.invalidate_caches()

        found = _shared_module().installed_platform_tag(name)

        assert found == expected, (
            f"{label}: expected {expected!r} from Tag lines {tags!r}, got {found!r}"
        )

    def test_a_fat_install_is_reported_rather_than_guessed_at(
        self, tmp_path, monkeypatch
    ):
        """Two platforms in one install must yield "", not whichever the set happens to pop.

        Test scenario:
            A `delocate`-fused universal2 install carries both arch tags. Picking one would resolve
            that install's exceptions under a platform it is only half of, and the choice would vary
            run to run because it comes out of a set.
        """
        monkeypatch.syspath_prepend(
            str(
                _fake_distribution(
                    tmp_path,
                    "fakedistfat1627",
                    ["cp312-cp312-macosx_11_0_arm64", "cp312-cp312-macosx_11_0_x86_64"],
                )
            )
        )
        importlib.invalidate_caches()

        assert _shared_module().installed_platform_tag("fakedistfat1627") == "", (
            "a fat install must resolve no tag rather than half of one"
        )


class TestVerifyWheelAssertsTheGdalVersion:
    """`ci/verify-wheel.py` owns the installed-side assertion, because it runs in every job.

    The version check was first written as its own script and wired into two of the eight smoke and
    verify jobs, so GDAL parity was asserted for the musllinux source build alone -- the one route
    whose GDAL was never in doubt -- and the `win_arm64` GDAL exception was never evaluated by any
    job. `verify-wheel.py` already runs everywhere and already imports `pyramids` ahead of `osgeo`,
    which a published wheel requires (#1627).
    """

    def test_it_is_wired_to_the_shared_resolution(self):
        """A cheap wiring check: the function must exist and defer to the shared module.

        Test scenario:
            This is deliberately only a wiring check. `verify-wheel.py` runs every check at import
            and imports `pyramids`/`osgeo`, so it cannot be imported by a test; the behaviour it
            reports is therefore tested through `native_stack_baseline.expected_gdal`, by execution,
            in `TestTheInstalledGdalExpectationIsResolved`. Substring assertions alone were the whole
            guard once, and they could survive a logic inversion that stopped the exception applying.
        """
        source = _VERIFY_WHEEL.read_text(encoding="utf-8")

        assert "_check_gdal_version()" in source, (
            "verify-wheel.py must call the GDAL version check, or no job asserts it"
        )
        assert "native_stack_baseline.expected_gdal(" in source, (
            "the expectation must come from the shared, tested resolution rather than be hand-rolled"
        )

    def test_it_does_not_re_assert_the_driver_set(self):
        """`_check_driver_set` already covers drivers, over a superset, on every platform."""
        source = _VERIFY_WHEEL.read_text(encoding="utf-8")

        assert "installed.drivers" not in source, (
            "the driver contract belongs to _check_driver_set; a second, weaker list would be a "
            "third hand-maintained copy"
        )

    def test_the_baseline_no_longer_declares_drivers(self):
        """The driver lists were dropped in favour of verify-wheel.py's stronger contract."""
        baseline = _shared_module().load()

        assert "drivers" not in baseline.get("installed", {}), (
            "ci/verify-wheel.py owns the driver contract; the baseline must not duplicate it"
        )

    def test_pyramids_is_imported_before_osgeo(self):
        """The vendor bootstrap must come first, or the script cannot run against a wheel.

        Test scenario:
            A published wheel ships no top-level `osgeo`: it lives at `pyramids/_vendor/osgeo`,
            reachable only after `pyramids/__init__.py` injects it onto `sys.path`. A bare
            `from osgeo import ...` raises ModuleNotFoundError in exactly the environment this
            script is written for, while passing in a conda dev environment.
        """
        tree = ast.parse(_VERIFY_WHEEL.read_text(encoding="utf-8"))
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
            "verify-wheel.py must import pyramids to run the vendor bootstrap"
        )
        assert osgeo_at, "verify-wheel.py is expected to import osgeo"
        assert min(pyramids_at) < min(osgeo_at), (
            f"import pyramids (line {min(pyramids_at)}) must precede osgeo (line {min(osgeo_at)})"
        )


_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_SOURCE_BUILD_CONFIG = _REPO_ROOT / "ci" / "source-build" / "config.sh"
_VCPKG_MANIFEST = _REPO_ROOT / "ci" / "vcpkg.json"
_WHEEL_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "bundle-pypi-wheels.yml"


def _source_build_pin(library: str) -> str:
    """The version `ci/source-build/config.sh` pins for one library.

    Args:
        library: The `define <name> <version> <sha256>` key to read.

    Returns:
        The pinned version string, or an empty string when the library is not pinned there.
    """
    pattern = re.compile(rf"^define\s+{re.escape(library)}\s+(\S+)", re.M)
    match = pattern.search(_SOURCE_BUILD_CONFIG.read_text(encoding="utf-8"))
    return match.group(1) if match else ""


class TestTheBaselineAgreesWithEveryPinItRestates:
    """The baseline is a fifth pinning site, so something must compare it with the other four.

    Its own rationale is that the pins live in several places with nothing comparing them -- and it
    then became another place. Without these tests the next GDAL, HDF5, PROJ or GEOS bump leaves the
    baseline behind and the mismatch surfaces *after* the wheels are built, in a job that only runs
    on a push to main or at release (#1627).
    """

    def test_the_installed_gdal_version_satisfies_the_pyproject_spec(self):
        """`installed.gdal.version` restates the conda pin that pyproject calls the single source."""
        with _PYPROJECT.open("rb") as handle:
            spec = tomllib.load(handle)["tool"]["pixi"]["feature"]["gdal"][
                "dependencies"
            ]["gdal"]
        pinned = _shared_module().load()["installed"]["gdal"]["version"]
        floor = spec.split(",")[0].lstrip(">=")

        assert pinned == floor, (
            f"the baseline pins gdal {pinned} but pyproject's gdal feature pins {spec!r}; the "
            "smoke jobs would fail after the wheels are built. Update both together."
        )

    @pytest.mark.parametrize("library", ["hdf5", "proj", "geos"])
    def test_the_baseline_matches_the_source_build_pin(self, library):
        """The source build compiles these from pinned tarballs; the baseline must expect those.

        Args:
            library: The library whose two pins must agree.
        """
        declared = _shared_module().load()["libraries"][library]["version"]
        compiled = _source_build_pin(library)

        assert compiled, f"ci/source-build/config.sh pins no {library}"
        assert declared == compiled, (
            f"the baseline expects {library} {declared} but the source build compiles {compiled}, "
            "so every manylinux and musllinux wheel would fail its own repair step"
        )

    @pytest.mark.parametrize("library", ["proj", "geos"])
    def test_the_baseline_matches_the_vcpkg_override(self, library):
        """win_arm64 takes these from vcpkg, whose overrides must match the baseline.

        Args:
            library: The library whose vcpkg override must agree with the baseline.
        """
        manifest = json.loads(_VCPKG_MANIFEST.read_text(encoding="utf-8"))
        overrides = {
            entry["name"]: entry.get("version", entry.get("version-semver", ""))
            for entry in manifest.get("overrides", [])
        }
        if library not in overrides:
            pytest.skip(f"ci/vcpkg.json declares no override for {library}")
        declared = _shared_module().load()["libraries"][library]["version"]

        assert declared == overrides[library], (
            f"the baseline expects {library} {declared} but ci/vcpkg.json overrides it to "
            f"{overrides[library]}, so the win_arm64 wheel would fail its own repair step"
        )


class TestTheCheckersAreActuallyWiredIntoCi:
    """Both top findings of this branch's review were wiring, not logic, so the wiring is guarded.

    The wheel workflow has no `pull_request` trigger, so neither checker runs in anger before a
    merge. A checker that is correct but invoked nowhere -- or invoked in two jobs out of eight --
    asserts nothing, and nothing but a test like this notices (#1627).
    """

    def test_every_repair_command_invokes_the_bundled_checker(self):
        """All three pyproject repair commands must run it, or that platform goes unchecked."""
        with _PYPROJECT.open("rb") as handle:
            cibuildwheel = tomllib.load(handle)["tool"]["cibuildwheel"]
        repair_commands = {
            platform: settings["repair-wheel-command"]
            for platform, settings in cibuildwheel.items()
            if isinstance(settings, dict) and "repair-wheel-command" in settings
        }

        assert len(repair_commands) >= 3, (
            f"expected a repair command per platform, found {sorted(repair_commands)}"
        )
        for platform, command in repair_commands.items():
            rendered = " ".join(command) if isinstance(command, list) else command
            assert "check-bundled-stack.py" in rendered, (
                f"the {platform} repair command does not run check-bundled-stack.py, so that "
                "platform's bundled stack is never compared against the baseline"
            )

    def test_the_musllinux_repair_override_invokes_the_bundled_checker(self):
        """The workflow's CIBW_REPAIR_WHEEL_COMMAND replaces the TOML command wholesale."""
        workflow = _WHEEL_WORKFLOW.read_text(encoding="utf-8")
        overrides = [
            line
            for line in workflow.splitlines()
            if "CIBW_REPAIR_WHEEL_COMMAND" in line
        ]

        assert overrides, "expected a musllinux repair override in the wheel workflow"
        for line in overrides:
            index = workflow.splitlines().index(line)
            block = "\n".join(workflow.splitlines()[index : index + 6])
            assert "check-bundled-stack.py" in block, (
                "a CIBW_REPAIR_WHEEL_COMMAND override replaces the repair command from "
                "pyproject.toml entirely, so it must invoke the checker itself:\n"
                + block
            )

    def test_every_smoke_and_verify_job_runs_verify_wheel(self):
        """The installed-side assertion lives in verify-wheel.py, so every such job must run it."""
        workflow = yaml.safe_load(_WHEEL_WORKFLOW.read_text(encoding="utf-8"))
        candidates = {
            name: job
            for name, job in workflow["jobs"].items()
            if name.startswith(("test-wheels", "verify-"))
        }
        without = [
            name
            for name, job in candidates.items()
            if "verify-wheel.py" not in yaml.safe_dump(job.get("steps", []))
        ]

        assert candidates, "expected smoke and verify jobs in the wheel workflow"
        assert not without, (
            f"these jobs install a wheel but never run ci/verify-wheel.py, so GDAL's version and "
            f"the driver set go unasserted there: {without}"
        )

    def test_the_bundled_checker_runs_as_a_subprocess(self):
        """Executing it is what makes an import-time break visible to this suite.

        Test scenario:
            The critical finding of this branch's review was an import that could not resolve in a
            wheel environment. A test that only reads the source would not have caught it; running
            the script would.
        """
        result = subprocess.run(
            [sys.executable, str(_CHECKER)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0, (
            "with no arguments the checker must refuse, not pass"
        )
        assert "Traceback" not in result.stderr, (
            f"the checker must refuse cleanly rather than crash: {result.stderr}"
        )
        assert "usage" in (result.stdout + result.stderr).lower(), (
            f"the refusal should state the usage: {result.stdout} {result.stderr}"
        )


class TestTheInstalledGdalExpectationIsResolved:
    """`expected_gdal` decides what GDAL each platform must report, and says where that came from.

    This is the installed half of #1627 -- the half that was found broken, where a separate checker
    was wired into two of eight jobs. It is also the only evaluation of `[exceptions.win_arm64.gdal]`,
    a relaxation that exists for exactly one platform. It previously lived inline in
    `ci/verify-wheel.py`, which cannot be imported by a test, so it was guarded only by matching that
    file's source text -- assertions that a reflow would break and a logic inversion would survive.
    """

    def test_a_platform_without_an_exception_gets_the_baseline_pin(self):
        """The ordinary case: four of the five platforms resolve `installed.gdal`."""
        baseline = _shared_module().load()

        assert _shared_module().expected_gdal(baseline, "win_amd64") == (
            baseline["installed"]["gdal"]["version"],
            "installed.gdal",
        )

    def test_win_arm64_gets_its_declared_relaxation(self):
        """The one live exception must actually apply, and name itself as the source.

        Test scenario:
            vcpkg's pinned baseline carries no `gdal 3.13.x` port, so this platform is a minor
            version behind by force. If the relaxation stopped applying, every win_arm64 wheel would
            fail its own repair step against a version it provably cannot reach.
        """
        module = _shared_module()
        baseline = module.load()
        declared = baseline["exceptions"]["win_arm64"]["gdal"]["version"]

        version, source = module.expected_gdal(baseline, "win_arm64")

        assert version == declared, (
            f"win_arm64 must resolve its declared {declared}, got {version}"
        )
        assert source == "exceptions.win_arm64.gdal", (
            f"the source must name the exception so the failure message can cite it, got {source}"
        )
        assert version != baseline["installed"]["gdal"]["version"], (
            "this test is vacuous unless the exception differs from the baseline pin"
        )

    @pytest.mark.parametrize(
        "tag", ["", "macosx_11_0_x86_64", "manylinux_2_28_aarch64"]
    )
    def test_a_tag_with_no_exception_falls_back_to_the_pin(self, tag):
        """An unresolvable or unrelaxed tag must compare against the baseline, not against nothing.

        Args:
            tag: A platform tag the baseline declares no GDAL exception for.
        """
        module = _shared_module()
        baseline = module.load()

        assert module.expected_gdal(baseline, tag) == (
            baseline["installed"]["gdal"]["version"],
            "installed.gdal",
        )

    def test_a_baseline_pinning_no_gdal_yields_an_empty_version(self):
        """The caller must be able to refuse rather than silently compare nothing."""
        version, source = _shared_module().expected_gdal(
            {"libraries": {"hdf5": {"version": "2.2.0"}}}, "win_amd64"
        )

        assert version == "", (
            f"an absent [installed.gdal] must resolve to no version, got {version}"
        )
        assert source == "installed.gdal", (
            f"the source should still be nameable, got {source}"
        )

    def test_an_exception_for_another_platform_does_not_leak(self):
        """win_arm64's relaxation must not reach the platform next to it."""
        module = _shared_module()
        baseline = module.load()

        leaked = module.expected_gdal(baseline, "win_amd64")[0]

        assert leaked != baseline["exceptions"]["win_arm64"]["gdal"]["version"], (
            "win_amd64 resolved win_arm64's relaxed version, so the exception is not keyed by tag"
        )
