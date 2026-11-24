#!/usr/bin/env python
"""Fail the wheel build when a bundled native library drifts from the declared baseline.

pyramids-core builds its native stack three different ways -- from source for the Linux wheels,
through vcpkg for ``win_arm64``, and from conda-forge binaries for ``win_amd64``/macOS -- and each
route inherits its upstream's defaults for anything not explicitly requested. So the 0.81.0 wheels
shipped three different HDF5s under one wheel version: 2.2.0 thread-safe from conda-forge
(``win_amd64`` and both macOS arches), 2.1.0 non-thread-safe from the two Linux source builds, and
2.1.1 non-thread-safe from vcpkg (``win_arm64``) -- #1620. The pins that would have shown it live in
three separate places, and nothing compared the built artifacts against any of them (#1627).

This reads what a repaired wheel actually contains and compares it against
``ci/native-stack-baseline.toml``. Every property comes out of the library's own embedded
build-settings string, and only the archive members that look like a checked library are read, so
the check needs no install and no import of the wheel, and costs a fraction of a second per wheel.
Run it on the *repaired* wheel -- before ``auditwheel``/``delocate``/``delvewheel`` there are
no bundled libraries to inspect.

GDAL is deliberately not checked here. Its version is not reliably greppable (a manylinux
``libgdal`` yields only a spurious ``GDAL 3.6.0``, macOS yields two candidates, and the Windows DLLs
carry no version in their filename), so GDAL parity and the driver list are asserted from the
installed wheel instead -- see ``ci/check-installed-stack.py``.

Every comparison is printed as an ``ok``/``FAIL`` line on stdout; the summary and any ``WARN`` go to
stderr. Exits 0 when every wheel matches the baseline, and 1 on any mismatch or when the arguments
name no wheels at all.

Usage:
    python ci/check-bundled-stack.py <wheel-or-directory> [...]
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib
import zipfile

_BASELINE_PATH = pathlib.Path(__file__).resolve().parent / "native-stack-baseline.toml"

#: How to read each library's own build-settings string out of its binary. The key is the token that
#: must appear in the library's filename; `version` and `threadsafety` are the properties the
#: baseline may pin. Each pattern was verified against the published wheels of all five platforms.
_EXTRACTORS = {
    "hdf5": {
        "version": re.compile(rb"HDF5 Version:\s*([0-9][0-9.]*)"),
        "threadsafety": re.compile(rb"Threadsafety:\s*(\w+)"),
    },
    "proj": {"version": re.compile(rb"PROJ (\d+\.\d+\.\d+)")},
    "geos": {"version": re.compile(rb"(\d+\.\d+\.\d+)-CAPI")},
}

_LIBRARY_SUFFIXES = (".so", ".dll", ".dylib")


def load_baseline(path: pathlib.Path = _BASELINE_PATH) -> dict:
    """Read the declared baseline.

    Args:
        path: The baseline file. Defaults to the one beside this script.

    Returns:
        The parsed document. `libraries` holds the properties every wheel must report, the optional
        `exceptions` table the per-platform relaxations, and `installed` the GDAL version and driver
        lists that only `ci/check-installed-stack.py` reads.

    Raises:
        FileNotFoundError: If `path` does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    return tomllib.loads(path.read_text(encoding="utf-8"))


def platform_tag(wheel: pathlib.Path) -> str:
    """The platform tag of a wheel, used to resolve declared exceptions.

    Args:
        wheel: A wheel path named per PEP 427, `name-version-python-abi-platform.whl`, optionally
            with a build tag as its third component.

    Returns:
        The last `-`-separated component of the filename, which is the platform tag in both forms,
        or an empty string when the name has too few components to be a wheel name -- a renamed
        copy, for instance, whose exceptions `check_wheel` then reports as unresolvable. A
        compressed tag set is returned whole, dots included.
    """
    parts = wheel.stem.split("-")
    return parts[-1] if len(parts) >= 5 else ""


def expected_for(baseline: dict, tag: str) -> dict:
    """The properties each library must report on one platform.

    Args:
        baseline: The parsed baseline.
        tag: The wheel's platform tag.

    Returns:
        A `{library: {property: value}}` mapping, with any declared exception for `tag` applied
        over the baseline. An exception may relax a version; `threadsafety` is never waived, so a
        platform cannot opt out of the property that has already regressed. Every other key of an
        exception is dropped, including the `reason` prose it must carry, and an exception naming a
        library the `libraries` table does not list is ignored -- which is how the baseline's
        `exceptions.<tag>.gdal` entry stays inert here while `ci/check-installed-stack.py` honours
        it. An empty `tag` matches no exception, so the wheel is compared against the baseline.
    """
    expected = {name: dict(props) for name, props in baseline["libraries"].items()}
    for name, overrides in baseline.get("exceptions", {}).get(tag, {}).items():
        if name not in expected:
            continue
        for prop, value in overrides.items():
            if prop in ("version",):
                expected[name][prop] = value
    return expected


def _members_for(archive: zipfile.ZipFile, token: str) -> list[str]:
    """The archive members that look like the named native library.

    A member qualifies when its lower-cased basename contains `token` and carries one of the native
    suffixes, either at the end (`geos_c-<hash>.dll`) or as a versioned soname (`libproj.so.25.9.8.1`,
    `libhdf5.320.2.0.dylib`).

    Args:
        archive: The opened wheel.
        token: The substring identifying the library, lower-cased to match the basename, e.g. `hdf5`.

    Returns:
        Matching member names. Companion and satellite libraries match too -- `libhdf5_hl`, the
        `libgeos` core beneath `libgeos_c`, GDAL's own `gdal_hdf5` driver plugin -- but none of them
        embeds a settings string, so they contribute nothing and are harmless.
    """
    found = []
    for member in archive.namelist():
        base = member.rsplit("/", 1)[-1].lower()
        if token not in base:
            continue
        if base.endswith(_LIBRARY_SUFFIXES) or any(
            f"{s}." in base for s in _LIBRARY_SUFFIXES
        ):
            found.append(member)
    return found


def read_reported(wheel: pathlib.Path, libraries: list[str]) -> dict:
    """Read what each library in a wheel reports about itself.

    Args:
        wheel: The wheel to inspect.
        libraries: The library tokens to look for.

    Returns:
        A `{library: {property: value}}` mapping, each inner dict also carrying a `_member` key
        naming the file the values were read from, for the comparison messages. A library maps to an
        empty dict when the wheel bundles no copy of it, or bundles one whose bytes carry no settings
        string, which the caller reports as a failure in its own right. When several members report
        the same property the last one read wins.
    """
    reported: dict = {name: {} for name in libraries}
    with zipfile.ZipFile(wheel) as archive:
        for name in libraries:
            for member in _members_for(archive, name):
                blob = archive.read(member)
                for prop, pattern in _EXTRACTORS[name].items():
                    match = pattern.search(blob)
                    if match:
                        reported[name][prop] = match.group(1).decode("ascii", "replace")
                        reported[name]["_member"] = member.rsplit("/", 1)[-1]
    return reported


def check_wheel(wheel: pathlib.Path, baseline: dict) -> list[str]:
    """Compare one wheel's bundled stack against the baseline.

    Every comparison is also printed as an `ok` or `FAIL` line on stdout, prefixed with the wheel's
    name and platform tag. A `reason` key is documentation rather than a property, so it is skipped
    instead of compared -- which matters for a `reason` written into a `libraries` entry, an
    exception's own never reaching here. A wheel whose filename yields no platform tag draws a `WARN`
    on stderr when the baseline declares exceptions at all, since it is then compared against the
    unrelaxed baseline and any relaxation that should have applied looks like real drift.

    Args:
        wheel: The repaired wheel to check.
        baseline: The parsed baseline.

    Returns:
        A list of human-readable failures -- one per mismatched property, plus one per library the
        wheel does not bundle; empty when the wheel conforms.
    """
    tag = platform_tag(wheel)
    if not tag and baseline.get("exceptions"):
        # Silently applying no exceptions is the same class of failure this check exists to catch:
        # a platform that should have been compared against its own declared values gets compared
        # against the baseline instead, and the mismatch looks like real drift.
        print(
            f"WARN {wheel.name}: no platform tag in the filename, so declared exceptions "
            f"({', '.join(sorted(baseline['exceptions']))}) cannot be resolved for it",
            file=sys.stderr,
        )
    expected = expected_for(baseline, tag)
    reported = read_reported(wheel, list(expected))
    failures = []
    for name, wanted in sorted(expected.items()):
        got = reported.get(name, {})
        if not got:
            failures.append(f"{wheel.name}: no bundled {name} found")
            print(f"FAIL {wheel.name}: no bundled {name} found")
            continue
        member = got.get("_member", name)
        for prop, want in sorted(wanted.items()):
            if prop == "reason":
                continue
            have = got.get(prop)
            line = f"{member} {prop}={have} (want {want})"
            if have == want:
                print(f"ok   {wheel.name} [{tag}]: {line}")
            else:
                failures.append(f"{wheel.name} [{tag}]: {line}")
                print(f"FAIL {wheel.name} [{tag}]: {line}")
    return failures


def _wheels(targets: list[str]) -> list[pathlib.Path]:
    """Expand each argument into the wheels it names.

    Args:
        targets: Wheel paths, or directories to scan for `*.whl`. A directory is scanned one level
            deep and may legitimately yield nothing, which `main` treats as a failure.

    Returns:
        The wheels to check, sorted and deduplicated.

    Raises:
        SystemExit: If a target is neither an existing directory nor a path ending in `.whl`.
    """
    found: set[pathlib.Path] = set()
    for target in targets:
        path = pathlib.Path(target)
        if path.is_dir():
            found.update(path.glob("*.whl"))
        elif path.suffix == ".whl":
            found.add(path)
        else:
            raise SystemExit(
                f"check-bundled-stack: {target!r} is neither a wheel nor a directory"
            )
    return sorted(found)


def main(argv: list[str]) -> int:
    """Check every wheel named on the command line against the baseline.

    Args:
        argv: Wheel paths and/or directories containing wheels.

    Returns:
        0 only when at least one wheel was found and every one of them conforms. 1 when any wheel
        mismatches the baseline, and 1 as well when the targets name no wheels at all: an empty
        directory is a refusal, not a no-op, because reporting success having compared nothing is
        the outcome this guard exists to prevent.

    Raises:
        SystemExit: If `argv` is empty, carrying this script's usage text, or if a target is neither
            a wheel nor a directory.
    """
    if not argv:
        raise SystemExit(__doc__)
    baseline = load_baseline()
    wheels = _wheels(argv)
    if not wheels:
        # A repair step is always pointed at a directory holding the wheel it just repaired, so an
        # empty one means the wiring broke, not that there is nothing to check. Passing here would
        # be a vacuous success -- the same silently-unchecked outcome this guard exists to prevent.
        print(
            f"check-bundled-stack: no wheels found in {', '.join(argv)}; refusing to report "
            "success having compared nothing.",
            file=sys.stderr,
        )
        return 1
    failures: list[str] = []
    for wheel in wheels:
        failures.extend(check_wheel(wheel, baseline))
    if failures:
        print(
            f"\ncheck-bundled-stack: {len(failures)} bundled-library mismatch(es) against "
            f"{_BASELINE_PATH.name}.\nEither fix the build route that drifted, or -- if the platform "
            "genuinely cannot meet the baseline -- add a declared exception with its reason to that "
            "file (see issue #1627).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
