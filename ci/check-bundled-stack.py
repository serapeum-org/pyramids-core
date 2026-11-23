#!/usr/bin/env python
"""Fail the wheel build when a bundled native library drifts from the declared baseline.

pyramids-core builds its native stack three different ways -- from source for the Linux wheels,
through vcpkg for ``win_arm64``, and from conda-forge binaries for ``win_amd64``/macOS -- and each
route inherits its upstream's defaults for anything not explicitly requested. One release shipped
HDF5 with thread-safety ON for three platforms and OFF for the other two while every version pin
agreed (#1620), because nothing compared the built artifacts (#1627).

This reads what a repaired wheel actually contains and compares it against
``ci/native-stack-baseline.toml``. Every property comes out of the library's own embedded
build-settings string, so the check needs no install and no import of the wheel: about a second per
wheel. Run it on the *repaired* wheel -- before ``auditwheel``/``delocate``/``delvewheel`` there are
no bundled libraries to inspect.

GDAL is deliberately not checked here. Its version is not reliably greppable (a manylinux
``libgdal`` yields only a spurious ``GDAL 3.6.0``, macOS yields two candidates, and the Windows DLLs
carry no version in their filename), so GDAL parity and the driver list are asserted from the
installed wheel instead.

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
        The parsed document, with `libraries` and optional `exceptions` tables.
    """
    return tomllib.loads(path.read_text(encoding="utf-8"))


def platform_tag(wheel: pathlib.Path) -> str:
    """The platform tag of a wheel, used to resolve declared exceptions.

    Args:
        wheel: A wheel path named per PEP 427, `name-version-python-abi-platform.whl`.

    Returns:
        The platform component, or an empty string when the name does not parse.
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
        platform cannot opt out of the property that has already regressed.
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

    Args:
        archive: The opened wheel.
        token: The substring identifying the library, e.g. `hdf5`.

    Returns:
        Matching member names. Companion libraries such as `libhdf5_hl` match too; they embed no
        settings string, so they contribute nothing and are harmless.
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
        A `{library: {property: value}}` mapping. A library contributes nothing when it is absent
        from the wheel, which the caller reports as a failure in its own right.
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

    Args:
        wheel: The repaired wheel to check.
        baseline: The parsed baseline.

    Returns:
        A list of human-readable failures; empty when the wheel conforms.
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
        targets: Wheel paths, or directories to scan for `*.whl`.

    Returns:
        The wheels to check, sorted and deduplicated.
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
        0 when every wheel conforms; 1 otherwise.
    """
    if not argv:
        raise SystemExit(__doc__)
    baseline = load_baseline()
    failures: list[str] = []
    for wheel in _wheels(argv):
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
