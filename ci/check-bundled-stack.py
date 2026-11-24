#!/usr/bin/env python
"""Fail the wheel build when a bundled native library drifts from the declared baseline.

pyramids-core builds its native stack three different ways -- from source for the Linux wheels,
through vcpkg for `win_arm64`, and from conda-forge binaries for `win_amd64`/macOS -- and each
route inherits its upstream's defaults for anything not explicitly requested. So the 0.81.0 wheels
shipped three different HDF5s under one wheel version: 2.2.0 thread-safe from conda-forge
(`win_amd64` and both macOS arches), 2.1.0 non-thread-safe from the two Linux source builds, and
2.1.1 non-thread-safe from vcpkg (`win_arm64`) -- #1620. The pins that would have shown it live in
three separate places, and nothing compared the built artifacts against any of them (#1627).

This reads what a repaired wheel actually contains and compares it against
`ci/native-stack-baseline.toml`. Every property comes out of the library's own embedded
build-settings string, and only the archive members that look like a checked library are read, so
the check needs no install and no import of the wheel, and costs a fraction of a second per wheel.
Run it on the *repaired* wheel -- before `auditwheel`/`delocate`/`delvewheel` there are
no bundled libraries to inspect.

GDAL is deliberately not checked here. Its version is not reliably greppable (a manylinux
`libgdal` yields only a spurious `GDAL 3.6.0`, macOS yields two candidates, and the Windows DLLs
carry no version in their filename), so GDAL parity is asserted from the *installed* wheel instead,
by `ci/verify-wheel.py`'s `_check_gdal_version`. The required-driver set is not declared in the
baseline at all: that same script's `_check_driver_set` already gates a superset of it on every
platform.

Every comparison is printed as an `ok`/`FAIL` line on stdout; the summary and any `WARN` go to
stderr. Exits 0 when every wheel matches the baseline, and 1 on any mismatch or when the arguments
name no wheels at all.

Usage:
    python ci/check-bundled-stack.py <wheel-or-directory> [...]
"""

from __future__ import annotations

import pathlib
import re
import sys
import zipfile

# `import native_stack_baseline` needs ci/ on sys.path. CPython prepends a script's own
# directory for a plain `python ci/check-bundled-stack.py`, which is how CI invokes this, but NOT under
# `-I`/`-P`, and not when the module is loaded by path -- which is exactly what the guard
# tests do. So add it explicitly rather than depend on the invocation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import native_stack_baseline  # noqa: E402 -- needs the sys.path line above

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

# The shared module refuses a baseline naming a library with no extractor, so the two lists must
# agree; this keeps them honest at import rather than at the first wheel.
assert set(_EXTRACTORS) == set(native_stack_baseline.SUPPORTED_LIBRARIES), (
    "_EXTRACTORS and native_stack_baseline.SUPPORTED_LIBRARIES disagree"
)


def load_baseline(path: pathlib.Path = native_stack_baseline.BASELINE_PATH) -> dict:
    """Read the declared baseline, refusing one that would compare nothing.

    Args:
        path: The baseline file. Defaults to the one beside this script.

    Returns:
        The parsed document. `libraries` holds the properties every wheel must report, the optional
        `exceptions` table the per-platform relaxations, and `installed` the GDAL version
        `ci/verify-wheel.py` asserts after installing.

    Raises:
        FileNotFoundError: If `path` does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        ValueError: If the baseline is unusable -- an unsupported library, an unknown property, an
            exception with no reason, a malformed platform tag. See `native_stack_baseline.validate`.
    """
    return native_stack_baseline.load(path)


def platform_tag(wheel: pathlib.Path) -> str:
    """The platform tag of a wheel, used to resolve declared exceptions.

    Args:
        wheel: The wheel path; only its name is read.

    Returns:
        The tag, or an empty string when the filename is not PEP 427 shaped.
    """
    return native_stack_baseline.wheel_name_platform_tag(wheel.name)


def expected_for(baseline: dict, tag: str) -> dict:
    """The properties each library must report on one platform, exceptions applied.

    Args:
        baseline: The parsed baseline.
        tag: The wheel's platform tag.

    Returns:
        A `{library: {property: value}}` mapping. An exception may relax `version` only, never
        `threadsafety`.
    """
    return native_stack_baseline.expected_for(baseline, tag)


def _members_for(archive: zipfile.ZipFile, token: str) -> list[str]:
    """The archive members that look like the named native library.

    A member qualifies when its lower-cased basename contains `token` and carries one of the native
    suffixes, either at the end (`geos_c-<hash>.dll`) or as a versioned soname (`libproj.so.25.9.8.1`,
    `libhdf5.320.2.0.dylib`).

    Args:
        archive: The opened wheel.
        token: The substring identifying the library, lower-cased to match the basename, e.g. `hdf5`.

    Returns:
        Matching member names. Companions and satellites match too -- `libhdf5_hl`, the `libgeos`
        core beneath `libgeos_c`, GDAL's own `gdal_HDF5` driver plugin, shapely's `_geos` extension
        module -- but none of them embeds a settings string, so they contribute nothing and are
        harmless. Measured across the published wheels of all five platforms, exactly one matched
        member per library carries a reading.
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
    """Read what every bundled copy of each library reports about itself.

    Args:
        wheel: The wheel to inspect.
        libraries: The library tokens to look for.

    Returns:
        A `{library: [{property: value}, ...]}` mapping with **one entry per bundled copy** that
        carries a settings string, each also holding a `_member` key naming the file it was read
        from. A library maps to an empty list when the wheel bundles no copy of it, or only copies
        whose bytes carry no settings string -- companions such as `libhdf5_hl` and GDAL's
        `gdal_hdf5` plugin fall out here -- which the caller reports as a failure in its own right.

        Every copy is kept deliberately. Collapsing them to one reading made the verdict depend on
        archive order: a non-threadsafe `aa_libhdf5.so` beside a conforming `zz_libhdf5.so` passed,
        and the same pair renamed failed. The property this whole check exists to defend could
        therefore ship broken in a wheel that reported `ok` (#1627).
    """
    reported: dict = {name: [] for name in libraries}
    with zipfile.ZipFile(wheel) as archive:
        for name in libraries:
            for member in _members_for(archive, name):
                blob = archive.read(member)
                values = {}
                for prop, pattern in _EXTRACTORS[name].items():
                    match = pattern.search(blob)
                    if match:
                        values[prop] = match.group(1).decode("ascii", "replace")
                if values:
                    values["_member"] = member.rsplit("/", 1)[-1]
                    reported[name].append(values)
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
        readings = reported.get(name) or []
        if not readings:
            failures.append(f"{wheel.name}: no bundled {name} found")
            print(f"FAIL {wheel.name}: no bundled {name} found")
            continue
        for got in readings:
            member = got.get("_member", name)
            _verdict(wheel, tag, member, wanted, got, failures)
    return failures


def _verdict(
    wheel: pathlib.Path,
    tag: str,
    member: str,
    wanted: dict,
    got: dict,
    failures: list[str],
) -> None:
    """Compare one bundled copy against the baseline, printing and recording each property.

    Args:
        wheel: The wheel the copy came from, for the message prefix.
        tag: The wheel's platform tag.
        member: The bundled file being verdicted.
        wanted: The baseline properties this library must satisfy.
        got: What this copy reported.
        failures: Accumulator appended to for every mismatch.
    """
    for prop, want in sorted(wanted.items()):
        if prop == "reason":
            continue
        have = got.get(prop)
        line = f"{member} {prop}={have} (want {want})"
        ok = have == want
        if ok:
            print(f"ok   {wheel.name} [{tag}]: {line}")
        else:
            failures.append(f"{wheel.name} [{tag}]: {line}")
            print(f"FAIL {wheel.name} [{tag}]: {line}")


def _wheels(targets: list[str]) -> list[pathlib.Path]:
    """Expand each argument into the wheels it names.

    Args:
        targets: Wheel paths, or directories to scan for `*.whl`. A directory is scanned one level
            deep and may legitimately yield nothing, which `main` treats as a failure.

    Returns:
        The wheels to check, sorted and deduplicated.

    Raises:
        SystemExit: If a target is neither an existing directory nor an existing `.whl` file.
    """
    found: set[pathlib.Path] = set()
    for target in targets:
        path = pathlib.Path(target)
        if path.is_dir():
            found.update(path.glob("*.whl"))
        elif path.suffix == ".whl" and path.is_file():
            found.add(path)
        elif path.suffix == ".whl":
            # Without this arm a typo'd wheel path reached zipfile.ZipFile and surfaced as a
            # FileNotFoundError traceback rather than the script's own refusal.
            raise SystemExit(f"check-bundled-stack: {target!r} does not exist")
        else:
            raise SystemExit(
                f"check-bundled-stack: {target!r} is neither a wheel nor a directory"
            )
    return sorted(found)


def main(argv: list[str]) -> int:
    """Check every wheel named on the command line against the baseline.

    Args:
        argv: Wheel paths, or directories to scan one level deep for `*.whl`. The repair step passes
            cibuildwheel's `{dest_dir}`.

    Returns:
        0 when at least one wheel was found and all of them conform; 1 when any wheel fails a
        comparison, or when the targets yielded no wheels at all -- refusing to report success having
        compared nothing, which is the failure mode this check exists to prevent.

    Raises:
        SystemExit: If `argv` is empty, or a target is neither an existing directory nor an existing
            `.whl` file.
    """
    if not argv:
        raise SystemExit(__doc__)
    baseline = load_baseline()
    wheels = _wheels(argv)
    failures = []
    for wheel in wheels:
        failures.extend(check_wheel(wheel, baseline))
    if not wheels:
        print(
            f"check-bundled-stack: no wheels found in {', '.join(argv)} -- refusing to report "
            "success having compared nothing.",
            file=sys.stderr,
        )
        status = 1
    elif failures:
        print(
            f"\ncheck-bundled-stack: {len(failures)} bundled-library mismatch(es) against "
            f"{native_stack_baseline.BASELINE_PATH.name}.\nEither fix the build route that drifted, "
            "or -- if the platform genuinely cannot meet the baseline -- add a declared exception "
            "with its reason to that file (see issue #1627).",
            file=sys.stderr,
        )
        status = 1
    else:
        print(
            f"check-bundled-stack: {len(wheels)} wheel(s) match the declared native stack."
        )
        status = 0
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
