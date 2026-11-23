#!/usr/bin/env python
"""Fail the wheel build when a bundled HDF5 is not thread-safe.

HDF5's CMake default is ``HDF5_ENABLE_THREADSAFE=OFF``, and a non-threadsafe HDF5 must not be
entered from more than one OS thread over the process lifetime -- even when the calls are
serialized -- so dask's threaded scheduler eventually segfaults inside GDAL ``OpenMDArray``
(serapeum-org/pyramids-core#1620). The flag is invisible in the wheel's metadata, so nothing but
reading the shipped binary catches a regression to a non-threadsafe build.

Each HDF5 library embeds its own build-settings summary, including a ``Threadsafety:`` line, so
the check is a byte search: no install, no import of the wheel, about a second per wheel. Run it
on the *repaired* wheel -- before ``auditwheel``/``delocate``/``delvewheel`` runs there are no
bundled libraries to inspect.

Usage:
    python ci/check-bundled-hdf5.py <wheel-or-directory> [...]
"""

from __future__ import annotations

import pathlib
import re
import sys
import zipfile

#: HDF5 writes its configure summary into the binary, e.g. "    Threadsafety: ON".
_SETTING = re.compile(rb"Threadsafety:\s*(\w+)")
#: Reported alongside it, to make the log say which HDF5 was measured.
_VERSION = re.compile(rb"HDF5 Version:\s*([0-9][0-9.]*)")


def _wheels(targets: list[str]) -> list[pathlib.Path]:
    """Expand each argument into the wheels it names.

    Args:
        targets: Paths to wheels, or directories to scan for ``*.whl``.

    Returns:
        The wheels to check, sorted, without duplicates.
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
                f"check-bundled-hdf5: {target!r} is neither a wheel nor a directory"
            )
    return sorted(found)


def _verdicts(wheel: pathlib.Path) -> list[tuple[str, str, str]]:
    """Read the thread-safety setting of every HDF5 library bundled in one wheel.

    Args:
        wheel: The wheel to inspect.

    Returns:
        One ``(member, threadsafety, version)`` triple per bundled HDF5 that carries a build
        summary. Companion libraries such as ``libhdf5_hl`` embed no summary and are skipped.
    """
    results = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if "hdf5" not in member.rsplit("/", 1)[-1].lower():
                continue
            blob = archive.read(member)
            setting = _SETTING.search(blob)
            if setting is None:
                continue
            version = _VERSION.search(blob)
            results.append(
                (
                    member,
                    setting.group(1).decode("ascii", "replace").upper(),
                    version.group(1).decode("ascii", "replace")
                    if version
                    else "unknown",
                )
            )
    return results


def main(argv: list[str]) -> int:
    """Check every wheel named on the command line.

    Args:
        argv: Wheel paths and/or directories containing wheels.

    Returns:
        0 when every bundled HDF5 reports ``ON``; 1 otherwise.
    """
    if not argv:
        raise SystemExit(__doc__)
    failures = []
    for wheel in _wheels(argv):
        verdicts = _verdicts(wheel)
        if not verdicts:
            # A binary wheel without HDF5 cannot read NetCDF at all, so its absence is as much a
            # regression as a non-threadsafe build. A pure wheel carries no native libraries and
            # is never repaired, so this script is not run on one.
            failures.append(f"{wheel.name}: no bundled HDF5 found")
            print(f"FAIL {wheel.name}: no bundled HDF5 found")
            continue
        for member, threadsafety, version in verdicts:
            library = member.rsplit("/", 1)[-1]
            line = f"{library} -> Threadsafety: {threadsafety} (HDF5 {version})"
            if threadsafety == "ON":
                print(f"ok   {wheel.name}: {line}")
            else:
                failures.append(f"{wheel.name}: {line}")
                print(f"FAIL {wheel.name}: {line}")
    if failures:
        print(
            "\ncheck-bundled-hdf5: a bundled HDF5 is not thread-safe.\n"
            "Build it with -DHDF5_ENABLE_THREADSAFE:BOOL=ON -DHDF5_ALLOW_UNSUPPORTED:BOOL=ON\n"
            "(see ci/source-build/config.sh::src_hdf5 and issue #1620).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
