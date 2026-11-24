#!/usr/bin/env python
"""Assert an installed wheel's GDAL version and driver set against the declared baseline.

The companion to ``ci/check-bundled-stack.py``, which reads a wheel's bundled binaries without
installing them. Two properties cannot be read that way and are checked here instead, from the
installed package:

* **GDAL's version.** Byte-reading it is unreliable -- a manylinux ``libgdal`` yields only a
  spurious ``GDAL 3.6.0``, macOS yields two candidates, and the Windows DLLs carry no version in
  their filename, because the real version lives in the SONAME on Linux/macOS and nowhere in the
  name on Windows. Asking ``osgeo.gdal`` is exact on every platform.
* **The driver set.** ``ci/source-build/build-gdal-stack.sh`` already refuses a source build that
  is missing a required driver, but that check runs *inside* the source build, so it says nothing
  about the vcpkg or conda-forge wheels. Running it against the installed wheel extends the same
  guarantee to every route (issue #1627).

Run it where a wheel is already installed and importable -- the wheel smoke jobs -- not in the
repair step, which has no installed package.

Usage:
    python ci/check-installed-stack.py
"""

from __future__ import annotations

import pathlib
import sys
import sysconfig
import tomllib

# A published wheel ships no top-level `osgeo`: it lives at `pyramids/_vendor/osgeo`, which
# `pyramids/__init__.py` injects onto `sys.path`. So `import pyramids` MUST precede the osgeo
# import, or this checker raises ModuleNotFoundError in the very wheel jobs it runs in -- while
# passing in a conda dev environment, where `osgeo` is a real top-level package. `isort: off`
# keeps isort's first-party regrouping from floating `import pyramids` below the osgeo block,
# the same constraint ci/verify-wheel.py pins for the same reason.
# isort: off
import pyramids  # noqa: F401 -- side-effecting vendor bootstrap; must precede osgeo

from osgeo import gdal, ogr

# isort: on

_BASELINE_PATH = pathlib.Path(__file__).resolve().parent / "native-stack-baseline.toml"


def _platform_tag() -> str:
    """The platform tag that matches this interpreter, for resolving declared exceptions.

    Derived from `sysconfig.get_platform()` with `-` and `.` replaced by `_`, rather than
    reimplementing PEP 425: only the tags the baseline's exceptions actually key on have to come out
    exact, and today those are the Windows ones.

    Returns:
        `win_arm64` or `win_amd64` on Windows, where the two arches are told apart by the
        interpreter's own platform string and nothing else is collapsed into them. Elsewhere, the
        normalised `sysconfig` string as-is: `linux_x86_64` on every Linux (so neither a manylinux
        nor a musllinux tag -- harmless, as the baseline declares no Linux exception, which leaves
        those wheels compared against the unrelaxed baseline), and the interpreter's deployment
        target on macOS, e.g. `macosx_11_0_arm64`.
    """
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if platform.startswith("win"):
        return "win_arm64" if "arm64" in platform else "win_amd64"
    return platform


def _expected_gdal(baseline: dict, tag: str) -> str | None:
    """The GDAL version this platform must report.

    Args:
        baseline: The parsed baseline document.
        tag: This platform's wheel tag, as `_platform_tag` reports it.

    Returns:
        The version from `exceptions.<tag>.gdal` when this platform declares one, otherwise the
        `installed.gdal` pin, or None when neither exists -- which leaves GDAL's version unchecked
        rather than failing, so the driver comparison still runs.
    """
    pinned = baseline.get("installed", {}).get("gdal", {})
    override = baseline.get("exceptions", {}).get(tag, {}).get("gdal", {})
    return override.get("version", pinned.get("version"))


def main() -> int:
    """Check the installed GDAL's version and driver set against the baseline.

    Compares three things, printing an `ok` or `FAIL` line for each on stdout: the version
    `osgeo.gdal` reports, truncated at its first hyphen so a `3.14.0-dev` style suffix does not read
    as drift, against this platform's expected version; and the `installed.drivers.raster` and
    `installed.drivers.vector` lists against the drivers GDAL and OGR have actually registered. A
    baseline that pins no GDAL version reports the version it found and compares nothing, and a
    baseline that lists no drivers for a kind passes it trivially. The summary and the instruction to
    either fix the route or declare an exception go to stderr.

    Returns:
        0 when the installed stack conforms; 1 when the version differs or any required driver is
        missing, with one failure counted per mismatched version and per driver kind.
    """
    baseline = tomllib.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    tag = _platform_tag()
    failures: list[str] = []

    reported = gdal.__version__.split("-")[0]
    expected = _expected_gdal(baseline, tag)
    if expected is None:
        print(f"ok   [{tag}] gdal {reported} (baseline pins no version)")
    elif reported == expected:
        print(f"ok   [{tag}] gdal version={reported} (want {expected})")
    else:
        failures.append(f"gdal version={reported} (want {expected})")
        print(f"FAIL [{tag}] gdal version={reported} (want {expected})")

    installed_raster = {
        gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())
    }
    installed_vector = {ogr.GetDriver(i).GetName() for i in range(ogr.GetDriverCount())}
    required = baseline.get("installed", {}).get("drivers", {})
    for kind, installed in (("raster", installed_raster), ("vector", installed_vector)):
        missing = [name for name in required.get(kind, []) if name not in installed]
        if missing:
            failures.append(f"missing {kind} drivers: {', '.join(missing)}")
            print(f"FAIL [{tag}] missing {kind} drivers: {', '.join(missing)}")
        else:
            print(
                f"ok   [{tag}] all {len(required.get(kind, []))} required {kind} drivers present"
            )

    if failures:
        print(
            f"\ncheck-installed-stack: {len(failures)} mismatch(es) against {_BASELINE_PATH.name}.\n"
            "Either fix the build route that drifted, or -- if the platform genuinely cannot meet "
            "the baseline -- add a declared exception with its reason to that file (see #1627).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
