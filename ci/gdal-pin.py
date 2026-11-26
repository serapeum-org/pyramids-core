"""Print the conda pins the wheels are built against (single source of truth).

The native build/test stack is pinned once in two ``pyproject.toml`` features:
``[tool.pixi.feature.gdal.dependencies]`` (the shared gdal / libgdal-* libs every
environment lists) and ``[tool.pixi.feature.wheel-build.dependencies]`` (build-only
``swig``). This script reads those features and prints each requested package's version
spec verbatim, one per line, so every consumer installs the exact pins the published
wheels vendor instead of re-encoding them:

- ``.github/workflows/pure-wheel-test.yml`` — the conda-forge-shape wheel test
- ``ci/setup-gdal-micromamba.sh`` — the macOS x86_64 cross-compile env

``pyproject.toml`` is located relative to this file, so the script works from any cwd.

Usage::

    spec=$(python ci/gdal-pin.py)                    # default: gdal ->  >=3.13.3,<3.13.4
    python ci/gdal-pin.py gdal libgdal-netcdf swig   # one spec per line

The `--locked` mode reads `pixi.lock` instead, printing the exact `==<version>` a named
environment and platform resolve to. The cross-compile branch needs it because a range
spec does not pin a *solve*: re-resolving `gdal >=3.13.3,<3.13.4` for osx-64 picks the
newest matching build, which linked proj 9.9.0 and geos 3.15.0, while every platform
installed from `pixi.lock` got the older build with proj 9.8.1 and geos 3.14.1 -- one
release shipping two different native stacks (#1632)::

    python ci/gdal-pin.py --locked --platform osx-64 proj geos   # ==9.8.1 / ==3.14.1
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
LOCK = Path(__file__).resolve().parent.parent / "pixi.lock"

_DEFAULT_ENVIRONMENT = "wheel-build"


def feature_pins(pyproject: Path = PYPROJECT) -> dict[str, str]:
    """Return the conda pins from the gdal and wheel-build features, merged by name."""
    with pyproject.open("rb") as fh:
        feature = tomllib.load(fh)["tool"]["pixi"]["feature"]
    return {**feature["gdal"]["dependencies"], **feature["wheel-build"]["dependencies"]}


def gdal_spec(pyproject: Path = PYPROJECT) -> str:
    """Return the shared gdal version spec from ``pyproject``."""
    return feature_pins(pyproject)["gdal"]


def locked_versions(
    platform: str,
    environment: str = _DEFAULT_ENVIRONMENT,
    lock: Path = LOCK,
) -> dict[str, str]:
    """The conda package versions `pixi.lock` resolves for one environment and platform.

    The lock is read with a regular expression rather than a YAML parser because this runs on a
    bare CI runner before any environment exists, so only the standard library is available. That
    is safe here: each entry is a conda URL whose filename encodes the platform, package name and
    version (`.../osx-64/proj-9.8.1-h856bcef_1.conda`), so the values come from the URL itself
    rather than from interpreting the surrounding YAML.

    Args:
        platform: The conda subdir to read, e.g. `osx-64`.
        environment: The locked pixi environment, defaulting to the one the wheels are built in.
        lock: Path to `pixi.lock`.

    Returns:
        A `{package: version}` mapping, holding the environment's `.conda` entries served from that
        platform's own subdir. Requiring the subdir in the URL is what makes the version readable
        from the filename, and it excludes the two kinds of entry that carry no platform of their
        own: the environment's `noarch` conda packages and its PyPI wheels. Nothing the baseline
        asserts is `noarch`, so the exclusion costs this caller nothing -- but a `noarch` name asked
        for here resolves to nothing and `locked_pins` refuses it rather than guessing.

    Raises:
        FileNotFoundError: If `lock` does not exist.
        KeyError: If the environment or platform has no entry in the lock, rather than returning
            an empty mapping -- a pin silently resolving to nothing is the failure this is meant
            to prevent.
    """
    text = lock.read_text(encoding="utf-8", errors="replace")
    # Splitting on the `packages:` header consumes the newline that terminated the previous
    # line, while the entry pattern below needs one per line -- so without re-adding it the
    # final entry of the last platform block of the last environment is silently dropped.
    environments = text.split("\npackages:", 1)[0] + "\n"
    block = re.search(
        rf"\n  {re.escape(environment)}:\n(.*?)(?=\n  [A-Za-z][\w-]*:\n|\Z)",
        environments,
        re.S,
    )
    if not block:
        raise KeyError(f"{lock.name} declares no environment {environment!r}")
    entries = re.search(
        rf"\n      {re.escape(platform)}:\n((?:      - .*\n)+)", block.group(1)
    )
    if not entries:
        raise KeyError(
            f"{lock.name} environment {environment!r} has no platform {platform!r}"
        )
    versions = {}
    for filename in re.findall(
        rf"/{re.escape(platform)}/([^/\s]+?)\.conda", entries.group(1)
    ):
        name, _, remainder = filename.rpartition("-")
        package, _, version = name.rpartition("-")
        if package and re.fullmatch(r"[0-9][^-]*", version) and remainder:
            versions[package] = version
    return versions


def locked_pins(
    names: list[str],
    platform: str,
    environment: str = _DEFAULT_ENVIRONMENT,
    lock: Path = LOCK,
) -> dict[str, str]:
    """The exact `==<version>` specs the named packages are locked to.

    Args:
        names: The conda package names to look up.
        platform: The conda subdir, e.g. `osx-64`.
        environment: The locked pixi environment.
        lock: Path to `pixi.lock`.

    Returns:
        A `{package: "==<version>"}` mapping, in the order requested.

    Raises:
        KeyError: If any requested package is absent from that environment and platform. Every
            name must resolve, so a typo cannot quietly drop a pin and let the solver choose.
    """
    available = locked_versions(platform, environment, lock)
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(
            f"{lock.name} environment {environment!r} / {platform!r} pins no "
            f"{', '.join(missing)} (found {len(available)} packages)"
        )
    return {name: f"=={available[name]}" for name in names}


def _locked_main(argv: list[str]) -> int:
    """Print one locked `==<version>` per requested package.

    Args:
        argv: The argument list after `--locked`, carrying `--platform` and optionally
            `--environment`, followed by the package names.

    Returns:
        0 on success.

    Raises:
        SystemExit: If `--platform` is missing or no package names are given.
        KeyError: If the lock has no such environment or platform, or does not pin every requested
            package -- propagated from `locked_pins` so the traceback names what failed to resolve.
    """
    platform = ""
    environment = _DEFAULT_ENVIRONMENT
    names = []
    remaining = list(argv)
    while remaining:
        token = remaining.pop(0)
        if token == "--platform":
            platform = remaining.pop(0) if remaining else ""
        elif token == "--environment":
            environment = remaining.pop(0) if remaining else environment
        else:
            names.append(token)
    if not platform or not names:
        raise SystemExit(
            "usage: gdal-pin.py --locked --platform <subdir> [--environment <env>] <package>..."
        )
    print("\n".join(locked_pins(names, platform, environment).values()))
    return 0


def main(argv: list[str]) -> int:
    """Print one version spec per requested package, newline-separated.

    Args:
        argv: The conda package names to print, defaulting to `gdal` alone. A `--locked` token
            anywhere in the list switches to the lock reader: it is stripped and everything else is
            handed to `_locked_main`, which prints the exact `==<version>` instead of the
            `pyproject.toml` range and takes `--platform` and optional `--environment`.

    Returns:
        0 on success.

    Raises:
        KeyError: If a requested name is in neither feature table, or -- under `--locked` -- is not
            pinned by the lock for that environment and platform.
        SystemExit: Under `--locked`, if `--platform` or the package names are missing.
    """
    if "--locked" in argv:
        rest = [token for token in argv if token != "--locked"]
        status = _locked_main(rest)
    else:
        pins = feature_pins()
        print("\n".join(pins[name] for name in (argv or ["gdal"])))
        status = 0
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
