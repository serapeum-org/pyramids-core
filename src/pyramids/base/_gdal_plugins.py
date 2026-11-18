"""Composition of GDAL's ``GDAL_DRIVER_PATH`` from every directory that may hold driver plugins.

GDAL loads compiled driver plugins (``gdal_netCDF.so``, ``ogr_Parquet.dll``, …) from the
directories listed in ``GDAL_DRIVER_PATH``. pyramids can learn about three independent kinds of
such directory, and before this module each was written by a different piece of code that simply
overwrote whatever the previous one had set:

1. **The bundled directory** — ``pyramids/_data/gdalplugins/``, shipped inside a platform wheel and
   ABI-locked to that wheel's vendored libgdal. Set early by :mod:`pyramids.base._bootstrap`, because
   the SWIG extension import needs the environment sane before ``from osgeo import …`` runs.
   Absent on the from-source Linux wheels, where the same drivers are compiled *into* libgdal.
2. **A discovered system directory** — a conda ``Library/lib/gdalplugins`` or ``/usr/lib/gdalplugins``,
   found by :meth:`pyramids.base.config.Config.dynamic_env_variables`. This matches the GDAL of a
   conda / system install.
3. **Companion directories** — contributed by a separate distribution that ships an extra GDAL plugin
   (for example an OGR Parquet driver), registered through the :data:`ENTRY_POINT_GROUP` entry point.

:func:`compose_driver_path` turns those into one ordered, de-duplicated path list, and
:class:`~pyramids.base.config.Config` is the single place that writes it. The authoritative directory
comes first — the bundled one when a wheel supplied it, otherwise the discovered one — and companion
directories are always appended, never prepended, so a companion can *add* a driver but can never
shadow the version-locked drivers the active libgdal was built with (issue #465).

Discovery deliberately performs **no import of the companion package**. It runs at ``import pyramids``
time, before GDAL registers drivers, so importing a companion here would drag that distribution's
whole dependency tree into every ``import pyramids``. Entry-point *values* are therefore module names,
resolved with :func:`importlib.util.find_spec`, not attributes to be loaded.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from importlib.metadata import entry_points
from importlib.util import find_spec
from pathlib import Path

__all__ = [
    "DEFAULT_PLUGIN_SUBDIR",
    "ENTRY_POINT_GROUP",
    "clear_companion_plugin_dir_cache",
    "companion_plugin_dirs",
    "compose_driver_path",
]

#: The entry-point group a distribution registers its GDAL plugin directory under. Part of the
#: published plugin contract; mirrors the ``pyramids.compute_backend`` / ``pyramids.plot_backends``
#: groups. The value is a module name, optionally followed by ``:<relative-subdir>``:
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."pyramids.gdal_plugin_path"]
#:     parquet = "pyramids_parquet"              # -> <pyramids_parquet>/gdalplugins/
#:     other   = "some_pkg:native/plugins"       # -> <some_pkg>/native/plugins/
ENTRY_POINT_GROUP = "pyramids.gdal_plugin_path"

#: The subdirectory used when an entry-point value names only a module.
DEFAULT_PLUGIN_SUBDIR = "gdalplugins"

_COMPANION_CACHE: tuple[Path, ...] | None = None


def _module_dir(module_name: str) -> Path | None:
    """Locate a module's directory without importing it.

    Uses :func:`importlib.util.find_spec`, so a companion distribution's package is never
    executed — only its location on disk is read. A probe that raises (a missing parent
    package, a namespace package with no location) counts as not found.

    Args:
        module_name: The importable module name, e.g. ``"pyramids_parquet"``.

    Returns:
        pathlib.Path | None: The package's directory, or ``None`` when it cannot be located.

    Examples:
        - A real package resolves to the directory holding its ``__init__``:
            ```python
            >>> from pyramids.base._gdal_plugins import _module_dir
            >>> _module_dir("json").name
            'json'

            ```
        - A missing module resolves to ``None``:
            ```python
            >>> from pyramids.base._gdal_plugins import _module_dir
            >>> _module_dir("no_such_companion_xyz") is None
            True

            ```
    """
    found: Path | None = None
    try:
        spec = find_spec(module_name)
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        locations = list(spec.submodule_search_locations or ())
        if locations:
            found = Path(locations[0])
        elif spec.origin:
            found = Path(spec.origin).parent
    return found


def _entry_point_dir(value: str) -> Path | None:
    """Resolve one entry-point value to an existing plugin directory.

    Args:
        value: Either ``"<module>"`` or ``"<module>:<relative-subdir>"``. The subdir is a plain
            relative path, **not** a Python attribute — nothing is imported. An absolute or
            parent-escaping subdir is rejected, so a registration cannot point GDAL at an
            arbitrary location on the machine.

    Returns:
        pathlib.Path | None: The directory, or ``None`` when it is unresolvable or missing.

    Examples:
        - An unknown module yields nothing:
            ```python
            >>> from pyramids.base._gdal_plugins import _entry_point_dir
            >>> _entry_point_dir("no_such_companion_xyz") is None
            True

            ```
        - A subdir that climbs out of its own package is refused, so a registration can never
          aim GDAL at an arbitrary directory:
            ```python
            >>> from pyramids.base._gdal_plugins import _entry_point_dir
            >>> _entry_point_dir("json:../../etc") is None
            True

            ```
        - A real package that simply ships no plugin directory is skipped rather than failing:
            ```python
            >>> from pyramids.base._gdal_plugins import _entry_point_dir
            >>> _entry_point_dir("json") is None
            True

            ```
    """
    module_name, _, subdir = value.partition(":")
    module_name = module_name.strip()
    subdir = subdir.strip() or DEFAULT_PLUGIN_SUBDIR
    candidate: Path | None = None
    base = _module_dir(module_name) if module_name else None
    if (
        base is not None
        and not Path(subdir).is_absolute()
        and ".." not in Path(subdir).parts
    ):
        plugin_dir = base / subdir
        if plugin_dir.is_dir():
            candidate = plugin_dir
    return candidate


def companion_plugin_dirs(*, refresh: bool = False) -> tuple[Path, ...]:
    """Return the existing plugin directories registered under :data:`ENTRY_POINT_GROUP`.

    A pure metadata scan: the entry points are read, their values are resolved to directories
    with :func:`importlib.util.find_spec`, and the companion packages themselves are never
    imported. The result is cached because it is consulted at import time and the installed
    distribution set does not change within a process.

    Args:
        refresh: Re-scan instead of returning the cached result.

    Returns:
        tuple[pathlib.Path, ...]: Existing directories, in entry-point name order for stability.

    Examples:
        - Every directory handed back actually exists -- GDAL is never pointed at a missing path:
            ```python
            >>> from pyramids.base._gdal_plugins import companion_plugin_dirs
            >>> dirs = companion_plugin_dirs(refresh=True)
            >>> all(path.is_dir() for path in dirs)
            True

            ```
        - Repeated calls are served from the cache, so the answer is stable within a process:
            ```python
            >>> from pyramids.base._gdal_plugins import companion_plugin_dirs
            >>> companion_plugin_dirs() == companion_plugin_dirs()
            True

            ```
        - With no companion distribution installed there is simply nothing to add:
            ```python
            >>> from pyramids.base._gdal_plugins import companion_plugin_dirs
            >>> [path.name for path in companion_plugin_dirs(refresh=True)]
            []

            ```
    """
    global _COMPANION_CACHE
    if refresh or _COMPANION_CACHE is None:
        resolved: list[Path] = []
        try:
            points = sorted(
                entry_points(group=ENTRY_POINT_GROUP), key=lambda point: point.name
            )
        except (
            Exception
        ):  # pragma: no cover - a broken dist-info must not break `import pyramids`
            points = []
        for point in points:
            plugin_dir = _entry_point_dir(point.value)
            if plugin_dir is not None and plugin_dir not in resolved:
                resolved.append(plugin_dir)
        _COMPANION_CACHE = tuple(resolved)
    return _COMPANION_CACHE


def clear_companion_plugin_dir_cache() -> None:
    """Drop the cached companion scan, so the next call re-reads the entry points.

    Examples:
        - Clearing is always safe, cached or not:
            ```python
            >>> from pyramids.base._gdal_plugins import clear_companion_plugin_dir_cache
            >>> clear_companion_plugin_dir_cache()
            >>> clear_companion_plugin_dir_cache()

            ```
        - After clearing, the next scan re-reads the entry points and still agrees with itself:
            ```python
            >>> from pyramids.base._gdal_plugins import (
            ...     clear_companion_plugin_dir_cache,
            ...     companion_plugin_dirs,
            ... )
            >>> before = companion_plugin_dirs()
            >>> clear_companion_plugin_dir_cache()
            >>> companion_plugin_dirs() == before
            True

            ```
    """
    global _COMPANION_CACHE
    _COMPANION_CACHE = None


def _existing_dirs(values: Iterable[Path | str | None]) -> list[str]:
    """Return the string forms of the entries that name an existing directory, order preserved."""
    kept: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and Path(text).is_dir():
            kept.append(text)
    return kept


def compose_driver_path(
    *,
    bundled: Path | None,
    companions: Sequence[Path] = (),
    discovered: Path | None = None,
    inherited: str | None = None,
    keep_env: bool = False,
) -> str | None:
    """Compose the authoritative ``GDAL_DRIVER_PATH`` value.

    Precedence, highest first:

    1. ``inherited`` — only when ``keep_env`` is set, which is the documented
       ``PYRAMIDS_KEEP_GDAL_ENV=1`` escape hatch for power users who want their own value to win.
    2. ``bundled`` — the wheel's own plugin directory, ABI-locked to its vendored libgdal.
    3. ``discovered`` — a conda / system plugin directory matching the active GDAL.
    4. ``companions`` — always last, so an extra driver can be added but never shadows the
       version-locked drivers the active libgdal was built with (#465).

    Non-existent directories and duplicates are dropped, so the result never points GDAL at a path
    that is not there.

    Args:
        bundled: The wheel's `_data/gdalplugins` directory, or ``None`` on a source / conda install
            and on the from-source wheels, which compile their drivers into libgdal.
        companions: Directories contributed through :data:`ENTRY_POINT_GROUP`.
        discovered: A conda / system plugin directory, or ``None`` when none was found.
        inherited: The value of ``GDAL_DRIVER_PATH`` before pyramids' own discovery ran.
        keep_env: Whether ``PYRAMIDS_KEEP_GDAL_ENV=1`` is in effect.

    Returns:
        str | None: An ``os.pathsep``-joined path list, or ``None`` when no directory exists and
            ``GDAL_DRIVER_PATH`` should therefore be left alone.

    Examples:
        - Nothing to point at yields ``None`` rather than an empty string:
            ```python
            >>> from pyramids.base._gdal_plugins import compose_driver_path
            >>> compose_driver_path(bundled=None) is None
            True

            ```
        - A real directory comes back as itself, and a missing one is dropped:
            ```python
            >>> import pathlib
            >>> from pyramids.base._gdal_plugins import compose_driver_path
            >>> here = pathlib.Path(".").resolve()
            >>> compose_driver_path(bundled=here, discovered=here / "nope") == str(here)
            True

            ```
    """
    if keep_env and inherited:
        ordered: list[Path | str | None] = [*inherited.split(os.pathsep), *companions]
    else:
        authoritative = _existing_dirs([bundled, discovered])
        if not authoritative and inherited:
            # Neither a wheel nor discovery supplied a directory, so the value the process
            # inherited is the only thing that knows where the active GDAL's plugins are.
            # Dropping it here would hand GDAL a companions-only path and silently unload
            # the drivers it was working with: on a conda Linux environment the netCDF and
            # HDF5 drivers are separate .so plugins reachable ONLY through the activation
            # script's GDAL_DRIVER_PATH, and `set_env_conda` cannot find them because it
            # probes the Windows `Library/lib/gdalplugins` layout. It is still never allowed
            # to outrank a bundled or discovered directory -- that is the #465 invariant, and
            # ci/verify-wheel.py asserts a foreign path cannot reach the final value.
            authoritative = [*inherited.split(os.pathsep)]
        ordered = [*authoritative, *companions]
    parts: list[str] = []
    for text in _existing_dirs(ordered):
        if text not in parts:
            parts.append(text)
    return os.pathsep.join(parts) if parts else None
