"""Runtime vendor bootstrap for the pyramids platform wheel.

When `pyramids` is installed from a platform wheel,
`pyramids/_vendor/osgeo/` ships GDAL's Python SWIG bindings and
`pyramids/_data/` ships GDAL_DATA + PROJ_DATA.
`activate_vendored_osgeo(pkg_dir)` injects `_vendor/` into `sys.path`
and configures the env vars + Windows DLL directories that the bundled
GDAL needs. It is called once from `pyramids/__init__.py` BEFORE any
`from osgeo import ...` anywhere downstream — `pyramids.base._configure`
does that import at module load and would fail without the bootstrap.

The bootstrap is a no-op on editable / dev installs that don't have
`_vendor/` populated (the normal `pixi shell -e dev` setup) and skips
with a warning if the vendored `_gdal.<EXT>` doesn't match the
current Python ABI (defensive against stale leftover dirs from a
prior local cibuildwheel run).

Lives under `pyramids/base/` (not at the package root) so internal
infrastructure modules are grouped together. `pkg_dir` is passed
explicitly by the caller (resolved via `pyramids.__path__`) rather
than derived from `__file__`, so this module can move anywhere within
the package without the vendor-dir resolution breaking.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import warnings
from pathlib import Path

# Captured at import time, BEFORE pyramids puts its vendored osgeo on sys.path and before
# any ``from osgeo import ...`` runs downstream: ``pyramids/__init__.py`` imports this module
# above those imports. If ``osgeo`` is already in ``sys.modules`` here, a host application
# (QGIS being the motivating case) loaded its own GDAL first, and pyramids must not clobber
# the host's process-global GDAL environment. See :func:`gdal_env_is_host_managed` and #1588.
_HOST_GDAL_PRELOADED = "osgeo" in sys.modules


def host_gdal_preloaded() -> bool:
    """Whether a host application's GDAL (``osgeo``) was imported before pyramids.

    Captured once at this module's import, which ``pyramids/__init__.py`` runs before its own
    ``from osgeo import ...``. ``True`` means pyramids is embedded in a process that already
    owns GDAL -- QGIS is the motivating case (#1588).

    Returns:
        bool: ``True`` when ``osgeo`` was already imported before pyramids loaded.
    """
    return _HOST_GDAL_PRELOADED


def gdal_env_is_host_managed() -> bool:
    """Whether pyramids must leave the process-global GDAL environment to a host.

    When ``True``, pyramids does not set or overwrite ``GDAL_DRIVER_PATH``, ``GDAL_DATA``,
    ``PROJ_LIB`` / ``PROJ_DATA`` or ``PATH`` and does not activate its vendored GDAL tree --
    it uses whatever GDAL the host already provides (#1588). Resolution, highest precedence
    first:

    - ``PYRAMIDS_MANAGE_GDAL_ENV`` -- ``0`` / ``false`` / ``no`` / ``off`` forces host-managed;
      any other value forces pyramids-managed, overriding auto-detection either way.
    - auto-detection -- a host's ``osgeo`` was imported before pyramids
      (:func:`host_gdal_preloaded`).

    ``PYRAMIDS_KEEP_GDAL_ENV`` is deliberately not consulted here: it only re-ranks the
    composed ``GDAL_DRIVER_PATH`` so an inherited value leads, which is orthogonal to leaving
    the whole GDAL environment untouched.

    Returns:
        bool: ``True`` when pyramids should leave the GDAL environment to the host.
    """
    override = os.environ.get("PYRAMIDS_MANAGE_GDAL_ENV")
    if override is not None:
        return override.strip().lower() in ("0", "false", "no", "off")
    return _HOST_GDAL_PRELOADED


# Module-level handle from `os.add_dll_directory()` on Windows.
# add_dll_directory returns a context-manager-style handle; if the
# handle gets garbage-collected the directory falls off the DLL
# search path mid-process. Anchor it at module scope so the bootstrap
# module's lifetime (= the Python process's lifetime) keeps the
# directory registered.
_DLL_HANDLE = None


def _configure_gdal_data_env(data_dir: Path) -> None:
    """Point GDAL / PROJ / libcurl at the wheel's bundled data directories.

    ``GDAL_DATA`` / ``PROJ_DATA`` / ``PROJ_LIB`` and the CA bundle use
    ``setdefault`` so a user/system override wins; ``GDAL_DRIVER_PATH`` is
    forced to the bundled plugins dir (they are ABI-locked to the bundled
    libgdal) unless ``PYRAMIDS_KEEP_GDAL_ENV=1``.
    """
    gdal_data = data_dir / "gdal_data"
    proj_data = data_dir / "proj_data"
    gdal_plugins = data_dir / "gdalplugins"
    ca_bundle = data_dir / "ssl" / "cacert.pem"
    if gdal_data.is_dir():
        os.environ.setdefault("GDAL_DATA", str(gdal_data))
    if proj_data.is_dir():
        os.environ.setdefault("PROJ_DATA", str(proj_data))
        os.environ.setdefault("PROJ_LIB", str(proj_data))
    if gdal_plugins.is_dir():
        # GDAL_DRIVER_PATH points at compiled driver plugins (gdal_netCDF.so,
        # gdal_HDF5.so, gdal_GRIB.so, ...) ABI-locked to the exact bundled
        # libgdal build, so it is FORCED to the bundled dir rather than
        # setdefault'd — a foreign (e.g. conda-activated) GDAL_DRIVER_PATH is
        # never safe for the bundled libgdal (issue #465). Power users can set
        # PYRAMIDS_KEEP_GDAL_ENV=1 to restore "inherited value wins".
        #
        # This is the EARLY value, covering the window before `from osgeo import …`
        # runs. It is not the last word: `Config.initialize_gdal` recomposes the
        # final path from the bundled dir + any companion plugin dirs + a discovered
        # conda/system dir (see `pyramids.base._gdal_plugins`) and writes it once,
        # before `gdal.AllRegister()` actually loads the plugins.
        # A host-managed process never reaches here: activate_vendored_osgeo returns before
        # calling this helper when gdal_env_is_host_managed() (#1588). So the only choice left
        # is the #465 force vs. the PYRAMIDS_KEEP_GDAL_ENV opt-out.
        if os.environ.get("PYRAMIDS_KEEP_GDAL_ENV") == "1":
            os.environ.setdefault("GDAL_DRIVER_PATH", str(gdal_plugins))
        else:
            os.environ["GDAL_DRIVER_PATH"] = str(gdal_plugins)
    if ca_bundle.is_file():
        # The bundled libcurl bakes its CA path to the wheel-build prefix,
        # absent in the consuming env, so /vsicurl HTTPS fails with "error
        # adding trust anchors". Re-point GDAL / PROJ / libcurl at the vendored
        # cacert.pem; setdefault keeps any user override authoritative (#412).
        ca_str = str(ca_bundle)
        os.environ.setdefault("GDAL_HTTP_CAINFO", ca_str)
        os.environ.setdefault("PROJ_CURL_CA_BUNDLE", ca_str)
        os.environ.setdefault("CURL_CA_BUNDLE", ca_str)
        os.environ.setdefault("SSL_CERT_FILE", ca_str)


def _register_windows_dll_dirs(pkg_dir: Path) -> None:  # pragma: no cover
    """Register the wheel's native DLL directory on Windows.

    delvewheel places DLLs at ``<site-packages>/pyramids_core.libs/``; this is a
    safety net for that layout and the older ``pyramids/.libs/`` convention. The
    dir is also prepended to ``PATH`` because GDAL's native plugin loader uses
    raw ``LoadLibrary`` (no ``SEARCH_USER_DIRS``), which ignores
    ``add_dll_directory``. The handle is anchored at module scope so the
    directory stays registered for the process lifetime.
    """
    # os.add_dll_directory exists only on Windows. The sole call site already guards on
    # sys.platform, but repeat it here so mypy (run on Linux in CI) narrows sys.platform
    # and stops flagging the attribute as undefined.
    if sys.platform != "win32":
        return
    global _DLL_HANDLE
    for candidate in (pkg_dir / ".libs", pkg_dir.parent / "pyramids_core.libs"):
        if candidate.is_dir():
            _DLL_HANDLE = os.add_dll_directory(str(candidate))
            candidate_str = str(candidate)
            path = os.environ.get("PATH", "")
            if candidate_str not in path.split(os.pathsep):
                os.environ["PATH"] = candidate_str + os.pathsep + path
            break


def activate_vendored_osgeo(pkg_dir: Path) -> bool:
    """Activate the vendored osgeo if `_vendor/osgeo/` exists + ABI-matches.

    Args:
        pkg_dir: directory of the `pyramids` package (containing
            `_vendor/` and `_data/`). The caller is responsible for
            resolving this — typically ``Path(pyramids.__path__[0])``
            from `pyramids/__init__.py`. No default is provided so the
            resolution stays at the caller and this module never has
            to guess its own location (which would be wrong now that
            it lives under `pyramids/base/`).

    Returns:
        ``True`` if the vendor bootstrap activated (sys.path was
        modified, env vars were set). ``False`` if no `_vendor/`
        directory exists or if its `_gdal.<EXT>` doesn't match the
        current Python ABI.
    """
    vendored_osgeo = pkg_dir / "_vendor" / "osgeo"

    if not vendored_osgeo.is_dir():
        return False

    if gdal_env_is_host_managed():
        # A host already owns GDAL (osgeo was imported before pyramids, or the embedder set
        # PYRAMIDS_MANAGE_GDAL_ENV=0). Use the host's GDAL and leave its process-global env --
        # GDAL_DRIVER_PATH / GDAL_DATA / PROJ_LIB / PATH -- untouched: skip the vendored tree
        # entirely rather than activating it (#1588). The host's already-imported osgeo stays
        # the one `from osgeo import ...` resolves to.
        return False

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    if not (vendored_osgeo / f"_gdal{ext_suffix}").is_file():
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        warnings.warn(
            f"pyramids: ignoring src/pyramids/_vendor/ — no "
            f"_gdal{ext_suffix} found (vendored for a different Python "
            f"ABI). Remove src/pyramids/_vendor and src/pyramids/_data "
            f"to silence this warning, or rebuild for Python {py_ver}.",
            stacklevel=2,
        )
        return False

    vendor_str = str(pkg_dir / "_vendor")
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

    _configure_gdal_data_env(pkg_dir / "_data")

    if sys.platform == "win32":  # pragma: no cover
        _register_windows_dll_dirs(pkg_dir)

    if os.environ.get("PYRAMIDS_DEBUG_BOOTSTRAP"):  # pragma: no cover
        print(f"[pyramids] vendor dir: {vendor_str}")
        print(f"[pyramids] GDAL_DATA: {os.environ.get('GDAL_DATA')}")
        print(f"[pyramids] PROJ_DATA: {os.environ.get('PROJ_DATA')}")
        print(f"[pyramids] GDAL_HTTP_CAINFO: {os.environ.get('GDAL_HTTP_CAINFO')}")

    return True
