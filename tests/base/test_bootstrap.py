"""Unit tests for pyramids.base._bootstrap (vendored-osgeo activation).

Focus: the curl CA-bundle wiring added for issue #412. The bootstrap must
re-point GDAL / PROJ / libcurl at the cacert.pem vendored into the wheel,
without clobbering a user/system override (the setdefault contract).
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from pyramids.base import _bootstrap
from pyramids.base._bootstrap import activate_vendored_osgeo, host_gdal_preloaded

pytestmark = pytest.mark.core

_CA_ENV_VARS = (
    "GDAL_HTTP_CAINFO",
    "PROJ_CURL_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
)

# Env vars the driver-path / data-dir logic touches, plus the opt-out.
_GDAL_ENV_VARS = (
    "GDAL_DRIVER_PATH",
    "GDAL_DATA",
    "PROJ_DATA",
    "PROJ_LIB",
    "PYRAMIDS_KEEP_GDAL_ENV",
    "PYRAMIDS_MANAGE_GDAL_ENV",
)


@pytest.fixture
def vendored_pkg(tmp_path: Path):
    """Build a fake vendored-wheel package dir (ABI-matching _gdal ext).

    Yields the package dir. `with_ca` controls whether the cacert.pem is
    present so individual tests can toggle it.
    """

    def _build(with_ca: bool = False, with_plugins: bool = False) -> Path:
        osgeo = tmp_path / "_vendor" / "osgeo"
        osgeo.mkdir(parents=True)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        (osgeo / f"_gdal{ext_suffix}").touch()
        if with_ca:
            ssl_dir = tmp_path / "_data" / "ssl"
            ssl_dir.mkdir(parents=True)
            (ssl_dir / "cacert.pem").write_text("# fake bundle\n", encoding="utf-8")
        if with_plugins:
            plugins = tmp_path / "_data" / "gdalplugins"
            plugins.mkdir(parents=True)
            (plugins / "gdal_netCDF.so").touch()
        return tmp_path

    return _build


@pytest.fixture(autouse=True)
def _isolate_env_and_syspath():
    """Snapshot the CA env vars + sys.path; restore after each test.

    activate_vendored_osgeo mutates os.environ (via setdefault) and
    inserts the vendor dir into sys.path — both are process-global, so
    restore them to keep tests order-independent.
    """
    tracked = _CA_ENV_VARS + _GDAL_ENV_VARS
    saved_env = {k: os.environ.get(k) for k in tracked}
    saved_path = list(sys.path)
    for k in tracked:
        os.environ.pop(k, None)
    yield
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    sys.path[:] = saved_path


class TestCaBundle:
    """CA-bundle env wiring (#412)."""

    def test_sets_all_ca_vars_to_bundled_cert(self, vendored_pkg):
        """All CA env vars point at the vendored cacert.pem when present.

        Test scenario:
            A vendored package dir ships _data/ssl/cacert.pem; activation
            sets every CA env var to that file.
        """
        pkg = vendored_pkg(with_ca=True)
        expected = str(pkg / "_data" / "ssl" / "cacert.pem")

        assert activate_vendored_osgeo(pkg) is True
        for var in _CA_ENV_VARS:
            assert os.environ.get(var) == expected, f"{var} not pointed at bundle"

    def test_does_not_override_user_setting(self, vendored_pkg):
        """A pre-set CA var wins over the bundled cert (setdefault contract).

        Test scenario:
            CURL_CA_BUNDLE is already set; activation leaves it untouched
            while still setting the unset vars.
        """
        pkg = vendored_pkg(with_ca=True)
        os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

        activate_vendored_osgeo(pkg)

        assert os.environ["CURL_CA_BUNDLE"] == "/etc/ssl/certs/ca-certificates.crt"
        assert os.environ["GDAL_HTTP_CAINFO"] == str(
            pkg / "_data" / "ssl" / "cacert.pem"
        )

    def test_no_ca_vars_when_cert_absent(self, vendored_pkg):
        """No CA var is set when the wheel ships no cacert.pem.

        Test scenario:
            A vendored package dir without _data/ssl/cacert.pem activates
            (the osgeo ext is present) but sets none of the CA env vars.
        """
        pkg = vendored_pkg(with_ca=False)

        assert activate_vendored_osgeo(pkg) is True
        for var in _CA_ENV_VARS:
            assert var not in os.environ, f"{var} should be unset without a bundle"


class TestDriverPath:
    """GDAL_DRIVER_PATH override for the bundled, version-locked plugins (#465)."""

    def test_sets_driver_path_when_unset(self, vendored_pkg):
        """GDAL_DRIVER_PATH points at the bundled plugin dir when unset.

        Test scenario:
            A vendored package dir ships _data/gdalplugins; activation
            with no inherited GDAL_DRIVER_PATH sets it to that dir.
        """
        pkg = vendored_pkg(with_plugins=True)
        expected = str(pkg / "_data" / "gdalplugins")

        assert activate_vendored_osgeo(pkg) is True
        assert os.environ["GDAL_DRIVER_PATH"] == expected

    def test_overrides_inherited_driver_path(self, vendored_pkg):
        """A conda-style inherited GDAL_DRIVER_PATH is FORCED to the bundle.

        Test scenario:
            GDAL_DRIVER_PATH is pre-set to a foreign dir (as an activated
            conda env exports it). Because the bundled plugins are
            ABI-locked to the bundled libgdal, activation overrides it
            rather than honoring the inherited value (#465).
        """
        pkg = vendored_pkg(with_plugins=True)
        os.environ["GDAL_DRIVER_PATH"] = "/opt/conda/envs/x/lib/gdalplugins"

        activate_vendored_osgeo(pkg)

        assert os.environ["GDAL_DRIVER_PATH"] == str(pkg / "_data" / "gdalplugins")

    def test_opt_out_keeps_inherited_driver_path(self, vendored_pkg):
        """PYRAMIDS_KEEP_GDAL_ENV=1 restores the old setdefault behavior.

        Test scenario:
            With the opt-out set, a pre-set GDAL_DRIVER_PATH is preserved.
        """
        pkg = vendored_pkg(with_plugins=True)
        os.environ["PYRAMIDS_KEEP_GDAL_ENV"] = "1"
        os.environ["GDAL_DRIVER_PATH"] = "/opt/conda/envs/x/lib/gdalplugins"

        activate_vendored_osgeo(pkg)

        assert os.environ["GDAL_DRIVER_PATH"] == "/opt/conda/envs/x/lib/gdalplugins"

    def test_data_dirs_still_respect_user_override(self, vendored_pkg):
        """GDAL_DATA / PROJ_DATA keep setdefault semantics (not forced).

        Test scenario:
            A pre-set GDAL_DATA is left untouched even though the driver
            path is forced — the data dirs are version-tolerant and the
            override contract for them is intentional.
        """
        pkg = vendored_pkg(with_plugins=True)
        (pkg / "_data" / "gdal_data").mkdir(parents=True)
        (pkg / "_data" / "proj_data").mkdir(parents=True)
        os.environ["GDAL_DATA"] = "/custom/gdal_data"

        activate_vendored_osgeo(pkg)

        assert os.environ["GDAL_DATA"] == "/custom/gdal_data"
        assert os.environ["PROJ_DATA"] == str(pkg / "_data" / "proj_data")
        assert os.environ["GDAL_DRIVER_PATH"] == str(pkg / "_data" / "gdalplugins")


class TestHostManaged:
    """Host-aware opt-out: pyramids leaves a host application's GDAL env alone (#1588)."""

    def test_host_gdal_preloaded_reflects_module_flag(self, monkeypatch):
        """host_gdal_preloaded() returns the import-time _HOST_GDAL_PRELOADED capture.

        Test scenario:
            Flipping the module-level flag flips what the accessor reports — the capture is
            the single source of truth for "a host imported osgeo first".
        """
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", True)
        assert host_gdal_preloaded() is True, (
            "should report host-preloaded when flag is True"
        )
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", False)
        assert host_gdal_preloaded() is False, (
            "should report not-preloaded when flag is False"
        )

    def test_activate_skips_when_env_opt_out(self, monkeypatch, vendored_pkg):
        """PYRAMIDS_MANAGE_GDAL_ENV=0 makes activation a no-op: driver path + sys.path untouched.

        Test scenario:
            A full vendored wheel is present, but the embedder opted out via the env var, so
            activation returns False and leaves the host's GDAL_DRIVER_PATH and sys.path as-is.
        """
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", False)
        monkeypatch.setenv("PYRAMIDS_MANAGE_GDAL_ENV", "0")
        pkg = vendored_pkg(with_plugins=True)
        monkeypatch.setenv("GDAL_DRIVER_PATH", "/host/gdalplugins")
        before_path = list(sys.path)
        assert activate_vendored_osgeo(pkg) is False, (
            "activation must be skipped when host-managed"
        )
        assert os.environ["GDAL_DRIVER_PATH"] == "/host/gdalplugins", (
            "driver path must be untouched"
        )
        assert sys.path == before_path, (
            "vendor dir must not be inserted when host-managed"
        )

    def test_activate_skips_when_host_preloaded(self, monkeypatch, vendored_pkg):
        """Auto-detection: a host osgeo preloaded makes activation a no-op (no env writes).

        Test scenario:
            No env override is set, but osgeo was imported before pyramids. Activation is
            skipped, so neither GDAL_DRIVER_PATH nor the CA-bundle vars are written.
        """
        monkeypatch.delenv("PYRAMIDS_MANAGE_GDAL_ENV", raising=False)
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", True)
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_ENV_PRESENT", True)
        pkg = vendored_pkg(with_plugins=True, with_ca=True)
        monkeypatch.setenv("GDAL_DRIVER_PATH", "/host/gdalplugins")
        assert activate_vendored_osgeo(pkg) is False, (
            "activation must be skipped when host-preloaded"
        )
        assert os.environ["GDAL_DRIVER_PATH"] == "/host/gdalplugins", (
            "driver path must be untouched"
        )
        for var in _CA_ENV_VARS:
            assert var not in os.environ, f"{var} must stay unset when host-managed"

    def test_managed_activates_when_no_host(self, monkeypatch, vendored_pkg):
        """No host and no opt-out: activation proceeds and forces the bundled driver path.

        Test scenario:
            The standalone path — nothing preloaded osgeo and no override — still activates
            the vendored tree exactly as before, confirming the gate does not regress it.
        """
        monkeypatch.delenv("PYRAMIDS_MANAGE_GDAL_ENV", raising=False)
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", False)
        pkg = vendored_pkg(with_plugins=True)
        assert activate_vendored_osgeo(pkg) is True, (
            "standalone activation must still proceed"
        )
        assert os.environ["GDAL_DRIVER_PATH"] == str(pkg / "_data" / "gdalplugins")

    def test_suite_imports_pyramids_before_osgeo(self):
        """Guard: the suite must import pyramids before osgeo, or auto-detection flips it host-managed.

        The module-level capture is only `False` because the root `tests/conftest.py` imports
        pyramids (hence `_bootstrap`) before any `from osgeo import ...`. If a future reflow or a new
        top-level conftest/plugin imported osgeo first, `_HOST_GDAL_PRELOADED` would flip to True and
        the whole GDAL-dependent suite could silently run host-managed. This reads the real captured
        value (not a monkeypatched one) so that regression fails loudly here (#1588 review L4).
        """
        assert host_gdal_preloaded() is False, (
            "osgeo was imported before pyramids in the test session; keep the conftest ordering "
            "(import pyramids before `from osgeo import ...`)"
        )


class TestHostEnvCapture:
    """The import-time `_HOST_GDAL_ENV_PRESENT` capture probes only GDAL driver/data vars (#1588 L1-L3)."""

    @pytest.mark.parametrize(
        "env, expected",
        [
            ({}, False),
            ({"GDAL_DATA": "/x"}, True),
            ({"GDAL_DRIVER_PATH": "/x"}, True),
            ({"PROJ_LIB": "/x"}, False),
            ({"GDAL_DATA": ""}, True),
        ],
    )
    def test_capture_probes_gdal_driver_data_only(self, env, expected):
        """A fresh interpreter reads host-env presence from GDAL_DRIVER_PATH/GDAL_DATA only, not PROJ_*.

        Spawns a clean child with the three vars stripped then the parametrized ones set, imports
        `_bootstrap`, and prints the captured flag. Proves PROJ_LIB alone does not qualify (L3) and an
        explicit empty GDAL_DATA still does (L2, membership not truthiness).
        """
        stripped = {
            k: v
            for k, v in os.environ.items()
            if k not in ("GDAL_DRIVER_PATH", "GDAL_DATA", "PROJ_LIB")
        }
        stripped.update(env)
        code = "import pyramids.base._bootstrap as b; print(b._HOST_GDAL_ENV_PRESENT)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=stripped
        )
        got = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        assert got == str(expected), (
            f"env={env}: expected {expected}, got {got!r}; stderr={out.stderr[-300:]}"
        )
