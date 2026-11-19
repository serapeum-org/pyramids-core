"""Host-aware GDAL env management: the opt-out knob and auto-detection (#1588).

pyramids must not clobber a host application's process-global GDAL environment
(`GDAL_DRIVER_PATH` / `GDAL_DATA` / `PROJ_LIB` / `PATH`) when it is embedded in a process that
already owns GDAL (QGIS). These tests exercise the resolver and the `initialize_gdal` gate in
isolation, with GDAL mocked so nothing touches the real environment.
"""

import logging
import os
from unittest import mock

from pyramids.base import _bootstrap
from pyramids.base.config import Config


class TestGdalEnvIsHostManaged:
    """The `PYRAMIDS_MANAGE_GDAL_ENV` override and the preloaded-host auto-detection."""

    def test_env_var_off_forces_host_managed(self, monkeypatch):
        """`PYRAMIDS_MANAGE_GDAL_ENV` off-values mark the process host-managed regardless of detection."""
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", False)
        for off in ("0", "false", "False", "no", "OFF"):
            monkeypatch.setenv("PYRAMIDS_MANAGE_GDAL_ENV", off)
            assert _bootstrap.gdal_env_is_host_managed() is True

    def test_env_var_on_forces_managed(self, monkeypatch):
        """Any other `PYRAMIDS_MANAGE_GDAL_ENV` value forces pyramids-managed, overriding detection."""
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", True)
        monkeypatch.setenv("PYRAMIDS_MANAGE_GDAL_ENV", "1")
        assert _bootstrap.gdal_env_is_host_managed() is False

    def test_auto_detection_follows_preloaded_flag(self, monkeypatch):
        """With no override, the decision follows whether a host `osgeo` was preloaded."""
        monkeypatch.delenv("PYRAMIDS_MANAGE_GDAL_ENV", raising=False)
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", True)
        assert _bootstrap.gdal_env_is_host_managed() is True
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", False)
        assert _bootstrap.gdal_env_is_host_managed() is False


def _bare_config():
    """A Config built without __init__ side effects, with just a logger attached."""
    cfg = object.__new__(Config)
    cfg.logger = logging.getLogger("tests.config.host_gdal")
    cfg.settings = {"gdal": {}, "ogr": {}}
    return cfg


class TestShouldManageGdalEnv:
    """`Config._should_manage_gdal_env` precedence: explicit flag over auto-detection."""

    def test_explicit_flag_wins(self, monkeypatch):
        """`manage_gdal_env=True/False` overrides auto-detection on that Config instance."""
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", True)
        cfg = _bare_config()
        cfg._manage_gdal_env = True
        assert cfg._should_manage_gdal_env() is True
        cfg._manage_gdal_env = False
        assert cfg._should_manage_gdal_env() is False

    def test_defaults_to_autodetection(self, monkeypatch):
        """`manage_gdal_env=None` defers to `gdal_env_is_host_managed`."""
        monkeypatch.delenv("PYRAMIDS_MANAGE_GDAL_ENV", raising=False)
        cfg = _bare_config()
        cfg._manage_gdal_env = None
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", True)
        assert cfg._should_manage_gdal_env() is False
        monkeypatch.setattr(_bootstrap, "_HOST_GDAL_PRELOADED", False)
        assert cfg._should_manage_gdal_env() is True


class TestInitializeGdalHostManaged:
    """The `initialize_gdal` gate: host-managed skips discovery and leaves the env alone."""

    def test_host_managed_leaves_env_untouched(self, monkeypatch, tmp_path):
        """Host-managed init writes none of GDAL_DRIVER_PATH / GDAL_DATA / PROJ_LIB and skips discovery."""
        host = {
            "GDAL_DRIVER_PATH": str(tmp_path / "dp"),
            "GDAL_DATA": str(tmp_path / "gd"),
            "PROJ_LIB": str(tmp_path / "pl"),
        }
        for key, value in host.items():
            monkeypatch.setenv(key, value)
        cfg = _bare_config()
        cfg._manage_gdal_env = False
        dyn = mock.Mock()
        monkeypatch.setattr(Config, "dynamic_env_variables", dyn)
        with mock.patch("pyramids.base.config.gdal") as g, mock.patch("pyramids.base.config.ogr"):
            cfg.initialize_gdal()
        dyn.assert_not_called()
        for key, value in host.items():
            assert os.environ[key] == value
        assert not [
            c for c in g.SetConfigOption.call_args_list if c.args[:1] == ("GDAL_DRIVER_PATH",)
        ]
        g.AllRegister.assert_called_once()

    def test_managed_runs_discovery(self, monkeypatch):
        """When managing, init still runs discovery (the gate's opposite branch)."""
        cfg = _bare_config()
        cfg._manage_gdal_env = True
        monkeypatch.delenv("GDAL_DRIVER_PATH", raising=False)
        monkeypatch.setattr(Config, "_bundled_plugin_dir", staticmethod(lambda: None))
        dyn = mock.Mock(return_value=None)
        monkeypatch.setattr(Config, "dynamic_env_variables", dyn)
        with mock.patch("pyramids.base.config.gdal"), mock.patch("pyramids.base.config.ogr"):
            cfg.initialize_gdal()
        dyn.assert_called_once()
