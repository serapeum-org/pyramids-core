"""Tests for the CLI ergonomics and bug-fix tasks.

Covers E-1 (`--version`), E-2 (`python -m pyramids`), T-1 (`--help` on every
subcommand), T-2 (top-level help / no-args), E-3 (shared resampling vocabulary),
E-4 (`--quiet`), B-1 (calc note ordering), B-3 (`edit-info --nodata` tokens),
B-4 (`edit-info` driver guard), and M-7 (`render`).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from osgeo import gdal

import pyramids
from pyramids import __version__
from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestVersion:
    """`pyramids --version` (E-1)."""

    def test_version_exits_zero(self, capsys):
        """--version prints 'pyramids <ver>' and exits 0."""
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_short_flag_exits_zero(self, capsys):
        """-V is an alias for --version."""
        with pytest.raises(SystemExit) as exc:
            main(["-V"])
        assert exc.value.code == 0


class TestPythonMEntrypoint:
    """`python -m pyramids` (E-2)."""

    def test_module_form_runs(self, cli_raster):
        """`python -m pyramids info FILE --json` behaves like the console script."""
        src_dir = os.path.dirname(os.path.dirname(pyramids.__file__))
        env = {**os.environ, "PYTHONPATH": src_dir}
        proc = subprocess.run(
            [sys.executable, "-m", "pyramids", "info", cli_raster, "--json"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert '"epsg": 4326' in proc.stdout


class TestTopLevelHelp:
    """Top-level help surface (T-2)."""

    def test_help_lists_groups(self, capsys):
        """`pyramids --help` names the raster, cog, and new command groups."""
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for token in ("cog", "vector", "nc", "collection", "stac", "fetch", "apply"):
            assert token in out, f"top-level help should list {token!r}"

    def test_no_args_errors_cleanly(self, capsys):
        """`pyramids` with no args exits non-zero with usage, not a traceback."""
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert "usage" in err.lower()
        assert "Traceback" not in err


_HELP_ARGVS = [
    ["info", "--help"],
    ["bounds", "--help"],
    ["clip", "--help"],
    ["warp", "--help"],
    ["merge", "--help"],
    ["overview", "--help"],
    ["sample", "--help"],
    ["convert", "--help"],
    ["georeference", "--help"],
    ["orthorectify", "--help"],
    ["edit-info", "--help"],
    ["calc", "--help"],
    ["shapes", "--help"],
    ["rasterize", "--help"],
    ["render", "--help"],
    ["cog", "--help"],
    ["cog", "create", "--help"],
    ["cog", "validate", "--help"],
    ["cog", "info", "--help"],
    ["run", "--help"],
    ["apply", "--help"],
    ["tools", "--help"],
    ["tool", "--help"],
    ["vector", "--help"],
    ["vector", "info", "--help"],
    ["vector", "warp", "--help"],
    ["vector", "clip", "--help"],
    ["nc", "--help"],
    ["nc", "extract", "--help"],
    ["collection", "--help"],
    ["collection", "stats", "--help"],
    ["stac", "--help"],
    ["stac", "search", "--help"],
    ["fetch", "--help"],
    ["fetch", "wcs", "--help"],
    ["fetch", "wfs", "--help"],
]


class TestHelpSurfaceAll:
    """Every subcommand exposes --help and exits 0 (T-1)."""

    @pytest.mark.parametrize("argv", _HELP_ARGVS)
    def test_help_exits_zero(self, argv, capsys):
        """Each subcommand (and subgroup) prints usage and exits 0."""
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 0, f"{argv} --help must exit 0"
        assert "usage" in capsys.readouterr().out.lower()


class TestQuiet:
    """Global --quiet flag (E-4)."""

    def test_quiet_suppresses_wrote_line(self, cli_raster, tmp_path, capsys):
        """--quiet suppresses the 'wrote' chatter but still writes the output."""
        out = str(tmp_path / "q.tif")
        assert main(["--quiet", "warp", cli_raster, out, "--crs", "3857"]) == 0
        assert capsys.readouterr().out == "", "no stdout under --quiet"
        assert os.path.exists(out)


class TestResamplingVocab:
    """A shared resampling vocabulary across warp and overview (E-3)."""

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average"])
    def test_name_accepted_by_warp_and_overview(self, cli_raster, tmp_path, method):
        """The same name works on both warp and overview."""
        assert (
            main(
                [
                    "warp",
                    cli_raster,
                    str(tmp_path / f"w_{method}.tif"),
                    "--crs",
                    "3857",
                    "--resampling",
                    method,
                ]
            )
            == 0
        )
        assert (
            main(["overview", cli_raster, "--levels", "2", "--resampling", method]) == 0
        )

    def test_cubic_spline_alias_on_overview(self, cli_raster):
        """The warp spelling 'cubic_spline' maps to the overview 'cubicspline'."""
        assert (
            main(
                [
                    "overview",
                    cli_raster,
                    "--levels",
                    "2",
                    "--resampling",
                    "cubic_spline",
                ]
            )
            == 0
        )


class TestCalcNote:
    """calc CRS-note ordering (B-1)."""

    def test_untagged_companion_grid_mismatch_has_no_note(self, tmp_path, capsys):
        """A CRS-less companion on a different grid raises without the 'assuming' note."""
        a = str(tmp_path / "a.tif")
        Dataset.from_array(
            np.full((2, 2), 4.0, "float32"),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
        ).to_file(a)
        b = str(tmp_path / "b_nocrs.tif")
        out = gdal.GetDriverByName("GTiff").Create(b, 2, 2, 1, gdal.GDT_Float32)
        out.SetGeoTransform(
            (50.0, 1.0, 0.0, 2.0, 0.0, -1.0)
        )  # different origin, no CRS
        out.GetRasterBand(1).WriteArray(np.full((2, 2), 2.0, "float32"))
        out.FlushCache()
        out = None
        res = str(tmp_path / "diff.tif")

        rc = main(["calc", "A - B", a, b, res])
        err = capsys.readouterr().err

        assert rc == 1, "a grid mismatch must exit 1"
        assert "does not share the grid" in err
        assert "assuming it is on" not in err


class TestEditInfoNodata:
    """edit-info --nodata tokens (B-3)."""

    def test_unset_removes_sentinel(self, cli_raster):
        """--nodata none clears the no-data value."""
        assert main(["edit-info", cli_raster, "--nodata", "none"]) == 0
        assert Dataset.read_file(cli_raster).no_data_value[0] is None

    def test_nan_token(self, cli_raster):
        """--nodata nan sets a NaN sentinel."""
        assert main(["edit-info", cli_raster, "--nodata", "nan"]) == 0
        assert math.isnan(Dataset.read_file(cli_raster).no_data_value[0])

    def test_numeric_still_works(self, cli_raster):
        """A numeric --nodata still sets the value."""
        assert main(["edit-info", cli_raster, "--nodata", "0"]) == 0
        assert Dataset.read_file(cli_raster).no_data_value[0] == 0

    def test_bad_token_errors(self, cli_raster, capsys):
        """A non-numeric, non-token --nodata exits 1 cleanly."""
        assert main(["edit-info", cli_raster, "--nodata", "banana"]) == 1
        assert "not a number" in capsys.readouterr().err


class TestEditInfoDriverGuard:
    """edit-info on a non-updatable driver (B-4)."""

    def test_non_updatable_driver_clear_message(self, cli_raster, monkeypatch, capsys):
        """An update-mode open failure yields a CLI-level message, not raw GDAL text."""
        import pyramids.cli as cli_mod

        def boom(_path, *args, **kwargs):
            raise RuntimeError("driver X does not support update access")

        monkeypatch.setattr(cli_mod.Dataset, "read_file", staticmethod(boom))
        rc = main(["edit-info", cli_raster, "--nodata", "0"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "update-mode" in err and "pyramids convert" in err


class TestCogJson:
    """`cog create --json` / `cog validate --json` (E-4)."""

    def test_cog_create_json_result(self, cli_raster, tmp_path, capsys):
        """cog create --json emits a parseable result with the validity flag."""
        out = str(tmp_path / "out.cog.tif")
        rc = main(["cog", "create", cli_raster, out, "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0, "a valid COG must exit 0"
        assert payload["is_valid"] is True
        assert payload["output"].endswith(".cog.tif")

    def test_cog_validate_json_result(self, cli_raster, tmp_path, capsys):
        """cog validate --json emits is_valid / errors / warnings as JSON."""
        out = str(tmp_path / "out.cog.tif")
        assert main(["cog", "create", cli_raster, out]) == 0
        capsys.readouterr()
        rc = main(["cog", "validate", out, "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["is_valid"] is True
        assert payload["errors"] == []


@pytest.mark.plot
class TestRender:
    """`pyramids render` quick-look PNG (M-7)."""

    def test_writes_png(self, cli_raster, tmp_path):
        """render saves a non-empty PNG headlessly."""
        out = str(tmp_path / "ql.png")
        assert main(["render", cli_raster, out]) == 0
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_refuses_existing(self, cli_raster, tmp_path):
        """render refuses an existing output without --overwrite."""
        out = str(tmp_path / "exists.png")
        main(["render", cli_raster, out])
        assert main(["render", cli_raster, out]) == 1
