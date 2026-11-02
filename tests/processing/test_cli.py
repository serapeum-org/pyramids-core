"""CLI smoke tests for the processing subcommands (tools / tool / run)."""

import geopandas as gpd
import numpy as np
import pytest
import yaml
from shapely.geometry import Point

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.processing import Pipeline


@pytest.fixture
def dem_raster(tmp_path):
    """An 8x8 float32 ramp GeoTIFF on EPSG:4326 (a stand-in elevation model).

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "dem.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return path


@pytest.fixture
def points_geojson(tmp_path):
    """Write a small point layer to GeoJSON and return its path.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written .geojson.
    """
    gdf = gpd.GeoDataFrame(
        {"elevation": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(0, 0), Point(4, 0), Point(0, 4), Point(4, 4), Point(2, 2)],
        crs="EPSG:4326",
    )
    path = tmp_path / "pts.geojson"
    FeatureCollection(gdf).to_file(str(path))
    return str(path)


class TestProcessingCli:
    """Tests for `pyramids tools|tool|run`."""

    def test_tools_lists_registry(self, capsys):
        """`pyramids tools` lists every registered tool.

        Test scenario:
            Output contains slope and interpolate_to_raster and exits 0.
        """
        rc = main(["tools"])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "slope" in out, out
        assert "interpolate_to_raster" in out, out

    def test_tool_prints_schema(self, capsys):
        """`pyramids tool slope` prints the tool's parameter schema.

        Test scenario:
            Output includes the input header and the 'band' param; exit 0.
        """
        rc = main(["tool", "slope"])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "Dataset -> Array" in out, out
        assert "band" in out, out

    def test_tool_unknown_returns_error(self, capsys):
        """`pyramids tool <unknown>` reports an error and exits non-zero.

        Test scenario:
            An unknown tool name prints 'unknown tool' to stderr and returns 1.
        """
        rc = main(["tool", "nope"])
        err = capsys.readouterr().err
        assert rc == 1, err
        assert "unknown tool" in err, err

    def test_run_end_to_end_writes_output(self, points_geojson, tmp_path):
        """`pyramids run` executes a pipeline YAML and writes an output.

        Args:
            points_geojson: input point layer path.
            tmp_path: pytest temp directory.

        Test scenario:
            An interpolate_to_raster pipeline over the input writes a .tif and
            exits 0.
        """
        yaml_path = tmp_path / "pipe.yaml"
        Pipeline(
            [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
        ).to_yaml(str(yaml_path))
        out_dir = tmp_path / "out"
        rc = main(
            ["run", str(yaml_path), "--inputs", points_geojson, "--out", str(out_dir)]
        )
        assert rc == 0, "run should exit 0"
        assert list(out_dir.glob("*.tif")), list(out_dir.iterdir())

    def test_run_bad_input_skip_reports_failure(self, tmp_path, capsys):
        """`pyramids run` with a bad input under skip reports the failure, exits 1.

        Args:
            tmp_path: pytest temp directory.
            capsys: stdout capture.

        Test scenario:
            A nonexistent input is skipped-and-reported; exit code is 1.
        """
        yaml_path = tmp_path / "pipe.yaml"
        Pipeline([("slope", {})]).to_yaml(str(yaml_path))
        out_dir = tmp_path / "out"
        rc = main(
            ["run", str(yaml_path), "--inputs", "C:/no/such.tif", "--out", str(out_dir)]
        )
        captured = capsys.readouterr()
        assert rc == 1, captured.err
        assert "FAILED" in captured.err, captured.err


def _interp_pipeline(tmp_path):
    """Write a one-step interpolate_to_raster pipeline YAML; return its path."""
    yaml_path = tmp_path / "pipe.yaml"
    Pipeline(
        [("interpolate_to_raster", {"column": "elevation", "cell_size": 1.0})]
    ).to_yaml(str(yaml_path))
    return str(yaml_path)


class TestApply:
    """`pyramids apply` — run one registered tool over a single input (F-1)."""

    def test_apply_slope_writes_output(self, dem_raster, tmp_path):
        """apply runs a single Array-output tool and writes a georeferenced raster."""
        out = str(tmp_path / "slope.tif")
        rc = main(["apply", "slope", dem_raster, out, "--set", "units=degrees"])
        assert rc == 0, "apply must exit 0"
        assert Dataset.read_file(out).band_count >= 1

    def test_unknown_tool_errors(self, dem_raster, tmp_path, capsys):
        """An unknown tool exits 1 naming the registry."""
        rc = main(["apply", "nope", dem_raster, str(tmp_path / "o.tif")])
        assert rc == 1
        assert "unknown tool" in capsys.readouterr().err

    def test_unknown_param_errors(self, dem_raster, tmp_path, capsys):
        """An unknown --set parameter exits 1 with a schema-driven message."""
        rc = main(
            ["apply", "slope", dem_raster, str(tmp_path / "o.tif"), "--set", "bogus=1"]
        )
        assert rc == 1
        assert "unknown parameter" in capsys.readouterr().err

    def test_bad_value_errors(self, dem_raster, tmp_path, capsys):
        """A value outside an OptionList's choices exits 1."""
        rc = main(
            [
                "apply",
                "slope",
                dem_raster,
                str(tmp_path / "o.tif"),
                "--set",
                "units=parsecs",
            ]
        )
        assert rc == 1

    def test_missing_required_param_errors(self, points_geojson, tmp_path, capsys):
        """Omitting a required parameter fails at pipeline construction."""
        rc = main(
            ["apply", "interpolate_to_raster", points_geojson, str(tmp_path / "o.tif")]
        )
        assert rc == 1
        assert "missing required parameter" in capsys.readouterr().err

    def test_refuses_existing(self, dem_raster, tmp_path):
        """A second apply without --overwrite is refused."""
        out = str(tmp_path / "s.tif")
        assert main(["apply", "slope", dem_raster, out]) == 0
        assert main(["apply", "slope", dem_raster, out]) == 1


class TestRunManifest:
    """`pyramids run --manifest` persists provenance (F-2)."""

    def test_manifest_written_and_reloadable(self, points_geojson, tmp_path):
        """--manifest writes a recipe that round-trips through Pipeline.from_dict."""
        yaml_path = _interp_pipeline(tmp_path)
        out_dir = tmp_path / "out"
        manifest = tmp_path / "manifest.yaml"
        rc = main(
            [
                "run",
                yaml_path,
                "--inputs",
                points_geojson,
                "--out",
                str(out_dir),
                "--manifest",
                str(manifest),
            ]
        )
        assert rc == 0
        data = yaml.safe_load(manifest.read_text())
        assert data["runs"][0]["source"].endswith(".geojson")
        assert data["runs"][0]["total_seconds"] >= 0.0
        reloaded = Pipeline.from_dict({"pipeline": data["pipeline"]})
        assert len(reloaded) == 1


class TestRunMultiInput:
    """`pyramids run --inputs a b` processes an explicit file list (B-2)."""

    def test_two_explicit_inputs(self, points_geojson, tmp_path):
        """Both listed inputs produce an output."""
        second = tmp_path / "pts2.geojson"
        FeatureCollection.read_file(points_geojson).to_file(str(second))
        yaml_path = _interp_pipeline(tmp_path)
        out_dir = tmp_path / "out"
        rc = main(
            [
                "run",
                yaml_path,
                "--inputs",
                points_geojson,
                str(second),
                "--out",
                str(out_dir),
            ]
        )
        assert rc == 0
        assert len(list(out_dir.glob("*.tif"))) == 2


class TestRunQuiet:
    """`pyramids --quiet run` output gating (L4)."""

    def test_quiet_suppresses_summary(self, points_geojson, tmp_path, capsys):
        """--quiet silences the 'N succeeded' summary chatter."""
        yaml_path = _interp_pipeline(tmp_path)
        out_dir = tmp_path / "out"
        rc = main(
            [
                "--quiet",
                "run",
                yaml_path,
                "--inputs",
                points_geojson,
                "--out",
                str(out_dir),
            ]
        )
        assert rc == 0
        assert "succeeded" not in capsys.readouterr().out

    def test_quiet_still_reports_failures(self, tmp_path, capsys):
        """A FAILED line is error reporting and prints even under --quiet."""
        yaml_path = tmp_path / "pipe.yaml"
        Pipeline([("slope", {})]).to_yaml(str(yaml_path))
        rc = main(
            [
                "--quiet",
                "run",
                str(yaml_path),
                "--inputs",
                "C:/no/such.tif",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "FAILED" in captured.err, (
            "failures must still be reported (stderr) under --quiet"
        )
        assert "succeeded" not in captured.out, "the summary chatter stays silenced"


class TestRunParallel:
    """`pyramids run --parallel` end to end (F-3 / T-3)."""

    def test_parallel_writes_outputs(self, points_geojson, tmp_path):
        """A parallel batch runs across a process pool and writes both outputs."""
        second = tmp_path / "pts2.geojson"
        FeatureCollection.read_file(points_geojson).to_file(str(second))
        yaml_path = _interp_pipeline(tmp_path)
        out_dir = tmp_path / "out"
        rc = main(
            [
                "run",
                yaml_path,
                "--inputs",
                points_geojson,
                str(second),
                "--out",
                str(out_dir),
                "--parallel",
                "--max-workers",
                "2",
            ]
        )
        assert rc == 0, "a parallel batch must succeed"
        assert len(list(out_dir.glob("*.tif"))) == 2

    def test_bad_max_workers_errors(self, points_geojson, tmp_path, capsys):
        """--max-workers 0 is rejected with a one-line error."""
        yaml_path = _interp_pipeline(tmp_path)
        rc = main(
            [
                "run",
                yaml_path,
                "--inputs",
                points_geojson,
                "--out",
                str(tmp_path / "o"),
                "--parallel",
                "--max-workers",
                "0",
            ]
        )
        assert rc == 1
        assert "max_workers must be >= 1" in capsys.readouterr().err
