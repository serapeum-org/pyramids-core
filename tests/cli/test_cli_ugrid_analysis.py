"""Tests for the ``pyramids mesh`` analysis / terrain verbs (``cli_ugrid_analysis``).

The registrator is wired onto a throwaway ``mesh`` sub-command action and the parsed
handler is dispatched directly, so the suite passes before the orchestrator wires the
module into the real CLI. Each mesh fixture is a four-face strip written to a temporary
NetCDF, giving every verb a face-located variable to act on.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["MPLBACKEND"] = "Agg"

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from pyramids._cli_groups import add_group
from pyramids.cli_ugrid_analysis import add_ugrid_analysis_commands
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def _build(values, *, nodata=None):
    """Return a four-face strip MeshDataset carrying ``values`` as a face variable."""
    ds = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values, dtype="float64")},
        data_locations={"v": "face"},
    )
    if nodata is not None:
        ds["v"].nodata = nodata
    return ds


@pytest.fixture
def mesh_file(tmp_path):
    """A written four-face mesh NetCDF with values ``[5, 0, 0, 7]``."""
    path = tmp_path / "mesh.nc"
    _build([5.0, 0.0, 0.0, 7.0]).to_file(str(path))
    return str(path)


@pytest.fixture
def ramp_file(tmp_path):
    """A written four-face mesh NetCDF with a monotonic ramp ``[1, 2, 3, 4]``."""
    path = tmp_path / "ramp.nc"
    _build([1.0, 2.0, 3.0, 4.0]).to_file(str(path))
    return str(path)


@pytest.fixture
def mask_file(tmp_path):
    """A written four-face boolean mask mesh ``[1, 0, 0, 0]`` for morphology verbs."""
    path = tmp_path / "mask.nc"
    _build([1.0, 0.0, 0.0, 0.0]).to_file(str(path))
    return str(path)


@pytest.fixture
def zones_file(tmp_path):
    """A one-polygon zones vector covering the whole strip."""
    path = tmp_path / "zones.geojson"
    poly = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (0.0, 1.0)])
    FeatureCollection(gpd.GeoDataFrame(geometry=[poly])).to_file(
        str(path), driver="geojson"
    )
    return str(path)


def _run(argv):
    """Build a throwaway ``mesh`` parser, dispatch ``argv``, and return the exit code."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_analysis_commands(
        add_group(msub, "analysis", "terrain/field analysis over the mesh")
    )
    args = parser.parse_args(argv)
    return args.func(args)


def _run_capture(argv, capsys):
    """Dispatch ``argv`` and return ``(exit_code, parsed_json_stdout)``."""
    rc = _run(argv)
    return rc, json.loads(capsys.readouterr().out)


class TestTerrainWriters:
    """``slope`` / ``aspect`` / ``hillshade`` / ``focal-*`` / ``proximity`` write a mesh."""

    def test_slope_writes_a_mesh_with_the_derived_variable(self, ramp_file, tmp_path):
        out = str(tmp_path / "slope.nc")
        assert _run(["mesh", "analysis", "slope", ramp_file, out]) == 0
        assert "slope" in MeshDataset.read_file(out).data_variable_names

    def test_slope_radians(self, ramp_file, tmp_path):
        out = str(tmp_path / "slope.nc")
        assert (
            _run(["mesh", "analysis", "slope", ramp_file, out, "--units", "radians"])
            == 0
        )
        assert os.path.exists(out)

    def test_aspect_writes_a_mesh(self, ramp_file, tmp_path):
        out = str(tmp_path / "aspect.nc")
        assert _run(["mesh", "analysis", "aspect", ramp_file, out]) == 0
        assert "aspect" in MeshDataset.read_file(out).data_variable_names

    def test_hillshade_writes_a_mesh(self, ramp_file, tmp_path):
        out = str(tmp_path / "hs.nc")
        assert (
            _run(["mesh", "analysis", "hillshade", ramp_file, out, "--azimuth", "270"])
            == 0
        )
        assert "hillshade" in MeshDataset.read_file(out).data_variable_names

    def test_focal_mean_writes_a_mesh(self, ramp_file, tmp_path):
        out = str(tmp_path / "fm.nc")
        assert (
            _run(["mesh", "analysis", "focal-mean", ramp_file, out, "--rings", "2"])
            == 0
        )
        assert "focal_mean" in MeshDataset.read_file(out).data_variable_names

    def test_focal_std_writes_a_mesh(self, ramp_file, tmp_path):
        out = str(tmp_path / "fs.nc")
        assert _run(["mesh", "analysis", "focal-std", ramp_file, out]) == 0
        assert "focal_std" in MeshDataset.read_file(out).data_variable_names

    def test_proximity_writes_distances(self, ramp_file, tmp_path):
        out = str(tmp_path / "prox.nc")
        assert (
            _run(
                ["mesh", "analysis", "proximity", ramp_file, out, "--target-faces", "0"]
            )
            == 0
        )
        values = MeshDataset.read_file(out)["proximity"].data
        assert values.tolist() == [0, 1, 2, 3]

    def test_refuses_existing_without_overwrite(self, ramp_file, tmp_path):
        out = str(tmp_path / "slope.nc")
        assert _run(["mesh", "analysis", "slope", ramp_file, out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "analysis", "slope", ramp_file, out])

    def test_overwrite_replaces(self, ramp_file, tmp_path):
        out = str(tmp_path / "slope.nc")
        assert _run(["mesh", "analysis", "slope", ramp_file, out]) == 0
        assert _run(["mesh", "analysis", "slope", ramp_file, out, "--overwrite"]) == 0

    def test_unknown_variable_is_a_clean_error(self, ramp_file, tmp_path):
        out = str(tmp_path / "slope.nc")
        with pytest.raises(ValueError):
            _run(["mesh", "analysis", "slope", ramp_file, out, "--var", "nope"])


class TestMorphologyWriters:
    """``sieve`` / ``cluster`` / ``binary-*`` write a mesh."""

    def test_cluster_labels_two_pools(self, mesh_file, tmp_path):
        out = str(tmp_path / "cl.nc")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "cluster",
                    mesh_file,
                    out,
                    "--lower",
                    "1",
                    "--upper",
                    "10",
                ]
            )
            == 0
        )
        labels = MeshDataset.read_file(out)["v"].data
        assert sorted(set(labels.tolist())) == [0, 1, 2]

    def test_sieve_writes_a_mesh(self, ramp_file, tmp_path):
        out = str(tmp_path / "sv.nc")
        assert (
            _run(["mesh", "analysis", "sieve", ramp_file, out, "--threshold", "1"]) == 0
        )
        assert os.path.exists(out)

    def test_binary_dilation_writes_a_mesh(self, mask_file, tmp_path):
        out = str(tmp_path / "dil.nc")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "binary-dilation",
                    mask_file,
                    out,
                    "--iterations",
                    "1",
                ]
            )
            == 0
        )
        assert os.path.exists(out)

    def test_binary_erosion_writes_a_mesh(self, mask_file, tmp_path):
        out = str(tmp_path / "ero.nc")
        assert _run(["mesh", "analysis", "binary-erosion", mask_file, out]) == 0
        assert os.path.exists(out)


class TestSummaries:
    """Printing / vector-summary verbs."""

    def test_domain_area_json(self, ramp_file, capsys):
        rc, payload = _run_capture(
            ["mesh", "analysis", "domain-area", ramp_file, "--json"], capsys
        )
        assert rc == 0
        assert payload["area"] == pytest.approx(4.0)

    def test_domain_area_text(self, ramp_file, capsys):
        assert _run(["mesh", "analysis", "domain-area", ramp_file]) == 0
        assert "area:" in capsys.readouterr().out

    def test_count_domain_cells_json(self, ramp_file, capsys):
        rc, payload = _run_capture(
            ["mesh", "analysis", "count-domain-cells", ramp_file, "--json"], capsys
        )
        assert rc == 0
        assert payload["count"] == 4

    def test_histogram_json(self, ramp_file, capsys):
        rc, records = _run_capture(
            ["mesh", "analysis", "histogram", ramp_file, "--bins", "4", "--json"],
            capsys,
        )
        assert rc == 0
        assert sum(r["count"] for r in records) == 4

    def test_histogram_csv(self, ramp_file, tmp_path):
        out = str(tmp_path / "hist.csv")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "histogram",
                    ramp_file,
                    "--bins",
                    "4",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        frame = pd.read_csv(out)
        assert list(frame.columns) == ["bin_left", "bin_right", "count"]

    def test_histogram_range(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "histogram",
                ramp_file,
                "--bins",
                "2",
                "--range",
                "0",
                "4",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert len(records) == 2

    def test_zonal_stats_json(self, ramp_file, zones_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "zonal-stats",
                ramp_file,
                zones_file,
                "--var",
                "v",
                "--stats",
                "mean",
                "sum",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert "mean" in records[0]
        assert "sum" in records[0]

    def test_zonal_stats_csv(self, ramp_file, zones_file, tmp_path):
        out = str(tmp_path / "zs.csv")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "zonal-stats",
                    ramp_file,
                    zones_file,
                    "--var",
                    "v",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert "mean" in pd.read_csv(out).columns

    def test_zonal_stats_unweighted(self, ramp_file, zones_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "zonal-stats",
                ramp_file,
                zones_file,
                "--var",
                "v",
                "--unweighted",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert "mean" in records[0]

    def test_zonal_stats_reprojects_mismatched_zones(self, ramp_file, tmp_path, capsys):
        """Zones in a different CRS are reprojected onto the mesh CRS before summarising."""
        mesh = MeshDataset.read_file(ramp_file)
        mesh.set_crs(epsg=4326)
        geo_mesh = str(tmp_path / "geo.nc")
        mesh.to_file(geo_mesh)
        zones = str(tmp_path / "zones3857.geojson")
        poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        FeatureCollection(gpd.GeoDataFrame(geometry=[poly], crs=3857)).to_file(
            zones, driver="geojson"
        )
        rc = _run(
            ["mesh", "analysis", "zonal-stats", geo_mesh, zones, "--var", "v", "--json"]
        )
        assert rc == 0

    def test_histogram_prints_records_as_text(self, ramp_file, capsys):
        """Without --json or --output the records print one per line."""
        assert _run(["mesh", "analysis", "histogram", ramp_file, "--bins", "2"]) == 0
        assert "count" in capsys.readouterr().out

    def test_footprint_json(self, ramp_file, capsys):
        rc, payload = _run_capture(
            ["mesh", "analysis", "footprint", ramp_file, "--json"], capsys
        )
        assert rc == 0
        assert payload["area"] == pytest.approx(4.0)
        assert payload["wkt"].startswith("POLYGON")

    def test_footprint_vector(self, ramp_file, tmp_path):
        out = str(tmp_path / "fp.geojson")
        assert _run(["mesh", "analysis", "footprint", ramp_file, "--output", out]) == 0
        assert len(FeatureCollection.read_file(out)) == 1

    def test_footprint_vector_refuses_existing(self, ramp_file, tmp_path):
        out = str(tmp_path / "fp.geojson")
        assert _run(["mesh", "analysis", "footprint", ramp_file, "--output", out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "analysis", "footprint", ramp_file, "--output", out])


class TestSampling:
    """Point / line / random sampling verbs."""

    def test_sample_point_json(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample-point",
                ramp_file,
                "--x",
                "1.0",
                "--y",
                "0.5",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert "value" in records[0]

    def test_sample_point_outside_is_nan(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample-point",
                ramp_file,
                "--x",
                "-100",
                "--y",
                "-100",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert records[0]["value"] is None  # NaN scrubbed to null by json_sanitize

    def test_sample_point_csv(self, ramp_file, tmp_path):
        out = str(tmp_path / "sp.csv")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "sample-point",
                    ramp_file,
                    "--x",
                    "1.0",
                    "2.0",
                    "--y",
                    "0.5",
                    "0.5",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert list(pd.read_csv(out).columns) == ["x", "y", "value"]

    def test_sample_json(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample",
                ramp_file,
                "--var",
                "v",
                "--x",
                "1.0",
                "--y",
                "0.5",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert len(records) == 1

    def test_sample_nearest_method(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample",
                ramp_file,
                "--var",
                "v",
                "--x",
                "1.0",
                "--y",
                "0.5",
                "--method",
                "nearest",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert "value" in records[0]

    def test_sample_csv(self, ramp_file, tmp_path):
        out = str(tmp_path / "s.csv")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "sample",
                    ramp_file,
                    "--var",
                    "v",
                    "--x",
                    "1.0",
                    "--y",
                    "0.5",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert list(pd.read_csv(out).columns) == ["x", "y", "value"]

    def test_sample_line_csv(self, ramp_file, tmp_path):
        out = str(tmp_path / "sl.csv")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "sample-line",
                    ramp_file,
                    "--coords",
                    "0",
                    "0.5",
                    "5",
                    "0.5",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert list(pd.read_csv(out).columns) == ["distance", "value", "face_index"]

    def test_sample_line_json(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample-line",
                ramp_file,
                "--coords",
                "0",
                "0.5",
                "5",
                "0.5",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert all({"distance", "value", "face_index"} <= set(r) for r in records)

    def test_sample_line_rejects_odd_coords(self, ramp_file):
        with pytest.raises(ValueError, match="even number"):
            _run(
                [
                    "mesh",
                    "analysis",
                    "sample-line",
                    ramp_file,
                    "--coords",
                    "0",
                    "0.5",
                    "5",
                ]
            )

    def test_sample_line_rejects_single_point(self, ramp_file):
        with pytest.raises(ValueError, match="pairs"):
            _run(["mesh", "analysis", "sample-line", ramp_file, "--coords", "0", "0.5"])

    def test_sample_random_json(self, ramp_file, capsys):
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample-random",
                ramp_file,
                "-n",
                "5",
                "--seed",
                "1",
                "--json",
            ],
            capsys,
        )
        assert rc == 0
        assert len(records) == 5

    def test_sample_random_long_n_flag(self, ramp_file, capsys):
        """--n is accepted as the long spelling of -n (parity with head/tail) (N2)."""
        rc, records = _run_capture(
            [
                "mesh",
                "analysis",
                "sample-random",
                ramp_file,
                "--n",
                "4",
                "--seed",
                "1",
                "--json",
            ],
            capsys,
        )
        assert rc == 0, "the --n long flag must be accepted"
        assert len(records) == 4

    def test_sample_random_csv(self, ramp_file, tmp_path):
        out = str(tmp_path / "sr.csv")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "sample-random",
                    ramp_file,
                    "-n",
                    "3",
                    "--seed",
                    "2",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert len(pd.read_csv(out)) == 3


class TestContour:
    """``contour`` traces iso-value lines to a vector file."""

    @pytest.mark.plot
    def test_contour_writes_lines(self, ramp_file, tmp_path):
        out = str(tmp_path / "contour.geojson")
        assert (
            _run(["mesh", "analysis", "contour", ramp_file, out, "--levels", "3"]) == 0
        )
        assert len(FeatureCollection.read_file(out)) >= 1

    @pytest.mark.plot
    def test_contour_explicit_levels(self, ramp_file, tmp_path):
        out = str(tmp_path / "contour.geojson")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "contour",
                    ramp_file,
                    out,
                    "--levels",
                    "1.5",
                    "2.5",
                    "3.5",
                ]
            )
            == 0
        )
        assert os.path.exists(out)

    @pytest.mark.plot
    def test_contour_default_levels(self, ramp_file, tmp_path):
        out = str(tmp_path / "contour.geojson")
        assert _run(["mesh", "analysis", "contour", ramp_file, out]) == 0
        assert os.path.exists(out)

    def test_contour_unknown_driver_is_refused(self, ramp_file, tmp_path):
        out = str(tmp_path / "contour.weird")
        with pytest.raises(ValueError, match="driver"):
            _run(["mesh", "analysis", "contour", ramp_file, out, "--levels", "3"])


class TestPolyfit:
    """``mesh polyfit`` — per-element polynomial trend along time (#1479)."""

    @staticmethod
    def _ramp_file(tmp_path):
        """A written four-face mesh with a 3-step temporal ramp on variable ``v``."""
        ds = MeshDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={
                "v": np.array(
                    [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]
                )
            },
            data_locations={"v": "face"},
        )
        ds["v"].attributes = {**ds["v"].attributes, "time_values": [0.0, 1.0, 2.0]}
        path = tmp_path / "ramp_t.nc"
        ds.to_file(str(path))
        return str(path)

    def test_polyfit_writes_a_coefficient_mesh(self, tmp_path):
        """A degree-1 fit writes a mesh whose time axis became a coefficient axis."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "pf.nc")

        assert _run(["mesh", "analysis", "polyfit", src, out, "--degree", "1"]) == 0

        back = MeshDataset.read_file(out)
        assert back["v"].dimensions[0] == "coefficient", f"got {back['v'].dimensions}"
        assert np.asarray(back["v"].data).shape == (2, 4)

    def test_polyfit_requires_a_degree(self, tmp_path):
        """`--degree` is required; argparse rejects the call without it."""
        src = self._ramp_file(tmp_path)
        with pytest.raises(SystemExit):
            _run(["mesh", "analysis", "polyfit", src, str(tmp_path / "pf.nc")])

    def test_polyfit_refuses_an_existing_output(self, tmp_path):
        """Without `--overwrite`, writing over an existing file is refused (R1 N4)."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "pf.nc")
        assert _run(["mesh", "analysis", "polyfit", src, out, "--degree", "1"]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "analysis", "polyfit", src, out, "--degree", "1"])


class TestFocalReduce:
    """``mesh focal-reduce`` — neighbourhood aggregated with a named reducer (#1542)."""

    def test_writes_a_reducer_named_variable(self, mesh_file, tmp_path):
        """The output mesh carries a ``focal_<reducer>`` face variable."""
        out = str(tmp_path / "fr.nc")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "focal-reduce",
                    mesh_file,
                    out,
                    "--reducer",
                    "median",
                ]
            )
            == 0
        )
        assert "focal_median" in MeshDataset.read_file(out).data_variable_names

    @pytest.mark.parametrize("reducer", ["median", "min", "max", "sum", "range", "var"])
    def test_every_named_reducer_runs(self, mesh_file, reducer, tmp_path):
        """Each registered reducer writes its own named variable.

        Args:
            mesh_file: A written four-face mesh.
            reducer: The reducer name under test.
            tmp_path: pytest temp directory.
        """
        out = str(tmp_path / f"{reducer}.nc")
        assert (
            _run(
                [
                    "mesh",
                    "analysis",
                    "focal-reduce",
                    mesh_file,
                    out,
                    "--reducer",
                    reducer,
                ]
            )
            == 0
        )
        assert f"focal_{reducer}" in MeshDataset.read_file(out).data_variable_names

    def test_median_values_match_a_two_ring_neighbourhood(self, mesh_file, tmp_path):
        """``median`` over two rings on ``[5, 0, 0, 7]`` yields the known per-face medians."""
        out = str(tmp_path / "fr.nc")
        _run(
            [
                "mesh",
                "analysis",
                "focal-reduce",
                mesh_file,
                out,
                "--reducer",
                "median",
                "--rings",
                "2",
            ]
        )
        values = np.asarray(MeshDataset.read_file(out)["focal_median"].data)
        assert np.allclose(values, [0.0, 2.5, 2.5, 0.0]), f"got {values}"

    def test_requires_a_reducer(self, mesh_file, tmp_path):
        """``--reducer`` is required; argparse rejects the call without it."""
        with pytest.raises(SystemExit):
            _run(
                ["mesh", "analysis", "focal-reduce", mesh_file, str(tmp_path / "x.nc")]
            )

    def test_rejects_an_unknown_reducer(self, mesh_file, tmp_path):
        """An unregistered reducer name is rejected by argparse's ``choices``."""
        with pytest.raises(SystemExit):
            _run(
                [
                    "mesh",
                    "analysis",
                    "focal-reduce",
                    mesh_file,
                    str(tmp_path / "x.nc"),
                    "--reducer",
                    "nope",
                ]
            )

    def test_refuses_an_existing_output(self, mesh_file, tmp_path):
        """Without ``--overwrite``, writing over an existing file is refused."""
        out = str(tmp_path / "fr.nc")
        _run(["mesh", "analysis", "focal-reduce", mesh_file, out, "--reducer", "max"])
        with pytest.raises(ValueError, match="already exists"):
            _run(
                ["mesh", "analysis", "focal-reduce", mesh_file, out, "--reducer", "max"]
            )


class TestCurvefit:
    """``mesh curvefit`` — per-element fit of a named model along time (#1542)."""

    @staticmethod
    def _ramp_file(tmp_path):
        """A written four-face mesh with a 3-step temporal ramp on variable ``v``."""
        ds = MeshDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={
                "v": np.array(
                    [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]
                )
            },
            data_locations={"v": "face"},
        )
        ds["v"].attributes = {**ds["v"].attributes, "time_values": [0.0, 1.0, 2.0]}
        path = tmp_path / "ramp_t.nc"
        ds.to_file(str(path))
        return str(path)

    def test_writes_a_parameter_axis(self, tmp_path):
        """A fit replaces the time axis with a ``parameter`` axis of the model's arity."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "cf.nc")
        assert (
            _run(["mesh", "analysis", "curvefit", src, out, "--model", "linear"]) == 0
        )
        back = MeshDataset.read_file(out)
        assert back["v"].dimensions[0] == "parameter", f"got {back['v'].dimensions}"
        assert np.asarray(back["v"].data).shape == (2, 4)

    def test_linear_recovers_the_slope(self, tmp_path):
        """A straight-line fit of the exact ramp recovers each face's slope (param 0)."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "cf.nc")
        _run(["mesh", "analysis", "curvefit", src, out, "--model", "linear"])
        slope = np.asarray(MeshDataset.read_file(out)["v"].data)[0]
        assert np.allclose(slope, [1.0, 2.0, 3.0, 4.0]), f"got {slope}"

    def test_gaussian_has_three_parameters(self, tmp_path):
        """A three-parameter model yields a length-3 ``parameter`` axis."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "cf.nc")
        assert (
            _run(["mesh", "analysis", "curvefit", src, out, "--model", "gaussian"]) == 0
        )
        assert np.asarray(MeshDataset.read_file(out)["v"].data).shape[0] == 3

    def test_p0_is_parsed_and_accepted(self, tmp_path):
        """A comma-separated ``--p0`` seeds the fit without error."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "cf.nc")
        rc = _run(
            [
                "mesh",
                "analysis",
                "curvefit",
                src,
                out,
                "--model",
                "exponential",
                "--p0",
                "1,0.1",
            ]
        )
        assert rc == 0

    def test_requires_a_model(self, tmp_path):
        """``--model`` is required; argparse rejects the call without it."""
        src = self._ramp_file(tmp_path)
        with pytest.raises(SystemExit):
            _run(["mesh", "analysis", "curvefit", src, str(tmp_path / "cf.nc")])

    def test_rejects_an_unknown_model(self, tmp_path):
        """An unregistered model name is rejected by argparse's ``choices``."""
        src = self._ramp_file(tmp_path)
        with pytest.raises(SystemExit):
            _run(
                [
                    "mesh",
                    "analysis",
                    "curvefit",
                    src,
                    str(tmp_path / "cf.nc"),
                    "--model",
                    "nope",
                ]
            )

    def test_refuses_an_existing_output(self, tmp_path):
        """Without ``--overwrite``, writing over an existing file is refused."""
        src = self._ramp_file(tmp_path)
        out = str(tmp_path / "cf.nc")
        _run(["mesh", "analysis", "curvefit", src, out, "--model", "linear"])
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "analysis", "curvefit", src, out, "--model", "linear"])
