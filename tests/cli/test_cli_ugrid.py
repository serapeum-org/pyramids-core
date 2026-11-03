"""Tests for the ``pyramids mesh`` command group (UGRID).

Marked ``lazy``: reading/rasterizing a UGRID mesh goes through the dask-backed
lazy path, which needs the ``[lazy]`` extra, so these run only in the dask-equipped
envs (dev / lazy) — the core-wheel job still covers mesh parser registration via
the ``--help`` surface test. Mirrors the collection-stats gating (#1435).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.lazy

# A small UGRID mesh shipped with the test suite (mesh2d; node/edge data vars).
_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


class TestMeshInfo:
    """`pyramids mesh info`."""

    def test_json_payload(self, capsys):
        """--json reports node/face/edge counts, mesh name, bounds, and variables."""
        assert main(["mesh", "info", str(_MESH), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["mesh_name"] == "mesh2d"
        assert payload["n_node"] > 0
        assert payload["n_face"] > 0
        assert "mesh2d_node_z" in payload["variables"]
        assert len(payload["bounds"]) == 4

    def test_plain_output(self, capsys):
        """The human form prints key: value lines."""
        assert main(["mesh", "info", str(_MESH)]) == 0
        out = capsys.readouterr().out
        assert "n_node:" in out and "mesh_name:" in out


class TestMeshConvert:
    """`pyramids mesh convert`."""

    def test_writes_vector(self, tmp_path):
        """convert writes the mesh faces as a readable vector file."""
        out = str(tmp_path / "mesh.geojson")
        assert main(["mesh", "convert", str(_MESH), out]) == 0
        fc = FeatureCollection.read_file(out)
        assert len(fc) > 0

    def test_refuses_existing(self, tmp_path):
        """A second convert without --overwrite is refused."""
        out = str(tmp_path / "m.geojson")
        assert main(["mesh", "convert", str(_MESH), out]) == 0
        assert main(["mesh", "convert", str(_MESH), out]) == 1

    def test_unknown_extension_errors(self, tmp_path, capsys):
        """An unknown output extension with no --driver exits 1 with a clear error."""
        rc = main(["mesh", "convert", str(_MESH), str(tmp_path / "m.weird")])
        assert rc == 1
        assert "cannot infer a vector driver" in capsys.readouterr().err

    def test_location_node(self, tmp_path):
        """--location node converts the node-located geometries."""
        out = str(tmp_path / "nodes.geojson")
        rc = main(
            [
                "mesh",
                "convert",
                str(_MESH),
                out,
                "--var",
                "mesh2d_node_z",
                "--location",
                "node",
            ]
        )
        assert rc == 0
        assert len(FeatureCollection.read_file(out)) > 0


class TestMeshExtract:
    """`pyramids mesh extract`."""

    def _template(self, tmp_path) -> str:
        """Write a coarse template raster covering the mesh bounds (no CRS)."""
        path = str(tmp_path / "template.tif")
        Dataset.from_array(
            np.zeros((20, 20), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(17325.0, 392034.0), cell_size=3500.0),
        ).to_file(path)
        return path

    def test_rasterizes_onto_template(self, tmp_path):
        """extract rasterizes a mesh variable onto the --like template grid."""
        out = str(tmp_path / "z.tif")
        rc = main(
            [
                "mesh",
                "extract",
                str(_MESH),
                out,
                "--like",
                self._template(tmp_path),
                "--var",
                "mesh2d_node_z",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).band_count >= 1

    def test_refuses_existing(self, tmp_path):
        """A second extract without --overwrite is refused."""
        template = self._template(tmp_path)
        out = str(tmp_path / "z.tif")
        assert (
            main(
                [
                    "mesh",
                    "extract",
                    str(_MESH),
                    out,
                    "--like",
                    template,
                    "--var",
                    "mesh2d_node_z",
                ]
            )
            == 0
        )
        assert (
            main(
                [
                    "mesh",
                    "extract",
                    str(_MESH),
                    out,
                    "--like",
                    template,
                    "--var",
                    "mesh2d_node_z",
                ]
            )
            == 1
        )

    def test_unknown_variable_errors(self, tmp_path, capsys):
        """An unknown --var exits 1 with a one-line error, not a traceback."""
        rc = main(
            [
                "mesh",
                "extract",
                str(_MESH),
                str(tmp_path / "x.tif"),
                "--like",
                self._template(tmp_path),
                "--var",
                "does_not_exist",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert err.startswith("error: ")
        assert "Traceback" not in err
