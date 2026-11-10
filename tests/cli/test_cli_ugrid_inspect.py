"""Tests for the ``pyramids mesh`` inspection verbs (UGRID).

Print verbs (``summary``/``mesh-names``/``keys``/``items``/``values``/``get``),
geometry verbs that write a vector file (``exterior-edges``/``exterior-faces``/
``circumcenters``/``bounding-polygon``), and array verbs that summarise and
optionally persist an array (``sparse-adjacency``/``connected-components``/
``active-mask``/``masked-values``). The registrator is wired onto a throwaway
parser so these tests do not depend on the top-level CLI assembly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import load_npz

from pyramids._cli_groups import add_group
from pyramids.cli_ugrid_inspect import add_ugrid_inspect_commands
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _run(argv: list[str]) -> int:
    """Parse ``argv`` against a throwaway ``mesh`` parser and run the handler."""
    parser = argparse.ArgumentParser(prog="pyramids")
    sub = parser.add_subparsers(dest="group")
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    advanced = add_group(
        msub, "advanced", "power-user primitives (accessors, array/dim mechanics)"
    )
    add_ugrid_inspect_commands(
        add_group(msub, "inspect", "topology, connectivity, variable metadata"),
        advanced,
    )
    args = parser.parse_args(argv)
    return args.func(args)


def _triangle_mesh(tmp_path) -> str:
    """Write a 2-triangle mesh NetCDF (triangular, for circumcenters)."""
    mesh = MeshDataset.from_arrays(
        np.array([0.0, 1.0, 0.5, 1.5, 2.0]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        np.array([[0, 1, 2], [1, 4, 3]], dtype=np.intp),
        epsg=4326,
    )
    path = str(tmp_path / "tri.nc")
    mesh.to_file(path)
    return path


class TestSummary:
    """``pyramids mesh summary``."""

    def test_json(self, capsys):
        """--json reports one record per variable with the default metrics."""
        assert _run(["mesh", "inspect", "summary", str(_MESH), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        names = {record["variable"] for record in payload["summary"]}
        assert {"mesh2d_node_z", "mesh2d_edge_type"} <= names

    def test_plain_subset_and_options(self, capsys):
        """--variables / --metrics / --no-skipna / --ddof all flow through."""
        rc = _run(
            [
                "mesh",
                "inspect",
                "summary",
                str(_MESH),
                "--variables",
                "mesh2d_edge_type",
                "--metrics",
                "min",
                "max",
                "--no-skipna",
                "--ddof",
                "1",
            ]
        )
        assert rc == 0
        assert "mesh2d_edge_type" in capsys.readouterr().out

    def test_unknown_variable_errors(self):
        """An unknown variable is a clean (mapped) error, not a KeyError."""
        with pytest.raises(ValueError):
            _run(["mesh", "inspect", "summary", str(_MESH), "--variables", "nope"])


class TestMeshNames:
    """``pyramids mesh mesh-names``."""

    def test_json(self, capsys):
        """--json lists the topology names."""
        assert _run(["mesh", "inspect", "mesh-names", str(_MESH), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["mesh_names"] == ["mesh2d"]

    def test_plain(self, capsys):
        """The human form prints one name per line."""
        assert _run(["mesh", "inspect", "mesh-names", str(_MESH)]) == 0
        assert "mesh2d" in capsys.readouterr().out


class TestGet:
    """``pyramids mesh get``."""

    def test_json(self, capsys):
        """get describes a single variable's metadata."""
        assert (
            _run(["mesh", "advanced", "get", str(_MESH), "mesh2d_edge_type", "--json"])
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "mesh2d_edge_type"
        assert payload["location"] == "edge"

    def test_plain(self, capsys):
        """The human form prints key: value lines."""
        assert _run(["mesh", "advanced", "get", str(_MESH), "mesh2d_node_z"]) == 0
        assert "location:" in capsys.readouterr().out

    def test_unknown_variable_errors(self):
        """An unknown variable is a clean (mapped) error."""
        with pytest.raises(ValueError):
            _run(["mesh", "advanced", "get", str(_MESH), "nope"])


class TestKeysItemsValues:
    """``pyramids mesh keys|items|values``."""

    def test_keys_json(self, capsys):
        """keys lists the variable names."""
        assert _run(["mesh", "advanced", "keys", str(_MESH), "--json"]) == 0
        assert "mesh2d_edge_type" in json.loads(capsys.readouterr().out)["keys"]

    def test_keys_plain(self, capsys):
        """keys plain form prints one name per line."""
        assert _run(["mesh", "advanced", "keys", str(_MESH)]) == 0
        assert "mesh2d_node_z" in capsys.readouterr().out

    def test_items_json(self, capsys):
        """items describes every (name, variable) pair."""
        assert _run(["mesh", "advanced", "items", str(_MESH), "--json"]) == 0
        items = json.loads(capsys.readouterr().out)["items"]
        assert {record["name"] for record in items} == {
            "mesh2d_node_z",
            "mesh2d_edge_type",
        }

    def test_items_plain(self, capsys):
        """items plain form prints one record per variable."""
        assert _run(["mesh", "advanced", "items", str(_MESH)]) == 0
        assert "mesh2d_node_z" in capsys.readouterr().out

    def test_values_json(self, capsys):
        """values describes every variable (no key)."""
        assert _run(["mesh", "advanced", "values", str(_MESH), "--json"]) == 0
        values = json.loads(capsys.readouterr().out)["values"]
        assert len(values) == 2

    def test_values_plain(self, capsys):
        """values plain form prints one record per variable."""
        assert _run(["mesh", "advanced", "values", str(_MESH)]) == 0
        assert "location" in capsys.readouterr().out


class TestGeometryVerbs:
    """Geometry verbs that write a vector file."""

    def test_exterior_edges(self, tmp_path):
        """exterior-edges writes the boundary edges as LineStrings."""
        out = str(tmp_path / "edges.geojson")
        assert _run(["mesh", "inspect", "exterior-edges", str(_MESH), out]) == 0
        fc = FeatureCollection.read_file(out)
        assert len(fc) > 0
        assert set(fc.geometry.geom_type) == {"LineString"}

    def test_exterior_faces(self, tmp_path):
        """exterior-faces writes the boundary faces as Polygons."""
        out = str(tmp_path / "faces.geojson")
        assert _run(["mesh", "inspect", "exterior-faces", str(_MESH), out]) == 0
        fc = FeatureCollection.read_file(out)
        assert len(fc) > 0
        assert set(fc.geometry.geom_type) == {"Polygon"}

    def test_bounding_polygon(self, tmp_path):
        """bounding-polygon writes the single outline polygon."""
        out = str(tmp_path / "bp.geojson")
        assert _run(["mesh", "inspect", "bounding-polygon", str(_MESH), out]) == 0
        fc = FeatureCollection.read_file(out)
        assert set(fc.geometry.geom_type) == {"Polygon"}

    def test_circumcenters_triangular(self, tmp_path):
        """circumcenters writes one point per triangular face."""
        out = str(tmp_path / "cc.geojson")
        assert (
            _run(["mesh", "inspect", "circumcenters", _triangle_mesh(tmp_path), out])
            == 0
        )
        fc = FeatureCollection.read_file(out)
        assert len(fc) == 2
        assert set(fc.geometry.geom_type) == {"Point"}

    def test_circumcenters_non_triangular_errors(self, tmp_path):
        """circumcenters on a non-triangular mesh is a clean error."""
        with pytest.raises(ValueError, match="triangular"):
            _run(
                [
                    "mesh",
                    "inspect",
                    "circumcenters",
                    str(_MESH),
                    str(tmp_path / "x.geojson"),
                ]
            )

    def test_refuses_existing(self, tmp_path):
        """A second geometry write without --overwrite is refused."""
        out = str(tmp_path / "bp.geojson")
        assert _run(["mesh", "inspect", "bounding-polygon", str(_MESH), out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "inspect", "bounding-polygon", str(_MESH), out])

    def test_unknown_extension_errors(self, tmp_path):
        """An unknown output extension with no --driver is refused."""
        with pytest.raises(ValueError, match="cannot infer a vector driver"):
            _run(
                [
                    "mesh",
                    "inspect",
                    "bounding-polygon",
                    str(_MESH),
                    str(tmp_path / "o.weird"),
                ]
            )


class TestSparseAdjacency:
    """``pyramids mesh sparse-adjacency``."""

    def test_json(self, capsys):
        """--json reports the matrix kind, shape, and non-zero count."""
        assert _run(["mesh", "advanced", "sparse-adjacency", str(_MESH), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["kind"] == "face"
        assert payload["nnz"] > 0

    def test_plain(self, capsys):
        """The human form prints key: value lines."""
        assert _run(["mesh", "advanced", "sparse-adjacency", str(_MESH)]) == 0
        assert "nnz:" in capsys.readouterr().out

    def test_writes_npz(self, tmp_path):
        """--output persists the matrix as a .npz readable by scipy."""
        out = str(tmp_path / "adj.npz")
        assert (
            _run(["mesh", "advanced", "sparse-adjacency", str(_MESH), "--output", out])
            == 0
        )
        assert load_npz(out).nnz > 0

    def test_refuses_existing(self, tmp_path):
        """A second --output write without --overwrite is refused."""
        out = str(tmp_path / "adj.npz")
        assert (
            _run(["mesh", "advanced", "sparse-adjacency", str(_MESH), "--output", out])
            == 0
        )
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "advanced", "sparse-adjacency", str(_MESH), "--output", out])


class TestConnectedComponents:
    """``pyramids mesh connected-components``."""

    def test_json(self, capsys):
        """--json reports the element and component counts."""
        assert (
            _run(["mesh", "advanced", "connected-components", str(_MESH), "--json"])
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_components"] >= 1
        assert payload["n_elements"] > 0

    def test_plain(self, capsys):
        """The human form prints key: value lines."""
        assert _run(["mesh", "advanced", "connected-components", str(_MESH)]) == 0
        assert "n_components:" in capsys.readouterr().out

    def test_writes_npy(self, tmp_path):
        """--output persists the label array."""
        out = str(tmp_path / "labels.npy")
        assert (
            _run(
                [
                    "mesh",
                    "advanced",
                    "connected-components",
                    str(_MESH),
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert np.load(out).size > 0


class TestActiveMask:
    """``pyramids mesh active-mask``."""

    def test_none_json(self, capsys):
        """A variable with no active mask reports null under --json."""
        rc = _run(
            [
                "mesh",
                "advanced",
                "active-mask",
                str(_MESH),
                "mesh2d_edge_type",
                "--json",
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["active_mask"] is None

    def test_none_plain(self, capsys):
        """The human form prints the null mask as key: value lines."""
        assert (
            _run(["mesh", "advanced", "active-mask", str(_MESH), "mesh2d_edge_type"])
            == 0
        )
        assert "active_mask:" in capsys.readouterr().out

    def test_output_without_mask_errors(self, tmp_path):
        """--output on a variable with no mask is a clean error."""
        with pytest.raises(ValueError, match="no active mask"):
            _run(
                [
                    "mesh",
                    "advanced",
                    "active-mask",
                    str(_MESH),
                    "mesh2d_edge_type",
                    "--output",
                    str(tmp_path / "m.npy"),
                ]
            )

    def test_unknown_variable_errors(self):
        """An unknown variable is a clean (mapped) error."""
        with pytest.raises(ValueError):
            _run(["mesh", "advanced", "active-mask", str(_MESH), "nope"])

    def _masked_mesh(self):
        """Build an in-memory mesh whose ``depth`` variable carries an active mask."""
        mesh = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"depth": np.array([2.0, 4.0])},
        )
        mesh["depth"].active = np.array([True, False])
        return mesh

    def test_present_json(self, capsys, monkeypatch):
        """A variable with a mask reports its active-cell count under --json."""
        monkeypatch.setattr(
            MeshDataset, "read_file", lambda *a, **k: self._masked_mesh()
        )
        assert (
            _run(["mesh", "advanced", "active-mask", "ignored.nc", "depth", "--json"])
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_elements"] == 2
        assert payload["n_active"] == 1

    def test_present_writes_npy(self, tmp_path, monkeypatch):
        """--output persists the boolean mask when the variable has one."""
        monkeypatch.setattr(
            MeshDataset, "read_file", lambda *a, **k: self._masked_mesh()
        )
        out = str(tmp_path / "mask.npy")
        assert (
            _run(
                [
                    "mesh",
                    "advanced",
                    "active-mask",
                    "ignored.nc",
                    "depth",
                    "--output",
                    out,
                ]
            )
            == 0
        )
        assert np.load(out).tolist() == [True, False]


class TestMaskedValues:
    """``pyramids mesh masked-values``."""

    def test_json(self, capsys):
        """--json reports the element and finite-value counts."""
        rc = _run(
            [
                "mesh",
                "advanced",
                "masked-values",
                str(_MESH),
                "mesh2d_edge_type",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_elements"] > 0
        assert payload["n_finite"] >= 0

    def test_plain(self, capsys):
        """The human form prints key: value lines."""
        assert (
            _run(["mesh", "advanced", "masked-values", str(_MESH), "mesh2d_edge_type"])
            == 0
        )
        assert "n_finite:" in capsys.readouterr().out

    def test_writes_npy(self, tmp_path):
        """--output persists the masked value array."""
        out = str(tmp_path / "vals.npy")
        rc = _run(
            [
                "mesh",
                "advanced",
                "masked-values",
                str(_MESH),
                "mesh2d_edge_type",
                "--output",
                out,
            ]
        )
        assert rc == 0
        assert np.load(out).size > 0

    def test_unknown_variable_errors(self):
        """An unknown variable is a clean (mapped) error."""
        with pytest.raises(ValueError):
            _run(["mesh", "advanced", "masked-values", str(_MESH), "nope"])
