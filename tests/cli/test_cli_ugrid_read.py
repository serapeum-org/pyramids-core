"""Tests for the ``pyramids mesh`` read / construct verbs (UGRID).

Each verb is a thin wrapper over a :class:`pyramids.netcdf.ugrid.MeshDataset`
constructor classmethod that writes a UGRID NetCDF. The registrator is wired onto
a throwaway parser (the orchestrator wires it into ``add_ugrid_commands`` in
production), so these tests do not depend on the top-level CLI assembly.
"""

from __future__ import annotations

import argparse
import io
import types
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from pyramids._cli_groups import add_group
from pyramids.base.georeference import GeoReference
from pyramids.cli_ugrid_read import _parse_kv, add_ugrid_read_commands
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _run(argv: list[str]) -> int:
    """Parse ``argv`` against a throwaway ``mesh`` parser and run the handler."""
    parser = argparse.ArgumentParser(prog="pyramids")
    sub = parser.add_subparsers(dest="group")
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_read_commands(add_group(msub, "read", "read/open a mesh from a source"))
    args = parser.parse_args(argv)
    return args.func(args)


def _triangle_arrays(tmp_path) -> tuple[str, str, str]:
    """Write node_x / node_y / faces ``.npy`` files for a 2-triangle mesh."""
    node_x = tmp_path / "nx.npy"
    node_y = tmp_path / "ny.npy"
    faces = tmp_path / "faces.npy"
    np.save(node_x, np.array([0.0, 1.0, 0.5, 1.5, 2.0]))
    np.save(node_y, np.array([0.0, 0.0, 1.0, 1.0, 0.0]))
    np.save(faces, np.array([[0, 1, 2], [1, 4, 3]], dtype=np.intp))
    return str(node_x), str(node_y), str(faces)


class TestParseKv:
    """``_parse_kv`` helper."""

    def test_parses_pairs(self):
        """NAME=VALUE tokens become a mapping; None is an empty mapping."""
        assert _parse_kv(["a=1", "b=2"]) == {"a": "1", "b": "2"}
        assert _parse_kv(None) == {}

    def test_rejects_bad_token(self):
        """A token without '=' is a clean ValueError."""
        with pytest.raises(ValueError, match="NAME=VALUE"):
            _parse_kv(["noequals"])


class TestReadAll:
    """``pyramids mesh read-all``."""

    def test_writes_each_topology(self, tmp_path):
        """read-all writes one NetCDF per topology into the output directory."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        assert _run(["mesh", "read", "read-all", str(_MESH), str(out_dir)]) == 0
        assert (out_dir / "mesh2d.nc").exists()

    def test_eager_rejected_by_native_reader(self, tmp_path):
        """--eager forwards lazy=False; the built-in UGRID reader rejects it cleanly.

        The flag exists for driver-backed formats that honour an eager read; the
        native NetCDF reader always defers values, so it raises a clean TypeError
        (which the CLI's main() maps to a one-line error / exit 1).
        """
        out_dir = tmp_path / "eager"
        out_dir.mkdir()
        with pytest.raises(TypeError, match="lazy=False"):
            _run(["mesh", "read", "read-all", str(_MESH), str(out_dir), "--eager"])

    def test_refuses_existing(self, tmp_path):
        """A second read-all without --overwrite is refused."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        assert _run(["mesh", "read", "read-all", str(_MESH), str(out_dir)]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "read", "read-all", str(_MESH), str(out_dir)])


class TestFromArrays:
    """``pyramids mesh read arrays``."""

    def test_builds_from_arrays(self, tmp_path):
        """read arrays with a per-face variable writes a readable mesh."""
        node_x, node_y, faces = _triangle_arrays(tmp_path)
        values = tmp_path / "z.npy"
        np.save(values, np.array([10.0, 20.0]))
        out = str(tmp_path / "mesh.nc")
        rc = _run(
            [
                "mesh",
                "read",
                "arrays",
                node_x,
                node_y,
                faces,
                out,
                "--data",
                f"z={values}",
                "--data-location",
                "z=face",
            ]
        )
        assert rc == 0
        mesh = MeshDataset.read_file(out)
        assert mesh.n_face == 2
        assert "z" in mesh.keys()

    def test_builds_without_data(self, tmp_path):
        """read arrays with no --data builds a bare topology."""
        node_x, node_y, faces = _triangle_arrays(tmp_path)
        out = str(tmp_path / "bare.nc")
        assert _run(["mesh", "read", "arrays", node_x, node_y, faces, out]) == 0
        assert MeshDataset.read_file(out).n_face == 2

    def test_refuses_existing(self, tmp_path):
        """A second read arrays without --overwrite is refused."""
        node_x, node_y, faces = _triangle_arrays(tmp_path)
        out = str(tmp_path / "m.nc")
        assert _run(["mesh", "read", "arrays", node_x, node_y, faces, out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "read", "arrays", node_x, node_y, faces, out])


class TestFromGeodataframe:
    """``pyramids mesh read geodataframe``."""

    def _polygons(self, tmp_path) -> str:
        """Write a two-polygon GeoJSON layer with a CRS."""
        gdf = gpd.GeoDataFrame(
            {"v": [1.0, 2.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        path = str(tmp_path / "polys.geojson")
        FeatureCollection(gdf).to_file(path, driver="geojson")
        return path

    def test_builds_from_polygons(self, tmp_path):
        """read geodataframe turns polygons into faces."""
        out = str(tmp_path / "mg.nc")
        assert (
            _run(["mesh", "read", "geodataframe", self._polygons(tmp_path), out]) == 0
        )
        assert MeshDataset.read_file(out).n_face == 2

    def test_refuses_existing(self, tmp_path):
        """A second read geodataframe without --overwrite is refused."""
        polygons = self._polygons(tmp_path)
        out = str(tmp_path / "mg.nc")
        assert _run(["mesh", "read", "geodataframe", polygons, out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "read", "geodataframe", polygons, out])


class TestFromStructured:
    """``pyramids mesh read structured``."""

    def _raster(self, tmp_path) -> str:
        """Write a tiny 3x3 raster with a CRS."""
        path = str(tmp_path / "r.tif")
        Dataset.from_array(
            np.arange(9, dtype="float32").reshape(3, 3),
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        ).to_file(path)
        return path

    def test_builds_quad_mesh(self, tmp_path):
        """read structured makes one quad face per raster cell."""
        out = str(tmp_path / "fs.nc")
        assert _run(["mesh", "read", "structured", self._raster(tmp_path), out]) == 0
        assert MeshDataset.read_file(out).n_face == 9

    def test_band_names(self, tmp_path):
        """--band-names labels the per-face variable."""
        out = str(tmp_path / "fsb.nc")
        rc = _run(
            [
                "mesh",
                "read",
                "structured",
                self._raster(tmp_path),
                out,
                "--band-names",
                "depth",
            ]
        )
        assert rc == 0
        assert "depth" in MeshDataset.read_file(out).keys()


class TestFromBytes:
    """``pyramids mesh read bytes``."""

    def test_from_file(self, tmp_path):
        """read bytes reads a file of raw bytes and writes the mesh."""
        out = str(tmp_path / "fb.nc")
        assert _run(["mesh", "read", "bytes", str(_MESH), out]) == 0
        assert MeshDataset.read_file(out).n_node > 0

    def test_from_stdin(self, tmp_path, monkeypatch):
        """--stdin reads the bytes from standard input."""
        data = _MESH.read_bytes()
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(buffer=io.BytesIO(data)))
        out = str(tmp_path / "fbs.nc")
        assert _run(["mesh", "read", "bytes", out, "--stdin"]) == 0
        assert MeshDataset.read_file(out).n_node > 0

    def test_requires_source(self, tmp_path):
        """With neither an input path nor --stdin, read bytes errors cleanly."""
        with pytest.raises(ValueError, match="input path or --stdin"):
            _run(["mesh", "read", "bytes", str(tmp_path / "x.nc")])

    def test_refuses_existing(self, tmp_path):
        """A second read bytes without --overwrite is refused."""
        out = str(tmp_path / "fb.nc")
        assert _run(["mesh", "read", "bytes", str(_MESH), out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "read", "bytes", str(_MESH), out])


class TestOpenMf:
    """``pyramids mesh open-mf``."""

    def test_single_file(self, tmp_path):
        """open-mf over a single file writes a combined mesh."""
        out = str(tmp_path / "omf.nc")
        assert _run(["mesh", "read", "open-mf", str(_MESH), out]) == 0
        assert MeshDataset.read_file(out).n_node > 0

    def test_no_sort(self, tmp_path):
        """--no-sort keeps the given order and still writes."""
        out = str(tmp_path / "omf2.nc")
        assert _run(["mesh", "read", "open-mf", str(_MESH), out, "--no-sort"]) == 0
        assert MeshDataset.read_file(out).n_node > 0

    def test_refuses_existing(self, tmp_path):
        """A second open-mf without --overwrite is refused."""
        out = str(tmp_path / "omf.nc")
        assert _run(["mesh", "read", "open-mf", str(_MESH), out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "read", "open-mf", str(_MESH), out])


class TestFromZarr:
    """``pyramids mesh read zarr``."""

    def _store(self, tmp_path) -> str:
        """Write a pyramids Zarr store from the shipped mesh (needs the [lazy] extra)."""
        store = str(tmp_path / "store.zarr")
        MeshDataset.read_file(_MESH).to_zarr(store)
        return store

    def test_round_trip(self, tmp_path):
        """read zarr reads a store written by to_zarr and writes a NetCDF."""
        out = str(tmp_path / "fz.nc")
        assert _run(["mesh", "read", "zarr", self._store(tmp_path), out]) == 0
        assert MeshDataset.read_file(out).n_node > 0

    def test_refuses_existing(self, tmp_path):
        """A second read zarr without --overwrite is refused."""
        store = self._store(tmp_path)
        out = str(tmp_path / "fz.nc")
        assert _run(["mesh", "read", "zarr", store, out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(["mesh", "read", "zarr", store, out])

    def test_missing_extra_is_clean_error(self, tmp_path, monkeypatch):
        """A missing [lazy] extra (ImportError) is remapped to a clean ValueError."""

        def _boom(*_args, **_kwargs):
            raise ImportError("No module named 'zarr'")

        monkeypatch.setattr(
            MeshDataset, "from_zarr", classmethod(lambda cls, *a, **k: _boom())
        )
        with pytest.raises(ValueError, match=r"\[lazy\] extra"):
            _run(
                [
                    "mesh",
                    "read",
                    "zarr",
                    str(tmp_path / "s.zarr"),
                    str(tmp_path / "o.nc"),
                ]
            )
