"""Tests for the ``pyramids mesh`` convert/export commands (issue #1470).

These exercise :func:`pyramids.cli_ugrid_export.add_ugrid_export_commands` on a
throwaway parser, so they pass before the registrator is wired into the top-level
CLI. Each happy path re-opens the written artifact; each error path asserts the
handler raises (``pyramids.cli.main`` turns those into the one-line ``error:``
contract in production).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pyramids.cli_ugrid_export import add_ugrid_export_commands
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset, _drivers
from pyramids.netcdf.ugrid._drivers import MESH_DRIVER_API_VERSION
from pyramids.netcdf.ugrid.models import MeshVariable

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _dispatch(argv: list[str]) -> int:
    """Build a throwaway ``mesh`` parser, register the verbs, and dispatch ``argv``.

    Args:
        argv: The CLI tokens (starting with ``mesh`` or ``--quiet``).

    Returns:
        int: The handler's exit code.
    """
    parser = argparse.ArgumentParser(prog="pyramids")
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="group", required=True)
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_export_commands(msub)
    args = parser.parse_args(argv)
    return args.func(args)


def _face_mesh(tmp_path: Path) -> str:
    """Write a copy of the shipped mesh carrying a face-located variable ``fv``.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        str: Path to the written mesh.
    """
    mesh = MeshDataset.read_file(str(_MESH))
    values = np.arange(mesh.n_face, dtype="float64") % 3
    out = str(tmp_path / "face_mesh.nc")
    mesh.with_variable("fv", values, location="face").to_file(out)
    return out


def _crs_mesh(tmp_path: Path) -> str:
    """Write a copy of the shipped mesh with EPSG:4326 declared.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        str: Path to the written mesh.
    """
    mesh = MeshDataset.read_file(str(_MESH))
    mesh.set_crs(epsg=4326)
    out = str(tmp_path / "crs_mesh.nc")
    mesh.to_file(out)
    return out


def _geometry_file(tmp_path: Path) -> str:
    """Write a single-polygon vector file covering part of the mesh bounds.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        str: Path to the written GeoJSON.
    """
    xmin, ymin, xmax, ymax = MeshDataset.read_file(str(_MESH)).bounds
    poly = box(xmin, ymin, (xmin + xmax) / 2, (ymin + ymax) / 2)
    out = str(tmp_path / "geom.geojson")
    FeatureCollection(gpd.GeoDataFrame({"geometry": [poly]})).to_file(
        out, driver="geojson"
    )
    return out


class _AttachDriver:
    """A minimal results-only driver, injected straight into the registry cache."""

    name = "attach"
    long_name = "Fake Results"
    suffixes = (".res",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"needs_mesh", "read_data"})

    def can_open(self, path: str) -> bool:
        """Accept any path offered.

        Args:
            path: The file to inspect.

        Returns:
            True, always.
        """
        return True

    def read_data(self, path, dataset, *, lazy=True):
        """Return one face-located variable sized to the mesh.

        Args:
            path: The results file.
            dataset: The mesh to attach to.
            lazy: Unused.

        Returns:
            A mapping with a single ``velocity`` variable.
        """
        return {
            "velocity": MeshVariable(
                name="velocity",
                location="face",
                mesh_name=dataset.mesh_name,
                shape=(dataset.n_face,),
                _data=np.zeros(dataset.n_face),
            )
        }


@pytest.fixture
def attach_registered(monkeypatch):
    """Inject the fake ``attach`` driver into the registry cache for one test."""
    monkeypatch.setattr(_drivers, "_CACHE", {"attach": _AttachDriver()})


class TestToFile:
    """`pyramids mesh to-file`."""

    def test_round_trips(self, tmp_path):
        """to-file re-writes a readable UGRID NetCDF."""
        out = str(tmp_path / "copy.nc")
        assert _dispatch(["mesh", "to-file", str(_MESH), out]) == 0
        assert set(MeshDataset.read_file(out).data_variable_names)

    def test_refuses_existing(self, tmp_path):
        """A second write without --overwrite raises."""
        out = str(tmp_path / "c.nc")
        assert _dispatch(["mesh", "to-file", str(_MESH), out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _dispatch(["mesh", "to-file", str(_MESH), out])

    def test_overwrite_allows_replacement(self, tmp_path):
        """--overwrite replaces an existing output."""
        out = str(tmp_path / "c.nc")
        assert _dispatch(["mesh", "to-file", str(_MESH), out]) == 0
        assert _dispatch(["mesh", "to-file", str(_MESH), out, "--overwrite"]) == 0

    def test_quiet_suppresses_chatter(self, tmp_path, capsys):
        """--quiet silences the 'wrote …' line."""
        out = str(tmp_path / "q.nc")
        assert _dispatch(["--quiet", "mesh", "to-file", str(_MESH), out]) == 0
        assert capsys.readouterr().out == ""


class TestToGeodataframe:
    """`pyramids mesh to-geodataframe`."""

    def test_writes_faces(self, tmp_path):
        """The default face location writes polygons."""
        out = str(tmp_path / "faces.geojson")
        assert _dispatch(["mesh", "to-geodataframe", str(_MESH), out]) == 0
        assert set(FeatureCollection.read_file(out).geometry.geom_type) == {"Polygon"}

    def test_writes_nodes_with_var(self, tmp_path):
        """--location node with --var writes points."""
        out = str(tmp_path / "nodes.geojson")
        rc = _dispatch(
            [
                "mesh",
                "to-geodataframe",
                str(_MESH),
                out,
                "--location",
                "node",
                "--var",
                "mesh2d_node_z",
            ]
        )
        assert rc == 0
        assert set(FeatureCollection.read_file(out).geometry.geom_type) == {"Point"}

    def test_unknown_extension_raises(self, tmp_path):
        """An unknown extension with no --driver raises before any read."""
        with pytest.raises(ValueError, match="cannot infer a vector driver"):
            _dispatch(
                ["mesh", "to-geodataframe", str(_MESH), str(tmp_path / "x.weird")]
            )


class TestToDataframe:
    """`pyramids mesh to-dataframe`."""

    def test_writes_csv(self, tmp_path):
        """A CSV is written with one row per element value."""
        out = str(tmp_path / "t.csv")
        rc = _dispatch(
            ["mesh", "to-dataframe", str(_MESH), out, "--variables", "mesh2d_node_z"]
        )
        assert rc == 0
        assert len(pd.read_csv(out)) > 0

    @pytest.mark.parquet
    def test_writes_parquet(self, tmp_path):
        """A .parquet extension is written through the Parquet writer (needs [parquet])."""
        out = str(tmp_path / "t.parquet")
        assert _dispatch(["mesh", "to-dataframe", str(_MESH), out, "--dropna"]) == 0
        assert len(pd.read_parquet(out)) >= 0

    def test_missing_parquet_extra_is_clean(self, tmp_path, monkeypatch):
        """A missing pyarrow is remapped to a clean ValueError naming the extra."""

        def _boom(*args, **kwargs):
            raise ImportError("No module named 'pyarrow'")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
        with pytest.raises(ValueError, match=r"\[parquet\] extra"):
            _dispatch(["mesh", "to-dataframe", str(_MESH), str(tmp_path / "t.parquet")])


class TestToDataset:
    """`pyramids mesh to-dataset`."""

    def test_rasterizes_to_geotiff(self, tmp_path):
        """to-dataset writes a single-band raster."""
        out = str(tmp_path / "grid.tif")
        rc = _dispatch(
            [
                "mesh",
                "to-dataset",
                str(_MESH),
                out,
                "--var",
                "mesh2d_node_z",
                "--cell-size",
                "2000",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).band_count >= 1

    def test_honours_bounds_and_epsg(self, tmp_path):
        """--bounds and --epsg are forwarded to the rasterizer."""
        out = str(tmp_path / "g2.tif")
        xmin, ymin, xmax, ymax = MeshDataset.read_file(str(_MESH)).bounds
        rc = _dispatch(
            [
                "mesh",
                "to-dataset",
                str(_MESH),
                out,
                "--var",
                "mesh2d_node_z",
                "--cell-size",
                "3000",
                "--bounds",
                str(xmin),
                str(ymin),
                str(xmax),
                str(ymax),
                "--epsg",
                "32636",
            ]
        )
        assert rc == 0
        assert Dataset.read_file(out).epsg == 32636

    def test_unknown_variable_raises(self, tmp_path):
        """A typo'd --var becomes a clean ValueError (via map_lookup_errors)."""
        with pytest.raises(ValueError):
            _dispatch(
                [
                    "mesh",
                    "to-dataset",
                    str(_MESH),
                    str(tmp_path / "x.tif"),
                    "--var",
                    "nope",
                    "--cell-size",
                    "2000",
                ]
            )


class TestToZarr:
    """`pyramids mesh to-zarr`."""

    def test_writes_store(self, tmp_path):
        """to-zarr writes a Zarr store directory."""
        out = str(tmp_path / "m.zarr")
        assert _dispatch(["mesh", "to-zarr", str(_MESH), out]) == 0
        assert Path(out).exists()


class TestToImage:
    """`pyramids mesh to-image`."""

    def test_writes_png(self, tmp_path):
        """to-image writes a readable colour-mapped image."""
        # Pillow — and the cleopatra/matplotlib viz stack the `to-image` command renders
        # with — only ships in the [viz] extra. Import it here, not at module top level, so
        # this module still *collects* (and its non-viz export tests still run) in the core /
        # lazy / interop wheel envs, which have no Pillow.
        Image = pytest.importorskip("PIL.Image")
        out = str(tmp_path / "m.png")
        rc = _dispatch(
            [
                "mesh",
                "to-image",
                str(_MESH),
                out,
                "--var",
                "mesh2d_node_z",
                "--cell-size",
                "3000",
            ]
        )
        assert rc == 0
        with Image.open(out) as img:
            assert img.size[0] > 0


class TestToBytes:
    """`pyramids mesh to-bytes`."""

    def test_writes_file(self, tmp_path):
        """to-bytes writes the NetCDF buffer to a file."""
        out = str(tmp_path / "m.nc")
        assert _dispatch(["mesh", "to-bytes", str(_MESH), out]) == 0
        assert Path(out).stat().st_size > 0

    def test_writes_stdout(self, capsysbinary):
        """With no output path the bytes go to stdout."""
        assert _dispatch(["mesh", "to-bytes", str(_MESH)]) == 0
        assert len(capsysbinary.readouterr().out) > 0

    def test_refuses_existing(self, tmp_path):
        """A second write to the same file without --overwrite raises."""
        out = str(tmp_path / "m.nc")
        assert _dispatch(["mesh", "to-bytes", str(_MESH), out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _dispatch(["mesh", "to-bytes", str(_MESH), out])


class TestToLocation:
    """`pyramids mesh to-location`."""

    def test_remaps_to_face(self, tmp_path):
        """A node variable remapped onto faces writes a readable mesh."""
        out = str(tmp_path / "loc.nc")
        rc = _dispatch(
            [
                "mesh",
                "to-location",
                str(_MESH),
                out,
                "--location",
                "face",
                "--var",
                "mesh2d_node_z",
            ]
        )
        assert rc == 0
        assert "mesh2d_node_z" in MeshDataset.read_file(out).data_variable_names


class TestToCrs:
    """`pyramids mesh to-crs`."""

    def test_reprojects(self, tmp_path):
        """to-crs reprojects a CRS-bearing mesh to the target EPSG."""
        out = str(tmp_path / "rep.nc")
        rc = _dispatch(["mesh", "to-crs", _crs_mesh(tmp_path), out, "--epsg", "3857"])
        assert rc == 0
        assert MeshDataset.read_file(out).epsg == 3857

    def test_without_source_crs_raises(self, tmp_path):
        """Reprojecting a mesh with no CRS is a clean error."""
        with pytest.raises(ValueError, match="source CRS"):
            _dispatch(
                ["mesh", "to-crs", str(_MESH), str(tmp_path / "x.nc"), "--epsg", "3857"]
            )


class TestSetCrs:
    """`pyramids mesh set-crs`."""

    def test_assigns_epsg(self, tmp_path):
        """set-crs declares a CRS on a mesh that had none."""
        out = str(tmp_path / "declared.nc")
        assert _dispatch(["mesh", "set-crs", str(_MESH), out, "--epsg", "4326"]) == 0
        assert MeshDataset.read_file(out).epsg == 4326

    def test_neither_epsg_nor_crs_raises(self, tmp_path):
        """Omitting both --epsg and --crs is a clean error."""
        with pytest.raises(ValueError):
            _dispatch(["mesh", "set-crs", str(_MESH), str(tmp_path / "x.nc")])


class TestBurn:
    """`pyramids mesh burn`."""

    def test_burns_geometry(self, tmp_path):
        """burn adds a face variable from a vector layer."""
        out = str(tmp_path / "burned.nc")
        rc = _dispatch(
            [
                "mesh",
                "burn",
                str(_MESH),
                out,
                "--geometry",
                _geometry_file(tmp_path),
                "--value",
                "1",
                "--fill",
                "0",
                "--name",
                "mask",
            ]
        )
        assert rc == 0
        assert "mask" in MeshDataset.read_file(out).data_variable_names

    def test_no_all_touched(self, tmp_path):
        """--no-all-touched flips the all_touched flag off and still writes."""
        out = str(tmp_path / "b2.nc")
        rc = _dispatch(
            [
                "mesh",
                "burn",
                str(_MESH),
                out,
                "--geometry",
                _geometry_file(tmp_path),
                "--value",
                "5",
                "--no-all-touched",
            ]
        )
        assert rc == 0
        assert "burned" in MeshDataset.read_file(out).data_variable_names


class TestRegrid:
    """`pyramids mesh regrid`."""

    def test_regrids_onto_target(self, tmp_path):
        """regrid carries a face variable onto the target mesh."""
        face = _face_mesh(tmp_path)
        out = str(tmp_path / "regridded.nc")
        rc = _dispatch(["mesh", "regrid", face, out, "--target", face, "--var", "fv"])
        assert rc == 0
        assert "fv" in MeshDataset.read_file(out).data_variable_names


class TestPolygonize:
    """`pyramids mesh polygonize`."""

    def test_dissolves_to_polygons(self, tmp_path):
        """polygonize writes one feature per merged region."""
        out = str(tmp_path / "poly.geojson")
        rc = _dispatch(["mesh", "polygonize", _face_mesh(tmp_path), out, "--var", "fv"])
        assert rc == 0
        assert len(FeatureCollection.read_file(out)) > 0


class TestAttachData:
    """`pyramids mesh attach-data`."""

    def test_attaches_results(self, tmp_path, attach_registered):
        """attach-data merges a results file's variable onto the mesh."""
        results = tmp_path / "v.res"
        results.write_text("results")
        out = str(tmp_path / "attached.nc")
        rc = _dispatch(
            [
                "mesh",
                "attach-data",
                str(_MESH),
                out,
                "--data",
                str(results),
                "--driver",
                "attach",
                "--no-lazy",
            ]
        )
        assert rc == 0
        assert "velocity" in MeshDataset.read_file(out).data_variable_names

    def test_missing_results_file_raises(self, tmp_path):
        """A non-existent results file raises (a clean OSError subclass)."""
        with pytest.raises(FileNotFoundError):
            _dispatch(
                [
                    "mesh",
                    "attach-data",
                    str(_MESH),
                    str(tmp_path / "o.nc"),
                    "--data",
                    str(tmp_path / "nope.res"),
                ]
            )


class TestWithVariable:
    """`pyramids mesh with-variable`."""

    def test_adds_from_npy(self, tmp_path):
        """with-variable loads an array from .npy and adds it as a face variable."""
        mesh = MeshDataset.read_file(str(_MESH))
        npy = str(tmp_path / "v.npy")
        np.save(npy, np.arange(mesh.n_face, dtype="float64"))
        out = str(tmp_path / "wv.nc")
        rc = _dispatch(
            [
                "mesh",
                "with-variable",
                str(_MESH),
                out,
                "--name",
                "wv",
                "--data",
                npy,
                "--location",
                "face",
                "--nodata",
                "-1",
            ]
        )
        assert rc == 0
        assert "wv" in MeshDataset.read_file(out).data_variable_names


class TestRenameVariable:
    """`pyramids mesh rename-variable`."""

    def test_renames(self, tmp_path):
        """rename-variable swaps a variable's name."""
        out = str(tmp_path / "ren.nc")
        rc = _dispatch(
            [
                "mesh",
                "rename-variable",
                str(_MESH),
                out,
                "--old",
                "mesh2d_node_z",
                "--new",
                "zz",
            ]
        )
        assert rc == 0
        names = MeshDataset.read_file(out).data_variable_names
        assert "zz" in names
        assert "mesh2d_node_z" not in names

    def test_unknown_old_name_raises(self, tmp_path):
        """Renaming a variable that does not exist is a clean error."""
        with pytest.raises(ValueError):
            _dispatch(
                [
                    "mesh",
                    "rename-variable",
                    str(_MESH),
                    str(tmp_path / "x.nc"),
                    "--old",
                    "nope",
                    "--new",
                    "zz",
                ]
            )


class TestVoronoiPolygons:
    """``mesh voronoi-polygons`` — Voronoi cells of the mesh nodes, as a vector file (#1479)."""

    @staticmethod
    def _mesh_file(tmp_path):
        """A written four-node two-triangle mesh in EPSG:4326."""
        ds = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 1.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 2, 3]]),
            epsg=4326,
        )
        path = tmp_path / "m.nc"
        ds.to_file(str(path))
        return str(path)

    def test_voronoi_polygons_writes_a_vector_file(self, tmp_path):
        """The verb writes one polygon per node to a vector file."""
        src = self._mesh_file(tmp_path)
        out = tmp_path / "vor.gpkg"

        assert _dispatch(["mesh", "voronoi-polygons", src, str(out)]) == 0

        cells = FeatureCollection.read_file(str(out))
        assert len(cells) == 4
        assert set(cells.geometry.geom_type) == {"Polygon"}

    def test_voronoi_polygons_infers_driver_from_geojson(self, tmp_path):
        """The output driver is inferred from the extension (R1 N4)."""
        src = self._mesh_file(tmp_path)
        out = tmp_path / "vor.geojson"
        assert _dispatch(["mesh", "voronoi-polygons", src, str(out)]) == 0
        assert out.exists(), "the .geojson output must be created"
        assert out.stat().st_size > 0, "the output must not be empty"

    def test_voronoi_polygons_refuses_an_existing_output(self, tmp_path):
        """Without `--overwrite`, writing over an existing file is refused (R1 N4)."""
        src = self._mesh_file(tmp_path)
        out = tmp_path / "vor.gpkg"
        assert _dispatch(["mesh", "voronoi-polygons", src, str(out)]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _dispatch(["mesh", "voronoi-polygons", src, str(out)])
