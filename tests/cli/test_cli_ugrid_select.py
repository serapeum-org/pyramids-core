"""Tests for the ``pyramids mesh`` selection / subset commands (``cli_ugrid_select``).

The registrator is exercised on a throwaway parser (it is wired into the real CLI by the
orchestrator), so these tests stand on their own before that wiring lands. Each happy path
asserts the handler returns ``0`` and re-opens the written mesh to check the subset; each
error path asserts the handler raises the expected exception (the handlers raise — the real
``cli.main`` is what turns a raised ``ValueError`` into the one-line ``error:`` exit).

Three fixtures cover the three selection axes: the shipped static ``ugrid.nc`` for space and
value ops, an in-memory temporal mesh (written to a tmp ``.nc``) for the time ops, and an
in-memory ragged 3-D volume mesh for the layer ops.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.cli_ugrid_select import (
    _COND_PATTERN,
    _build_condition,
    _resolve_variable,
    _scalar_or_list,
    add_ugrid_select_commands,
)
from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset, VolumeTopology
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _run(argv: list[str]) -> int:
    """Dispatch ``argv`` through a throwaway ``mesh`` parser carrying the registrator.

    Args:
        argv: The CLI tokens after ``pyramids`` (e.g. ``["mesh", "squeeze", src, out]``).

    Returns:
        int: The handler's return code.
    """
    parser = argparse.ArgumentParser(prog="pyramids")
    groups = parser.add_subparsers(dest="group")
    mesh = groups.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_select_commands(msub)
    args = parser.parse_args(argv)
    return args.func(args)


def _temporal_mesh(stamps: list[float]) -> MeshDataset:
    """A two-face mesh with one temporal variable and explicit time coordinates.

    Args:
        stamps: The time coordinate values; the variable gets shape ``(len(stamps), 2)``.

    Returns:
        MeshDataset: The temporal mesh.
    """
    mesh = MeshDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": np.arange(len(stamps) * 2, dtype="float64").reshape(len(stamps), 2)},
        data_locations={"d": "face"},
    )
    mesh["d"].attributes = {**mesh["d"].attributes, "time_values": list(stamps)}
    return mesh


def _volume_mesh() -> MeshDataset:
    """A 3-face strip mesh carrying a ragged 3-D volume variable (columns of 3/1/2 layers).

    Returns:
        MeshDataset: The volume mesh, which round-trips through ``to_file``.
    """
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        node_y=np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        face_node_connectivity=fnc,
        face_x=np.array([1.0, 2.0, 3.0]),
        face_y=np.array([0.3, 0.6, 0.3]),
    )
    topology = VolumeTopology(
        layer_count=np.array([3, 1, 2]), volume_offset=np.array([0, 3, 4, 6])
    )
    var = MeshVariable(
        name="sal",
        location="volume",
        mesh_name="mesh2d",
        shape=(6,),
        dimensions=("mesh2d_nVolumes",),
        _data=np.arange(6.0),
    )
    return MeshDataset(mesh, {"sal": var}, {}, volume_topology=topology)


def _write_mesh(mesh: MeshDataset, tmp_path, name: str = "src.nc") -> str:
    """Write a mesh to ``tmp_path`` and return its path as a string.

    Args:
        mesh: The mesh to persist.
        tmp_path: The pytest temp directory.
        name: The file name to use.

    Returns:
        str: The absolute path of the written mesh.
    """
    path = str(tmp_path / name)
    mesh.to_file(path)
    return path


def _nsteps(mesh: MeshDataset) -> int:
    """Return the number of time steps of a temporal mesh's first variable.

    Args:
        mesh: An opened mesh with a temporal variable ``d``.

    Returns:
        int: The leading (time) dimension length of ``d``'s values.
    """
    return int(mesh.get_data("d").data.shape[0])


def _bbox_left_half() -> tuple[float, float, float, float]:
    """Return a bbox covering the western half of ``ugrid.nc``.

    Returns:
        tuple[float, float, float, float]: ``(minx, miny, midx, maxy)``.
    """
    minx, miny, maxx, maxy = MeshDataset.read_file(str(_MESH)).bounds
    return (minx, miny, (minx + maxx) / 2, maxy)


class TestHelpers:
    """Unit coverage for the module's small pure helpers."""

    def test_scalar_or_list(self):
        """None stays None, one value unwraps to a scalar, several stay a list."""
        assert _scalar_or_list(None) is None
        assert _scalar_or_list([3]) == 3
        assert _scalar_or_list([1, 2]) == [1, 2]

    def test_resolve_variable_explicit(self):
        """An explicit name is returned unchanged without inspecting the mesh."""
        assert (
            _resolve_variable(MeshDataset.read_file(str(_MESH)), "chosen") == "chosen"
        )

    def test_resolve_variable_multi_requires_var(self):
        """A multi-variable mesh with no --var is a clean ValueError."""
        mesh = MeshDataset.read_file(str(_MESH))
        with pytest.raises(ValueError, match="pass --var"):
            _resolve_variable(mesh, None)

    def test_resolve_variable_sole(self):
        """A single-variable mesh resolves to that one variable."""
        mesh = _temporal_mesh([0.0, 1.0])
        assert _resolve_variable(mesh, None) == "d"

    def test_build_condition(self):
        """A comparison becomes the matching boolean mask over the variable's values."""
        mesh = _temporal_mesh([0.0, 1.0])
        mask = _build_condition(mesh, "d", "d > 3")
        assert mask.tolist() == (mesh.get_data("d").data > 3).tolist()

    def test_build_condition_bad_expression(self):
        """A non-comparison --cond is refused with a clear ValueError."""
        mesh = _temporal_mesh([0.0, 1.0])
        with pytest.raises(ValueError, match="--cond must be"):
            _build_condition(mesh, "d", "not an expression")

    def test_cond_pattern_bare_operator(self):
        """A bare 'OP NUMBER' matches with no variable captured."""
        match = _COND_PATTERN.match("> 3")
        assert match is not None
        assert match.group("var") is None


class TestTimeSelect:
    """Time-axis commands on a temporal mesh (``sel``/``isel``/``sel-time``/…)."""

    def test_sel_scalar_collapses_time(self, tmp_path):
        """sel with one coordinate value writes a static (time-collapsed) mesh."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "sel", src, out, "--time", "1"]) == 0
        assert MeshDataset.read_file(out).get_data("d").data.ndim == 1

    def test_sel_sequence_keeps_time(self, tmp_path):
        """sel with several values keeps the time axis, trimmed to those steps."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "sel", src, out, "--time", "1", "2"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 2

    def test_sel_nearest_tolerance(self, tmp_path):
        """--method nearest with a tolerance snaps a near-miss value to a step."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        rc = _run(
            [
                "mesh",
                "sel",
                src,
                out,
                "--time",
                "1.4",
                "--method",
                "nearest",
                "--tolerance",
                "1",
            ]
        )
        assert rc == 0

    def test_isel_time_slice(self, tmp_path):
        """isel over several time positions keeps the time axis."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "isel", src, out, "--time", "0", "1"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 2

    def test_isel_requires_an_axis(self, tmp_path):
        """isel with neither --time nor --element raises (mapped to ValueError)."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0]), tmp_path)
        with pytest.raises(ValueError):
            _run(["mesh", "isel", src, str(tmp_path / "o.nc")])

    def test_isel_single_time_collapses(self, tmp_path):
        """isel with one --time value collapses the axis and still serialises.

        ``sel-time`` is the dedicated single-step counterpart; both collapse the time axis
        and write back cleanly.
        """
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "isel", src, out, "--time", "0"]) == 0
        assert MeshDataset.read_file(out).get_data("d").data.ndim == 1

    def test_sel_time_collapses_and_serialises(self, tmp_path):
        """sel-time picks one step by position, collapses the axis, and round-trips."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "sel-time", src, out, "--index", "1"]) == 0
        written = MeshDataset.read_file(out)
        assert written.get_data("d").data.ndim == 1
        assert written.get_data("d").data.tolist() == [2.0, 3.0]

    def test_sel_time_bad_index_is_value_error(self, tmp_path):
        """An out-of-range --index is remapped to a clean ValueError."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0]), tmp_path)
        with pytest.raises(ValueError):
            _run(["mesh", "sel-time", src, str(tmp_path / "o.nc"), "--index", "9"])

    def test_sel_time_non_temporal_is_value_error(self, tmp_path):
        """sel-time on a static mesh is a clean ValueError, matching isel --time."""
        static = MeshDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"depth": np.array([2.0, 4.0])},
            data_locations={"depth": "face"},
        )
        src = _write_mesh(static, tmp_path, name="static.nc")
        with pytest.raises(ValueError, match="time dimension"):
            _run(["mesh", "sel-time", src, str(tmp_path / "o.nc"), "--index", "0"])

    def test_sel_time_refuses_existing_output(self, tmp_path):
        """sel-time refuses to clobber an existing output unless --overwrite is given."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0]), tmp_path)
        out = tmp_path / "o.nc"
        out.write_text("sentinel")
        with pytest.raises(ValueError, match="exists"):
            _run(["mesh", "sel-time", src, str(out), "--index", "0"])
        assert (
            _run(["mesh", "sel-time", src, str(out), "--index", "0", "--overwrite"])
            == 0
        )
        assert MeshDataset.read_file(str(out)).get_data("d").data.ndim == 1

    def test_sel_time_range(self, tmp_path):
        """sel-time-range keeps the half-open [start, stop) window."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert (
            _run(["mesh", "sel-time-range", src, out, "--start", "1", "--stop", "3"])
            == 0
        )
        assert _nsteps(MeshDataset.read_file(out)) == 2

    def test_thin(self, tmp_path):
        """thin keeps every N-th step."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "thin", src, out, "--step", "2"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 2

    def test_thin_step_zero_raises(self, tmp_path):
        """A stride below 1 is refused by the kernel (ValueError)."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0]), tmp_path)
        with pytest.raises(ValueError, match="step"):
            _run(["mesh", "thin", src, str(tmp_path / "o.nc"), "--step", "0"])

    def test_head(self, tmp_path):
        """head keeps the leading N steps."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "head", src, out, "--n", "2"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 2

    def test_tail(self, tmp_path):
        """tail keeps the trailing N steps."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "tail", src, out, "--n", "2"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 2

    def test_drop_isel(self, tmp_path):
        """drop-isel removes steps by position."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "drop-isel", src, out, "--indices", "0"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 3

    def test_drop_sel(self, tmp_path):
        """drop-sel removes steps by coordinate value."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "drop-sel", src, out, "--values", "0"]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 3

    def test_drop_duplicates(self, tmp_path):
        """drop-duplicates runs and keeps every step when the stamps are unique.

        A written mesh's time coordinate is read back as positional (0, 1, 2, …), so the
        on-disk stamps are always unique and the kernel keeps them all — exercising the
        full read -> op -> write path. Genuine duplicate stamps cannot be fixtured through
        the public constructors (see the report's fixture-gap note).
        """
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "drop-duplicates", src, out]) == 0
        assert _nsteps(MeshDataset.read_file(out)) == 4

    def test_squeeze(self, tmp_path):
        """squeeze collapses a single-step temporal variable to static."""
        src = _write_mesh(_temporal_mesh([5.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "squeeze", src, out]) == 0
        assert MeshDataset.read_file(out).get_data("d").data.ndim == 1

    def test_drop_isel_on_static_raises(self, tmp_path):
        """A time op on a mesh with no time dimension is a clean ValueError."""
        with pytest.raises(ValueError, match="time dimension"):
            _run(
                [
                    "mesh",
                    "drop-isel",
                    str(_MESH),
                    str(tmp_path / "o.nc"),
                    "--indices",
                    "0",
                ]
            )


class TestLayerSelect:
    """Layer-axis commands on a ragged 3-D volume mesh."""

    def test_isel_layer(self, tmp_path):
        """isel-layer selects one layer per column and writes a readable mesh."""
        src = _write_mesh(_volume_mesh(), tmp_path, "vol.nc")
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "isel-layer", src, out, "--index", "0"]) == 0
        assert "sal" in MeshDataset.read_file(out).data_variable_names

    def test_sel_layer(self, tmp_path):
        """sel-layer selects a layer by (positional) coordinate value."""
        src = _write_mesh(_volume_mesh(), tmp_path, "vol.nc")
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "sel-layer", src, out, "--value", "1"]) == 0
        assert "sal" in MeshDataset.read_file(out).data_variable_names

    def test_collapse_layer(self, tmp_path):
        """collapse-layer flattens the volume onto a 2-D face field."""
        src = _write_mesh(_volume_mesh(), tmp_path, "vol.nc")
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "collapse-layer", src, out, "--layer", "0"]) == 0
        assert MeshDataset.read_file(out).get_data("sal").data.shape == (3,)

    def test_collapse_layer_rejects_non_face(self, tmp_path):
        """A non-face --location is refused by the kernel."""
        src = _write_mesh(_volume_mesh(), tmp_path, "vol.nc")
        with pytest.raises(ValueError):
            _run(
                [
                    "mesh",
                    "collapse-layer",
                    src,
                    str(tmp_path / "o.nc"),
                    "--layer",
                    "0",
                    "--location",
                    "node",
                ]
            )

    def test_sel_layer_on_flat_mesh_raises(self, tmp_path):
        """sel-layer on a mesh with no layer dimension is a clean ValueError."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0]), tmp_path)
        with pytest.raises(ValueError, match="layer"):
            _run(["mesh", "sel-layer", src, str(tmp_path / "o.nc"), "--value", "0"])


class TestSpatialSubset:
    """Space-axis commands on the shipped static mesh."""

    def test_subset_by_bounds(self, tmp_path):
        """subset-by-bounds keeps a strict subset of the faces."""
        out = str(tmp_path / "o.nc")
        minx, miny, maxx, maxy = _bbox_left_half()
        rc = _run(
            [
                "mesh",
                "subset-by-bounds",
                str(_MESH),
                out,
                "--bbox",
                str(minx),
                str(miny),
                str(maxx),
                str(maxy),
            ]
        )
        assert rc == 0
        full = MeshDataset.read_file(str(_MESH)).n_face
        assert 0 < MeshDataset.read_file(out).n_face < full

    def test_crop_bbox(self, tmp_path):
        """crop with a --bbox keeps the faces inside it."""
        out = str(tmp_path / "o.nc")
        minx, miny, maxx, maxy = _bbox_left_half()
        rc = _run(
            [
                "mesh",
                "crop",
                str(_MESH),
                out,
                "--bbox",
                str(minx),
                str(miny),
                str(maxx),
                str(maxy),
            ]
        )
        assert rc == 0
        assert MeshDataset.read_file(out).n_face > 0

    def test_crop_mask(self, tmp_path):
        """crop with a --mask polygon keeps the faces inside it."""
        mask = _mask_file(tmp_path)
        out = str(tmp_path / "o.nc")
        assert (
            _run(["mesh", "crop", str(_MESH), out, "--mask", mask, "--no-touch"]) == 0
        )
        assert MeshDataset.read_file(out).n_face > 0

    def test_clip(self, tmp_path):
        """clip keeps the faces inside a polygon mask file."""
        mask = _mask_file(tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "clip", str(_MESH), out, "--mask", mask]) == 0
        assert MeshDataset.read_file(out).n_face > 0

    def test_isel_element(self, tmp_path):
        """isel --element renumbers the mesh around the chosen faces."""
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "isel", str(_MESH), out, "--element", "0", "1", "2"]) == 0
        assert MeshDataset.read_file(out).n_face == 3


class TestValueOps:
    """Value-masking and variable-dropping commands."""

    def test_clip_values(self, tmp_path):
        """clip-values clamps a variable to [lower, upper]."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert (
            _run(["mesh", "clip-values", src, out, "--lower", "1", "--upper", "5"]) == 0
        )
        assert float(np.nanmax(MeshDataset.read_file(out).get_data("d").data)) <= 5.0

    def test_where_with_variable_in_cond(self, tmp_path):
        """where masks the elements the comparison does not select."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "where", src, out, "--cond", "d > 3"]) == 0
        data = MeshDataset.read_file(out).get_data("d").data
        assert np.isnan(data).any(), "elements failing the comparison must be masked"

    def test_where_bare_operator(self, tmp_path):
        """A bare '> N' --cond works, applying to the resolved variable."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0, 2.0, 3.0]), tmp_path)
        out = str(tmp_path / "o.nc")
        assert _run(["mesh", "where", src, out, "--cond", "> 3", "--other", "0"]) == 0

    def test_where_bad_cond_raises(self, tmp_path):
        """A malformed --cond is refused before the mesh is touched."""
        src = _write_mesh(_temporal_mesh([0.0, 1.0]), tmp_path)
        with pytest.raises(ValueError, match="--cond must be"):
            _run(["mesh", "where", src, str(tmp_path / "o.nc"), "--cond", "garbage"])

    def test_drop_variables(self, tmp_path):
        """drop-variables removes the named variable."""
        out = str(tmp_path / "o.nc")
        assert (
            _run(
                [
                    "mesh",
                    "drop-variables",
                    str(_MESH),
                    out,
                    "--names",
                    "mesh2d_edge_type",
                ]
            )
            == 0
        )
        assert "mesh2d_edge_type" not in MeshDataset.read_file(out).data_variable_names

    def test_drop_variables_unknown_raises(self, tmp_path):
        """Dropping an absent variable is remapped to a clean ValueError."""
        with pytest.raises(ValueError):
            _run(
                [
                    "mesh",
                    "drop-variables",
                    str(_MESH),
                    str(tmp_path / "o.nc"),
                    "--names",
                    "nope",
                ]
            )


class TestWriterContract:
    """The shared writer guard applies to every command."""

    def test_refuses_existing(self, tmp_path):
        """A second write to the same path without --overwrite raises."""
        out = str(tmp_path / "o.nc")
        minx, miny, maxx, maxy = _bbox_left_half()
        argv = [
            "mesh",
            "subset-by-bounds",
            str(_MESH),
            out,
            "--bbox",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
        ]
        assert _run(argv) == 0
        with pytest.raises(ValueError, match="already exists"):
            _run(argv)

    def test_overwrite_allows_replacement(self, tmp_path):
        """--overwrite lets a command replace an existing output."""
        out = str(tmp_path / "o.nc")
        minx, miny, maxx, maxy = _bbox_left_half()
        base = [
            "mesh",
            "subset-by-bounds",
            str(_MESH),
            out,
            "--bbox",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
        ]
        assert _run(base) == 0
        assert _run(base + ["--overwrite"]) == 0


def _mask_file(tmp_path) -> str:
    """Write a GeoJSON polygon covering the western half of ``ugrid.nc``.

    Args:
        tmp_path: The pytest temp directory.

    Returns:
        str: The path of the mask file.
    """
    minx, miny, maxx, maxy = _bbox_left_half()
    path = str(tmp_path / "mask.geojson")
    gdf = gpd.GeoDataFrame({"geometry": [box(minx, miny, maxx, maxy)]}, crs=None)
    FeatureCollection(gdf).to_file(path, driver="geojson")
    return path
