"""Tests for the ``pyramids mesh`` plotting commands (``cli_ugrid_plot``).

Each verb is dispatched through a throwaway parser that mirrors how the
orchestrator wires the registrator onto the ``mesh`` sub-parser, then the handler
is run directly. Happy paths assert ``0`` and that the image / animation artifact
was written; error paths assert the handler raises (``pyramids.cli.main`` owns the
exit-code contract, so the handlers themselves raise).

The shipped ``ugrid.nc`` has no temporal variable and no finite face/node vector
pair, so the ``plot-vector-field`` and ``animate`` happy paths build small meshes
with ``MeshDataset.from_arrays`` and write them to disk first.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

os.environ["MPLBACKEND"] = "Agg"
# matplotlib ships only with the [viz] extra; skip this plot module where it is absent.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyramids.cli_ugrid_plot import add_ugrid_plot_commands  # noqa: E402
from pyramids.netcdf.ugrid import MeshDataset  # noqa: E402

pytestmark = [
    pytest.mark.plot,
    pytest.mark.filterwarnings("ignore:Animation was deleted without rendering"),
]

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"
_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _dispatch(argv: list[str]) -> int:
    """Register the plot commands on a throwaway parser and run the matched handler.

    Args:
        argv: The argument vector, starting with ``"mesh"``.

    Returns:
        int: The handler's exit code.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_plot_commands(msub)
    args = parser.parse_args(argv)
    return args.func(args)


def _write_mesh(path: Path, data: dict, locations: dict) -> str:
    """Write a 4-node / 2-triangle mesh carrying ``data`` to ``path``.

    Args:
        path: Destination ``.nc`` path.
        data: Variable-name → array mapping.
        locations: Variable-name → location mapping.

    Returns:
        str: The written path.
    """
    MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={
            name: np.asarray(values, dtype="float64") for name, values in data.items()
        },
        data_locations=locations,
        epsg=4326,
    ).to_file(str(path))
    return str(path)


class TestMeshPlot:
    """``pyramids mesh plot``."""

    def test_writes_png(self, tmp_path):
        """plot renders a variable and saves the figure."""
        out = str(tmp_path / "mesh.png")
        assert (
            _dispatch(["mesh", "plot", str(_MESH), out, "--var", "mesh2d_node_z"]) == 0
        )
        assert os.path.exists(out)

    def test_refuses_existing(self, tmp_path):
        """A plot onto an existing path without --overwrite is refused."""
        out = str(tmp_path / "m.png")
        assert (
            _dispatch(["mesh", "plot", str(_MESH), out, "--var", "mesh2d_node_z"]) == 0
        )
        with pytest.raises(ValueError):
            _dispatch(["mesh", "plot", str(_MESH), out, "--var", "mesh2d_node_z"])

    def test_overwrite_allows_replacement(self, tmp_path):
        """--overwrite replaces an existing output."""
        out = str(tmp_path / "m.png")
        assert (
            _dispatch(["mesh", "plot", str(_MESH), out, "--var", "mesh2d_node_z"]) == 0
        )
        rc = _dispatch(
            ["mesh", "plot", str(_MESH), out, "--var", "mesh2d_node_z", "--overwrite"]
        )
        assert rc == 0

    def test_unknown_variable_raises(self, tmp_path):
        """A typo'd --var is remapped to a clean ValueError (not a bare KeyError)."""
        out = str(tmp_path / "x.png")
        with pytest.raises(ValueError):
            _dispatch(["mesh", "plot", str(_MESH), out, "--var", "nope"])


class TestMeshPlotOutline:
    """``pyramids mesh plot-outline``."""

    def test_writes_png(self, tmp_path):
        """plot-outline saves the wireframe figure."""
        out = str(tmp_path / "outline.png")
        assert _dispatch(["mesh", "plot-outline", str(_MESH), out]) == 0
        assert os.path.exists(out)

    def test_refuses_existing(self, tmp_path):
        """A second outline without --overwrite is refused."""
        out = str(tmp_path / "o.png")
        assert _dispatch(["mesh", "plot-outline", str(_MESH), out]) == 0
        with pytest.raises(ValueError):
            _dispatch(["mesh", "plot-outline", str(_MESH), out])


class TestMeshPlotVectorField:
    """``pyramids mesh plot-vector-field``."""

    def test_writes_png(self, tmp_path):
        """plot-vector-field draws two face variables and saves the figure."""
        mesh = _write_mesh(
            tmp_path / "vf.nc",
            {"u": [1.0, 2.0], "v": [0.5, -0.5]},
            {"u": "face", "v": "face"},
        )
        out = str(tmp_path / "vf.png")
        assert (
            _dispatch(["mesh", "plot-vector-field", mesh, out, "--u", "u", "--v", "v"])
            == 0
        )
        assert os.path.exists(out)

    def test_refuses_existing(self, tmp_path):
        """A second vector-field plot without --overwrite is refused."""
        mesh = _write_mesh(
            tmp_path / "vf.nc",
            {"u": [1.0, 2.0], "v": [0.5, -0.5]},
            {"u": "face", "v": "face"},
        )
        out = str(tmp_path / "vf.png")
        assert (
            _dispatch(["mesh", "plot-vector-field", mesh, out, "--u", "u", "--v", "v"])
            == 0
        )
        with pytest.raises(ValueError):
            _dispatch(["mesh", "plot-vector-field", mesh, out, "--u", "u", "--v", "v"])

    def test_unknown_variable_raises(self, tmp_path):
        """An unknown component variable is a clean ValueError."""
        out = str(tmp_path / "x.png")
        with pytest.raises(ValueError):
            _dispatch(
                [
                    "mesh",
                    "plot-vector-field",
                    str(_MESH),
                    out,
                    "--u",
                    "nope",
                    "--v",
                    "nope",
                ]
            )


class TestMeshPlotHistogram:
    """``pyramids mesh plot-histogram``."""

    def test_writes_png(self, tmp_path):
        """plot-histogram saves a variable's value distribution."""
        out = str(tmp_path / "hist.png")
        rc = _dispatch(
            ["mesh", "plot-histogram", str(_MESH), out, "--var", "mesh2d_edge_type"]
        )
        assert rc == 0
        assert os.path.exists(out)

    def test_default_variable_on_single_var_mesh(self, tmp_path):
        """With no --var, the only data variable is histogrammed (bins forwarded)."""
        mesh = _write_mesh(tmp_path / "single.nc", {"d": [1.0, 2.0]}, {"d": "face"})
        out = str(tmp_path / "h.png")
        assert _dispatch(["mesh", "plot-histogram", mesh, out, "--bins", "5"]) == 0
        assert os.path.exists(out)

    def test_refuses_existing(self, tmp_path):
        """A second histogram without --overwrite is refused."""
        out = str(tmp_path / "h.png")
        assert (
            _dispatch(
                ["mesh", "plot-histogram", str(_MESH), out, "--var", "mesh2d_edge_type"]
            )
            == 0
        )
        with pytest.raises(ValueError):
            _dispatch(
                ["mesh", "plot-histogram", str(_MESH), out, "--var", "mesh2d_edge_type"]
            )

    def test_unknown_variable_raises(self, tmp_path):
        """A typo'd --var is a clean ValueError."""
        out = str(tmp_path / "x.png")
        with pytest.raises(ValueError):
            _dispatch(["mesh", "plot-histogram", str(_MESH), out, "--var", "nope"])


class TestMeshAnimate:
    """``pyramids mesh animate``."""

    def test_writes_gif(self, tmp_path):
        """animate saves a temporal variable as a GIF."""
        mesh = _write_mesh(
            tmp_path / "temporal.nc",
            {"d": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
            {"d": "face"},
        )
        out = str(tmp_path / "anim.gif")
        assert _dispatch(["mesh", "animate", mesh, out, "--interval", "50"]) == 0
        assert os.path.exists(out)

    def test_refuses_existing(self, tmp_path):
        """A second animate without --overwrite is refused."""
        mesh = _write_mesh(
            tmp_path / "temporal.nc",
            {"d": [[1.0, 2.0], [3.0, 4.0]]},
            {"d": "face"},
        )
        out = str(tmp_path / "anim.gif")
        assert _dispatch(["mesh", "animate", mesh, out]) == 0
        with pytest.raises(ValueError):
            _dispatch(["mesh", "animate", mesh, out])

    def test_no_time_axis_raises(self, tmp_path):
        """A mesh variable with no time axis is a clean ValueError, not a KeyError."""
        out = str(tmp_path / "x.gif")
        with pytest.raises(ValueError):
            _dispatch(["mesh", "animate", str(_MESH), out, "--var", "mesh2d_node_z"])
