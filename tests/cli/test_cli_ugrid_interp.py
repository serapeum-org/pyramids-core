"""Tests for the interp/fill/time/calculus ``pyramids mesh`` commands.

Each command is a thin 1:1 wrapper over a :class:`~pyramids.netcdf.ugrid.MeshDataset`
method: it reads a mesh, applies one transform, and writes a new mesh file. The registrator
(:func:`pyramids.cli_ugrid_interp.add_ugrid_interp_commands`) is wired onto a throwaway
``mesh`` subparser here — exactly as the top-level CLI wires it — and each case dispatches
through ``args.func(args)`` so the handlers are exercised without depending on the orchestrator
wiring. Happy paths assert a ``0`` exit and that the written mesh re-opens; error paths assert
the handler raises (``pyramids.cli.main`` owns turning that into the ``error:`` exit).

These are extras-free (core): meshes are built in memory, written with ``to_file`` and re-read
via the GDAL multidimensional path, and the time/value kernels fall back to NumPy/SciPy.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from pyramids._cli_groups import add_group
from pyramids.cli_ugrid_interp import add_ugrid_interp_commands
from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _dispatch(argv: list[str]) -> int:
    """Register the group on a throwaway parser and run the selected command.

    Args:
        argv: The argument vector, e.g. ``["mesh", "advanced", "integrate", src, dst]``.

    Returns:
        int: The handler's exit code.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    advanced = add_group(
        msub, "advanced", "power-user primitives (accessors, array/dim mechanics)"
    )
    add_ugrid_interp_commands(
        add_group(msub, "interp", "interpolation, regridding, sampling"), advanced
    )
    args = parser.parse_args(argv)
    return args.func(args)


def _temporal(values: list) -> MeshDataset:
    """A two-face mesh carrying one temporal ``depth`` variable of shape ``(n_time, 2)``.

    Args:
        values: The per-step face values.

    Returns:
        MeshDataset: The in-memory mesh with a positional time axis.
    """
    data = np.asarray(values, dtype="float64")
    mesh = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"depth": data},
        data_locations={"depth": "face"},
    )
    mesh["depth"].dimensions = ("time", "mesh2d_nFaces")
    mesh["depth"].attributes = {
        **mesh["depth"].attributes,
        "time_values": list(range(len(values))),
    }
    return mesh


def _cf(values: list) -> MeshDataset:
    """A temporal mesh whose time axis carries a decoded CF calendar.

    The CF ``units``/``calendar`` attributes survive the NetCDF roundtrip, so the read-back
    mesh presents real dates — which ``resample`` / ``convert-calendar`` / ``interp-calendar``
    require.

    Args:
        values: The per-step face values.

    Returns:
        MeshDataset: The in-memory mesh with a CF time axis.
    """
    mesh = _temporal(values)
    mesh["depth"].attributes = {
        **mesh["depth"].attributes,
        TIME_UNITS_ATTR: "days since 2000-01-01",
        TIME_CALENDAR_ATTR: "standard",
    }
    return mesh


def _reopen_has_depth(path: str) -> bool:
    """Return whether the mesh at ``path`` re-opens with a ``depth`` variable.

    Args:
        path: The written mesh path.

    Returns:
        bool: ``True`` when the file reads back carrying ``depth``.
    """
    return "depth" in MeshDataset.read_file(path).data_variable_names


@pytest.fixture
def plain(tmp_path) -> str:
    """A written three-step mesh with finite values."""
    path = str(tmp_path / "plain.nc")
    _temporal([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).to_file(path)
    return path


@pytest.fixture
def gappy(tmp_path) -> str:
    """A written three-step mesh whose middle step is all missing."""
    path = str(tmp_path / "gappy.nc")
    _temporal([[1.0, 2.0], [np.nan, np.nan], [5.0, 6.0]]).to_file(path)
    return path


@pytest.fixture
def cf(tmp_path) -> str:
    """A written three-step mesh carrying a CF (dated) time axis."""
    path = str(tmp_path / "cf.nc")
    _cf([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).to_file(path)
    return path


@pytest.fixture
def unitful(tmp_path) -> str:
    """A written mesh whose ``depth`` variable declares Kelvin units."""
    path = str(tmp_path / "unit.nc")
    mesh = _temporal([[300.0, 301.0], [302.0, 303.0], [304.0, 305.0]])
    mesh["depth"].units = "K"
    mesh.to_file(path)
    return path


class TestInterpolation:
    """``interp-like`` / ``interp-time`` / ``interp-calendar``."""

    def test_interp_like(self, plain, tmp_path):
        """interp-like interpolates the mesh onto another mesh's grid."""
        out = str(tmp_path / "out.nc")
        assert (
            _dispatch(["mesh", "interp", "interp-like", plain, out, "--other", plain])
            == 0
        )
        assert _reopen_has_depth(out)

    def test_interp_time(self, plain, tmp_path):
        """interp-time resamples onto new (positional) time coordinates."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            ["mesh", "interp", "interp-time", plain, out, "--time", "0.5", "1.5"]
        )
        assert rc == 0
        assert _reopen_has_depth(out)

    def test_interp_time_method_and_bounds(self, plain, tmp_path):
        """The --method / --bounds choices are forwarded to interp_time."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            [
                "mesh",
                "interp",
                "interp-time",
                plain,
                out,
                "--time",
                "1.0",
                "--method",
                "nearest",
                "--bounds",
                "clamp",
            ]
        )
        assert rc == 0
        assert _reopen_has_depth(out)

    def test_interp_calendar(self, cf, tmp_path):
        """interp-calendar interpolates onto a target mesh's time axis."""
        out = str(tmp_path / "out.nc")
        assert (
            _dispatch(["mesh", "interp", "interp-calendar", cf, out, "--target", cf])
            == 0
        )
        assert _reopen_has_depth(out)


class TestCalendar:
    """``convert-calendar``."""

    def test_convert_calendar(self, cf, tmp_path):
        """convert-calendar recasts the time axis to the requested calendar."""
        out = str(tmp_path / "out.nc")
        assert (
            _dispatch(
                ["mesh", "interp", "convert-calendar", cf, out, "--calendar", "noleap"]
            )
            == 0
        )
        assert _reopen_has_depth(out)


class TestFillFamily:
    """``interpolate-na`` / ``fillna`` / ``ffill`` / ``bfill`` / ``fill-na-spatial`` /
    ``laplace-interpolate`` — the post-#1463 fill family."""

    def test_interpolate_na(self, gappy, tmp_path):
        """interpolate-na fills gaps by interpolating along the time axis."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "interp", "interpolate-na", gappy, out]) == 0
        assert _reopen_has_depth(out)

    def test_interpolate_na_no_use_coordinate(self, gappy, tmp_path):
        """--no-use-coordinate switches interpolation to index positions."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            ["mesh", "interp", "interpolate-na", gappy, out, "--no-use-coordinate"]
        )
        assert rc == 0
        assert _reopen_has_depth(out)

    def test_fillna(self, gappy, tmp_path):
        """fillna replaces missing values with the --value constant."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "interp", "fillna", gappy, out, "--value", "0"]) == 0
        assert _reopen_has_depth(out)

    def test_ffill(self, gappy, tmp_path):
        """ffill forward-fills gaps along the time axis."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "ffill", gappy, out]) == 0
        assert _reopen_has_depth(out)

    def test_bfill(self, gappy, tmp_path):
        """bfill back-fills gaps along the time axis."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "bfill", gappy, out, "--limit", "1"]) == 0
        assert _reopen_has_depth(out)

    def test_fill_na_spatial(self, gappy, tmp_path):
        """fill-na-spatial fills gaps from nearby cells in space."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "interp", "fill-na-spatial", gappy, out]) == 0
        assert _reopen_has_depth(out)

    def test_laplace_interpolate(self, gappy, tmp_path):
        """laplace-interpolate fills gaps via a graph-Laplacian solve."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            [
                "mesh",
                "interp",
                "laplace-interpolate",
                gappy,
                out,
                "--no-weights",
                "--direct-solve",
            ]
        )
        assert rc == 0
        assert _reopen_has_depth(out)


class TestWindowed:
    """``resample`` / ``rolling`` / ``coarsen``."""

    def test_resample(self, cf, tmp_path):
        """resample bins the dated time axis to a frequency and reduces."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "interp", "resample", cf, out, "--freq", "2D"]) == 0
        assert _reopen_has_depth(out)

    def test_resample_no_skipna(self, cf, tmp_path):
        """--no-skipna is forwarded to resample."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            ["mesh", "interp", "resample", cf, out, "--freq", "2D", "--no-skipna"]
        )
        assert rc == 0
        assert _reopen_has_depth(out)

    def test_rolling(self, plain, tmp_path):
        """rolling applies a moving-window reduction along the time axis."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            [
                "mesh",
                "advanced",
                "rolling",
                plain,
                out,
                "--window",
                "2",
                "--center",
                "--min-periods",
                "1",
            ]
        )
        assert rc == 0
        assert _reopen_has_depth(out)

    def test_coarsen(self, plain, tmp_path):
        """coarsen block-reduces the time axis by a window."""
        out = str(tmp_path / "out.nc")
        assert (
            _dispatch(["mesh", "advanced", "coarsen", plain, out, "--window", "2"]) == 0
        )
        assert _reopen_has_depth(out)


class TestCalculus:
    """``integrate`` / ``cumulative-integrate`` / ``differentiate``."""

    def test_integrate(self, plain, tmp_path):
        """integrate collapses the time axis to a single integral."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "integrate", plain, out]) == 0
        assert _reopen_has_depth(out)

    def test_cumulative_integrate(self, plain, tmp_path):
        """cumulative-integrate produces a running integral along the time axis."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "cumulative-integrate", plain, out]) == 0
        assert _reopen_has_depth(out)

    def test_differentiate(self, plain, tmp_path):
        """differentiate produces the derivative along the time axis."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "differentiate", plain, out]) == 0
        assert _reopen_has_depth(out)


class TestValueTransforms:
    """``convert-units`` / ``shift`` / ``dropna``."""

    def test_convert_units(self, unitful, tmp_path):
        """convert-units converts the named variable to target units."""
        out = str(tmp_path / "out.nc")
        rc = _dispatch(
            [
                "mesh",
                "interp",
                "convert-units",
                unitful,
                out,
                "--to",
                "celsius",
                "--var",
                "depth",
            ]
        )
        assert rc == 0
        reopened = MeshDataset.read_file(out)
        np.testing.assert_allclose(
            np.asarray(reopened["depth"].data)[0], [26.85, 27.85]
        )

    def test_shift(self, plain, tmp_path):
        """shift moves values along the time axis by --periods."""
        out = str(tmp_path / "out.nc")
        assert (
            _dispatch(["mesh", "advanced", "shift", plain, out, "--periods", "1"]) == 0
        )
        assert _reopen_has_depth(out)

    def test_dropna(self, gappy, tmp_path):
        """dropna drops time steps that carry missing values."""
        out = str(tmp_path / "out.nc")
        assert (
            _dispatch(["mesh", "advanced", "dropna", gappy, out, "--how", "any"]) == 0
        )
        assert _reopen_has_depth(out)


class TestErrors:
    """Shared guards: the overwrite refusal and the clean unknown-variable error."""

    def test_refuses_existing_without_overwrite(self, plain, tmp_path):
        """A second write to the same path without --overwrite raises ValueError."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "integrate", plain, out]) == 0
        with pytest.raises(ValueError, match="already exists"):
            _dispatch(["mesh", "advanced", "integrate", plain, out])

    def test_overwrite_allows_replacing(self, plain, tmp_path):
        """--overwrite replaces an existing output."""
        out = str(tmp_path / "out.nc")
        assert _dispatch(["mesh", "advanced", "integrate", plain, out]) == 0
        assert (
            _dispatch(["mesh", "advanced", "integrate", plain, out, "--overwrite"]) == 0
        )

    def test_unknown_variable_is_clean_valueerror(self, plain, tmp_path):
        """A typo'd --var surfaces as ValueError (remapped from KeyError), not KeyError."""
        out = str(tmp_path / "out.nc")
        with pytest.raises(ValueError, match="data variable"):
            _dispatch(["mesh", "advanced", "integrate", plain, out, "--var", "nope"])
