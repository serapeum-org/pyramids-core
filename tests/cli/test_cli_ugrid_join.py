"""Tests for the ``pyramids mesh concat`` / ``merge`` commands (``cli_ugrid_join``).

Each verb is dispatched through a throwaway parser that mirrors how the orchestrator
wires the registrator onto the ``mesh`` sub-parser. The joins need several same-topology
inputs, so the tests build two copies of the shipped mesh and join them. Happy paths
assert ``0`` and that the output is a readable mesh; error paths assert the handler
raises (too few inputs, the overwrite guard, and merge's conflict policy).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pyramids._cli_groups import add_group
from pyramids.cli_ugrid_join import add_ugrid_join_commands
from pyramids.netcdf.ugrid import MeshDataset

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _dispatch(argv: list[str]) -> int:
    """Register the join commands on a throwaway parser and run the matched handler.

    Args:
        argv: The argument vector, starting with ``"mesh"``.

    Returns:
        int: The handler's exit code.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_join_commands(add_group(msub, "join", "merge/join meshes and data"))
    args = parser.parse_args(argv)
    return args.func(args)


def _two_copies(tmp_path) -> tuple[str, str]:
    """Write two copies of the shipped mesh and return their paths.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        tuple[str, str]: Two same-topology mesh paths.
    """
    first = str(tmp_path / "a.nc")
    second = str(tmp_path / "b.nc")
    MeshDataset.read_file(str(_MESH)).to_file(first)
    MeshDataset.read_file(str(_MESH)).to_file(second)
    return first, second


class TestMeshConcat:
    """``pyramids mesh concat``."""

    def test_joins_two_meshes(self, tmp_path):
        """concat writes a readable same-topology mesh from two inputs."""
        first, second = _two_copies(tmp_path)
        out = str(tmp_path / "concat.nc")
        assert (
            _dispatch(["mesh", "join", "concat", out, "--inputs", first, second]) == 0
        )
        assert MeshDataset.read_file(out).n_face > 0

    def test_refuses_existing(self, tmp_path):
        """A second concat without --overwrite is refused."""
        first, second = _two_copies(tmp_path)
        out = str(tmp_path / "c.nc")
        assert (
            _dispatch(["mesh", "join", "concat", out, "--inputs", first, second]) == 0
        )
        with pytest.raises(ValueError):
            _dispatch(["mesh", "join", "concat", out, "--inputs", first, second])

    def test_single_input_raises(self, tmp_path):
        """A join with fewer than two inputs is a clean ValueError."""
        first, _second = _two_copies(tmp_path)
        out = str(tmp_path / "x.nc")
        with pytest.raises(ValueError):
            _dispatch(["mesh", "join", "concat", out, "--inputs", first])


class TestMeshMerge:
    """``pyramids mesh merge``."""

    def test_override_merges_shared_variables(self, tmp_path):
        """merge --compat override unions the variables of two copies."""
        first, second = _two_copies(tmp_path)
        out = str(tmp_path / "merge.nc")
        rc = _dispatch(
            [
                "mesh",
                "join",
                "merge",
                out,
                "--inputs",
                first,
                second,
                "--compat",
                "override",
            ]
        )
        assert rc == 0
        assert len(MeshDataset.read_file(out).data_variable_names) > 0

    def test_default_compat_rejects_conflict(self, tmp_path):
        """The default no_conflicts policy refuses a variable shared by both inputs."""
        first, second = _two_copies(tmp_path)
        out = str(tmp_path / "m.nc")
        with pytest.raises(ValueError):
            _dispatch(["mesh", "join", "merge", out, "--inputs", first, second])

    def test_refuses_existing(self, tmp_path):
        """A second merge without --overwrite is refused."""
        first, second = _two_copies(tmp_path)
        out = str(tmp_path / "m.nc")
        assert (
            _dispatch(
                [
                    "mesh",
                    "join",
                    "merge",
                    out,
                    "--inputs",
                    first,
                    second,
                    "--compat",
                    "override",
                ]
            )
            == 0
        )
        with pytest.raises(ValueError):
            _dispatch(
                [
                    "mesh",
                    "join",
                    "merge",
                    out,
                    "--inputs",
                    first,
                    second,
                    "--compat",
                    "override",
                ]
            )

    def test_single_input_raises(self, tmp_path):
        """A merge with fewer than two inputs is a clean ValueError."""
        first, _second = _two_copies(tmp_path)
        out = str(tmp_path / "x.nc")
        with pytest.raises(ValueError):
            _dispatch(["mesh", "join", "merge", out, "--inputs", first])
