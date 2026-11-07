"""Tests for the ``pyramids mesh set-attr`` / ``del-attr`` commands (``cli_ugrid_attrs``).

Each verb is dispatched through a throwaway parser that mirrors how the orchestrator
wires the registrator onto the ``mesh`` sub-parser. Happy paths assert ``0`` and that
the written mesh carries (or no longer carries) the attribute; the error path asserts
the overwrite guard raises. These are extras-free: the mesh reads/writes go through the
built-in UGRID writer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pyramids.cli_ugrid_attrs import add_ugrid_attrs_commands
from pyramids.netcdf.ugrid import MeshDataset

_MESH = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _dispatch(argv: list[str]) -> int:
    """Register the attr commands on a throwaway parser and run the matched handler.

    Args:
        argv: The argument vector, starting with ``"mesh"``.

    Returns:
        int: The handler's exit code.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    mesh = sub.add_parser("mesh")
    msub = mesh.add_subparsers(dest="command", required=True)
    add_ugrid_attrs_commands(msub)
    args = parser.parse_args(argv)
    return args.func(args)


class TestMeshSetAttr:
    """``pyramids mesh set-attr``."""

    def test_writes_attribute(self, tmp_path):
        """set-attr writes a new mesh whose global attributes include the value."""
        out = str(tmp_path / "set.nc")
        rc = _dispatch(
            ["mesh", "set-attr", str(_MESH), out, "--name", "author", "--value", "me"]
        )
        assert rc == 0
        assert MeshDataset.read_file(out).global_attributes.get("author") == "me"

    def test_source_is_not_modified(self, tmp_path):
        """The source mesh keeps its original attributes (immutable derivation)."""
        out = str(tmp_path / "set.nc")
        before = dict(MeshDataset.read_file(str(_MESH)).global_attributes)
        assert (
            _dispatch(
                ["mesh", "set-attr", str(_MESH), out, "--name", "k", "--value", "v"]
            )
            == 0
        )
        assert "k" not in before
        assert dict(MeshDataset.read_file(str(_MESH)).global_attributes) == before

    def test_refuses_existing(self, tmp_path):
        """A second set-attr onto the same path without --overwrite is refused."""
        out = str(tmp_path / "s.nc")
        assert (
            _dispatch(
                ["mesh", "set-attr", str(_MESH), out, "--name", "a", "--value", "1"]
            )
            == 0
        )
        with pytest.raises(ValueError):
            _dispatch(
                ["mesh", "set-attr", str(_MESH), out, "--name", "a", "--value", "1"]
            )

    def test_overwrite_allows_replacement(self, tmp_path):
        """--overwrite replaces an existing output."""
        out = str(tmp_path / "s.nc")
        assert (
            _dispatch(
                ["mesh", "set-attr", str(_MESH), out, "--name", "a", "--value", "1"]
            )
            == 0
        )
        rc = _dispatch(
            [
                "mesh",
                "set-attr",
                str(_MESH),
                out,
                "--name",
                "a",
                "--value",
                "2",
                "--overwrite",
            ]
        )
        assert rc == 0
        assert MeshDataset.read_file(out).global_attributes.get("a") == "2"


class TestMeshDelAttr:
    """``pyramids mesh del-attr``."""

    def test_removes_attribute(self, tmp_path):
        """del-attr writes a new mesh without the named attribute."""
        with_attr = str(tmp_path / "with.nc")
        assert (
            _dispatch(
                [
                    "mesh",
                    "set-attr",
                    str(_MESH),
                    with_attr,
                    "--name",
                    "tag",
                    "--value",
                    "x",
                ]
            )
            == 0
        )
        out = str(tmp_path / "without.nc")
        assert _dispatch(["mesh", "del-attr", with_attr, out, "--name", "tag"]) == 0
        assert "tag" not in MeshDataset.read_file(out).global_attributes

    def test_missing_attribute_is_noop(self, tmp_path):
        """Dropping an absent attribute writes an unchanged copy (exit 0)."""
        out = str(tmp_path / "noop.nc")
        assert _dispatch(["mesh", "del-attr", str(_MESH), out, "--name", "absent"]) == 0
        assert MeshDataset.read_file(out).n_node > 0

    def test_refuses_existing(self, tmp_path):
        """A second del-attr onto the same path without --overwrite is refused."""
        out = str(tmp_path / "d.nc")
        assert _dispatch(["mesh", "del-attr", str(_MESH), out, "--name", "x"]) == 0
        with pytest.raises(ValueError):
            _dispatch(["mesh", "del-attr", str(_MESH), out, "--name", "x"])
