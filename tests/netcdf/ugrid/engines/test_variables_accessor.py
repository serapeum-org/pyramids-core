"""The ``mesh.variables`` accessor owns the data-variable editing family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): ``with_variable`` / ``attach_data`` /
``rename_variable`` / ``drop_variables`` move onto a dedicated ``mesh.variables`` engine, mirroring
the raster surface, and the flat ``mesh.with_variable`` / ``mesh.attach_data`` /
``mesh.rename_variable`` / ``mesh.drop_variables`` methods become deprecating facades that delegate
to it, so both call forms do the same thing.

``attach_data`` is the one method whose body stays on ``MeshDataset`` (as ``_attach_data_impl``)
because it reads a results file through module-level helpers the engine cannot import without a
cycle; the engine method delegates to it, which these tests pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.variables import _MeshVariables

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh() -> MeshDataset:
    """A two-triangle mesh carrying a face-located ``v`` variable, with a CRS."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([1.0, 2.0])},
        data_locations={"v": "face"},
        epsg=4326,
    )


def test_accessor_is_a_variables_engine():
    """``mesh.variables`` is a fresh ``_MeshVariables`` bound to the dataset."""
    mesh = _mesh()
    assert isinstance(mesh.variables, _MeshVariables)


def test_with_variable_accessor_matches_flat_facade():
    """``mesh.variables.with_variable()`` adds a variable; the deprecated ``mesh.with_variable()`` agrees."""
    mesh = _mesh()
    extra = np.array([3.0, 4.0])
    with pytest.warns(DeprecationWarning, match="mesh.with_variable"):
        flat = mesh.with_variable("w", extra, location="face")
    accessor = mesh.variables.with_variable("w", extra, location="face")
    assert accessor.variable_names == flat.variable_names == ["v", "w"]
    np.testing.assert_array_equal(
        np.asarray(accessor["w"].data), np.asarray(flat["w"].data)
    )


def test_rename_variable_accessor_matches_flat_facade():
    """``mesh.variables.rename_variable()`` relabels a variable; the flat method agrees."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.rename_variable"):
        flat = mesh.rename_variable("v", "z")
    accessor = mesh.variables.rename_variable("v", "z")
    assert accessor.variable_names == flat.variable_names == ["z"]


def test_drop_variables_accessor_matches_flat_facade():
    """``mesh.variables.drop_variables()`` removes a variable; the flat method agrees."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.drop_variables"):
        flat = mesh.drop_variables("v")
    accessor = mesh.variables.drop_variables("v")
    assert accessor.variable_names == flat.variable_names == []


def test_attach_data_accessor_matches_flat_facade(monkeypatch):
    """``mesh.variables.attach_data()`` and the deprecated ``mesh.attach_data()`` both route to the impl."""
    mesh = _mesh()
    monkeypatch.setattr(
        MeshDataset, "_attach_data_impl", lambda self, *a, **k: "sentinel"
    )
    with pytest.warns(DeprecationWarning, match="mesh.attach_data"):
        flat = mesh.attach_data("velocity.dat")
    accessor = mesh.variables.attach_data("velocity.dat")
    assert accessor == flat == "sentinel"


def test_attach_data_accessor_delegates_to_the_dataset_impl(monkeypatch):
    """``mesh.variables.attach_data`` routes through ``MeshDataset._attach_data_impl`` (the cycle break)."""
    mesh = _mesh()
    calls: list[tuple] = []
    monkeypatch.setattr(
        MeshDataset,
        "_attach_data_impl",
        lambda self, *a, **k: calls.append((a, k)) or "sentinel",
    )
    assert mesh.variables.attach_data("velocity.dat", "mydriver", False) == "sentinel"
    assert calls == [(("velocity.dat", "mydriver", False), {})]


def test_accessor_form_does_not_warn(monkeypatch):
    """The engine call is the non-deprecated path -- it emits no ``mesh.*`` deprecation."""
    mesh = _mesh()
    monkeypatch.setattr(MeshDataset, "_attach_data_impl", lambda self, *a, **k: mesh)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh.variables.with_variable("w", np.array([3.0, 4.0]), location="face")
        mesh.variables.rename_variable("v", "z")
        mesh.variables.drop_variables("v")
        mesh.variables.attach_data("velocity.dat")
    mesh_deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and str(w.message).startswith("`mesh.")
    ]
    assert not mesh_deprecations, [str(w.message) for w in mesh_deprecations]
