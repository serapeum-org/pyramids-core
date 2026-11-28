"""The ``mesh.selection`` accessor owns the selection/reduction family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the selection / reduction / calculus /
calendar family moves onto a dedicated ``mesh.selection`` engine, mirroring the raster surface,
and the flat ``mesh.sel`` / ``mesh.isel`` / ``mesh.reduce`` / ``mesh.coarsen`` / ``mesh.head`` /
``mesh.weighted`` / ``mesh.sel_layer`` / … methods become deprecating facades that delegate to
it, so both call forms return identical results. One flat method per group is exercised under
``pytest.warns`` here; the engine path is asserted warning-free.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset, VolumeTopology
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.engines.selection import _MeshSelection
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])
_RAMP = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])
_FOUR = np.arange(8.0).reshape(4, 2)


def _static() -> MeshDataset:
    """A two-face mesh carrying a static per-face ``v`` variable."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([1.0, 2.0])},
        data_locations={"v": "face"},
    )


def _temporal(values=_RAMP) -> MeshDataset:
    """A two-face mesh carrying a temporal ``v`` variable shaped ``(n_time, 2)``."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values, dtype="float64")},
        data_locations={"v": "face"},
    )


def _layered(values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])) -> MeshDataset:
    """A two-face mesh carrying a layered ``t`` variable shaped ``(n_layer, 2)``."""
    mesh = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"t": np.asarray(values, dtype="float64")},
        data_locations={"t": "face"},
    )
    mesh["t"].dimensions = ("nLayers", "nMesh2d_face")
    return mesh


def _ragged_mesh() -> Mesh2d:
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        node_y=np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        face_node_connectivity=fnc,
        face_x=np.array([1.0, 2.0, 3.0]),
        face_y=np.array([0.3, 0.6, 0.3]),
    )


def _volume() -> MeshDataset:
    """A ragged 3-D mesh (columns of 3, 1 and 2 layers) carrying a ``volume`` variable."""
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
    return MeshDataset(_ragged_mesh(), {"sal": var}, {}, volume_topology=topology)


def _norm(result):
    """A comparable snapshot of a method result (dataset variables, or a scalar)."""
    if isinstance(result, MeshDataset):
        return {name: np.asarray(result[name].data) for name in result.variable_names}
    return result


def _assert_equal(a, b):
    na, nb = _norm(a), _norm(b)
    if isinstance(na, dict):
        assert na.keys() == nb.keys()
        for key in na:
            np.testing.assert_array_equal(na[key], nb[key])
    else:
        assert na == pytest.approx(nb)


# (name, builder, invocation) -- invocation is called once on the flat dataset and once on the
# ``mesh.selection`` engine; the two results must be identical.
_CASES = [
    ("sel", _temporal, lambda o: o.sel(1)),
    ("isel", _temporal, lambda o: o.isel(time=0)),
    ("reduce", _temporal, lambda o: o.reduce("mean")),
    ("coarsen", lambda: _temporal(_FOUR), lambda o: o.coarsen(2)),
    ("rolling", lambda: _temporal(_FOUR), lambda o: o.rolling(2)),
    ("head", lambda: _temporal(_FOUR), lambda o: o.head(2)),
    ("tail", lambda: _temporal(_FOUR), lambda o: o.tail(2)),
    ("thin", lambda: _temporal(_FOUR), lambda o: o.thin(2)),
    ("squeeze", lambda: _temporal(np.array([[1.0, 2.0]])), lambda o: o.squeeze()),
    ("cumsum", _temporal, lambda o: o.cumsum()),
    ("polyfit", _temporal, lambda o: o.polyfit(1)),
    ("weighted", _static, lambda o: o.weighted("v", how="mean")),
    ("sel_layer", _layered, lambda o: o.sel_layer(1)),
    ("isel_layer", _layered, lambda o: o.isel_layer(0)),
    ("collapse_layer", _volume, lambda o: o.collapse_layer(0)),
]


def test_accessor_is_a_selection_engine():
    """``mesh.selection`` is a fresh ``_MeshSelection`` bound to the dataset."""
    assert isinstance(_static().selection, _MeshSelection)


@pytest.mark.parametrize("name,builder,invoke", _CASES, ids=[c[0] for c in _CASES])
def test_accessor_matches_flat_facade(name, builder, invoke):
    """``mesh.selection.<name>()`` equals the deprecated flat ``mesh.<name>()``."""
    subject = builder()
    with pytest.warns(DeprecationWarning, match=rf"mesh\.{name}\b"):
        flat = invoke(subject)
    accessor = invoke(builder().selection)
    _assert_equal(accessor, flat)


@pytest.mark.parametrize(
    "name,builder,invoke",
    [
        c
        for c in _CASES
        if c[0] in {"sel", "isel", "reduce", "head", "weighted", "collapse_layer"}
    ],
    ids=["sel", "isel", "reduce", "head", "weighted", "collapse_layer"],
)
def test_accessor_form_does_not_warn(name, builder, invoke):
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        invoke(builder().selection)
