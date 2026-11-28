"""The shared ``_MeshEngine`` base: a directly-pickled engine round-trips to a placeholder.

Pins the ugrid engine-accessor migration (issue #1650): every ``MeshDataset`` engine subclasses
:class:`~pyramids.netcdf.ugrid.engines._base._MeshEngine`, which mirrors the raster
``pyramids.dataset.engines._base._Engine`` pickle contract. The accessors are recreated on
demand (they are ``MeshDataset`` properties), so an engine is never part of the dataset's own
pickle. If user code pickles a bare engine, ``__reduce__`` hands back a
:class:`~pyramids.netcdf.ugrid.engines._base._MeshPlaceholder` instead of pickling a circular
reference through the owning dataset. These tests pin that contract.
"""

import pickle

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines._base import (
    _MeshEngine,
    _MeshPlaceholder,
    _recreate_mesh_placeholder,
)

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh() -> MeshDataset:
    """A two-triangle mesh carrying a node-located ``z`` variable."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"z": _NODE_X},
        data_locations={"z": "node"},
    )


def test_engine_is_bound_to_its_dataset():
    """An engine holds a strong reference to the owning dataset in its single slot."""
    mesh = _mesh()
    engine = mesh.terrain
    assert isinstance(engine, _MeshEngine)
    assert engine._ds is mesh


def test_reduce_returns_the_placeholder_recipe():
    """``__reduce__`` reduces to ``_recreate_mesh_placeholder`` with no args (no dataset cycle)."""
    func, args = _mesh().terrain.__reduce__()
    assert func is _recreate_mesh_placeholder
    assert args == ()


def test_recreate_builds_a_bare_placeholder():
    """The reduce target constructs a fresh :class:`_MeshPlaceholder`."""
    assert isinstance(_recreate_mesh_placeholder(), _MeshPlaceholder)


def test_pickled_engine_round_trips_to_a_placeholder():
    """Pickling a bare engine yields a ``_MeshPlaceholder`` rather than pickling through ``_ds``."""
    restored = pickle.loads(pickle.dumps(_mesh().terrain))
    assert isinstance(restored, _MeshPlaceholder)
