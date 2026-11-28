"""The ``mesh.interop`` accessor owns to_xarray/to_dataframe/to_dask_dataframe; facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the PyData hand-off family moves onto a
dedicated ``mesh.interop`` engine, mirroring ``NetCDF.interop``, and the flat
``mesh.to_xarray`` / ``mesh.to_dataframe`` / ``mesh.to_dask_dataframe`` methods become
deprecating facades that delegate to it, so both call forms return identical results.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.interop import _MeshInterop

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh(node_z=None) -> MeshDataset:
    """A two-triangle mesh carrying a node-located ``z`` variable."""
    values = _NODE_X if node_z is None else np.asarray(node_z, dtype="float64")
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"z": np.asarray(values, dtype="float64")},
        data_locations={"z": "node"},
    )


def test_accessor_is_an_interop_engine():
    """``mesh.interop`` is a fresh ``_MeshInterop`` bound to the dataset."""
    mesh = _mesh()
    assert isinstance(mesh.interop, _MeshInterop)


def test_to_dataframe_accessor_matches_flat_facade():
    """``mesh.interop.to_dataframe()`` equals the deprecated ``mesh.to_dataframe()``."""
    import pandas as pd

    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_dataframe"):
        flat = mesh.to_dataframe()
    pd.testing.assert_frame_equal(mesh.interop.to_dataframe(), flat)


def test_to_xarray_accessor_matches_flat_facade():
    """``mesh.interop.to_xarray()`` equals the deprecated ``mesh.to_xarray()``."""
    pytest.importorskip("xarray")
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_xarray"):
        flat = mesh.to_xarray()
    assert mesh.interop.to_xarray().identical(flat)


def test_to_dask_dataframe_accessor_matches_flat_facade():
    """``mesh.interop.to_dask_dataframe()`` equals the deprecated ``mesh.to_dask_dataframe()``."""
    pytest.importorskip("dask.dataframe")
    import pandas as pd

    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_dask_dataframe"):
        lazy_flat = mesh.to_dask_dataframe(npartitions=2)
    flat = lazy_flat.compute()
    accessor = mesh.interop.to_dask_dataframe(npartitions=2).compute()
    pd.testing.assert_frame_equal(accessor, flat)


def test_npartitions_forwarded_through_accessor():
    """The ``npartitions`` argument reaches the engine."""
    pytest.importorskip("dask.dataframe")
    mesh = _mesh()
    assert mesh.interop.to_dask_dataframe(npartitions=3).npartitions == 3


def test_to_dask_dataframe_rejects_bad_npartitions():
    """``npartitions < 1`` is refused before any optional-dask import is attempted."""
    mesh = _mesh()
    with pytest.raises(ValueError, match="npartitions=0"):
        mesh.interop.to_dask_dataframe(npartitions=0)


def test_accessor_form_does_not_warn():
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    pytest.importorskip("xarray")
    mesh = _mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mesh.interop.to_dataframe()
        mesh.interop.to_xarray()
