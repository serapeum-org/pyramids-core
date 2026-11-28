"""The ``mesh.spatial`` accessor owns the spatial-op family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the spatial-subset / reprojection /
resampling family moves onto a dedicated ``mesh.spatial`` engine, mirroring the raster
``Dataset.spatial`` surface, and the flat ``mesh.crop`` / ``mesh.subset_by_bounds`` /
``mesh.to_crs`` / ``mesh.set_crs`` / ``mesh.resample`` methods become deprecating facades that
delegate to it, so both call forms return identical results.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.spatial import _MeshSpatial

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh() -> MeshDataset:
    """A two-triangle mesh carrying a face-located ``v`` variable."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([1.0, 2.0])},
        data_locations={"v": "face"},
    )


def _crs_mesh() -> MeshDataset:
    """The same mesh, already referenced to EPSG:4326."""
    mesh = _mesh()
    mesh.spatial.set_crs(epsg=4326)
    return mesh


def _dated_mesh() -> MeshDataset:
    """A three-step mesh variable spanning two calendar months."""
    mesh = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        data_locations={"v": "face"},
    )
    mesh["v"].attributes["time_values"] = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-15"),
        pd.Timestamp("2020-02-01"),
    ]
    return mesh


def test_accessor_is_a_spatial_engine():
    """``mesh.spatial`` is a fresh ``_MeshSpatial`` bound to the dataset."""
    mesh = _mesh()
    assert isinstance(mesh.spatial, _MeshSpatial)


def test_set_crs_accessor_matches_flat_facade():
    """``mesh.spatial.set_crs()`` assigns the same CRS the deprecated ``mesh.set_crs()`` does."""
    flat = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.set_crs"):
        flat.set_crs(epsg=4326)
    accessor = _mesh()
    assert accessor.spatial.set_crs(epsg=4326) is None
    assert accessor.epsg == flat.epsg == 4326
    assert accessor.crs_wkt == flat.crs_wkt


def test_to_crs_accessor_matches_flat_facade():
    """``mesh.spatial.to_crs()`` equals the deprecated ``mesh.to_crs()``."""
    subject = _crs_mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_crs"):
        flat = subject.to_crs(3857)
    accessor = _crs_mesh().spatial.to_crs(3857)
    assert accessor.epsg == flat.epsg == 3857
    np.testing.assert_array_equal(accessor.mesh.node_x, flat.mesh.node_x)
    np.testing.assert_array_equal(accessor.mesh.node_y, flat.mesh.node_y)


def test_subset_by_bounds_accessor_matches_flat_facade():
    """``mesh.spatial.subset_by_bounds()`` equals the deprecated ``mesh.subset_by_bounds()``."""
    subject = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.subset_by_bounds"):
        flat = subject.subset_by_bounds(-0.5, -0.5, 1.5, 1.5)
    accessor = _mesh().spatial.subset_by_bounds(-0.5, -0.5, 1.5, 1.5)
    assert accessor.n_face == flat.n_face
    assert accessor.analysis.equals(flat)


def test_crop_accessor_matches_flat_facade():
    """``mesh.spatial.crop()`` equals the deprecated ``mesh.crop()``."""
    subject = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.crop"):
        flat = subject.crop(bbox=(-0.5, -0.5, 1.5, 1.5))
    accessor = _mesh().spatial.crop(bbox=(-0.5, -0.5, 1.5, 1.5))
    assert accessor.n_face == flat.n_face
    assert accessor.analysis.equals(flat)


def test_resample_accessor_matches_flat_facade():
    """``mesh.spatial.resample()`` equals the deprecated ``mesh.resample()``."""
    subject = _dated_mesh()
    with pytest.warns(DeprecationWarning, match="mesh.resample"):
        flat = subject.resample("1MS")
    accessor = _dated_mesh().spatial.resample("1MS")
    np.testing.assert_array_equal(
        np.asarray(accessor["v"].data), np.asarray(flat["v"].data)
    )


def test_accessor_form_does_not_warn():
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _mesh().spatial.set_crs(epsg=4326)
        _crs_mesh().spatial.to_crs(3857)
        _mesh().spatial.subset_by_bounds(-0.5, -0.5, 1.5, 1.5)
        _mesh().spatial.crop(bbox=(-0.5, -0.5, 1.5, 1.5))
        _dated_mesh().spatial.resample("1MS")


def test_regrid_accessor_matches_flat_facade():
    """`mesh.spatial.regrid()` equals the deprecated flat `mesh.regrid()`."""
    target = _mesh()
    subject = _mesh()
    with pytest.warns(DeprecationWarning, match=r"mesh\.regrid"):
        flat = subject.regrid(target, method="overlap", variable_name="v")
    accessor = _mesh().spatial.regrid(target, method="overlap", variable_name="v")
    np.testing.assert_array_equal(
        np.asarray(accessor["v"].data), np.asarray(flat["v"].data)
    )


def test_regrid_accessor_form_does_not_warn():
    """The `mesh.spatial.regrid` path emits no deprecation."""
    target = _mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _mesh().spatial.regrid(target, method="overlap", variable_name="v")
