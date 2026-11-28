"""The ``mesh.io`` accessor owns the I/O family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650) for the last and most import-entangled
engine: the serialisation / hand-off family (``to_file`` / ``to_bytes`` / ``to_zarr`` /
``to_dataset`` / ``rasterize_like`` / ``burn``) moves onto a dedicated ``mesh.io`` engine,
mirroring the raster write surface, and the flat ``mesh.to_file`` / ``mesh.burn`` / … methods
become deprecating facades that delegate to it, so both call forms return identical results. Each
flat method is exercised under ``pytest.warns`` here against the engine path; the engine path is
asserted free of the migration's deprecation warning.
"""

import warnings

import numpy as np
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.io import _MeshIO

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 4.0, 4.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 2.0, 1.0])
_FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _static() -> MeshDataset:
    """Two triangles of unequal area carrying a static per-face ``v``."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([10.0, 20.0])},
        data_locations={"v": "face"},
        epsg=4326,
    )


def test_accessor_is_an_io_engine():
    """``mesh.io`` is a fresh ``_MeshIO`` bound to the dataset."""
    assert isinstance(_static().io, _MeshIO)


def test_to_file_matches_flat_facade(tmp_path):
    """``mesh.io.to_file`` round-trips to the same variables as the deprecated flat form."""
    flat_path = tmp_path / "flat.nc"
    engine_path = tmp_path / "engine.nc"
    subject = _static()
    with pytest.warns(DeprecationWarning, match=r"mesh\.to_file\b"):
        subject.to_file(flat_path)
    _static().io.to_file(engine_path)
    flat = MeshDataset.read_file(flat_path)
    engine = MeshDataset.read_file(engine_path)
    assert engine.variable_names == flat.variable_names


def test_to_bytes_matches_flat_facade(tmp_path):
    """``mesh.io.to_bytes`` serialises to a buffer that reads back like the flat form."""
    subject = _static()
    with pytest.warns(DeprecationWarning, match=r"mesh\.to_bytes\b"):
        flat = subject.to_bytes()
    engine = _static().io.to_bytes()
    assert isinstance(engine, bytes)
    assert isinstance(flat, bytes)
    flat_path = tmp_path / "flat_bytes.nc"
    engine_path = tmp_path / "engine_bytes.nc"
    flat_path.write_bytes(flat)
    engine_path.write_bytes(engine)
    assert (
        MeshDataset.read_file(engine_path).variable_names
        == MeshDataset.read_file(flat_path).variable_names
    )


def test_to_dataset_matches_flat_facade():
    """``mesh.io.to_dataset`` rasterises to the same grid as the deprecated flat form."""
    subject = _static()
    with pytest.warns(DeprecationWarning, match=r"mesh\.to_dataset\b"):
        flat = subject.to_dataset("v", cell_size=0.5, method="nearest")
    engine = _static().io.to_dataset("v", cell_size=0.5, method="nearest")
    assert isinstance(engine, Dataset)
    np.testing.assert_array_equal(engine.io.read_array(), flat.io.read_array())


def test_rasterize_like_matches_flat_facade():
    """``mesh.io.rasterize_like`` snaps to the reference grid like the flat form."""
    reference = _static().io.to_dataset("v", cell_size=0.5, method="nearest")
    subject = _static()
    with pytest.warns(DeprecationWarning, match=r"mesh\.rasterize_like\b"):
        flat = subject.rasterize_like(reference, "v")
    engine = _static().io.rasterize_like(reference, "v")
    np.testing.assert_array_equal(engine.io.read_array(), flat.io.read_array())


def test_burn_matches_flat_facade():
    """``mesh.io.burn`` burns the geometry onto the faces like the flat form."""
    geom = box(-1.0, -1.0, 1.05, 2.0)
    subject = _static()
    with pytest.warns(DeprecationWarning, match=r"mesh\.burn\b"):
        flat = subject.burn(geom, 9.0, name="b")
    engine = _static().io.burn(geom, 9.0, name="b")
    assert isinstance(engine, MeshDataset)
    np.testing.assert_array_equal(engine["b"].data, flat["b"].data)


def test_to_zarr_matches_flat_facade(tmp_path):
    """``mesh.io.to_zarr`` writes the same variables as the flat form (zarr only)."""
    pytest.importorskip("zarr")
    xr = pytest.importorskip("xarray")
    flat_store = tmp_path / "flat.zarr"
    engine_store = tmp_path / "engine.zarr"
    subject = _static()
    with pytest.warns(DeprecationWarning, match=r"mesh\.to_zarr\b"):
        subject.to_zarr(flat_store)
    _static().io.to_zarr(engine_store)
    flat = xr.open_dataset(flat_store, engine="zarr")
    engine = xr.open_dataset(engine_store, engine="zarr")
    assert set(engine.data_vars) == set(flat.data_vars)


def test_accessor_form_does_not_warn(tmp_path):
    """Every ``mesh.io`` call is the non-deprecated path -- no migration ``DeprecationWarning``.

    The I/O methods touch GDAL / netCDF, which can emit unrelated third-party
    ``DeprecationWarning``s, so this records warnings and asserts only that none of the
    engine-migration ones (the ``will be removed in 0.90.0`` facades) fire -- rather than turning
    every ``DeprecationWarning`` into an error as the pure-compute accessor tests can.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _static()
        ds.io.to_file(tmp_path / "engine_nowarn.nc")
        ds.io.to_bytes()
        reference = ds.io.to_dataset("v", cell_size=0.5, method="nearest")
        ds.io.rasterize_like(reference, "v")
        ds.io.burn(box(-1.0, -1.0, 1.05, 2.0), 9.0, name="b")
    offenders = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "will be removed in 0.90.0" in str(w.message)
    ]
    assert not offenders, offenders
