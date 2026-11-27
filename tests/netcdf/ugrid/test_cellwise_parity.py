"""The raster and the mesh must agree, cell for cell, on the shared missing-value kernels.

D1 of the netcdf-ugrid duplication audit collapses the cell-wise kernels in
``pyramids.base._array_ops`` into the single implementation both ``Dataset`` (raster) and
``MeshDataset`` (mesh) call. This module drives the two hierarchies from the SAME cases and
asserts they land on the same values and declare the same gap -- a standing invariant rather
than two suites that happen to match today. The seven repro rows from the audit are its first
cases; the ``astype(float16)`` row is the one the collapse flips (the mesh used to over-refuse
a sentinel the raster snaps and keeps).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from tests.netcdf.ugrid.conftest import make_two_triangle_ds

pytestmark = pytest.mark.core

_GEO = GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326)


def _raster(values, nodata, dtype=None):
    return Dataset.from_array(
        np.array([values], dtype=dtype), geo_ref=_GEO, no_data_value=nodata
    )


def _mesh(values, nodata, dtype=None):
    return make_two_triangle_ds(values=np.array(values, dtype=dtype), nodata=nodata)


def _raster_out(ds):
    return np.asarray(ds.io.read_array()).ravel()


def _mesh_out(ds):
    return np.asarray(ds["depth"].data).ravel()


# (name, (values, nodata, dtype), raster-call, mesh-call)
_OPS = [
    ("clip", lambda ds: ds.analysis.clip(8.0, 22.0), lambda ds: ds.clip_values(8.0, 22.0)),
    ("round", lambda ds: ds.analysis.round(0), lambda ds: ds.analysis.round(0)),
    ("astype_int32", lambda ds: ds.analysis.astype(np.int32), lambda ds: ds.analysis.astype(np.int32)),
    (
        "astype_float16",
        lambda ds: ds.analysis.astype(np.float16),
        lambda ds: ds.astype(np.float16),
    ),
    (
        "apply",
        lambda ds: ds.analysis.apply(lambda a: a * 2),
        lambda ds: ds.apply(lambda a: a * 2),
    ),
    ("fillna", lambda ds: ds.analysis.fillna(0.0), lambda ds: ds.analysis.fillna(0.0)),
]


@pytest.mark.parametrize(("name", "r_op", "m_op"), _OPS, ids=[o[0] for o in _OPS])
def test_raster_and_mesh_agree_on_a_gap_carrying_array(name, r_op, m_op):
    """A ``-9999`` sentinel, clipped / rounded / cast / transformed, lands the same place."""
    values, nodata = [5.0, -9999.0], -9999.0
    r_result = _raster_out(r_op(_raster(values, nodata)))
    m_result = _mesh_out(m_op(_mesh(values, nodata)))
    np.testing.assert_array_equal(
        m_result, r_result, err_msg=f"{name}: mesh {m_result} != raster {r_result}"
    )


def test_isin_agrees_on_the_gap():
    """A gap is in no set on either hierarchy, whatever band type each returns."""
    r = _raster_out(_raster([5.0, -9999.0], -9999.0).analysis.isin([-9999.0]))
    m = _mesh_out(_mesh([5.0, -9999.0], -9999.0).isin([-9999.0]))
    np.testing.assert_array_equal(m.astype(bool), r.astype(bool))
    assert not r.astype(bool).any()


def test_round_preserves_a_fractional_sentinel_on_both():
    """``round(0)`` leaves ``-9999.5`` where it was declared, raster and mesh alike."""
    r = _raster_out(_raster([5.0, -9999.5], -9999.5).analysis.round(0))
    m = _mesh_out(_mesh([5.0, -9999.5], -9999.5).round(0))
    np.testing.assert_array_equal(m, r)
    assert r.tolist() == [5.0, -9999.5]


def test_fillna_widens_a_float32_band_on_both():
    """``fillna(1e300)`` on a ``float32`` band widens to ``float64`` rather than storing inf."""
    r_ds = _raster([5.0, -9999.0], -9999.0, dtype="float32").analysis.fillna(1e300)
    m_ds = _mesh([5.0, -9999.0], -9999.0, dtype="float32").fillna(1e300)
    np.testing.assert_array_equal(_mesh_out(m_ds), _raster_out(r_ds))
    assert m_ds["depth"].data.dtype == np.float64


@pytest.mark.parametrize("bad", [(22.0, 8.0), (float("nan"), 8.0), (None, None)])
def test_both_refuse_bad_clip_bounds(bad):
    """Crossed, NaN and empty bounds are a ``ValueError`` on either hierarchy."""
    lower, upper = bad
    raster = _raster([5.0, 25.0], -9999.0)
    with pytest.raises(ValueError):
        raster.analysis.clip(lower, upper)
    mesh = _mesh([5.0, 25.0], -9999.0)
    with pytest.raises(ValueError):
        mesh.clip_values(lower, upper)
