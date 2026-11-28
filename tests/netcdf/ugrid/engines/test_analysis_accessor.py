"""The ``mesh.analysis`` accessor owns the analysis family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the per-element analysis family
(cell-wise transforms, comparisons, focal aggregations, morphology / coverage, spatial fillers,
sampling / extraction, domain descriptors, the topology ``clip`` and the whole-axis time
reductions) moves onto a dedicated ``mesh.analysis`` engine, mirroring the raster surface, and the
flat ``mesh.apply`` / ``mesh.round`` / ``mesh.mean`` / ``mesh.equals`` / … methods become
deprecating facades that delegate to it, so both call forms return identical results. One flat
method per group is exercised under ``pytest.warns`` here; the engine path is asserted
warning-free.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.analysis import _MeshAnalysis

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


def _gappy() -> MeshDataset:
    """A two-face mesh whose ``v`` carries one declared no-data cell."""
    mesh = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([5.0, -9999.0])},
        data_locations={"v": "face"},
        epsg=4326,
    )
    mesh["v"].nodata = -9999.0
    return mesh


def _temporal() -> MeshDataset:
    """A two-face mesh carrying a temporal ``v`` shaped ``(n_time, 2)``."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([[1.0, 2.0], [3.0, 4.0]])},
        data_locations={"v": "face"},
        epsg=4326,
    )


def _zones() -> FeatureCollection:
    """A single polygon covering both faces, in the mesh CRS."""
    return FeatureCollection(
        gpd.GeoDataFrame({"z": ["a"]}, geometry=[box(-1, -1, 5, 3)], crs="EPSG:4326")
    )


def _norm(result):
    """A comparable snapshot of a method result."""
    if isinstance(result, MeshDataset):
        return {name: np.asarray(result[name].data) for name in result.variable_names}
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, tuple):
        return tuple(np.asarray(part) for part in result)
    if isinstance(result, np.ndarray):
        return result
    return result


def _assert_equal(a, b):
    na, nb = _norm(a), _norm(b)
    if isinstance(na, dict):
        assert na.keys() == nb.keys()
        for key in na:
            np.testing.assert_array_equal(na[key], nb[key])
    elif isinstance(na, pd.DataFrame):
        pd.testing.assert_frame_equal(na, nb)
    elif isinstance(na, tuple):
        assert len(na) == len(nb)
        for left, right in zip(na, nb):
            np.testing.assert_array_equal(left, right)
    elif isinstance(na, np.ndarray):
        np.testing.assert_array_equal(na, nb)
    else:
        assert na == pytest.approx(nb) if isinstance(na, float) else na == nb


# (name, builder, invocation) -- invocation is called once on the flat dataset and once on the
# ``mesh.analysis`` engine; the two results must be identical.
_CASES = [
    ("apply", _static, lambda o: o.apply(lambda a: a * 2.0)),
    ("astype", _static, lambda o: o.astype("int32")),
    ("clip_values", _static, lambda o: o.clip_values(12.0, 18.0)),
    ("round", _static, lambda o: o.round(0)),
    ("fillna", _gappy, lambda o: o.fillna(0.0)),
    ("where", _static, lambda o: o.where(np.array([True, False]))),
    ("isin", _static, lambda o: o.isin([10.0])),
    ("isnull", _gappy, lambda o: o.isnull()),
    ("focal_mean", _static, lambda o: o.focal_mean(rings=1)),
    ("sieve", _static, lambda o: o.sieve(1)),
    ("proximity", _static, lambda o: o.proximity(np.array([0]))),
    (
        "zonal_stats",
        _static,
        lambda o: o.zonal_stats(_zones(), variable_name="v", stats=("mean", "count")),
    ),
    (
        "sample",
        _static,
        lambda o: o.sample("v", x=[3.0, 1.0], y=[0.5, 0.9], method="contains"),
    ),
    ("get_histogram", _static, lambda o: o.get_histogram(bins=4)),
    ("mean", _temporal, lambda o: o.mean()),
    ("sum", _temporal, lambda o: o.sum()),
    ("max", _temporal, lambda o: o.max()),
    ("equals", _static, lambda o: o.equals(_static())),
    ("to_location", _static, lambda o: o.to_location("node")["v"].data),
]


def test_accessor_is_an_analysis_engine():
    """``mesh.analysis`` is a fresh ``_MeshAnalysis`` bound to the dataset."""
    assert isinstance(_static().analysis, _MeshAnalysis)


@pytest.mark.parametrize("name,builder,invoke", _CASES, ids=[c[0] for c in _CASES])
def test_accessor_matches_flat_facade(name, builder, invoke):
    """``mesh.analysis.<name>()`` equals the deprecated flat ``mesh.<name>()``."""
    subject = builder()
    with pytest.warns(DeprecationWarning, match=rf"mesh\.{name}\b"):
        flat = invoke(subject)
    accessor = invoke(builder().analysis)
    _assert_equal(accessor, flat)


@pytest.mark.parametrize(
    "name,builder,invoke",
    [
        c
        for c in _CASES
        if c[0] in {"apply", "round", "mean", "equals", "focal_mean", "zonal_stats"}
    ],
    ids=["apply", "round", "mean", "equals", "focal_mean", "zonal_stats"],
)
def test_accessor_form_does_not_warn(name, builder, invoke):
    """The engine call is the non-deprecated path -- no ``DeprecationWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        invoke(builder().analysis)
