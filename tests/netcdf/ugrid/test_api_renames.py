"""The ugrid naming cleanup: the new names work, the old names still work but warn, and
both resolve to the same thing. MESH-7's GDAL constructors are the exception — they were
buried with no alias, so the public names are gone outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _layered(counts: dict[str, int]) -> MeshDataset:
    """A mesh whose variables sit on the given layer counts (centre/interface pair)."""
    data = {
        name: np.arange(float(count * 2)).reshape(count, 2)
        for name, count in counts.items()
    }
    dataset = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data=data,
        data_locations=dict.fromkeys(counts, "face"),
    )
    for name in counts:
        dataset[name].dimensions = ("mesh2d_nLayers", "mesh2d_nFaces")
    return dataset


def _triangle_mesh() -> Mesh2d:
    return Mesh2d(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=Connectivity(
            data=_TRIANGLES,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


class TestMesh4TimeAxisIndex:
    """MESH-4: ``MeshVariable.time_index`` -> ``time_axis_index``."""

    def _temporal_variable(self) -> MeshVariable:
        return MeshVariable(
            name="water_level",
            location="face",
            mesh_name="mesh2d",
            shape=(3, 2),
            dimensions=("time", "mesh2d_nFaces"),
            _data=np.zeros((3, 2)),
        )

    def test_new_name_returns_the_axis_number(self):
        assert self._temporal_variable().time_axis_index == 0

    def test_old_name_warns_and_matches(self):
        var = self._temporal_variable()
        with pytest.warns(DeprecationWarning, match="time_axis_index"):
            legacy = var.time_index
        assert legacy == var.time_axis_index == 0


class TestMesh5LayerCounts:
    """MESH-5: ``n_layers``/``n_layer_interfaces`` -> ``n_layer``/``n_layer_interface``."""

    def test_new_names(self):
        ds = _layered({"salinity": 3, "w": 4})
        assert ds.n_layer == 3
        assert ds.n_layer_interface == 4

    def test_old_names_warn_and_match(self):
        ds = _layered({"salinity": 3, "w": 4})
        with pytest.warns(DeprecationWarning, match="n_layer"):
            assert ds.n_layers == ds.n_layer == 3
        with pytest.warns(DeprecationWarning, match="n_layer_interface"):
            assert ds.n_layer_interfaces == ds.n_layer_interface == 4


class TestMesh6VariableAccess:
    """MESH-6: ``get_data`` deprecated toward ``mesh[name]``; ``get(name, default)`` added."""

    def _dataset(self) -> MeshDataset:
        return _layered({"salinity": 3})

    def test_subscription_and_get(self):
        ds = self._dataset()
        assert isinstance(ds["salinity"], MeshVariable)
        assert ds.get("salinity") is ds["salinity"]
        assert ds.get("missing") is None
        sentinel = object()
        assert ds.get("missing", sentinel) is sentinel

    def test_subscription_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self._dataset()["missing"]

    def test_get_data_warns_and_matches(self):
        ds = self._dataset()
        with pytest.warns(DeprecationWarning, match=r"mesh\[name\]"):
            legacy = ds.get_data("salinity")
        assert legacy is ds["salinity"]


class TestMesh7GdalConstructorsBuried:
    """MESH-7: the GDAL-fed constructors are private, with no public alias."""

    def test_public_names_are_gone(self):
        assert not hasattr(Mesh2d, "from_gdal_group")
        assert not hasattr(Connectivity, "from_gdal_array")

    def test_private_names_exist(self):
        assert hasattr(Mesh2d, "_from_gdal_group")
        assert hasattr(Connectivity, "_from_gdal_array")


class TestMesh8SpatialTrees:
    """MESH-8: ``face_tree`` -> ``face_centroid_tree``; ``face_strtree`` -> ``face_rtree``."""

    def test_new_names_build(self):
        idx = MeshSpatialIndex(_triangle_mesh())
        assert idx.face_centroid_tree is not None
        assert idx.face_rtree is not None

    def test_old_names_warn_and_match(self):
        idx = MeshSpatialIndex(_triangle_mesh())
        with pytest.warns(DeprecationWarning, match="face_centroid_tree"):
            assert idx.face_tree is idx.face_centroid_tree
        with pytest.warns(DeprecationWarning, match="face_rtree"):
            assert idx.face_strtree is idx.face_rtree
