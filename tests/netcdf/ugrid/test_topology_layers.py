"""C1 — multi-topology reading; C2 — layer (vertical) selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _layered(values: np.ndarray, layer_values: list | None = None) -> MeshDataset:
    """A two-face mesh carrying a layered variable (layer, element)."""
    mesh = MeshDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=FACES,
        data={"t": values},
        data_locations={"t": "face"},
    )
    mesh["t"].dimensions = ("nLayers", "nMesh2d_face")
    if layer_values is not None:
        mesh["t"].attributes = {
            **mesh["t"].attributes,
            "layer_values": list(layer_values),
        }
    return mesh


class TestMultiTopology:
    def test_mesh_names_lists_the_topology(self, tmp_path: Path):
        mesh = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        path = tmp_path / "one.nc"
        mesh.to_file(path)
        names = MeshDataset.mesh_names(path)
        assert len(names) == 1
        assert names[0] == mesh.mesh_name

    def test_read_all_returns_one_dataset_per_topology(self, tmp_path: Path):
        mesh = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        path = tmp_path / "one.nc"
        mesh.to_file(path)
        everything = MeshDataset.read_all(path)
        assert set(everything) == set(MeshDataset.mesh_names(path))
        only = next(iter(everything.values()))
        assert only["d"].data.tolist() == [1.0, 2.0]

    def test_read_file_accepts_the_mesh_name(self, tmp_path: Path):
        mesh = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        path = tmp_path / "one.nc"
        mesh.to_file(path)
        name = MeshDataset.mesh_names(path)[0]
        assert MeshDataset.read_file(path, mesh_name=name)["d"].data.tolist() == [
            1.0,
            2.0,
        ]

    def test_unknown_mesh_name_raises(self, tmp_path: Path):
        mesh = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        path = tmp_path / "one.nc"
        mesh.to_file(path)
        with pytest.raises(ValueError, match="No UGRID topology named"):
            MeshDataset.read_file(path, mesh_name="not_a_mesh")

    def test_default_read_is_unchanged(self, tmp_path: Path):
        mesh = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        path = tmp_path / "one.nc"
        mesh.to_file(path)
        assert isinstance(MeshDataset.read_file(path), MeshDataset)


class TestLayerSelection:
    def test_n_layers(self):
        assert _layered(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).n_layer == 3

    def test_n_layers_is_none_without_layers(self):
        flat = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        assert flat.n_layer is None

    def test_isel_layer_int_collapses_the_axis(self):
        out = _layered(np.array([[1.0, 2.0], [3.0, 4.0]])).isel_layer(1)
        assert out["t"].data.tolist() == [3.0, 4.0]

    def test_isel_layer_sequence_keeps_the_axis(self):
        out = _layered(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).isel_layer(
            [0, 2]
        )
        assert out["t"].data.tolist() == [[1.0, 2.0], [5.0, 6.0]]

    def test_isel_layer_slice(self):
        out = _layered(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).isel_layer(
            slice(0, 2)
        )
        assert out["t"].data.shape == (2, 2)

    def test_isel_layer_without_layers_raises(self):
        flat = MeshDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=FACES,
            data={"d": np.array([1.0, 2.0])},
        )
        with pytest.raises(
            ValueError, match="needs at least one variable with a layer"
        ):
            flat.isel_layer(0)

    def test_layer_values_and_sel_layer(self):
        mesh = _layered(np.array([[1.0, 2.0], [3.0, 4.0]]), layer_values=[-5.0, -10.0])
        assert mesh.layer_values == [-5.0, -10.0]
        assert mesh.sel_layer(-10.0)["t"].data.tolist() == [3.0, 4.0]

    def test_sel_layer_falls_back_to_positions(self):
        mesh = _layered(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert mesh.sel_layer(1)["t"].data.tolist() == [3.0, 4.0]

    def test_sel_layer_unknown_value_raises(self):
        mesh = _layered(np.array([[1.0, 2.0]]), layer_values=[-5.0])
        with pytest.raises(ValueError, match="not found in the layer coordinate"):
            mesh.sel_layer(-99.0)

    def test_static_variables_are_carried_through(self):
        mesh = _layered(np.array([[1.0, 2.0], [3.0, 4.0]]))
        both = mesh.with_variable("flat", np.array([7.0, 8.0]), location="face")
        out = both.isel_layer(0)
        assert out["flat"].data.tolist() == [7.0, 8.0]
        assert out["t"].data.tolist() == [1.0, 2.0]
