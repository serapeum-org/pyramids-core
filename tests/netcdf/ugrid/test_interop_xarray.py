"""Tests for the standalone UGRID xarray / Zarr / dask interop module.

Exercises :mod:`pyramids.netcdf.ugrid.interop_xarray`: the ``to_xarray`` / ``from_xarray`` round trip
(including the connectivity ``start_index`` / ``_FillValue`` encoding trap), the Zarr round trip, and
the dask ``chunked_variables`` lazy path.

``xarray`` is pulled at module scope via ``pytest.importorskip`` and the module carries the
``interop`` marker, so the whole file skips when the optional interop stack is absent (guard
:mod:`tests.guards.test_marker_hygiene`). The Zarr / dask tests add ``@pytest.mark.lazy`` and an
inner ``importorskip`` so they also skip without ``pyramids-lazy``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.errors import OptionalPackageDoesNotExist
from pyramids.netcdf.ugrid import interop_xarray
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import MeshDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop

_SAMPLE_NC = Path("tests/data/netcdf/ugrid/ugrid.nc")


def _mesh_with_two_variables() -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Build a 2-triangle mesh plus a static and a temporal face variable.

    The temporal variable carries explicit ``dimensions=("time", ...)`` so it is detected as temporal
    without relying on the leading-axis fallback (the ``from_arrays`` trap in S.3).

    Returns:
        A ``(mesh, data_variables)`` pair with a static ``depth`` and a temporal ``level``.
    """
    node_x = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    faces = np.array([[0, 1, 2], [1, 4, 3]], dtype=np.intp)
    mesh = Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    static = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(2,),
        units="m",
        nodata=-999.0,
        dimensions=("mesh2d_nFaces",),
        _data=np.array([1.5, 2.5]),
    )
    temporal = MeshVariable(
        name="level",
        location="face",
        mesh_name="mesh2d",
        shape=(3, 2),
        units="m",
        dimensions=("time", "mesh2d_nFaces"),
        _data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    return mesh, {"depth": static, "level": temporal}


class TestToFromXarray:
    """The ``to_xarray`` / ``from_xarray`` round trip."""

    def test_round_trips_nodes_faces_and_values(self):
        """``from_xarray(to_xarray(...))`` reproduces nodes, faces and both variables exactly."""
        mesh, data_vars = _mesh_with_two_variables()

        ds = interop_xarray.to_xarray(mesh, data_vars)
        assert isinstance(ds, xr.Dataset)

        mesh2, data_vars2 = interop_xarray.from_xarray(ds)

        np.testing.assert_array_equal(mesh2.node_x, mesh.node_x)
        np.testing.assert_array_equal(mesh2.node_y, mesh.node_y)
        np.testing.assert_array_equal(
            mesh2.face_node_connectivity.data, mesh.face_node_connectivity.data
        )
        assert mesh2.n_node == mesh.n_node
        assert mesh2.n_face == mesh.n_face

        assert set(data_vars2) == {"depth", "level"}
        np.testing.assert_array_equal(data_vars2["depth"].data, data_vars["depth"].data)
        np.testing.assert_array_equal(data_vars2["level"].data, data_vars["level"].data)

    def test_preserves_variable_metadata_and_time(self):
        """Location, units, nodata and the temporal flag survive the round trip."""
        mesh, data_vars = _mesh_with_two_variables()

        mesh2, data_vars2 = interop_xarray.from_xarray(
            interop_xarray.to_xarray(mesh, data_vars)
        )

        depth = data_vars2["depth"]
        level = data_vars2["level"]
        assert depth.location == "face"
        assert depth.units == "m"
        assert depth.nodata == -999.0
        assert depth.has_time is False
        assert level.has_time is True
        assert level.time_axis_index == 0
        # Element axis stays last.
        assert level.data.shape[-1] == mesh.n_face

    def test_topology_variable_carries_cf_role(self):
        """The exported dataset holds a dummy ``mesh_topology`` variable with UGRID attributes."""
        mesh, data_vars = _mesh_with_two_variables()

        ds = interop_xarray.to_xarray(mesh, data_vars)

        assert ds["mesh2d"].attrs["cf_role"] == "mesh_topology"
        assert ds["mesh2d"].attrs["topology_dimension"] == 2
        assert ds["mesh2d"].attrs["node_coordinates"] == "mesh2d_node_x mesh2d_node_y"
        assert ds["mesh2d"].attrs["face_node_connectivity"] == "mesh2d_face_nodes"

    def test_does_not_mutate_inputs(self):
        """Exporting leaves the source mesh connectivity and variable arrays untouched."""
        mesh, data_vars = _mesh_with_two_variables()
        original_conn = mesh.face_node_connectivity.data.copy()
        original_depth = data_vars["depth"].data.copy()

        interop_xarray.to_xarray(mesh, data_vars)

        np.testing.assert_array_equal(mesh.face_node_connectivity.data, original_conn)
        np.testing.assert_array_equal(data_vars["depth"].data, original_depth)

    def test_from_xarray_rejects_non_dataset(self):
        """A non-``xarray.Dataset`` argument raises ``TypeError``."""
        with pytest.raises(TypeError, match="xarray.Dataset"):
            interop_xarray.from_xarray({"not": "a dataset"})


class TestConnectivityEncoding:
    """The connectivity ``start_index`` / ``_FillValue`` round-trip trap."""

    def test_start_index_and_fill_restored_on_disk(self):
        """A 1-indexed, filled connectivity is emitted with on-disk ``start_index`` / ``_FillValue``."""
        node_x = np.arange(6, dtype=np.float64)
        node_y = np.zeros(6, dtype=np.float64)
        # Internal form: 0-indexed with -1 fill (a quad and a triangle).
        internal = np.array([[0, 1, 4, 3], [1, 2, 5, -1]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=internal,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=1,
            ),
        )

        ds = interop_xarray.to_xarray(mesh, {})
        face_nodes = ds["mesh2d_face_nodes"]

        assert face_nodes.attrs["start_index"] == 1
        assert face_nodes.attrs["_FillValue"] == -999
        # Valid entries shifted by +1; the -1 fill became the file fill -999.
        expected = np.array([[1, 2, 5, 4], [2, 3, 6, -999]])
        np.testing.assert_array_equal(face_nodes.values, expected)

        # And the inverse restores the internal 0-indexed / -1-fill form exactly.
        mesh2, _ = interop_xarray.from_xarray(ds)
        np.testing.assert_array_equal(mesh2.face_node_connectivity.data, internal)
        assert mesh2.face_node_connectivity.original_start_index == 1
        assert mesh2.face_node_connectivity.fill_value == -1


class TestConnectivityFromDiskNanHandling:
    """``_connectivity_from_disk`` against xarray's CF-decoded (float/NaN) and raw-int inputs.

    Exercises the private codec function directly, without a real Zarr round trip, so each
    edge case of the NaN-before-cast fix is isolated and fast: a decoded float array with a
    NaN fill cell, one without any NaN, the raw on-disk integer form (unaffected by the fix),
    and a NaN fill cell combined with a nonzero ``start_index``.
    """

    def test_nan_fill_cell_becomes_minus_one_without_warning(self):
        """A float array with a NaN fill cell decodes to ``-1`` there, raising no warning.

        Test scenario:
            Mirrors what xarray's default CF decoding (``mask_and_scale=True``) hands back
            for a connectivity variable: upcast to float, fill cell replaced by ``NaN``, and
            no ``_FillValue`` in ``attrs`` (it moved to ``.encoding``). The fix must detect
            the ``NaN`` before casting to ``np.intp`` so no ``RuntimeWarning`` fires and the
            cell still ends up as the internal ``-1`` sentinel.
        """
        raw_data = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, np.nan]])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            conn = interop_xarray._connectivity_from_disk(
                raw_data, attrs={"start_index": 0}, cf_role="face_node_connectivity"
            )

        expected = np.array([[0, 1, 2], [0, 1, -1]], dtype=np.intp)
        np.testing.assert_array_equal(
            conn.data, expected, err_msg=f"Expected {expected!r}, got {conn.data!r}"
        )
        assert conn.fill_value == -1, f"Expected fill_value -1, got {conn.fill_value}"

    def test_nan_fill_cell_with_nonzero_start_index(self):
        """A NaN fill cell combined with a nonzero ``start_index`` still decodes correctly.

        Test scenario:
            The NaN-masked cell must be excluded from the ``start_index`` subtraction
            (``data[~mask] -= start_index``) the same way the raw-int path already excludes
            the on-disk sentinel, not shifted along with the valid entries.
        """
        raw_data = np.array([[1.0, 2.0, 5.0, 4.0], [2.0, 3.0, 6.0, np.nan]])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            conn = interop_xarray._connectivity_from_disk(
                raw_data, attrs={"start_index": 1}, cf_role="face_node_connectivity"
            )

        expected = np.array([[0, 1, 4, 3], [1, 2, 5, -1]], dtype=np.intp)
        np.testing.assert_array_equal(
            conn.data, expected, err_msg=f"Expected {expected!r}, got {conn.data!r}"
        )
        assert conn.original_start_index == 1, (
            f"Expected original_start_index 1, got {conn.original_start_index}"
        )

    def test_float_array_without_nan_is_unaffected(self):
        """A float array with no ``NaN`` cells skips the substitution branch cleanly.

        Test scenario:
            A connectivity array that happens to be float-dtyped (e.g. a decoded variable
            whose fill value is absent from this particular slice) but carries no ``NaN``
            must cast and mask exactly as the raw-int path would, with no warning and no
            cells mistakenly treated as fill.
        """
        raw_data = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            conn = interop_xarray._connectivity_from_disk(
                raw_data,
                attrs={"start_index": 0, "_FillValue": -999},
                cf_role="face_node_connectivity",
            )

        expected = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.intp)
        np.testing.assert_array_equal(
            conn.data, expected, err_msg=f"Expected {expected!r}, got {conn.data!r}"
        )

    def test_raw_integer_array_is_unaffected_by_the_nan_guard(self):
        """The raw on-disk integer form (e.g. the GDAL read path) behaves exactly as before.

        Test scenario:
            An integer-dtyped array can never contain ``NaN``, so ``nan_mask`` must stay
            ``None`` and the function must take the original raw-fill-comparison path
            unchanged -- a regression guard that the new float/NaN branch never affects an
            integer input.
        """
        raw_data = np.array([[0, 1, 2], [0, 1, -999]], dtype=np.int64)

        conn = interop_xarray._connectivity_from_disk(
            raw_data,
            attrs={"start_index": 0, "_FillValue": -999},
            cf_role="face_node_connectivity",
        )

        expected = np.array([[0, 1, 2], [0, 1, -1]], dtype=np.intp)
        np.testing.assert_array_equal(
            conn.data, expected, err_msg=f"Expected {expected!r}, got {conn.data!r}"
        )

    def test_missing_fill_value_attr_defaults_to_file_fill_constant(self):
        """A NaN fill cell decodes correctly even when ``_FillValue`` is absent from ``attrs``.

        Test scenario:
            xarray moves a decoded variable's ``_FillValue`` out of ``attrs`` into
            ``.encoding``, so ``attrs`` alone carries no sentinel. ``raw_fill`` must fall
            back to the module's own ``_FILE_FILL`` constant, and the NaN cell must still be
            recognized as fill via ``nan_mask`` regardless of what ``raw_fill`` resolves to.
        """
        raw_data = np.array([[0.0, 1.0, np.nan]])

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            conn = interop_xarray._connectivity_from_disk(
                raw_data, attrs={"start_index": 0}, cf_role="face_node_connectivity"
            )

        expected = np.array([[0, 1, -1]], dtype=np.intp)
        np.testing.assert_array_equal(
            conn.data, expected, err_msg=f"Expected {expected!r}, got {conn.data!r}"
        )


class TestMissingXarray:
    """Guarded import wiring when xarray is unavailable."""

    def test_to_xarray_raises_optional_package(self, monkeypatch):
        """A missing xarray surfaces as ``OptionalPackageDoesNotExist`` from ``to_xarray``."""

        def _boom(_message):
            raise OptionalPackageDoesNotExist("xarray missing")

        monkeypatch.setattr(interop_xarray, "import_xarray", _boom)
        mesh, data_vars = _mesh_with_two_variables()
        with pytest.raises(OptionalPackageDoesNotExist):
            interop_xarray.to_xarray(mesh, data_vars)


@pytest.mark.lazy
class TestZarrRoundTrip:
    """The Zarr round trip, skipped without ``pyramids-lazy``."""

    def test_to_zarr_from_zarr(self, tmp_path):
        """A mesh and its variables round-trip through a Zarr store on disk."""
        pytest.importorskip("zarr")
        mesh, data_vars = _mesh_with_two_variables()
        store = str(tmp_path / "mesh.zarr")

        interop_xarray.to_zarr(mesh, data_vars, store)
        mesh2, data_vars2 = interop_xarray.from_zarr(store)

        np.testing.assert_array_equal(mesh2.node_x, mesh.node_x)
        np.testing.assert_array_equal(
            mesh2.face_node_connectivity.data, mesh.face_node_connectivity.data
        )
        np.testing.assert_array_equal(data_vars2["depth"].data, data_vars["depth"].data)
        np.testing.assert_array_equal(data_vars2["level"].data, data_vars["level"].data)
        assert data_vars2["level"].has_time is True

    def test_mixed_element_fill_survives_the_round_trip(self, tmp_path):
        """A padded (mixed-element) face *and* edge row keep their fill cells as ``-1``.

        xarray's default CF decoding (``mask_and_scale=True``) consumes the on-disk
        ``_FillValue`` on read, upcasting the connectivity array to float and replacing the
        fill cell with ``NaN`` before ``from_zarr`` ever sees it -- unlike
        ``to_xarray``/``from_xarray``, which never leaves memory and so never exercises this
        encode/decode machinery. Casting that ``NaN`` straight to ``np.intp`` is undefined: it
        both warns (``RuntimeWarning: invalid value encountered in cast``) and produces a huge
        negative "index" instead of the ``-1`` sentinel the rest of the mesh expects.

        ``_connectivity_from_disk`` has two call sites in ``_build_mesh_from_dataset`` --
        ``face_node_connectivity`` and ``edge_node_connectivity`` -- so the fixture gives both
        a fill cell to prove the fix through both, not just the face one.
        """
        pytest.importorskip("zarr")
        node_x = np.array([0.0, 1.0, 0.5, 1.5])
        node_y = np.array([0.0, 0.0, 1.0, 1.0])
        # A triangle (padded with -1) and a quad -- mixed element counts force a fill cell.
        internal_faces = np.array([[0, 1, 2, -1], [0, 1, 3, 2]], dtype=np.intp)
        # A synthetic padded edge row (real UGRID edges are always 2 nodes; the 3rd, -1-filled
        # column here exists purely to drive a fill cell through this call site too).
        internal_edges = np.array([[0, 1, -1], [1, 2, 3]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=internal_faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
            edge_node_connectivity=Connectivity(
                data=internal_edges,
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            ),
        )
        store = str(tmp_path / "mixed.zarr")
        interop_xarray.to_zarr(mesh, {}, store)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            mesh2, _ = interop_xarray.from_zarr(store)

        np.testing.assert_array_equal(mesh2.face_node_connectivity.data, internal_faces)
        assert mesh2.edge_node_connectivity is not None, "edge_node_connectivity did not survive the round trip"
        np.testing.assert_array_equal(mesh2.edge_node_connectivity.data, internal_edges)


@pytest.mark.lazy
class TestChunkedVariables:
    """The dask ``chunked_variables`` lazy path, skipped without ``pyramids-lazy``."""

    def test_chunked_read_computes_matching_reduction(self):
        """A chunked read of the sample file yields dask arrays whose reduction matches eager."""
        da = pytest.importorskip("dask.array")
        ug = MeshDataset.read_file(str(_SAMPLE_NC))
        data_vars = ug.data_vars

        chunked = interop_xarray.chunked_variables(data_vars, chunks="auto")

        assert chunked, "expected at least one chunked variable from the sample file"
        name = next(iter(chunked))
        lazy = chunked[name]
        assert isinstance(lazy, da.Array)

        eager = np.asarray(ug[name].data, dtype="float64")
        np.testing.assert_allclose(
            float(lazy.astype("float64").mean().compute()), float(np.nanmean(eager))
        )

    def test_in_memory_variable_is_wrapped_as_dask(self):
        """An in-memory variable (no source path) is wrapped via ``dask.array.from_array``."""
        da = pytest.importorskip("dask.array")
        _, data_vars = _mesh_with_two_variables()

        chunked = interop_xarray.chunked_variables(data_vars, chunks=1)

        assert isinstance(chunked["level"], da.Array)
        np.testing.assert_array_equal(
            chunked["level"].compute(), data_vars["level"].data
        )
