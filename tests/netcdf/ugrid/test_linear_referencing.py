"""place_nodes_on_network: branch+offset -> x/y linear referencing on a 1D network.

Ground truth is analytic — polylines whose interpolated coordinates can be computed by hand —
so the assertions do not depend on any fixture file or driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import place_nodes_on_network

pytestmark = pytest.mark.core


class TestPlaceNodesOnNetwork:
    """Reconstruction of node coordinates from branch index + along-branch offset."""

    def test_public_export(self):
        import pyramids.netcdf.ugrid as ugrid

        assert "place_nodes_on_network" in ugrid.__all__
        assert ugrid.place_nodes_on_network is place_nodes_on_network

    def test_straight_branch_offsets_at_and_between_vertices(self):
        """A single straight branch: offsets map to raw distance along it."""
        x, y = place_nodes_on_network(
            branch_index=np.array([0, 0, 0]),
            offset=np.array([0.0, 5.0, 10.0]),
            geom_x=np.array([0.0, 10.0]),
            geom_y=np.array([0.0, 0.0]),
            vertices_per_branch=np.array([2]),
        )
        np.testing.assert_allclose(x, [0.0, 5.0, 10.0])
        np.testing.assert_allclose(y, [0.0, 0.0, 0.0])

    def test_two_branches_sliced_from_the_flat_vertex_list(self):
        """Nodes on different branches index into the correct polyline slice."""
        # branch 0: (0,0)->(10,0); branch 1: (0,0)->(0,4)->(3,4), chord 4 then 3.
        x, y = place_nodes_on_network(
            branch_index=np.array([0, 1, 1, 1]),
            offset=np.array([2.5, 4.0, 5.5, 7.0]),
            geom_x=np.array([0.0, 10.0, 0.0, 0.0, 3.0]),
            geom_y=np.array([0.0, 0.0, 0.0, 4.0, 4.0]),
            vertices_per_branch=np.array([2, 3]),
        )
        np.testing.assert_allclose(x, [2.5, 0.0, 1.5, 3.0])
        np.testing.assert_allclose(y, [0.0, 4.0, 4.0, 4.0])

    def test_edge_length_rescales_chainage_to_a_fraction_of_the_polyline(self):
        """A declared length != chord length places the node by fraction, not raw distance."""
        # Polyline chord length is 10, but the branch declares length 20. Chainage 10 is
        # half of the declared length -> the midpoint of the polyline, (5, 0) -- not (10, 0).
        with_len_x, with_len_y = place_nodes_on_network(
            branch_index=np.array([0]),
            offset=np.array([10.0]),
            geom_x=np.array([0.0, 10.0]),
            geom_y=np.array([0.0, 0.0]),
            vertices_per_branch=np.array([2]),
            edge_length=np.array([20.0]),
        )
        np.testing.assert_allclose(with_len_x, [5.0])
        np.testing.assert_allclose(with_len_y, [0.0])

        # Without edge_length the same offset is raw distance, so it reaches the end.
        raw_x, _ = place_nodes_on_network(
            branch_index=np.array([0]),
            offset=np.array([10.0]),
            geom_x=np.array([0.0, 10.0]),
            geom_y=np.array([0.0, 0.0]),
            vertices_per_branch=np.array([2]),
        )
        np.testing.assert_allclose(raw_x, [10.0])

    def test_offset_past_branch_end_clamps_to_last_vertex(self):
        """numpy.interp behaviour: an offset beyond the branch length clamps to the end."""
        x, y = place_nodes_on_network(
            branch_index=np.array([0]),
            offset=np.array([15.0]),
            geom_x=np.array([0.0, 10.0]),
            geom_y=np.array([0.0, 0.0]),
            vertices_per_branch=np.array([2]),
        )
        np.testing.assert_allclose(x, [10.0])
        np.testing.assert_allclose(y, [0.0])

    def test_single_vertex_branch_degenerates_to_that_vertex(self):
        """A one-vertex branch returns its vertex for any offset."""
        x, y = place_nodes_on_network(
            branch_index=np.array([0, 0]),
            offset=np.array([0.0, 99.0]),
            geom_x=np.array([7.0]),
            geom_y=np.array([3.0]),
            vertices_per_branch=np.array([1]),
        )
        np.testing.assert_allclose(x, [7.0, 7.0])
        np.testing.assert_allclose(y, [3.0, 3.0])

    def test_empty_node_set_returns_empty_arrays(self):
        """No nodes -> empty x/y, no error even with a non-empty geometry."""
        x, y = place_nodes_on_network(
            branch_index=np.array([], dtype=np.int64),
            offset=np.array([]),
            geom_x=np.array([0.0, 1.0]),
            geom_y=np.array([0.0, 1.0]),
            vertices_per_branch=np.array([2]),
        )
        assert x.shape == (0,), f"expected empty node_x, got shape {x.shape}"
        assert y.shape == (0,), f"expected empty node_y, got shape {y.shape}"

    def test_accepts_lists_and_integer_valued_float_branch_index(self):
        """Inputs are coerced via np.asarray: plain lists and an integer-valued float branch
        index give the same result as int ndarrays -- so a driver's choice of array spelling or
        dtype does not change the output."""
        args = dict(
            geom_x=[0.0, 2.0],
            geom_y=[0.0, 0.0],
            vertices_per_branch=[2],
        )
        from_arrays = place_nodes_on_network(np.array([0]), np.array([1.0]), **args)
        from_lists = place_nodes_on_network([0], [1.0], **args)
        from_float_index = place_nodes_on_network(
            np.array([0.0]), np.array([1.0]), **args
        )
        np.testing.assert_allclose(from_lists[0], from_arrays[0])
        np.testing.assert_allclose(from_lists[1], from_arrays[1])
        np.testing.assert_allclose(from_float_index[0], from_arrays[0])
        np.testing.assert_allclose(from_float_index[1], from_arrays[1])


class TestPlaceNodesOnNetworkValidation:
    """Malformed geometry raises ValueError rather than emitting nan coordinates."""

    _GEOM = dict(
        geom_x=np.array([0.0, 10.0]),
        geom_y=np.array([0.0, 0.0]),
        vertices_per_branch=np.array([2]),
    )

    def test_branch_offset_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            place_nodes_on_network(np.array([0, 0]), np.array([1.0]), **self._GEOM)

    def test_geom_xy_shape_mismatch(self):
        with pytest.raises(ValueError, match="geom_x and geom_y"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0]),
                vertices_per_branch=np.array([2]),
            )

    def test_vertex_counts_do_not_tile_the_flat_list(self):
        with pytest.raises(ValueError, match="tile it exactly"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([3]),
            )

    def test_branch_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            place_nodes_on_network(np.array([2]), np.array([1.0]), **self._GEOM)

    def test_node_on_a_zero_vertex_branch(self):
        with pytest.raises(ValueError, match="no geometry vertices"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([0, 2]),
            )

    def test_edge_length_wrong_length(self):
        with pytest.raises(ValueError, match="edge_length must have shape"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                edge_length=np.array([1.0, 2.0]),
                **self._GEOM,
            )

    def test_negative_vertex_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([-2, 4]),
            )

    def test_two_dimensional_branch_or_offset(self):
        with pytest.raises(ValueError, match="1-D"):
            place_nodes_on_network(
                np.zeros((2, 2), int), np.zeros((2, 2)), **self._GEOM
            )

    def test_two_dimensional_geometry(self):
        with pytest.raises(ValueError, match="geom_x and geom_y must be 1-D"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.zeros((2, 2)),
                geom_y=np.zeros((2, 2)),
                vertices_per_branch=np.array([4]),
            )

    def test_two_dimensional_vertex_counts(self):
        with pytest.raises(ValueError, match="vertices_per_branch must be a 1-D"):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([[2]]),
            )

    def test_negative_branch_index(self):
        with pytest.raises(ValueError, match="out of range"):
            place_nodes_on_network(np.array([-1]), np.array([1.0]), **self._GEOM)

    def test_nan_offset_raises(self):
        """A non-finite offset (e.g. a _FillValue-coded chainage) raises, not a silent NaN (M1)."""
        with pytest.raises(ValueError, match="offset.*finite|finite.*offset"):
            place_nodes_on_network(np.array([0]), np.array([np.nan]), **self._GEOM)

    def test_inf_offset_raises(self):
        with pytest.raises(ValueError, match="offset.*finite|finite.*offset"):
            place_nodes_on_network(np.array([0]), np.array([np.inf]), **self._GEOM)

    def test_nan_geometry_raises(self):
        """A NaN vertex coordinate raises rather than producing a silent NaN node (M1)."""
        with pytest.raises(ValueError, match="geom_x.*finite|geom_y.*finite|finite"):
            place_nodes_on_network(
                np.array([0]),
                np.array([5.0]),
                geom_x=np.array([0.0, np.nan]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([2]),
            )

    def test_fractional_branch_index_raises(self):
        """A non-integral branch index raises rather than matching no branch and yielding NaN (M2)."""
        with pytest.raises(ValueError, match="integer|integral"):
            place_nodes_on_network(np.array([0.5]), np.array([1.0]), **self._GEOM)

    def test_negative_fractional_branch_index_raises(self):
        """A negative fractional index does not slip past the int()-truncated range check (M2)."""
        with pytest.raises(ValueError, match="integer|integral|out of range"):
            place_nodes_on_network(np.array([-0.5]), np.array([1.0]), **self._GEOM)

    def test_object_dtype_branch_index_raises_valueerror(self):
        """A non-numeric (object) branch index raises ValueError, not a numpy TypeError (R2 L1)."""
        with pytest.raises(ValueError, match="numeric"):
            place_nodes_on_network(
                np.array([0], dtype=object), np.array([1.0]), **self._GEOM
            )

    def test_boolean_branch_index_raises_valueerror(self):
        """A boolean branch index is rejected rather than silently coerced to 0/1 (R2 N1)."""
        with pytest.raises(ValueError, match="numeric"):
            place_nodes_on_network(np.array([True]), np.array([1.0]), **self._GEOM)

    def test_branch_index_exceeding_int64_range_raises(self):
        """An integer-valued float beyond int64 raises before astype wraps it silently (R2 N3)."""
        with pytest.raises(ValueError, match="int64 range"):
            place_nodes_on_network(np.array([1e30]), np.array([1.0]), **self._GEOM)


class TestPlaceNodesOnNetworkEdgeLengthEdgeCases:
    """edge_length branches: a non-positive declared length falls back to raw chord distance."""

    def test_zero_declared_length_uses_raw_chord(self):
        """A branch whose declared edge_length is 0 is not rescaled; raw chord distance applies.

        Test scenario:
            Two branches, the first declaring length 0: its node is placed by raw chord
            distance (the ``length > 0`` guard skips the rescale), the second rescales normally.
        """
        # branch 0: (0,0)->(10,0) chord 10, declared length 0 -> raw; offset 4 -> (4, 0).
        # branch 1: (0,0)->(10,0) chord 10, declared length 20 -> half -> offset 10 -> (5, 10).
        x, y = place_nodes_on_network(
            branch_index=np.array([0, 1]),
            offset=np.array([4.0, 10.0]),
            geom_x=np.array([0.0, 10.0, 0.0, 10.0]),
            geom_y=np.array([0.0, 0.0, 10.0, 10.0]),
            vertices_per_branch=np.array([2, 2]),
            edge_length=np.array([0.0, 20.0]),
        )
        np.testing.assert_allclose(x, [4.0, 5.0])
        np.testing.assert_allclose(y, [0.0, 10.0])

    def test_negative_edge_length_raises(self):
        """A negative declared length is a data error, not a silent raw-chord fallback (L1)."""
        with pytest.raises(
            ValueError, match="edge_length.*non-negative|edge_length.*finite"
        ):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([2]),
                edge_length=np.array([-20.0]),
            )

    def test_nan_edge_length_raises(self):
        """A NaN declared length raises rather than silently using raw chord distance (L1)."""
        with pytest.raises(
            ValueError, match="edge_length.*finite|edge_length.*non-negative"
        ):
            place_nodes_on_network(
                np.array([0]),
                np.array([1.0]),
                geom_x=np.array([0.0, 10.0]),
                geom_y=np.array([0.0, 0.0]),
                vertices_per_branch=np.array([2]),
                edge_length=np.array([np.nan]),
            )
