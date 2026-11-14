"""A2 — grouped `reduce(groupby=...)` and the `resample` sugar on MeshDataset."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import MeshDataset


def _temporal(values: np.ndarray, stamps: list | None = None) -> MeshDataset:
    """A two-face mesh carrying `values` of shape (n_time, 2) along time."""
    mesh = MeshDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": values},
        data_locations={"d": "face"},
    )
    if stamps is not None:
        mesh["d"].attributes = {**mesh["d"].attributes, "time_values": list(stamps)}
    return mesh


class TestGroupbyLabels:
    def test_label_sequence_groups_and_keeps_the_axis(self):
        values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
        out = _temporal(values).reduce("mean", groupby=["a", "a", "b", "b"])
        assert out["d"].has_time is True
        assert out["d"].data.tolist() == [[2.0, 20.0], [6.0, 60.0]]

    def test_group_labels_become_the_time_coordinate(self):
        values = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        out = _temporal(values).reduce("sum", groupby=["x", "y", "y"])
        assert out["d"].attributes["time_values"] == ["x", "y"]
        assert out["d"].data.tolist() == [[1.0, 1.0], [8.0, 8.0]]

    def test_first_appearance_order_is_kept(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        out = _temporal(values).reduce("max", groupby=["late", "early", "late"])
        assert out["d"].attributes["time_values"] == ["late", "early"]
        assert out["d"].data.tolist() == [[3.0, 3.0], [2.0, 2.0]]

    def test_gaps_are_skipped_within_a_group(self):
        values = np.array([[np.nan, 1.0], [4.0, 3.0]])
        out = _temporal(values).reduce("mean", groupby=["g", "g"])
        assert out["d"].data.tolist() == [[4.0, 2.0]]

    def test_wrong_label_count_raises(self):
        mesh = _temporal(np.array([[1.0, 1.0], [2.0, 2.0]]))
        with pytest.raises(ValueError, match="one label per time step"):
            mesh.reduce("mean", groupby=["only-one"])

    def test_ungrouped_reduce_still_collapses(self):
        values = np.array([[1.0, 1.0], [3.0, 3.0]])
        out = _temporal(values).reduce("mean")
        assert out["d"].has_time is False
        assert out["d"].data.tolist() == [2.0, 2.0]


class TestResample:
    def test_monthly_bins(self):
        stamps = ["2020-01-01", "2020-01-15", "2020-02-01"]
        values = np.array([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0]])
        out = _temporal(values, stamps=stamps).resample("1MS", "mean")
        assert out["d"].data.tolist() == [[2.0, 2.0], [10.0, 10.0]]
        assert len(out["d"].attributes["time_values"]) == 2

    def test_resample_defaults_to_mean(self):
        stamps = ["2020-01-01", "2020-01-02"]
        values = np.array([[2.0, 2.0], [4.0, 4.0]])
        out = _temporal(values, stamps=stamps).resample("1MS")
        assert out["d"].data.tolist() == [[3.0, 3.0]]

    def test_empty_bins_are_dropped(self):
        # Jan and Mar only: the Feb bin is empty and must not become an all-gap step.
        stamps = ["2020-01-01", "2020-03-01"]
        values = np.array([[1.0, 1.0], [5.0, 5.0]])
        out = _temporal(values, stamps=stamps).resample("1MS", "sum")
        assert out["d"].data.tolist() == [[1.0, 1.0], [5.0, 5.0]]

    def test_non_date_coordinates_raise(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0]])
        mesh = _temporal(values, stamps=["not-a-date", "also-not"])
        with pytest.raises(ValueError, match="date-like time coordinates"):
            mesh.resample("1MS")


class TestReduceMinCount:
    """`MeshDataset.reduce(min_count=)` gates a slice with too few valid samples."""

    def test_collapse_gates_sparse_face(self):
        """A face with one valid step is gated to no-data at min_count=2.

        Test scenario:
            face 0 is [5, NaN, NaN] (one valid), face 1 is [1, 2, 3]; collapsing
            with min_count=2 makes face 0 NaN while face 1 keeps its mean.
        """
        values = np.array([[5.0, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        out = _temporal(values).reduce("mean", min_count=2)["d"].data
        assert np.isnan(out[0]), f"one-sample face gated to NaN, got {out[0]}"
        assert out[1] == 2.0, f"three-sample face kept, got {out[1]}"

    def test_collapse_min_count_zero_keeps_sparse_face(self):
        """min_count=0 (default) leaves a one-sample face untouched.

        Test scenario:
            The one-valid-sample face reduces to that value, as today.
        """
        values = np.array([[5.0, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        out = _temporal(values).reduce("mean", min_count=0)["d"].data
        assert out[0] == 5.0, f"one-sample face kept at min_count=0, got {out[0]}"

    def test_grouped_gates_per_group(self):
        """Grouped reduction gates each group by its own valid count.

        Test scenario:
            values grouped ["g", "g", "h"]: face 0 has two valid in group g
            (kept) and one in group h (gated at min_count=2).
        """
        values = np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]])
        out = (
            _temporal(values)
            .reduce("mean", groupby=["g", "g", "h"], min_count=2)["d"]
            .data
        )
        assert out[0].tolist() == [2.0, 1.0], (
            f"group g (two samples) kept, got {out[0]}"
        )
        assert np.isnan(out[1][0]), f"group h (one sample) gated, got {out[1]}"

    def test_declared_no_data_excluded_and_rendered(self):
        """A declared sentinel is excluded from the count and rendered in a gated cell.

        Test scenario:
            With nodata=-9999, face 0 is [5, -9999, -9999] (one valid after the
            sentinel is masked); collapsing with min_count=2 gates it and the cell
            holds -9999 (not NaN), matching NetCDF.reduce. Face 1 [1, 2, 3] is kept.
        """
        mesh = _temporal(np.array([[5.0, 1.0], [-9999.0, 2.0], [-9999.0, 3.0]]))
        mesh["d"].nodata = -9999.0
        out = mesh.reduce("mean", min_count=2)["d"].data
        assert out[0] == -9999.0, f"gated face holds the declared no-data, got {out[0]}"
        assert out[1] == 2.0, f"three-sample face kept, got {out[1]}"

    def test_declared_no_data_gates_grouped_path(self):
        """The grouped mesh path also excludes the sentinel and renders no-data.

        Test scenario:
            nodata=-9999, grouped ["g", "g", "h"]: group h has only a sentinel for
            face 0, so it is gated to -9999 at min_count=1.
        """
        mesh = _temporal(np.array([[1.0, 1.0], [3.0, 1.0], [-9999.0, 1.0]]))
        mesh["d"].nodata = -9999.0
        out = mesh.reduce("mean", groupby=["g", "g", "h"], min_count=1)["d"].data
        assert out[0].tolist() == [2.0, 1.0], f"group g kept, got {out[0]}"
        assert out[1][0] == -9999.0, (
            f"group h sentinel-only face gated, got {out[1][0]}"
        )

    @pytest.mark.parametrize(
        "bad, exc", [(-1, ValueError), (2.5, TypeError), (True, TypeError)]
    )
    def test_invalid_min_count_raises(self, bad, exc):
        """A negative or non-integer min_count is rejected up front.

        Args:
            bad: The invalid min_count.
            exc: The expected exception type.
        """
        mesh = _temporal(np.array([[1.0, 1.0], [2.0, 2.0]]))
        with pytest.raises(exc, match="min_count"):
            mesh.reduce("mean", min_count=bad)
