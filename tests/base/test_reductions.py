"""Unit tests for the pure reduction kernels in ``pyramids.base._reductions``.

These run with no GDAL and no dataset object — the whole point of the module is that the
kernels are geometry-agnostic NumPy — so they exercise the maths directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._reductions import (
    reduce_axis,
    reduce_by_label,
    validate_min_count,
    weighted_statistic,
)


class TestReduceByLabelUnweighted:
    """Unweighted ``reduce_by_label`` matches a direct per-group NumPy reduction."""

    def test_matches_direct_numpy_per_group(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        labels = np.array([0, 0, 1, 1, -1, 1])  # last-but-one unassigned
        out = reduce_by_label(
            values, labels, 2, ["sum", "count", "mean", "min", "max", "std", "var"]
        )
        assert out["sum"].tolist() == [3.0, 13.0]
        assert out["count"].tolist() == [2.0, 3.0]
        assert out["mean"].tolist() == [1.5, pytest.approx(13.0 / 3.0)]
        assert out["min"].tolist() == [1.0, 3.0]
        assert out["max"].tolist() == [2.0, 6.0]
        np.testing.assert_allclose(
            out["std"], [np.std([1.0, 2.0]), np.std([3.0, 4.0, 6.0])]
        )
        np.testing.assert_allclose(
            out["var"], [np.var([1.0, 2.0]), np.var([3.0, 4.0, 6.0])]
        )

    def test_nan_values_are_skipped(self):
        values = np.array([1.0, np.nan, 3.0])
        labels = np.array([0, 0, 0])
        out = reduce_by_label(values, labels, 1, ["sum", "count", "mean"])
        assert out["sum"].tolist() == [4.0]
        assert out["count"].tolist() == [2.0]
        assert out["mean"].tolist() == [2.0]

    def test_empty_group_is_nan_and_count_zero(self):
        values = np.array([1.0, 2.0])
        labels = np.array([0, 0])  # group 1 gets nothing
        out = reduce_by_label(values, labels, 2, ["mean", "count", "min", "std"])
        assert np.isnan(out["mean"][1])
        assert out["count"][1] == 0.0
        assert np.isnan(out["min"][1])
        assert np.isnan(out["std"][1])

    def test_2d_inputs_are_ravelled(self):
        values = np.arange(6.0).reshape(2, 3)
        labels = np.array([[0, 0, 1], [1, 1, 0]])
        out = reduce_by_label(values, labels, 2, ["sum"])
        # group 0: 0 + 1 + 5 = 6 ; group 1: 2 + 3 + 4 = 9
        assert out["sum"].tolist() == [6.0, 9.0]


class TestReduceByLabelWeighted:
    """The weighted arm: sum/mean/std/var honour weights; count/min/max do not."""

    def test_weighted_mean_and_sum(self):
        values = np.array([1.0, 3.0])
        labels = np.array([0, 0])
        weights = np.array([1.0, 3.0])
        out = reduce_by_label(
            values, labels, 1, ["sum", "mean", "count"], weights=weights
        )
        assert out["sum"].tolist() == [1.0 * 1.0 + 3.0 * 3.0]  # 10.0
        assert out["mean"].tolist() == [10.0 / 4.0]  # 2.5
        assert out["count"].tolist() == [2.0]  # weight-independent

    def test_weighted_var_matches_formula(self):
        values = np.array([1.0, 3.0, 5.0])
        labels = np.array([0, 0, 0])
        weights = np.array([1.0, 1.0, 2.0])
        out = reduce_by_label(values, labels, 1, ["var", "std"], weights=weights)
        w = weights
        x = values
        wmean = np.sum(w * x) / np.sum(w)
        expected_var = np.sum(w * (x - wmean) ** 2) / np.sum(w)
        np.testing.assert_allclose(out["var"], [expected_var])
        np.testing.assert_allclose(out["std"], [np.sqrt(expected_var)])

    def test_min_max_are_weight_invariant(self):
        values = np.array([1.0, 9.0])
        labels = np.array([0, 0])
        unweighted = reduce_by_label(values, labels, 1, ["min", "max"])
        weighted = reduce_by_label(
            values, labels, 1, ["min", "max"], weights=np.array([100.0, 0.01])
        )
        assert unweighted["min"].tolist() == weighted["min"].tolist() == [1.0]
        assert unweighted["max"].tolist() == weighted["max"].tolist() == [9.0]


class TestReduceByLabelValidation:
    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="same size"):
            reduce_by_label(np.zeros(3), np.zeros(4), 1, ["sum"])

    def test_unknown_stat_raises(self):
        with pytest.raises(ValueError, match="unknown stat"):
            reduce_by_label(np.zeros(2), np.zeros(2, dtype=int), 1, ["bogus"])

    def test_label_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
            reduce_by_label(np.zeros(2), np.array([0, 5]), 2, ["sum"])

    def test_custom_unassigned_sentinel(self):
        values = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 255, 0])  # 255 marks "no group"
        out = reduce_by_label(values, labels, 1, ["sum", "count"], unassigned=255)
        assert out["sum"].tolist() == [4.0]
        assert out["count"].tolist() == [2.0]


class TestWeightedStatistic:
    """A direct check that the shared weighted kernel is callable GDAL-free."""

    def test_area_weighted_mean_over_axis_zero(self):
        values = np.array([[1.0, 3.0], [5.0, 7.0]])
        weights = np.array([[1.0], [3.0]])  # broadcast over the element axis
        out = weighted_statistic(values, weights, (0,), "mean", None, True)
        # column 0: (1*1 + 3*5)/4 = 4.0 ; column 1: (1*3 + 3*7)/4 = 6.0
        np.testing.assert_allclose(out.ravel(), [4.0, 6.0])


class TestReduceByLabelWeightedStability:
    def test_near_constant_group_var_is_nonnegative(self):
        # A group of equal values with distinct weights: variance is exactly 0, and the
        # stable formula must not return a tiny negative (which would make std NaN).
        values = np.full(6, 123.456)
        labels = np.zeros(6, dtype=int)
        weights = np.array([1.0, 1.3, 0.7, 2.1, 0.9, 1.1])
        out = reduce_by_label(values, labels, 1, ["var", "std"], weights=weights)
        assert out["var"][0] >= 0.0
        assert np.isclose(out["var"][0], 0.0, atol=1e-9)
        assert not np.isnan(out["std"][0])
        assert np.isclose(out["std"][0], 0.0, atol=1e-6)

    def test_weighted_var_matches_weighted_statistic(self):
        from pyramids.base._reductions import weighted_statistic

        values = np.array([1.0, 3.0, 5.0, 9.0])
        weights = np.array([1.0, 2.0, 1.0, 3.0])
        labels = np.zeros(4, dtype=int)
        by_label = reduce_by_label(values, labels, 1, ["var"], weights=weights)["var"][
            0
        ]
        direct = weighted_statistic(values, weights, (0,), "var", None, True)
        np.testing.assert_allclose(by_label, np.asarray(direct).ravel()[0])

    def test_weights_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="weights has"):
            reduce_by_label(
                np.zeros(3), np.zeros(3, dtype=int), 1, ["mean"], weights=np.zeros(2)
            )


class TestValidateMinCount:
    """`validate_min_count` accepts non-negative ints and rejects everything else."""

    @pytest.mark.parametrize("value", [0, 1, 20, np.int64(3)])
    def test_accepts_non_negative_int(self, value):
        """A non-negative Python or NumPy integer passes without raising.

        Args:
            value: The candidate min_count.

        Test scenario:
            0, a positive int, and a numpy integer are all valid and return None.
        """
        assert validate_min_count(value) is None, f"{value!r} should be accepted"

    def test_negative_raises_value_error(self):
        """A negative min_count raises ValueError naming the constraint.

        Test scenario:
            -1 is rejected with a '>= 0' message.
        """
        with pytest.raises(ValueError, match=r">= 0"):
            validate_min_count(-1)

    @pytest.mark.parametrize("value", [2.5, 3.0, "2", None])
    def test_non_int_raises_type_error(self, value):
        """A non-integer min_count raises TypeError.

        Args:
            value: The invalid candidate.

        Test scenario:
            A float (even a whole 3.0), a string, and None are all rejected.
        """
        with pytest.raises(TypeError, match="min_count must be an int"):
            validate_min_count(value)

    def test_bool_raises_type_error(self):
        """A bool is rejected even though it subclasses int.

        Test scenario:
            True must not sneak through as the integer 1.
        """
        with pytest.raises(TypeError, match="min_count must be an int"):
            validate_min_count(True)


class TestReduceAxisMinCount:
    """`reduce_axis` gates columns below `min_count` valid samples (skipna only)."""

    @staticmethod
    def _stack():
        """A `(3, 1, 2)` stack: column 0 has one valid cell, column 1 has three."""
        return np.array([[[5.0, 1.0]], [[np.nan, 2.0]], [[np.nan, 3.0]]])

    @pytest.mark.parametrize(
        "min_count, expected_col0",
        [(0, 5.0), (1, 5.0), (2, np.nan), (3, np.nan)],
    )
    def test_gates_sparse_column(self, min_count, expected_col0):
        """A one-sample column is kept at min_count<=1 and NaN at min_count>=2.

        Args:
            min_count: The threshold under test.
            expected_col0: The expected reduced value of the one-sample column.

        Test scenario:
            mean over axis 0 with skipna; the three-sample column stays 2.0
            throughout while the one-sample column flips to NaN once the
            threshold exceeds its single valid sample (>= semantics).
        """
        out = reduce_axis(self._stack(), 0, "mean", True, None, None, min_count)
        if np.isnan(expected_col0):
            assert np.isnan(out[0, 0]), f"col0 should be NaN at min_count={min_count}"
        else:
            assert out[0, 0] == expected_col0, f"col0 wrong at min_count={min_count}"
        assert out[0, 1] == 2.0, "the full column is never gated"

    def test_gated_cell_fills_declared_no_data(self):
        """With a sentinel, a gated cell holds the sentinel, not NaN.

        Test scenario:
            ndv=-9999 and the sentinel marks the gaps; a one-sample column below
            min_count resolves to -9999 (what the Dataset/NetCDF path renders).
        """
        arr = np.array([[[5.0, 1.0]], [[-9999.0, 2.0]], [[-9999.0, 3.0]]])
        out = reduce_axis(arr, 0, "mean", True, -9999.0, None, 2)
        assert out[0, 0] == -9999.0, (
            f"gated cell should be the sentinel, got {out[0, 0]}"
        )
        assert out[0, 1] == 2.0, "the full column keeps its mean"

    def test_boundary_count_equals_min_count_keeps_value(self):
        """A column with exactly min_count valid cells is kept (>=, not >).

        Test scenario:
            Column with two valid cells and min_count=2 keeps its mean.
        """
        arr = np.array([[[10.0]], [[20.0]], [[np.nan]]])
        out = reduce_axis(arr, 0, "mean", True, None, None, 2)
        assert out[0, 0] == 15.0, f"exactly-at-threshold column kept, got {out[0, 0]}"

    def test_ignored_without_skipna(self):
        """min_count has no effect when skipna is False.

        Test scenario:
            A full column reduced with skipna=False and a large min_count returns
            the plain mean (the gate only runs on the skipna path).
        """
        arr = np.array([[[10.0]], [[20.0]], [[30.0]]])
        out = reduce_axis(arr, 0, "mean", False, None, None, 99)
        assert out[0, 0] == 20.0, f"skipna=False must ignore min_count, got {out[0, 0]}"

    def test_ignored_for_count_reducer(self):
        """The counting reductions ignore min_count (a count is never gated).

        Test scenario:
            how='count' on the one-sample column still answers 1 at min_count=5.
        """
        out = reduce_axis(self._stack(), 0, "count", True, None, None, 5)
        assert int(out[0, 0]) == 1, f"count must ignore min_count, got {out[0, 0]}"
        assert int(out[0, 1]) == 3, "the full column counts 3"
