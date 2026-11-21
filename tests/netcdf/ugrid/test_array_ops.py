"""Tests for element-wise array operations on MeshDataset.

Covers the facade methods (fillna, isnull, notnull, clip_values, round,
astype, isin) *and* the underlying ``base._array_ops`` kernels including
branch coverage for nodata_mask, array_equals, and array_identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._array_ops import (
    array_equals,
    array_identical,
    nodata_mask,
)
from pyramids.netcdf.ugrid.dataset import MeshDataset
from tests.netcdf.ugrid.conftest import make_two_triangle_ds

pytestmark = pytest.mark.core


def _make_ds(
    values: np.ndarray | None = None,
    nodata: float | None = None,
) -> MeshDataset:
    ds = make_two_triangle_ds(values=values)
    if nodata is not None:
        ds["depth"].nodata = nodata
    return ds


class TestFillna:
    def test_replaces_nan(self):
        ds = _make_ds(np.array([1.0, np.nan]))
        filled = ds.fillna(0.0)
        np.testing.assert_array_equal(filled["depth"].data, [1.0, 0.0])

    def test_replaces_nodata_sentinel(self):
        ds = _make_ds(np.array([1.0, -9999.0]), nodata=-9999.0)
        filled = ds.fillna(0.0)
        np.testing.assert_array_equal(filled["depth"].data, [1.0, 0.0])

    def test_no_missing_unchanged(self):
        ds = _make_ds(np.array([5.0, 10.0]))
        filled = ds.fillna(0.0)
        np.testing.assert_array_equal(filled["depth"].data, [5.0, 10.0])


class TestIsnull:
    def test_nan_detected(self):
        ds = _make_ds(np.array([1.0, np.nan]))
        result = ds.isnull()
        np.testing.assert_array_equal(result["depth"].data, [False, True])

    def test_nodata_detected(self):
        ds = _make_ds(np.array([1.0, -9999.0]), nodata=-9999.0)
        result = ds.isnull()
        np.testing.assert_array_equal(result["depth"].data, [False, True])


class TestNotnull:
    def test_inverse_of_isnull(self):
        ds = _make_ds(np.array([1.0, np.nan]))
        result = ds.notnull()
        np.testing.assert_array_equal(result["depth"].data, [True, False])


class TestClipValues:
    def test_clamps_values(self):
        ds = _make_ds(np.array([5.0, 25.0]))
        result = ds.clip_values(8.0, 22.0)
        np.testing.assert_array_equal(result["depth"].data, [8.0, 22.0])

    def test_within_range_unchanged(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.clip_values(0.0, 100.0)
        np.testing.assert_array_equal(result["depth"].data, [10.0, 20.0])


class TestRound:
    def test_rounds_to_integer(self):
        ds = _make_ds(np.array([1.567, 2.345]))
        result = ds.round(0)
        np.testing.assert_array_equal(result["depth"].data, [2.0, 2.0])

    def test_rounds_to_one_decimal(self):
        ds = _make_ds(np.array([1.567, 2.345]))
        result = ds.round(1)
        np.testing.assert_array_almost_equal(result["depth"].data, [1.6, 2.3])


class TestAstype:
    def test_float_to_int(self):
        ds = _make_ds(np.array([1.0, 2.0]))
        result = ds.astype(np.int32)
        assert result["depth"].data.dtype == np.int32
        np.testing.assert_array_equal(result["depth"].data, [1, 2])

    def test_preserves_values(self):
        ds = _make_ds(np.array([1.5, 2.5]))
        result = ds.astype(np.float32)
        assert result["depth"].data.dtype == np.float32


class TestIsin:
    def test_membership(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.isin([10.0, 30.0])
        np.testing.assert_array_equal(result["depth"].data, [True, False])


class TestNodataMaskKernel:
    """Direct tests for base._array_ops.nodata_mask branch coverage."""

    def test_integer_no_nodata_returns_all_false(self):
        """Integer array with no sentinel — nothing is missing."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        mask = nodata_mask(arr, nodata=None)
        assert not mask.any(), f"Expected no missing, got {mask}"

    def test_integer_with_sentinel(self):
        """Integer array with sentinel — only exact matches."""
        arr = np.array([1, -9999, 3], dtype=np.int32)
        mask = nodata_mask(arr, nodata=-9999)
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_float_with_non_nan_sentinel(self):
        """Float array with a non-NaN sentinel — both NaN and sentinel."""
        arr = np.array([1.0, np.nan, -9999.0])
        mask = nodata_mask(arr, nodata=-9999.0)
        np.testing.assert_array_equal(mask, [False, True, True])

    def test_float_nan_sentinel_no_double_count(self):
        """Float with nodata=NaN — same as nodata=None for floats."""
        arr = np.array([1.0, np.nan, 2.0])
        mask = nodata_mask(arr, nodata=np.nan)
        np.testing.assert_array_equal(mask, [False, True, False])


class TestArrayEqualsKernel:
    """Direct tests for base._array_ops.array_equals branch coverage."""

    def test_different_shapes_returns_false(self):
        """Shape mismatch is an immediate False."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        assert not array_equals(a, b)

    def test_different_missing_positions_returns_false(self):
        """Same shape but NaN in different positions."""
        a = np.array([np.nan, 2.0])
        b = np.array([1.0, np.nan])
        assert not array_equals(a, b)

    def test_all_missing_returns_true(self):
        """Both arrays are entirely NaN — equal by convention."""
        a = np.full(3, np.nan)
        b = np.full(3, np.nan)
        assert array_equals(a, b)

    def test_same_values_same_mask(self):
        """Matching values and matching NaN positions."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, np.nan, 3.0])
        assert array_equals(a, b)


class TestArrayIdenticalKernel:
    """Direct tests for base._array_ops.array_identical branch coverage."""

    def test_different_shapes(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0])
        assert not array_identical(a, b)

    def test_different_dtypes(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([1, 2], dtype=np.int64)
        assert not array_identical(a, b)

    def test_same_with_nan(self):
        a = np.array([1.0, np.nan])
        b = np.array([1.0, np.nan])
        assert array_identical(a, b)


class TestCellwiseKernelsHonourTheSentinel:
    """D1/D5: the cell-wise kernels must preserve a declared gap, not consume it.

    These are the untested half the audit found: TestClipValues / TestRound / TestAstype /
    TestIsin above never construct a variable with a declared sentinel, which is exactly
    why the gap-blindness survived. Each case pins the corrupted answer it used to give.
    """

    def test_clip_preserves_the_gap(self):
        out = _make_ds(np.array([5.0, -9999.0]), nodata=-9999.0).clip_values(8.0, 22.0)
        assert out["depth"].data.tolist() == [8.0, -9999.0]  # was [8.0, 8.0]

    @pytest.mark.parametrize(
        ("lower", "upper", "exc"),
        [
            (22.0, 8.0, ValueError),
            (float("nan"), 8.0, ValueError),
            (None, None, ValueError),
        ],
    )
    def test_clip_validates_its_bounds(self, lower, upper, exc):
        with pytest.raises(exc):
            _make_ds(np.array([5.0, 25.0])).clip_values(lower, upper)

    def test_round_preserves_a_fractional_sentinel(self):
        out = _make_ds(np.array([5.0, -9999.5]), nodata=-9999.5).round(0)
        assert out["depth"].data.tolist() == [5.0, -9999.5]  # was [5.0, -10000.0]

    def test_isin_excludes_the_gap(self):
        out = _make_ds(np.array([5.0, -9999.0]), nodata=-9999.0).isin([-9999.0])
        assert out["depth"].data.tolist() == [False, False]  # was [False, True]

    def test_apply_leaves_the_gap_untouched(self):
        out = _make_ds(np.array([5.0, -9999.0]), nodata=-9999.0).apply(lambda a: a * 2)
        assert out["depth"].data.tolist() == [10.0, -9999.0]  # was [10.0, -19998.0]

    def test_astype_refuses_to_erase_a_gap(self):
        with pytest.raises(ValueError, match="cannot mark a gap"):
            _make_ds(np.array([5.0, -9999.0]), nodata=-9999.0).astype(np.uint8)

    def test_astype_is_fine_when_the_target_holds_the_sentinel(self):
        out = _make_ds(np.array([5.0, -9999.0]), nodata=-9999.0).astype(np.int32)
        assert out["depth"].data.tolist() == [5, -9999]

    def test_fillna_widens_an_integer_band_for_a_fractional_fill(self):
        ds = make_two_triangle_ds(values=np.array([5, -9999], dtype="int16"))
        ds["depth"].nodata = -9999
        out = ds.fillna(2.5)
        assert out["depth"].data.tolist() == [5.0, 2.5]  # was [5, 2] int16
        assert out["depth"].data.dtype == np.float64

    def test_fillna_widens_a_float32_band_for_an_overflowing_fill(self):
        ds = make_two_triangle_ds(values=np.array([5.0, -9999.0], dtype="float32"))
        ds["depth"].nodata = -9999.0
        out = ds.fillna(1e300)
        assert out["depth"].data[1] == 1e300  # was inf, behind a RuntimeWarning
        assert out["depth"].data.dtype == np.float64
