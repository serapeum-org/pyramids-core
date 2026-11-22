"""Direct unit tests for the gap-aware kernels and dtype helpers (audit D1/D5)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._array_ops import (
    _holds,
    apply_values,
    astype_values,
    clip_values,
    fill_dtype,
    fillna,
    isin_values,
    round_values,
)


class TestHolds:
    """`_holds` — whether a dtype can carry a sentinel."""

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("float32", -9999.0, True),
            ("float32", np.inf, True),
            ("float32", np.nan, True),
            ("float32", 1e300, False),
            ("float32", -9999.9, False),
            ("float16", -9999.0, False),
            ("float16", 70000.0, False),
            ("int16", -9999.0, True),
            ("int16", 0.5, False),
            ("uint8", 255.0, True),
            ("uint8", -9999.0, False),
            ("uint8", 300.0, False),
        ],
    )
    def test_holds_matrix(self, dtype, value, expected):
        """A float type holds any in-range value; an integer type only a whole one in range."""
        assert _holds(np.dtype(dtype), value) is expected, f"{dtype} holds {value}?"

    def test_a_large_int_beyond_float64_mantissa_is_not_held(self):
        """Review L1: ``float(2**53 + 1)`` rounds to ``2**53`` before any check could see it.

        The exactness guard must measure against the original integer, not a pre-rounded
        ``float`` of it, or it declares a value representable that ``float64`` cannot carry.
        """
        assert _holds(np.dtype("float64"), 2**53 + 1) is False
        assert _holds(np.dtype("float64"), 2**53) is True
        assert _holds(np.dtype("int64"), 2**62) is True

    @pytest.mark.parametrize(
        "value",
        [np.array([-9999]), np.array([[-9999.0]]), [1, 2], (1,), "not-a-number"],
    )
    def test_a_non_scalar_or_non_numeric_sentinel_is_rejected_not_raised(self, value):
        """Review N1: a one-element array, a sequence or a string is not a scalar sentinel.

        Probing such a value's finiteness used to surface a raw numpy ``TypeError``; it now
        answers ``False`` so the caller simply widens rather than crashing.
        """
        assert _holds(np.dtype("float64"), value) is False

    def test_a_zero_dimensional_array_sentinel_is_still_accepted(self):
        """A 0-d array wraps exactly one value, so it is a valid scalar sentinel."""
        assert _holds(np.dtype("float64"), np.array(-9999.0)) is True


class TestFillDtype:
    """`fill_dtype` — the widening policy before a masked write."""

    @pytest.mark.parametrize(
        ("dtype", "fill", "expected"),
        [
            ("float32", -9999.0, "float32"),
            ("float32", 2.5, "float32"),
            ("float32", np.inf, "float32"),
            ("float32", 1e300, "float64"),
            ("float16", 70000.0, "float64"),
            ("uint8", 255.0, "uint8"),
            ("uint8", np.nan, "float64"),
            ("int16", 0.5, "float64"),
            ("int16", -9999.0, "int16"),
        ],
    )
    def test_fill_dtype_matrix(self, dtype, fill, expected):
        """The source type is kept when it holds the fill, and widened only when it cannot."""
        assert str(fill_dtype(np.dtype(dtype), fill)) == expected


class TestClipValuesGuards:
    """`clip_values` bound validation and gap preservation."""

    def test_a_non_real_bound_raises_type_error(self):
        """A non-numeric bound is a TypeError, named for the offending side."""
        with pytest.raises(TypeError, match="lower must be a real number"):
            clip_values(np.array([1.0, 2.0]), "low", 8.0, nodata=None)

    def test_a_one_sided_bound_is_allowed(self):
        """Clipping with only an upper (or only a lower) bound works."""
        assert clip_values(np.array([1.0, 9.0]), None, 8.0).tolist() == [1.0, 8.0]
        assert clip_values(np.array([1.0, 9.0]), 2.0, None).tolist() == [2.0, 9.0]

    def test_gap_is_preserved_through_a_dtype_change(self):
        """An integer gap survives a float clip, still recognised as the sentinel."""
        out = clip_values(np.array([5, -9999]), 8.0, 22.0, nodata=-9999)
        assert out.tolist() == [8.0, -9999.0]


class TestGapKernels:
    """round / isin / apply / astype / fillna gap behaviour, directly."""

    def test_round_without_a_sentinel_is_plain_round(self):
        assert round_values(np.array([1.4, 2.6])).tolist() == [1.0, 3.0]

    def test_isin_without_a_sentinel_is_plain_isin(self):
        assert isin_values(np.array([1, 2, 3]), [2, 3]).tolist() == [False, True, True]

    def test_apply_refuses_a_shape_change(self):
        with pytest.raises(ValueError, match="changed the array shape"):
            apply_values(np.array([1.0, 2.0]), lambda a: a[:1])

    def test_apply_preserves_the_gap(self):
        out = apply_values(np.array([5.0, -9999.0]), lambda a: a * 2, nodata=-9999.0)
        assert out.tolist() == [10.0, -9999.0]

    def test_astype_without_a_gap_just_casts(self):
        assert astype_values(
            np.array([1.0, 2.0]), np.int32, nodata=-9999.0
        ).tolist() == [1, 2]

    def test_astype_refuses_when_the_target_cannot_hold_the_sentinel(self):
        with pytest.raises(ValueError, match="cannot mark a gap"):
            astype_values(np.array([5.0, -9999.0]), np.uint8, nodata=-9999.0)

    def test_astype_refuses_a_float_target_that_rounds_the_sentinel(self):
        """float16 fits 9999's magnitude but snaps -9999 to -10000, so the gap would be lost."""
        with pytest.raises(ValueError, match="cannot mark a gap"):
            astype_values(np.array([5.0, -9999.0]), np.float16, nodata=-9999.0)

    def test_astype_to_a_wider_float_keeps_the_gap(self):
        """A target that represents the sentinel exactly is allowed and preserves it."""
        out = astype_values(np.array([5.0, -9999.0]), np.float32, nodata=-9999.0)
        assert out.tolist() == [5.0, -9999.0] and out.dtype == np.float32

    def test_fillna_without_a_fitting_type_widens(self):
        out = fillna(np.array([5, -9999], dtype="int16"), 2.5, nodata=-9999)
        assert out.dtype == np.float64 and out.tolist() == [5.0, 2.5]


class TestOutputHygiene:
    """fillna and apply_values return fresh arrays and do not widen needlessly (L1/L4)."""

    def test_fillna_keeps_dtype_when_nothing_is_missing(self):
        """A gap-free integer band is not widened to float64 by a fill never written."""
        out = fillna(np.array([1, 2, 3], dtype="int16"), 2.5, nodata=-9999)
        assert out.dtype == np.int16 and out.tolist() == [1, 2, 3]

    def test_fillna_returns_a_fresh_array(self):
        """The no-fill path still returns a copy, not the caller's buffer."""
        src = np.array([1, 2, 3], dtype="int16")
        out = fillna(src, 0, nodata=-9999)
        assert out is not src

    def test_apply_does_not_alias_the_source_on_an_identity(self):
        """An identity func must not make the result share the caller's buffer."""
        src = np.array([1.0, 2.0, 3.0])
        out = apply_values(src, lambda a: a)
        assert out is not src and not np.shares_memory(out, src)
        out[0] = 99.0
        assert src[0] == 1.0, "mutating the result must not touch the source"
