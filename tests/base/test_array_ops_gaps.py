"""Direct unit tests for the shared gap-aware kernels and primitives (audit D1/D5).

After the D1 collapse the kernels in ``pyramids.base._array_ops`` are the single
implementation both the raster ``Analysis`` engine and the mesh ``MeshFacades`` call. Each
takes a REQUIRED ``domain`` mask -- computed by the caller, never guessed from
``values == sentinel`` -- and returns the array paired with the sentinel the result declares
(``isin_values`` is the exception: a membership flag declares no gap). The ``holds`` /
``regap`` / ``declared_gap`` primitives relocated here from the raster engine carry its
range-based judgement, so a float target accepts any in-range sentinel and snaps it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._array_ops import (
    apply_values,
    astype_values,
    clip_values,
    declared_gap,
    fill_dtype,
    fillna,
    holds,
    isin_values,
    regap,
    round_values,
)


class TestHolds:
    """``holds`` -- whether a dtype can carry a sentinel, by RANGE (the raster's rule)."""

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("float32", -9999.0, True),
            ("float32", np.inf, True),
            ("float32", np.nan, True),
            ("float32", -9999.9, True),  # a float target snaps; range is what matters
            ("float16", -9999.0, True),  # snaps to -10000.0, still in range
            ("float16", 65504.0, True),
            ("float16", -99999.0, False),  # past float16's range
            ("int16", -9999.0, True),
            ("int16", -9999.5, False),
            ("int16", np.nan, False),
            ("int32", np.inf, False),
            ("uint8", 255.0, True),
            ("uint8", -9999.0, False),
            ("uint8", 300.0, False),
        ],
    )
    def test_holds_matrix(self, dtype, value, expected):
        """A float type holds any in-range value; an integer type only a whole one in range."""
        assert holds(np.dtype(dtype), value) is expected, f"{dtype} holds {value}?"


class TestFillDtype:
    """``fill_dtype`` -- the widening policy before a masked write (unchanged by D1)."""

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


class TestRegapShapes:
    """``regap`` keys off the sentinel's sequence-ness, and never returns its input (risk #1/#5)."""

    def test_one_d_scalar_sentinel(self):
        """A 1-D mesh array with a scalar sentinel used to raise a numpy item-assignment error."""
        out = regap(np.array([8.0, 8.0]), np.array([True, False]), -9999.0)
        assert out.tolist() == [8.0, -9999.0]

    def test_two_d_scalar_sentinel(self):
        """A 2-D ``(time, n_face)`` mesh array, one scalar sentinel across it."""
        out = regap(
            np.array([[8.0, 8.0], [8.0, 8.0]]),
            np.array([[True, False], [True, True]]),
            -9999.0,
        )
        assert out.tolist() == [[8.0, -9999.0], [8.0, 8.0]]

    def test_three_d_scalar_sentinel_uniform_stack(self):
        """A 3-D raster stack with a SCALAR sentinel -- the path the shape re-key changed."""
        values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        domain = np.array([[[True, False]], [[False, True]]])
        out = regap(values, domain, -9999.0)
        assert out.tolist() == [[[1.0, -9999.0]], [[-9999.0, 4.0]]]

    def test_three_d_per_plane_list_sentinel(self):
        """A 3-D stack whose planes declare different sentinels keeps each its own."""
        values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        domain = np.array([[[True, False]], [[False, True]]])
        out = regap(values, domain, [-9999.0, -1.0])
        assert out.tolist() == [[[1.0, -9999.0]], [[-1.0, 4.0]]]

    def test_never_returns_its_input_even_on_a_full_domain(self):
        """A fresh array always, so the mesh facade cannot alias the source buffer."""
        values = np.array([1.0, 2.0])
        out = regap(values, np.array([True, True]), -9999.0)
        assert out is not values
        assert out.tolist() == [1.0, 2.0]


class TestDeclaredGap:
    """``declared_gap`` collapses agreeing planes to a scalar, else keeps the per-plane list."""

    def test_agreement_collapses_to_a_scalar(self):
        assert declared_gap([-9999.0, -9999.0]) == -9999.0
        assert declared_gap([None, None]) is None

    def test_disagreement_keeps_the_list(self):
        assert declared_gap([-9999.0, -1.0]) == [-9999.0, -1.0]
        assert declared_gap([-9999.0, None]) == [-9999.0, None]

    def test_agreement_is_judged_by_value_not_repr(self):
        assert declared_gap([-9999.0, np.float64(-9999.0)]) == -9999.0


class TestClipValues:
    """``clip_values`` validates bounds, widens, and preserves the gap; domain is required."""

    def test_domain_is_required(self):
        """An un-migrated call site that forgot ``domain`` fails loudly, not silently."""
        with pytest.raises(TypeError):
            clip_values(np.array([1.0, 2.0]), 0.0, 8.0)  # type: ignore[call-arg]

    def test_a_non_real_bound_raises_type_error_named_for_the_operation(self):
        with pytest.raises(TypeError, match=r"clip_values\(\) needs a number"):
            clip_values(
                np.array([1.0, 2.0]),
                "low",
                8.0,
                domain=np.array([True, True]),
                operation="clip_values",
            )

    def test_a_bool_bound_is_refused(self):
        """Review M2: a ``bool`` is a ``Real`` subclass, so the old ``float(bound)`` took it as
        ``1.0``; the shared kernel refuses it rather than silently clamping to a boolean.
        """
        with pytest.raises(TypeError, match=r"clip_values\(\) needs a number"):
            clip_values(
                np.array([1.0, 2.0]),
                True,
                8.0,
                domain=np.array([True, True]),
                operation="clip_values",
            )

    def test_a_one_sided_bound_is_allowed(self):
        out, sentinel = clip_values(
            np.array([1.0, 9.0]), None, 8.0, domain=np.array([True, True])
        )
        assert out.tolist() == [1.0, 8.0]
        assert sentinel is None

    def test_crossed_bounds_are_refused(self):
        with pytest.raises(ValueError, match="at or below max"):
            clip_values(np.array([1.0, 9.0]), 8.0, 2.0, domain=np.array([True, True]))

    def test_a_gap_is_preserved_through_a_dtype_change(self):
        out, sentinel = clip_values(
            np.array([5, -9999]),
            8.0,
            22.0,
            domain=np.array([True, False]),
            sentinel=-9999,
        )
        assert out.tolist() == [8.0, -9999.0]
        assert sentinel == -9999


class TestGapKernels:
    """round / isin / apply / astype / fillna gap behaviour, directly against the kernels."""

    def test_round_requires_domain(self):
        with pytest.raises(TypeError):
            round_values(np.array([1.4, 2.6]))  # type: ignore[call-arg]

    def test_round_without_a_sentinel_is_plain_round(self):
        out, _ = round_values(np.array([1.4, 2.6]), domain=np.array([True, True]))
        assert out.tolist() == [1.0, 3.0]

    def test_round_preserves_a_fractional_sentinel(self):
        out, _ = round_values(
            np.array([5.0, -9999.5]),
            0,
            domain=np.array([True, False]),
            sentinel=-9999.5,
        )
        assert out.tolist() == [5.0, -9999.5]

    def test_isin_without_a_sentinel_is_plain_isin(self):
        out = isin_values(
            np.array([1, 2, 3]), [2, 3], domain=np.array([True, True, True])
        )
        assert out.tolist() == [False, True, True]

    def test_isin_excludes_the_gap(self):
        out = isin_values(
            np.array([5.0, -9999.0]), [-9999.0], domain=np.array([True, False])
        )
        assert out.tolist() == [False, False]

    def test_isin_accepts_a_set(self):
        """Review M2: the old base ``isin_values`` left a ``set`` as a 0-d object array, so a
        mesh ``isin({...})`` returned all-``False``; the shared kernel converts it.
        """
        out = isin_values(
            np.array([5.0, -9999.0]), {5.0}, domain=np.array([True, False])
        )
        assert out.tolist() == [True, False]

    def test_apply_refuses_a_shape_change(self):
        with pytest.raises(ValueError, match="changed the array shape"):
            apply_values(
                np.array([1.0, 2.0]), lambda a: a[:1], domain=np.array([True, True])
            )

    def test_apply_preserves_the_gap(self):
        out, _ = apply_values(
            np.array([5.0, -9999.0]),
            lambda a: a * 2,
            domain=np.array([True, False]),
            sentinel=-9999.0,
        )
        assert out.tolist() == [10.0, -9999.0]

    def test_astype_without_a_gap_just_casts(self):
        out, _ = astype_values(
            np.array([1.0, 2.0]),
            np.int32,
            domain=np.array([True, True]),
            sentinel=-9999.0,
        )
        assert out.tolist() == [1, 2]

    def test_astype_refuses_when_the_target_cannot_hold_the_sentinel(self):
        with pytest.raises(ValueError, match="cannot mark a gap"):
            astype_values(
                np.array([5.0, -9999.0]),
                np.uint8,
                domain=np.array([True, False]),
                sentinel=-9999.0,
            )

    def test_astype_to_a_narrow_float_snaps_the_sentinel_and_keeps_the_gap(self):
        """float16 holds -9999 by RANGE (the raster's rule): it snaps to -10000 and marks it."""
        out, sentinel = astype_values(
            np.array([5.0, -9999.0]),
            np.float16,
            domain=np.array([True, False]),
            sentinel=-9999.0,
        )
        assert out.dtype == np.float16
        assert out.tolist() == [5.0, -10000.0]
        assert sentinel == -10000.0

    def test_astype_to_a_wider_float_keeps_the_gap(self):
        out, sentinel = astype_values(
            np.array([5.0, -9999.0]),
            np.float32,
            domain=np.array([True, False]),
            sentinel=-9999.0,
        )
        assert out.tolist() == [5.0, -9999.0]
        assert out.dtype == np.float32
        assert sentinel == -9999.0

    def test_fillna_widens_for_a_fill_the_type_cannot_hold(self):
        out, sentinel = fillna(
            np.array([5, -9999], dtype="int16"),
            2.5,
            domain=np.array([True, False]),
            sentinel=-9999,
        )
        assert out.dtype == np.float64
        assert out.tolist() == [5.0, 2.5]
        assert sentinel == -9999


class TestOutputHygiene:
    """The kernels return fresh arrays; the no-fill / identity paths do not alias the source."""

    def test_fillna_returns_a_fresh_array(self):
        src = np.array([1.0, 2.0, 3.0])
        out, _ = fillna(src, 0.0, domain=np.array([True, True, True]))
        assert out is not src

    def test_fillna_keeps_the_dtype_when_nothing_is_missing(self):
        """A gap-free array fills no cell, so the fill never lands -- the dtype must not widen.

        Review M1: the shared kernel used to widen a gap-free ``int16`` to ``float64`` for a
        fractional fill it never wrote. One owner now, so both hierarchies keep the dtype when
        there is nothing to fill.
        """
        src = np.array([1, 2, 3], dtype="int16")
        out, sentinel = fillna(
            src, 2.5, domain=np.array([True, True, True]), sentinel=-9999
        )
        assert out.dtype == np.int16
        assert out.tolist() == [1, 2, 3]
        assert out is not src, "still a fresh array, never the caller's buffer"

    def test_fillna_still_widens_when_a_gap_is_actually_filled(self):
        """The fast path is only for a gap-free array; a real fill still widens (D5)."""
        out, _ = fillna(
            np.array([1, -9999], dtype="int16"),
            2.5,
            domain=np.array([True, False]),
            sentinel=-9999,
        )
        assert out.dtype == np.float64
        assert out.tolist() == [1.0, 2.5]

    def test_apply_does_not_alias_the_source_on_an_identity(self):
        src = np.array([1.0, 2.0, 3.0])
        out, _ = apply_values(src, lambda a: a, domain=np.array([True, True, True]))
        assert out is not src
        assert not np.shares_memory(out, src)
        out[0] = 99.0
        assert src[0] == 1.0, "mutating the result must not touch the source"
