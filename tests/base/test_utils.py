"""Tests for helpers in :mod:`pyramids.base._utils`.

Covers :func:`apply_unpack` (the CF packed-value helper
install hint) and :func:`apply_unpack` (the shared scale/offset primitive behind both
the NetCDF CF unpack path and the raster ``read_array(scaled=True)`` path).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._utils import apply_unpack

pytestmark = pytest.mark.core


class TestApplyUnpack:
    """Tests for the shared scale/offset primitive ``apply_unpack``."""

    def test_none_none_passthrough(self):
        """Both None returns the array unchanged, with no float promotion."""
        arr = np.array([0, 1, 2], dtype="int16")
        out = apply_unpack(arr, None, None)
        assert out is arr, "identity: the same array object is returned"
        assert out.dtype == np.int16, "no float promotion when nothing is declared"

    def test_scalar_scale_and_offset(self):
        """A scalar scale/offset applies as float64."""
        out = apply_unpack(np.array([0, 1, 2]), 0.1, 5.0)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_allclose(out, [5.0, 5.1, 5.2])

    def test_scale_only_and_offset_only(self):
        """Scale-only and offset-only each apply their single operand."""
        np.testing.assert_allclose(
            apply_unpack(np.array([1, 2]), 2.0, None), [2.0, 4.0]
        )
        np.testing.assert_allclose(
            apply_unpack(np.array([1, 2]), None, 3.0), [4.0, 5.0]
        )

    def test_ndarray_broadcast(self):
        """A per-band (bands, 1, 1) scale/offset broadcasts over a 3-D array."""
        arr = np.ones((2, 1, 1))
        scale = np.array([2.0, 3.0]).reshape(-1, 1, 1)
        offset = np.array([1.0, -1.0]).reshape(-1, 1, 1)
        out = apply_unpack(arr, scale, offset)
        np.testing.assert_allclose(out.ravel(), [3.0, 2.0])

    def test_masked_array_mask_preserved(self):
        """A masked-array input keeps its mask across the transform."""
        arr = np.ma.MaskedArray([0, 1, 2], mask=[False, True, False])
        out = apply_unpack(arr, 0.1, 5.0)
        assert isinstance(out, np.ma.MaskedArray), "mask must survive"
        np.testing.assert_array_equal(out.mask, [False, True, False])

    def test_reexported_identity(self):
        """`_lazy` re-exports the canonical `base._utils` primitive (deprecated alias; one object)."""
        # This asserts a property of the pyramids-lazy backend, so it can only run where the backend
        # is installed. importorskip(exc_type=ImportError) skips — rather than errors — in a bare
        # core and in a backend without its [parquet] extra (whose guard raises a branded
        # ImportError, which importorskip re-raises unless told it is an "absent" signal).
        pytest.importorskip("pyramids_lazy._lazy", exc_type=ImportError)
        from pyramids_lazy._lazy import _apply_unpack
        from pyramids_lazy._lazy import apply_unpack as lazy_fn

        assert apply_unpack is lazy_fn is _apply_unpack, "one shared primitive"
