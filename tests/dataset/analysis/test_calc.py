"""`Dataset.calc` — string-expression band algebra (the Python side of `pyramids calc`)."""

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset._calc import eval_calc

REF = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)


def _ds(values: list) -> Dataset:
    """A single-band raster on `REF` holding `values` (shape (1, rows, cols))."""
    arr = np.asarray(values, dtype="float64").reshape(1, 2, 3)
    return Dataset.from_array(arr, geo_ref=REF)


class TestCalc:
    """`Dataset.calc(expr, *others)` evaluates a whitelisted string expression."""

    def test_ndvi_over_two_rasters(self):
        """`self` binds to A, the first other to B; `(B-A)/(B+A)` is NDVI."""
        red = _ds([[1.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
        nir = _ds([[3.0, 6.0, 4.0], [3.0, 3.0, 3.0]])
        out = red.calc("(B - A) / (B + A)", nir)
        expected = (
            (nir.read_array() - red.read_array())
            / (nir.read_array() + red.read_array())
        ).ravel()
        assert np.asarray(out.read_array()).ravel().tolist() == pytest.approx(
            expected.tolist()
        )

    def test_single_raster_scalar_and_np_func(self):
        """One input (A) with a scalar and a whitelisted `np` call works."""
        ds = _ds([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
        out = ds.calc("np.sqrt(A) * 10")
        assert np.asarray(out.read_array()).ravel().tolist() == pytest.approx(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        )

    def test_result_carries_template_grid_and_crs(self):
        """The result sits on `self`'s geotransform/CRS."""
        ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = ds.calc("A + 1")
        assert out.epsg == 4326
        assert out.geotransform == ds.geotransform

    def test_dataset_level_metadata_is_carried(self):
        """Dataset-level tags travel to the result (like `combine`)."""
        ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ds.metadata = {"SENSOR": "OLI"}
        assert ds.calc("A * 2").metadata.get("SENSOR") == "OLI"

    def test_dtype_cast(self):
        """`dtype=` casts the result array."""
        ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = ds.calc("A * 2", dtype="int16")
        assert np.asarray(out.read_array()).dtype == np.int16

    def test_unknown_name_raises(self):
        """A name not bound to an input raster is rejected."""
        with pytest.raises(ValueError, match="unknown name"):
            _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).calc("A + Z")

    def test_disallowed_expression_raises(self):
        """Anything outside the arithmetic/np whitelist is rejected, never executed."""
        with pytest.raises(ValueError, match="disallowed element"):
            _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).calc("__import__('os').getcwd()")

    def test_too_many_inputs_raises(self):
        """At most 26 inputs (A..Z) are allowed."""
        base = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        others = [_ds([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) for _ in range(26)]
        with pytest.raises(ValueError, match="at most 26"):
            base.calc("A + B", *others)

    def test_grid_mismatch_raises(self):
        """An operand on a different grid raises before evaluating."""
        base = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        other = Dataset.from_array(
            np.ones((1, 2, 3), dtype="float64"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=5.0, epsg=4326),
        )
        with pytest.raises(AlignmentError, match="grid"):
            base.calc("A + B", other)

    def test_facade_delegates_to_engine(self, mocker):
        """`Dataset.calc` forwards to `Analysis.calc` unchanged."""
        ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sentinel = object()
        spy = mocker.patch.object(ds.analysis, "calc", return_value=sentinel)
        result = ds.calc("A + 1", dtype="int16")
        assert result is sentinel
        spy.assert_called_once_with("A + 1", dtype="int16")

    def test_comparison_returns_a_uint8_mask(self):
        """A bare comparison yields a 0/1 uint8 mask, not a bool-dtype crash."""
        ds = _ds([[1.0, 2.0, 4.0], [0.0, 3.0, 5.0]])
        out = ds.calc("A > 2")
        arr = np.asarray(out.read_array())
        assert arr.dtype == np.uint8, f"expected uint8 mask, got {arr.dtype}"
        assert arr.ravel().tolist() == [0, 0, 1, 0, 1, 1]

    def test_scalar_expression_broadcasts_to_the_grid(self):
        """A constant/0-d expression fills the whole grid instead of raising."""
        ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = ds.calc("1 + 2")
        arr = np.asarray(out.read_array())
        assert arr.shape[-2:] == (2, 3), f"expected the template grid, got {arr.shape}"
        assert set(arr.ravel().tolist()) == {3.0}

    def test_multi_band_result_keeps_the_band_axis(self):
        """A multi-band input yields a multi-band result (the ndim-3 path)."""
        arr = np.arange(12.0, dtype="float64").reshape(2, 2, 3)
        ds = Dataset.from_array(arr, geo_ref=REF)
        out = ds.calc("A * 2")
        assert out.band_count == 2
        assert np.asarray(out.read_array()).ravel().tolist() == pytest.approx(
            (arr * 2).ravel().tolist()
        )


class TestEvalCalc:
    """The shared safe evaluator — grammar coverage over bound arrays (no GDAL)."""

    A = np.array([1.0, 2.0, 4.0])
    B = np.array([3.0, 6.0, 4.0])

    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("A + B", [4.0, 8.0, 8.0]),
            ("A - B", [-2.0, -4.0, 0.0]),
            ("A * B", [3.0, 12.0, 16.0]),
            ("B / A", [3.0, 3.0, 1.0]),
            ("A ** 2", [1.0, 4.0, 16.0]),
            ("B % A", [0.0, 0.0, 0.0]),
            ("B // A", [3.0, 3.0, 1.0]),
            ("-A", [-1.0, -2.0, -4.0]),
            ("+A", [1.0, 2.0, 4.0]),
            ("A > 2", [False, False, True]),
            ("A == B", [False, False, True]),
            ("np.maximum(A, B)", [3.0, 6.0, 4.0]),
            ("np.minimum(A, B)", [1.0, 2.0, 4.0]),
            ("np.clip(A, 2, 3)", [2.0, 2.0, 3.0]),
            ("np.where(A > 2, A, 0)", [0.0, 0.0, 4.0]),
            ("np.power(A, 2)", [1.0, 4.0, 16.0]),
            ("np.abs(-A)", [1.0, 2.0, 4.0]),
            ("np.sqrt(A)", [1.0, 2.0**0.5, 2.0]),
        ],
    )
    def test_grammar(self, expr, expected):
        """Each operator / comparison / np-func evaluates element-wise over A, B.

        Args:
            expr: The calc expression.
            expected: Its expected per-cell result.
        """
        out = eval_calc(expr, {"A": self.A, "B": self.B})
        assert out.tolist() == pytest.approx(expected), f"{expr!r} -> {out.tolist()}"

    def test_log_family(self):
        """`np.log`, `np.log10` and `np.exp` are whitelisted."""
        assert eval_calc("np.log(A)", {"A": np.array([1.0])}).tolist() == pytest.approx(
            [0.0]
        )
        assert eval_calc(
            "np.log10(A)", {"A": np.array([100.0])}
        ).tolist() == pytest.approx([2.0])
        assert eval_calc("np.exp(A)", {"A": np.array([0.0])}).tolist() == pytest.approx(
            [1.0]
        )

    def test_unknown_name_raises(self):
        """A name not in `variables` is rejected, not treated as zero."""
        with pytest.raises(ValueError, match="unknown name"):
            eval_calc("A + C", {"A": self.A})

    def test_disallowed_call_raises(self):
        """A non-whitelisted call (even on `np`) is rejected."""
        with pytest.raises(ValueError, match="disallowed element"):
            eval_calc("np.sum(A)", {"A": self.A})

    def test_attribute_access_raises(self):
        """Attribute access / dunder escape is rejected, never executed."""
        with pytest.raises(ValueError, match="disallowed element"):
            eval_calc("A.__class__", {"A": self.A})

    def test_keyword_argument_rejected(self):
        """A keyword argument to an np call is outside the grammar."""
        with pytest.raises(ValueError, match="disallowed element"):
            eval_calc("np.clip(A, a_min=0, a_max=1)", {"A": self.A})

    def test_syntax_error_propagates(self):
        """A malformed expression raises `SyntaxError` from the parse step."""
        with pytest.raises(SyntaxError):
            eval_calc("A +", {"A": self.A})
