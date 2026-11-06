"""`Dataset.calc` — string-expression band algebra (the Python side of `pyramids calc`)."""

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset, GeoReference

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
