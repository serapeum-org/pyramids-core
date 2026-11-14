"""Tests for ``NetCDF.reduce(min_count=)`` — gating sparse reduction windows.

A reduced cell fed by fewer than ``min_count`` valid (non-gap) samples resolves to the
variable's no-data value (NaN when it declares none) instead of a statistic computed from
too few samples. Built with in-memory ``NetCDF.from_array`` variables so each column's
valid-sample count is controlled exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

_GEO = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)


def _variable(values, *, no_data_value=-9999.0, size=3):
    """A single `(size, 1, N)` time variable `t` from a flat per-step list of rows."""
    arr = np.asarray(values, dtype="float64")
    return NetCDF.from_array(
        arr,
        geo_ref=_GEO,
        variable_name="t",
        no_data_value=no_data_value,
        dims=ExtraDimensions(name="time", values=[float(i) for i in range(size)]),
    ).get_variable("t")


class TestNetCDFReduceMinCount:
    """`NetCDF.reduce` gates columns below `min_count` valid samples (skipna only)."""

    @staticmethod
    def _sparse():
        """A `(3, 1, 2)` variable: x0 has one valid cell (sentinel gaps), x1 has three."""
        return _variable(np.array([[[5.0, 1.0]], [[-9999.0, 2.0]], [[-9999.0, 3.0]]]))

    def test_min_count_zero_keeps_sparse_column(self):
        """min_count=0 (default) leaves the one-sample column untouched.

        Test scenario:
            Collapsing time with min_count=0 reduces the one-sample column to
            that value, as today.
        """
        out = self._sparse().reduce("time", "mean", min_count=0).read_array()
        assert out[0, 0] == pytest.approx(5.0), "one-sample column kept at min_count=0"
        assert out[0, 1] == pytest.approx(2.0), "three-sample column is mean(1,2,3)"

    def test_min_count_renders_declared_no_data(self):
        """A gated column holds the declared no-data value, not NaN.

        Test scenario:
            min_count=2 drops the one-sample column to the variable's -9999
            sentinel while the three-sample column keeps its mean.
        """
        reduced = self._sparse().reduce("time", "mean", min_count=2)
        out = reduced.read_array()
        assert out[0, 0] == -9999.0, f"gated column should be -9999, got {out[0, 0]}"
        assert out[0, 1] == pytest.approx(2.0), "full column unaffected"

    def test_min_count_nan_when_no_sentinel(self):
        """Without a declared no-data value a gated column is NaN.

        Test scenario:
            A variable with no sentinel reduces the one-sample column to NaN at
            min_count=2.
        """
        var = _variable(
            np.array([[[5.0, 1.0]], [[np.nan, 2.0]], [[np.nan, 3.0]]]),
            no_data_value=None,
        )
        out = var.reduce("time", "mean", min_count=2).read_array()
        assert np.isnan(out[0, 0]), "gated column is NaN when no sentinel is declared"
        assert out[0, 1] == pytest.approx(2.0), "full column unaffected"

    def test_min_count_gates_per_group(self):
        """Grouped reduction gates each window by its own valid count.

        Test scenario:
            values [1, 2, 3] grouped ["a", "a", "b"] with min_count=2: group a
            (two samples) keeps its mean 1.5, group b (one sample) is no-data.
        """
        var = _variable(np.array([[[1.0]], [[2.0]], [[3.0]]]))
        out = var.reduce(
            "time", "mean", groupby=["a", "a", "b"], min_count=2
        ).read_array()
        assert out[0, 0] == pytest.approx(1.5), "two-sample group a keeps its mean"
        assert out[1, 0] == -9999.0, f"one-sample group b is no-data, got {out[1, 0]}"

    def test_min_count_boundary_keeps_exact(self):
        """A column with exactly min_count valid cells is kept (>=, not >).

        Test scenario:
            Two valid cells and min_count=2 keep the mean.
        """
        var = _variable(np.array([[[10.0]], [[20.0]], [[-9999.0]]]))
        out = var.reduce("time", "mean", min_count=2).read_array()
        assert out[0, 0] == pytest.approx(15.0), "exactly-at-threshold column kept"

    def test_min_count_ignored_for_count(self):
        """The counting reduction ignores min_count.

        Test scenario:
            how='count' on the one-sample column still answers 1 at min_count=5.
        """
        out = self._sparse().reduce("time", "count", min_count=5).read_array()
        assert int(out[0, 0]) == 1, f"count must ignore min_count, got {out[0, 0]}"
        assert int(out[0, 1]) == 3, "the full column counts 3"

    def test_min_count_ignored_without_skipna(self):
        """min_count has no effect when skipna is False.

        Test scenario:
            A full column reduced with skipna=False and a large min_count keeps
            the plain mean.
        """
        var = _variable(np.array([[[10.0]], [[20.0]], [[30.0]]]))
        out = var.reduce("time", "mean", skipna=False, min_count=99).read_array()
        assert out[0, 0] == pytest.approx(20.0), "skipna=False must ignore min_count"

    @pytest.mark.parametrize(
        "bad, exc", [(-1, ValueError), (2.5, TypeError), (True, TypeError)]
    )
    def test_invalid_min_count_raises(self, bad, exc):
        """A negative or non-integer min_count is rejected up front.

        Args:
            bad: The invalid min_count.
            exc: The expected exception type.

        Test scenario:
            -1 raises ValueError; a float and a bool raise TypeError.
        """
        with pytest.raises(exc, match="min_count"):
            self._sparse().reduce("time", "mean", min_count=bad)
