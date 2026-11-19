"""`read_array(squeeze=)` — eager/lazy shape convergence (#1226, #1241).

The eager path historically flattened every non-spatial dimension into one band axis and
squeezed a singleton to 2-D, while the lazy path kept `(*band_sizes, rows, cols)`. The `squeeze`
knob makes the two interchangeable and, since #1241, defaults to the dimension-preserving layout:
`squeeze=False` (the default) returns `(*band_sizes, rows, cols)` on both paths, keeping a size-1
axis; `squeeze=True` returns the classic flattened `(bands, rows, cols)` (a singleton squeezed to
2-D) on both paths.
"""

from __future__ import annotations

from pathlib import Path

from pyramids.netcdf import NetCDF

MULTIVAR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
)


class TestReadArraySqueeze:
    """`squeeze` converges the eager and lazy read shapes and defaults to the legacy behaviour."""

    def test_squeeze_false_preserves_band_dimensions_eager(self):
        """`squeeze=False` keeps one axis per band dimension on the eager read (`ua` → 4-D)."""
        ua = NetCDF.read_file(str(MULTIVAR)).get_variable("ua")
        assert ua.read_array(squeeze=False).ndim == len(ua._band_dim_names) + 2

    def test_squeeze_false_keeps_the_size1_axis_eager(self):
        """A size-1 band dimension is kept, not squeezed, under `squeeze=False`."""
        tas = NetCDF.read_file(str(MULTIVAR)).get_variable("tas")
        preserved = tas.read_array(squeeze=False)
        assert preserved.shape[0] == 1, f"size-1 time axis dropped: {preserved.shape}"
        assert preserved.ndim == len(tas._band_dim_names) + 2

    def test_the_default_is_dimension_preserving_eager(self):
        """Since #1241 the default eager read keeps one axis per band dimension (not 2-D)."""
        tas = NetCDF.read_file(str(MULTIVAR)).get_variable("tas")
        preserved = tas.read_array()
        assert preserved.ndim == len(tas._band_dim_names) + 2
        assert preserved.shape[0] == 1, (
            f"size-1 time axis must be kept by default now: {preserved.shape}"
        )
