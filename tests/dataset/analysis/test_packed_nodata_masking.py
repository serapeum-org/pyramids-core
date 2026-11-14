"""Regression tests: no-data masking on CF-packed bands uses the *physical* sentinel.

A CF-packed band stores its no-data sentinel in the packed datatype, so ``read_array`` returns
the gap in physical units and the stored ``no_data_value`` never appears in it. Masking a
physical array against the stored sentinel therefore leaks the gap. These guard the readers that
build their mask / exclude list from a physical read -- ``percentile_stretch`` and
``match_histogram`` (issue #1561) and ``animate`` (issue #1562) -- against reverting to the
stored sentinel.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines import analysis as analysis_module

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326)


def _packed_row(values, *, scale=0.5, no_data=-9999):
    """A single-band CF-packed ``int16`` raster (physical = stored * ``scale``)."""
    ds = Dataset.from_array(
        np.asarray(values, dtype="int16").reshape(1, -1),
        geo_ref=GEO,
        no_data_value=no_data,
    )
    ds.scale = [scale]
    return ds


@pytest.fixture
def captured_exclude(monkeypatch):
    """Capture the ``exclude_value`` ``animate`` forwards to ``render_array`` (no renderer runs).

    Returns:
        dict: populated with ``exclude_value`` once ``animate`` is called.
    """
    box: dict = {}

    def _capture(request, **kwargs):
        box["exclude_value"] = request.exclude_value
        return "glyph"

    monkeypatch.setattr(analysis_module, "render_array", _capture)
    return box


def _packed_stack(no_data=-9999, scale=(0.5, 0.5)):
    """A two-band CF-packed stack with a gap in the first band (physical = stored * scale)."""
    stack = np.array([[[100, 120, -9999]], [[140, 160, 180]]], dtype="int16")
    ds = Dataset.from_array(stack, geo_ref=GEO, no_data_value=no_data)
    ds.scale = list(scale)
    return ds


class TestPackedNoDataMasking:
    """The physical gap of a packed band is excluded, not treated as a sample."""

    def test_percentile_stretch_excludes_packed_gap(self):
        """The packed gap is masked, so the percentiles are not dragged down to it.

        Test scenario:
            A band packed at scale 0.5 with stored no-data -9999 reads the gap as -4999.5. The
            full-range stretch must ignore it: the gap stays no-data in the output and the valid
            cells span [0, 1] computed from the real values only (not clamped to 0 by the leaked
            gap).
        """
        ds = _packed_row([100, 120, 140, -9999, 160, 180])
        out = ds.percentile_stretch(0, 100)
        arr = out.read_array().ravel()
        valid = arr[[0, 1, 2, 4, 5]]
        assert np.isclose(arr[3], out.no_data_value[0]), "the gap must stay no-data"
        assert valid.min() == pytest.approx(0.0) and valid.max() == pytest.approx(
            1.0
        ), (
            f"valid cells must span [0, 1] from the real values only; got {valid.tolist()}"
        )

    def test_match_histogram_excludes_packed_gap(self):
        """Histogram matching drops the packed gap from both distributions.

        Test scenario:
            Matching a packed band against a packed reference must not treat the -4999.5 gap as a
            sample; the output keeps the gap as no-data and every real cell is finite.
        """
        ds = _packed_row([100, 120, 140, -9999, 160, 180])
        ref = _packed_row([40, 60, 80, -9999, 100, 120])
        out = ds.match_histogram(ref)
        arr = out.read_array().ravel()
        assert np.isclose(arr[3], out.no_data_value[0]), "the gap must stay no-data"
        assert np.isfinite(arr[[0, 1, 2, 4, 5]]).all(), "every real cell must be finite"

    def test_animate_forwards_physical_sentinel(self, captured_exclude):
        """``animate`` hands cleopatra the physical sentinel, not the stored one.

        Test scenario:
            On a packed stack the ``exclude_value`` forwarded to ``render_array`` must be the
            physical gap value (-4999.5), not the stored -9999 -- otherwise cleopatra compares
            -9999 against a physical array and draws the gap as data.
        """
        _packed_stack().animate()
        assert captured_exclude["exclude_value"] == [-4999.5], (
            f"animate must forward the physical sentinel; got {captured_exclude['exclude_value']}"
        )

    def test_percentile_stretch_unpacked_band_still_excludes_nodata(self):
        """An unpacked band (stored == physical) keeps excluding its no-data after the fix.

        Test scenario:
            With no packing the stored sentinel equals the physical value, so the fix must leave
            the common case unchanged: the -9999 gap stays no-data and the valid cells span
            [0, 1].
        """
        ds = Dataset.from_array(
            np.array([[10.0, 20.0, 30.0, -9999.0, 40.0]], dtype="float32"),
            geo_ref=GEO,
            no_data_value=-9999.0,
        )
        arr = ds.percentile_stretch(0, 100).read_array().ravel()
        valid = arr[[0, 1, 2, 4]]
        assert np.isclose(arr[3], -9999.0), "unpacked gap must stay no-data"
        assert valid.min() == pytest.approx(0.0) and valid.max() == pytest.approx(
            1.0
        ), f"valid cells must span [0, 1]; got {valid.tolist()}"

    def test_percentile_stretch_nan_cells_excluded(self):
        """NaN cells are dropped via ``isfinite`` and do not enter the percentiles.

        Test scenario:
            A float band with a NaN gap: the NaN is excluded from the stretch (via ``isfinite``),
            the valid cells span [0, 1] from the finite values, and the NaN cell becomes the
            output's no-data fill.
        """
        ds = Dataset.from_array(
            np.array([[1.0, 2.0, np.nan, 3.0, 4.0]], dtype="float32"), geo_ref=GEO
        )
        out = ds.percentile_stretch(0, 100)
        arr = out.read_array().ravel()
        valid = arr[[0, 1, 3, 4]]
        assert np.isclose(arr[2], out.no_data_value[0]), (
            "the NaN cell must become no-data"
        )
        assert valid.min() == pytest.approx(0.0) and valid.max() == pytest.approx(
            1.0
        ), f"finite cells must span [0, 1]; got {valid.tolist()}"

    def test_match_histogram_excludes_unpacked_reference_gap(self):
        """The reference band's physical sentinel is used, for any packing of the reference.

        Test scenario:
            A packed source matched against an *unpacked* reference that has its own -9999 gap:
            the reference gap must be dropped from its distribution (via the reference's physical
            sentinel), and the output keeps the source gap as no-data.
        """
        ds = _packed_row([100, 120, 140, -9999, 160, 180])
        ref = Dataset.from_array(
            np.array([[5.0, 10.0, 15.0, -9999.0, 20.0, 25.0]], dtype="float32"),
            geo_ref=GEO,
            no_data_value=-9999.0,
        )
        out = ds.match_histogram(ref)
        arr = out.read_array().ravel()
        assert np.isclose(arr[3], out.no_data_value[0]), (
            f"the source gap must stay no-data ({out.no_data_value[0]})"
        )
        assert np.isfinite(arr[[0, 1, 2, 4, 5]]).all(), (
            "every real source cell must be finite"
        )

    def test_animate_appends_extra_exclude_value(self, captured_exclude):
        """A caller ``exclude_value`` is appended after the physical sentinel.

        Test scenario:
            ``animate(exclude_value=X)`` on a packed stack forwards ``[physical_sentinel, X]`` so
            both the gap and the extra value are masked.
        """
        _packed_stack().animate(exclude_value=7.0)
        assert captured_exclude["exclude_value"] == [-4999.5, 7.0], (
            f"expected [physical sentinel, extra]; got {captured_exclude['exclude_value']}"
        )

    def test_animate_physical_sentinel_tracks_packing_scale(self, captured_exclude):
        """The forwarded sentinel reflects the band's actual packing scale, not a constant.

        Test scenario:
            With scale 0.1 the stored -9999 reads as physical -999.9, so ``animate`` must forward
            -999.9 -- proving the conversion uses the real packing (``_physical_no_data``) rather
            than a hardcoded factor.
        """
        _packed_stack(scale=(0.1, 0.1)).animate()
        assert captured_exclude["exclude_value"] == [pytest.approx(-999.9)], (
            f"expected [-999.9] for scale 0.1; got {captured_exclude['exclude_value']}"
        )
