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

    def test_animate_forwards_physical_sentinel(self, monkeypatch):
        """``animate`` hands cleopatra the physical sentinel, not the stored one.

        Test scenario:
            On a packed stack the ``exclude_value`` forwarded to ``render_array`` must be the
            physical gap value (-4999.5), not the stored -9999 -- otherwise cleopatra compares
            -9999 against a physical array and draws the gap as data. ``render_array`` is
            captured so the test needs no renderer.
        """
        captured: dict = {}

        def _capture(request, **kwargs):
            captured["exclude_value"] = request.exclude_value
            return "glyph"

        monkeypatch.setattr(analysis_module, "render_array", _capture)
        stack = np.array([[[100, 120, -9999]], [[140, 160, 180]]], dtype="int16")
        ds = Dataset.from_array(stack, geo_ref=GEO, no_data_value=-9999)
        ds.scale = [0.5, 0.5]
        ds.animate()
        assert captured["exclude_value"] == [-4999.5], (
            f"animate must forward the physical sentinel; got {captured['exclude_value']}"
        )
