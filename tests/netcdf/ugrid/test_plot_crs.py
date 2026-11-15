"""The returned ``MeshGlyph`` carries the dataset CRS (issue #1560).

cleopatra's ``MeshGlyph`` takes no ``crs`` in ``__init__`` and the ``Mesh2d`` topology carries
none, so pyramids must set ``glyph.crs`` from the dataset's ``epsg`` at the plotting boundary.
Otherwise a caller doing ``glyph.add_tiles()`` (or ``add_features`` / ``add_relief``) with no
explicit ``crs=`` gets cleopatra's EPSG:3857 default and a misplaced backdrop.
"""

import numpy as np
import pyproj
import pytest

matplotlib = pytest.importorskip("matplotlib")

_CUSTOM_WKT = pyproj.CRS.from_proj4(
    "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
).to_wkt()

matplotlib.use("Agg")

from pyramids.netcdf.ugrid import MeshDataset  # noqa: E402

pytestmark = pytest.mark.plot

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(epsg=28992):
    """Build a tiny two-face MeshDataset with one face variable and the given CRS."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"d": np.array([1.0, 2.0])},
        data_locations={"d": "face"},
        epsg=epsg,
    )


class TestGlyphCarriesDatasetCRS:
    """``plot`` / ``plot_outline`` tag the returned glyph with the dataset CRS."""

    def test_plot_outline_sets_glyph_crs(self):
        """``plot_outline()`` returns a glyph whose ``crs`` equals the dataset epsg."""
        glyph = _mesh(28992).plot_outline()
        assert glyph.crs == 28992, f"expected glyph.crs==28992, got {glyph.crs!r}"

    def test_plot_sets_glyph_crs(self):
        """``plot(variable)`` returns a glyph whose ``crs`` equals the dataset epsg."""
        glyph = _mesh(28992).plot("d")
        assert glyph.crs == 28992, f"expected glyph.crs==28992, got {glyph.crs!r}"

    def test_projected_crs_without_epsg_uses_wkt(self):
        """A projected CRS with no EPSG code tags the glyph with the WKT, not ``None``.

        ``MeshDataset.epsg`` is ``None`` for a custom/ESRI projection, but ``crs_wkt`` is
        populated; the glyph must still carry a CRS so a downstream ``add_tiles()`` does not
        fall back to EPSG:3857 and misproject.
        """
        ds = _mesh(28992)
        ds.set_crs(crs=_CUSTOM_WKT)
        assert ds.epsg is None, (
            "precondition: the custom CRS has no EPSG authority code"
        )
        glyph = ds.plot_outline()
        assert glyph.crs == _CUSTOM_WKT, f"expected the CRS WKT, got {glyph.crs!r}"

    def test_no_crs_when_dataset_has_none(self):
        """A dataset without a CRS leaves ``glyph.crs`` as ``None`` (no spurious default).

        ``from_arrays`` requires an epsg, but a mesh read from a CRS-less file has
        ``epsg=None``; drive the guard directly rather than constructing that here.
        """

        class _NoCrsDataset:
            epsg = None
            crs_wkt = None

        class _Glyph:
            crs = None

        tagged = MeshDataset._glyph_with_crs(_NoCrsDataset(), _Glyph())
        assert tagged.crs is None, f"expected glyph.crs is None, got {tagged.crs!r}"

    def test_non_glyph_render_result_is_left_untouched(self):
        """A render result without a ``crs`` attribute is returned as-is (hasattr guard).

        Existing plot tests patch ``_mesh_render`` with a string sentinel; this pins the
        branch directly so the helper never tries to set ``crs`` on a non-glyph object.
        """
        ds = _mesh(28992)
        sentinel = object()
        assert ds._glyph_with_crs(sentinel) is sentinel, (
            "non-glyph result must pass through untouched"
        )
        assert ds._glyph_with_crs("sentinel") == "sentinel", (
            "a str render result (the shape test_plot.py patches _mesh_render with) must pass through"
        )

    def test_preexisting_glyph_crs_is_not_overridden(self):
        """An already-set ``glyph.crs`` wins over the dataset epsg (defensive guard)."""
        ds = _mesh(28992)

        class _Glyph:
            crs = 4326

        tagged = ds._glyph_with_crs(_Glyph())
        assert tagged.crs == 4326, f"explicit crs must be preserved, got {tagged.crs!r}"
