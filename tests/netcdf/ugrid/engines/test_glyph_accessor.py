"""The ``mesh.glyph`` accessor owns the plotting family; the flat facades agree.

Pins the ugrid engine-accessor migration (issue #1650): the plotting family moves onto a
dedicated ``mesh.glyph`` engine, mirroring the raster ``Dataset.glyph`` surface, and the flat
``mesh.plot`` / ``mesh.plot_histogram`` / ``mesh.plot_vector_field`` / ``mesh.plot_outline`` /
``mesh.animate`` / ``mesh.to_image`` methods become deprecating facades that delegate to it, so
both call forms render the same thing.

``animate`` is the one method kept on ``MeshDataset`` (as ``_animate_impl``) because its body
reads the per-variable time coordinate through a module-level helper the engine cannot import
without a cycle; the engine method delegates to it, which these tests pin.

The Agg backend is forced before any pyplot import, so nothing here opens a window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

mesh_glyph = pytest.importorskip(
    "cleopatra.glyphs.gridded.mesh_glyph", reason="cleopatra not installed"
)
MeshGlyph = mesh_glyph.MeshGlyph

from pyramids.netcdf.ugrid import MeshDataset  # noqa: E402
from pyramids.netcdf.ugrid.engines.glyph import _MeshGlyph  # noqa: E402

pytestmark = [
    pytest.mark.plot,
    pytest.mark.filterwarnings("ignore:Animation was deleted without rendering"),
]

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _mesh() -> MeshDataset:
    """A two-triangle mesh carrying a face-located ``v`` variable, with a CRS."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.array([1.0, 2.0])},
        data_locations={"v": "face"},
        epsg=4326,
    )


def _velocity() -> MeshDataset:
    """A two-triangle mesh with face-located ``ucx`` / ``ucy`` velocity components."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"ucx": np.array([1.0, 2.0]), "ucy": np.array([0.5, 1.5])},
        data_locations={"ucx": "face", "ucy": "face"},
    )


def _temporal() -> MeshDataset:
    """A two-triangle mesh with a temporal face variable ``d`` (3 steps)."""
    return MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"d": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        data_locations={"d": "face"},
    )


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_accessor_is_a_glyph_engine():
    """``mesh.glyph`` is a fresh ``_MeshGlyph`` bound to the dataset."""
    mesh = _mesh()
    assert isinstance(mesh.glyph, _MeshGlyph)


def test_plot_accessor_matches_flat_facade():
    """``mesh.glyph.plot()`` renders a ``MeshGlyph``; the deprecated ``mesh.plot()`` agrees."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.plot"):
        flat = mesh.plot("v")
    accessor = mesh.glyph.plot("v")
    assert isinstance(accessor, MeshGlyph)
    assert isinstance(flat, MeshGlyph)


def test_plot_histogram_accessor_matches_flat_facade():
    """``mesh.glyph.plot_histogram()`` returns ``(fig, ax, hist)``; the flat method agrees."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.plot_histogram"):
        flat = mesh.plot_histogram(bins=3)
    accessor = mesh.glyph.plot_histogram(bins=3)
    assert len(accessor) == len(flat) == 3


def test_plot_vector_field_accessor_matches_flat_facade():
    """``mesh.glyph.plot_vector_field()`` returns ``(fig, ax, im)``; the flat method agrees."""
    velocity = _velocity()
    with pytest.warns(DeprecationWarning, match="mesh.plot_vector_field"):
        _fig, _ax, flat_im = velocity.plot_vector_field("ucx", "ucy")
    _fig2, _ax2, accessor_im = velocity.glyph.plot_vector_field("ucx", "ucy")
    assert accessor_im.N == flat_im.N


def test_plot_outline_accessor_matches_flat_facade():
    """``mesh.glyph.plot_outline()`` renders a mappable-free ``MeshGlyph``; the flat agrees."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.plot_outline"):
        flat = mesh.plot_outline()
    accessor = mesh.glyph.plot_outline()
    assert isinstance(accessor, MeshGlyph)
    assert accessor.im is None and flat.im is None


def test_animate_accessor_matches_flat_facade():
    """``mesh.glyph.animate()`` and the deprecated ``mesh.animate()`` return the same type."""
    mesh = _temporal()
    with pytest.warns(DeprecationWarning, match="mesh.animate"):
        flat = mesh.animate()
    accessor = mesh.glyph.animate()
    assert accessor is not None
    assert type(accessor) is type(flat)


def test_animate_accessor_delegates_to_the_dataset_impl(monkeypatch):
    """``mesh.glyph.animate`` routes through ``MeshDataset._animate_impl`` (the cycle break)."""
    mesh = _temporal()
    calls: list[tuple] = []
    monkeypatch.setattr(
        MeshDataset,
        "_animate_impl",
        lambda self, *a, **k: calls.append((a, k)) or "sentinel",
    )
    assert mesh.glyph.animate("d", interval=50) == "sentinel"
    assert calls == [(("d",), {"interval": 50, "backend": "matplotlib"})]


def test_to_image_accessor_matches_flat_facade():
    """``mesh.glyph.to_image()`` and the deprecated ``mesh.to_image()`` return equal images."""
    mesh = _mesh()
    with pytest.warns(DeprecationWarning, match="mesh.to_image"):
        flat = mesh.to_image(cell_size=0.5)
    accessor = mesh.glyph.to_image(cell_size=0.5)
    assert accessor.size == flat.size


def test_plot_dispatches_to_a_non_matplotlib_backend():
    """``backend != "matplotlib"`` marshals the kwargs and hands the mesh to the resolved backend.

    A fake backend is registered in the shared plot-backend cache so the dispatch path runs
    without needing an interactive renderer; it is torn down afterwards. The call carries only
    default (matplotlib-agnostic) parameters, so nothing is rejected, and the engine form emits
    no deprecation.
    """
    from pyramids.base import _plot_backends

    mesh = _mesh()
    _plot_backends._CACHE = {"demo": lambda obj, *, backend, **kw: (obj, backend, kw)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            obj, backend, forwarded = mesh.glyph.plot("v", backend="demo")
    finally:
        _plot_backends.clear_plot_backend_cache()
    assert obj is mesh
    assert backend == "demo"
    assert forwarded["variable_name"] == "v"


def test_accessor_form_does_not_warn():
    """The engine call is the non-deprecated path -- it emits NO deprecation at all.

    Not merely no ``mesh.*`` deprecation: an engine body that hands off across the stack must
    reach the other side's engine too. ``to_image`` rasterises and hands the grid to the raster
    ``Dataset``, so it must call ``grid.analysis.to_image`` -- routing through the deprecated
    ``grid.to_image`` facade made the sanctioned ``mesh.glyph.to_image`` path emit
    ``ds.to_image is deprecated``, which broke warnings-as-errors consumers.
    """
    mesh = _mesh()
    velocity = _velocity()
    temporal = _temporal()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh.glyph.plot("v")
        mesh.glyph.plot_histogram(bins=3)
        velocity.glyph.plot_vector_field("ucx", "ucy")
        mesh.glyph.plot_outline()
        temporal.glyph.animate()
        mesh.glyph.to_image(cell_size=0.5)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, [str(w.message) for w in deprecations]
