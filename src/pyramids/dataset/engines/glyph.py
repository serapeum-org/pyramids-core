"""Glyph engine.

Owns the plotting family on a Dataset, accessed as ``ds.glyph``. Named after
cleopatra's ``ArrayGlyph`` / ``Scatter`` objects, which every plot method builds -- all
drawing is delegated to cleopatra, never done in pyramids (see the repo CLAUDE.md).

``histogram``, ``vector_field`` and ``set_color_ramp`` are owned here: they forward to the
engine that holds the real implementation (``analysis`` for the renderers, ``bands`` for the
palette write), and the flat ``Dataset`` methods delegate to this accessor.

``plot`` and ``animate`` are exposed here too, but for now forward to the flat
``Dataset.plot`` / ``Dataset.animate`` implementations: those carry a large, rich signature
plus backend-routing and band-resolution logic that is relocated into this engine as part of
the later first-party migration (the step that also deprecates the flat facades). Until then
``ds.glyph.plot`` and ``ds.plot`` are equivalent entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyramids.dataset.engines._base import _Engine

if TYPE_CHECKING:
    from pyramids.dataset.dataset import (  # noqa: F401  (forward ref in _Engine["Dataset"])
        Dataset,
    )


class _Glyph(_Engine["Dataset"]):
    """Plotting operations on a Dataset (delegated to cleopatra)."""

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Render one band statically. See :meth:`Dataset.plot` for the full signature."""
        return self._ds.plot(*args, **kwargs)

    def animate(self, *args: Any, **kwargs: Any) -> Any:
        """Animate the bands as successive frames. See :meth:`Dataset.animate`."""
        return self._ds.animate(*args, **kwargs)

    def histogram(self, *args: Any, **kwargs: Any) -> Any:
        """Plot the band value histogram. See :meth:`Dataset.plot_histogram`."""
        return self._ds.analysis.plot_histogram(*args, **kwargs)

    def vector_field(self, *args: Any, **kwargs: Any) -> Any:
        """Plot a two-band raster as a vector field. See :meth:`Dataset.plot_vector_field`."""
        return self._ds.analysis.plot_vector_field(*args, **kwargs)

    def set_color_ramp(self, *args: Any, **kwargs: Any) -> None:
        """Write a colour ramp onto a band's palette. See :meth:`Dataset.set_color_ramp`.

        Raises:
            ReadOnlyError: The dataset is opened read-only on-disk.
        """
        self._ds._require_writable("set a color ramp")
        return self._ds.bands.set_color_ramp(*args, **kwargs)
