"""Basemap support for pyramids plots.

Adds web-tile basemaps (OpenStreetMap, CartoDB, Esri, etc.) underneath
data plotted on matplotlib axes. :func:`~pyramids.basemap.add_basemap`
wraps the cleopatra glyph ``add_tiles`` method
(:meth:`cleopatra.basemap.geo.GeoMixin.add_tiles`) and
:func:`~pyramids.basemap.get_provider` wraps
:func:`cleopatra.basemap.tiles.get_provider` — cleopatra does the tile
fetching, stitching, GDAL CRS warping, and rendering. (As of
``cleopatra >= 0.46`` web tiles are added through the glyph ``add_tiles``
method, not the now-private ``cleopatra.basemap.tiles`` module function.)

The tile functions require the cleopatra ``[tiles]`` extra (which pins
``cleopatra[tiles]``). Install with one of:

- PyPI: ``pip install 'pyramids-core[viz]'``
- conda-forge: ``conda install -c conda-forge pyramids-viz``

Natural Earth vector layers and the global relief raster have moved to
:mod:`cleopatra.basemap.reference` (the viz layer), where they belong with the rest of
the map-decoration helpers — use ``cleopatra.basemap.reference`` for those backdrops.
"""

from pyramids.basemap.basemap import add_basemap, get_provider

__all__ = [
    "add_basemap",
    "get_provider",
]
