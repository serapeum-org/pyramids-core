"""Engine classes for ``Dataset`` operations.

After the L-2 composition refactor, the seven public-API families on
``pyramids.dataset.Dataset`` live as engine instances accessed via
attribute (``ds.io``, ``ds.spatial``, ``ds.bands``, ``ds.analysis``,
``ds.cell``, ``ds.vectorize``, ``ds.cog``). Each engine owns the
methods of one family; ``Dataset`` exposes same-named facade methods
so ``ds.crop(mask)`` and ``ds.spatial.crop(mask)`` are equivalent.

Engines hold a weakref proxy back to the parent Dataset; the proxy
keeps GDAL handle release deterministic on Windows. See
:class:`pyramids.dataset.engines._base._Engine` for the contract.
"""

from pyramids.dataset.engines.analysis import _Analysis as _Analysis
from pyramids.dataset.engines.bands import _Bands as _Bands
from pyramids.dataset.engines.cell import _Cell as _Cell
from pyramids.dataset.engines.cog import _COG as _COG
from pyramids.dataset.engines.georef import _Georef as _Georef
from pyramids.dataset.engines.io import _IO as _IO
from pyramids.dataset.engines.spatial import _Spatial as _Spatial
from pyramids.dataset.engines.vectorize import _Vectorize as _Vectorize

# The engine classes are internal implementation detail. User code reaches their
# functionality through the main classes -- the facade methods on ``Dataset``
# (``ds.crop(...)``, ``ds.to_cog(...)``) or the per-family accessors it exposes
# (``ds.spatial``, ``ds.cog``, ``ds.bands``, …) -- never by importing the engine
# classes directly. They are therefore not part of the public API: ``__all__`` is
# empty, so ``from pyramids.dataset.engines import *`` exports nothing, and the
# classes carry no public-API guarantees even though the modules still define them
# for the accessors to instantiate.
__all__: list[str] = []
