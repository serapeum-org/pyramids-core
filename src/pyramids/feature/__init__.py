"""Feature subpackage.

Submodules:

- :mod:`pyramids.feature.collection` — the :class:`FeatureCollection`
  class (a :class:`geopandas.GeoDataFrame` subclass).
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors.
- :mod:`pyramids.feature._ogr` — private OGR bridge (internal only).

CRS / EPSG / reprojection helpers live in :mod:`pyramids.base.crs`.
"""

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import extra_hint, import_dask_geopandas
from pyramids.feature._read import VectorInfo
from pyramids.feature.collection import FeatureCollection
from pyramids.feature.geometry import (
    Coords,
    GeometryCoords,
    create_points,
    create_polygon,
    explode_gdf,
    geometry_collection_coords,
    get_coords,
    get_line_coords,
    get_point_coords,
    get_poly_coords,
    get_xy_coords,
    multi_geom_handler,
    point_collection,
    polygon_wkt,
)

_LAZY_FC_INSTALL_HINT = extra_hint(
    "LazyFeatureCollection requires the optional 'dask-geopandas' dependency.",
    "parquet",
)

# LazyFeatureCollection is only available when the [parquet] extra
# is installed. Two design goals:
#
# 1. Eager import on `from pyramids.feature import LazyFeatureCollection`
# should give the user the real class (when available) or a clear
# ImportError with install instructions — NOT a silent `None` sentinel
# that later breaks `isinstance` with a confusing `TypeError`.
# 2. `import pyramids.feature` on a minimal install must not fail; the
# :class:`FeatureCollection` path has to keep working.
#
# The PEP-562 `__getattr__` hook satisfies both: attribute lookups for
# `LazyFeatureCollection` try the real import, and raise a branded
# :class:`ImportError` when dask-geopandas is absent.
# Note: do NOT probe with `hasattr(module, "LazyFeatureCollection")` — on a
# minimal install `__getattr__` raises ImportError (not AttributeError), which
# `hasattr` does not swallow, so it propagates rather than returning False. Use
# `has_lazy_backend()` (or `is_lazy_fc(obj)` for dispatch) to check availability.
try:
    import_dask_geopandas(_LAZY_FC_INSTALL_HINT)
except OptionalPackageDoesNotExist:  # pragma: no cover - minimal install path
    _HAS_DASK_GEOPANDAS = False
else:
    _HAS_DASK_GEOPANDAS = True


def __getattr__(name: str) -> object:
    """PEP-562 hook — raise a clean ImportError for missing optional classes.

    Only `LazyFeatureCollection` is handled here. Any other unknown
    attribute falls through to the default :class:`AttributeError`.
    """
    if name != "LazyFeatureCollection":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not _HAS_DASK_GEOPANDAS:
        # Brand this like every other optional-dependency path. OptionalPackageDoesNotExist is an
        # ImportError *subclass*, so the plain ImportError raised here before was invisible to
        # every caller matching the branded type. Raised from the flag rather than by re-running
        # import_dask_geopandas, so the module-scope probe stays the single source of truth.
        raise OptionalPackageDoesNotExist(_LAZY_FC_INSTALL_HINT)
    # The lazy vector collection lives in the compute backend; forward the public name to whichever
    # backend is installed, resolved through the seam rather than by importing pyramids_lazy, so a
    # conforming third-party backend is honoured. Raises the branded error on a bare core.
    from pyramids.base._compute_backend import backend_lazy_type

    return backend_lazy_type(
        "lazy_feature_collection_type", "pyramids.feature.LazyFeatureCollection"
    )


def __dir__() -> list[str]:
    """List the module's names including the PEP-562 lazily-forwarded `LazyFeatureCollection`.

    `__getattr__`-provided names are invisible to the default `dir()`, so autocomplete and
    `dir(pyramids.feature)` would otherwise omit `LazyFeatureCollection`.
    """
    return sorted(set(globals()) | {"LazyFeatureCollection"})


def has_lazy_backend() -> bool:
    """Return True when the `[parquet]` extra is installed.

    public feature-detection helper that avoids the
    `try/except ImportError` dance for consumers that only need to
    know "can this session construct a :class:`LazyFeatureCollection`?"
    Cheap — just reads the module-level flag set at import time.

    Returns:
        bool: True iff :mod:`dask_geopandas` imported successfully
        when :mod:`pyramids.feature` was first loaded.

    Examples:
        - Branch on backend availability without importing the class:
            ```python
            >>> from pyramids.feature import has_lazy_backend
            >>> isinstance(has_lazy_backend(), bool)
            True

            ```
    """
    return _HAS_DASK_GEOPANDAS


def is_lazy_fc(obj: object) -> bool:
    """Return True if `obj` is a :class:`LazyFeatureCollection` instance.

    public dispatch helper for library code that consumes either
    eager or lazy FeatureCollections. Safe to call on minimal installs —
    returns False when :mod:`dask_geopandas` is absent (no object could
    be a LazyFC in that case). Equivalent of the eager-vs-lazy
    :func:`pyramids.base.protocols.is_lazy` helper that already exists
    for arrays.

    Args:
        obj: Any object.

    Returns:
        bool: True iff `obj` is a `LazyFeatureCollection`, False
        otherwise (including the "class not available" case).

    Examples:
        - An eager FeatureCollection is NOT a LazyFC:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature import FeatureCollection, is_lazy_fc
            >>> fc = FeatureCollection(gpd.GeoDataFrame(
            ...     {"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
            ... ))
            >>> is_lazy_fc(fc)
            False

            ```
        - Plain objects return False:
            ```python
            >>> from pyramids.feature import is_lazy_fc
            >>> is_lazy_fc("not a frame")
            False
            >>> is_lazy_fc(None)
            False

            ```
    """
    if not _HAS_DASK_GEOPANDAS:
        return False
    # Check against the installed backend's class, resolved through the seam. Importing
    # pyramids_lazy directly would answer False for a conforming third-party backend's own lazy
    # frames -- silently, which is the worst failure mode for a dispatch predicate. On a
    # no-backend install the branded error is the expected outcome and False is the right answer:
    # no object can be a lazy collection there.
    from pyramids.base._compute_backend import backend_lazy_type

    try:
        lazy_cls = backend_lazy_type(
            "lazy_feature_collection_type", "pyramids.feature.LazyFeatureCollection"
        )
    except ImportError:
        # No backend, or a backend whose [parquet] dependency is absent: the lazy class cannot be
        # resolved, so no object can be an instance -> False. ImportError is the one swallowed
        # family -- it covers both the branded OptionalPackageDoesNotExist (an ImportError subclass)
        # and the backend's own ModuleNotFoundError subclass. A malformed backend raising anything
        # else propagates rather than being hidden behind a dispatch predicate. A user who wants the
        # actionable "install/upgrade" error gets it from
        # accessing `pyramids.feature.LazyFeatureCollection` directly, which does raise.
        return False
    return isinstance(obj, lazy_cls)


__all__ = [
    "Coords",
    "FeatureCollection",
    "GeometryCoords",
    "VectorInfo",
    "create_points",
    "create_polygon",
    "explode_gdf",
    "geometry_collection_coords",
    "get_coords",
    "get_line_coords",
    "get_point_coords",
    "get_poly_coords",
    "get_xy_coords",
    "has_lazy_backend",
    "is_lazy_fc",
    "multi_geom_handler",
    "point_collection",
    "polygon_wkt",
]
