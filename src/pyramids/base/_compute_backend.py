"""Runtime plugin seam for the lazy/dask compute backend (plan Phase 1, tracker #4).

pyramids' lazy/dask operations — chunked reads, Zarr/kerchunk I/O, `open_mfdataset`, the
dask-backed `FeatureCollection` — are being prepared for extraction into a separate
``pyramids-lazy`` distribution (see issue #1455). This module is the seam core will dispatch
through once that happens: a single :class:`ComputeBackend` is resolved by, in order,

1. an :mod:`importlib.metadata` entry point in the ``pyramids.compute_backend`` group;
2. an optional import of ``pyramids_lazy`` (whose ``get_compute_backend()`` returns the backend),
   the way :mod:`pyramids.base._plot_backends` falls back to ``digitalearth``;
3. otherwise a branded :class:`~pyramids.base._errors.OptionalPackageDoesNotExist` telling the user
   to install the lazy stack.

It mirrors the raster/vector driver seam (:mod:`pyramids.dataset._drivers`) and the plot-backend
seam — same caching, publish lock, re-entrancy guard and first-wins collision rule — so there is
one provider in practice (``pyramids-lazy``) but a third party *could* register its own.

**Status:** the resolver and contract are defined here in Phase 1; the core method bodies are wired
to :func:`require_compute_backend` in the Phase 2 cutover, when ``pyramids-lazy`` ships the backend.
Until then nothing in core calls :func:`require_compute_backend`, and a bare install resolves no
backend (its lazy features keep working through the in-core implementations that still live here).
"""

from __future__ import annotations

import inspect
import threading
import warnings
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pyramids.base._driver_discovery import extra_missing
from pyramids.base._errors import OptionalPackageDoesNotExist

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing the lazy layer
    from pyramids.base.protocols import ArrayLike

__all__ = [
    "COMPUTE_BACKEND_API_VERSION",
    "ENTRY_POINT_GROUP",
    "ComputeBackend",
    "clear_compute_backend_cache",
    "compute_backends",
    "require_compute_backend",
]

#: The entry-point group a lazy backend registers under. Part of the published contract; mirrors the
#: ``pyramids.raster_drivers`` / ``pyramids.plot_backends`` groups.
ENTRY_POINT_GROUP = "pyramids.compute_backend"

#: The contract version core speaks. A backend must declare an ``api_version`` core still supports
#: (``>=`` this value). Bumped only on a breaking change to :class:`ComputeBackend`; a backend
#: declaring an older version is rejected with a clear "upgrade" message rather than an
#: :class:`AttributeError` deep in a lazy call.
COMPUTE_BACKEND_API_VERSION = 1

#: The import name of the first-party lazy backend, probed as the optional-import fallback (the
#: ``digitalearth`` analogue of the plot-backend seam). It does not exist yet; the fallback is inert
#: until ``pyramids-lazy`` ships.
_FALLBACK_DISTRIBUTION = "pyramids_lazy"


@runtime_checkable
class ComputeBackend(Protocol):
    """The lazy/dask operations core delegates to once the lazy layer is extracted.

    A backend is a single object (not a class discovered per-source like a driver) exposing one
    method per lazy operation plus an :attr:`api_version`. The method set is exactly the forwarders
    Phase 2 moves out of core; signatures mirror today's in-core implementations so the cutover is a
    straight delegation. ``@runtime_checkable`` so :func:`require_compute_backend` can verify the
    object structurally.

    Attributes:
        api_version: The contract version this backend implements; must be ``>=``
            :data:`COMPUTE_BACKEND_API_VERSION` for core to use it.
    """

    api_version: int

    def read_array_lazy(
        self, dataset: Any, *, chunks: Any, lock: Any = None, **kwargs: Any
    ) -> ArrayLike:
        """Lazy dask-backed read of a raster/variable (``Dataset``/``NetCDF`` ``read_array(chunks=)``)."""
        ...  # pragma: no cover - protocol

    def build_lazy_cube(self, netcdf: Any, chunks: Any) -> Any:
        """Build the lazy cube view for ``NetCDF.chunk`` (returns a ``LazyNetCDF``)."""
        ...  # pragma: no cover - protocol

    def to_dask_dataframe(
        self, netcdf: Any, *, variables: Any, dropna: bool, chunks: Any
    ) -> Any:
        """Back ``NetCDF.to_dask_dataframe``."""
        ...  # pragma: no cover - protocol

    def to_kerchunk(self, source: Any, output_path: Any, **kwargs: Any) -> dict:
        """Back ``NetCDF.to_kerchunk`` (single-file kerchunk manifest)."""
        ...  # pragma: no cover - protocol

    def combine_kerchunk(self, paths: Any, output_path: Any, **kwargs: Any) -> dict:
        """Back ``NetCDF.combine_kerchunk`` / ``DatasetCollection.to_kerchunk``."""
        ...  # pragma: no cover - protocol

    def open_mfdataset(self, paths: Any, variable: Any, **kwargs: Any) -> Any:
        """Back ``NetCDF.open_mfdataset`` (dask-stacked multi-file read)."""
        ...  # pragma: no cover - protocol

    def write_dataset_to_zarr(self, dataset: Any, store: Any, **kwargs: Any) -> Any:
        """Back ``Dataset.to_zarr``."""
        ...  # pragma: no cover - protocol

    def read_dataset_from_zarr(self, store: Any, **kwargs: Any) -> Any:
        """Back ``Dataset.from_zarr`` (returns a ``Dataset``)."""
        ...  # pragma: no cover - protocol

    def write_collection_to_zarr(
        self, collection: Any, store: Any, **kwargs: Any
    ) -> Any:
        """Back ``DatasetCollection.to_zarr``."""
        ...  # pragma: no cover - protocol

    def read_collection_from_zarr(
        self, collection_cls: Any, store: Any, **kwargs: Any
    ) -> Any:
        """Back ``DatasetCollection.from_zarr`` (returns a ``DatasetCollection``)."""
        ...  # pragma: no cover - protocol

    def read_vector_lazy(self, source: Any, **kwargs: Any) -> Any:
        """Back ``FeatureCollection.read_file(backend="dask")`` (returns a ``LazyFeatureCollection``)."""
        ...  # pragma: no cover - protocol

    def read_vector_parquet_lazy(self, source: Any, **kwargs: Any) -> Any:
        """Back ``FeatureCollection.read_parquet(backend="dask")``."""
        ...  # pragma: no cover - protocol


#: Resolved backend registry, filled lazily by :func:`compute_backends`; reset by
#: :func:`clear_compute_backend_cache` for tests.
_CACHE: dict[str, ComputeBackend] | None = None

#: Guards only the publish of the resolved mapping, never discovery (which imports plugin code).
_PUBLISH_LOCK = threading.Lock()

#: Per-thread marker that this thread is inside discovery, holding the backends resolved so far, so a
#: plugin that re-enters :func:`compute_backends` from its own import sees the partial mapping rather
#: than recursing into a second discovery.
_IN_PROGRESS = threading.local()


def clear_compute_backend_cache() -> None:
    """Reset the discovered-backend cache (for tests that publish in-memory entry points)."""
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    with _PUBLISH_LOCK:
        _CACHE = None


def _load(entry_point: EntryPoint) -> ComputeBackend | None:
    """Load one backend object, or warn and skip.

    Args:
        entry_point: The entry point to resolve. Its value may be the backend object itself or a
            zero-argument factory returning it.

    Returns:
        The loaded backend, or ``None`` when it could not be loaded or does not conform.
    """
    backend: ComputeBackend | None = None
    try:
        candidate = entry_point.load()
        # A zero-arg factory (the backend *class*, or a function returning the backend) is called to
        # construct it. Gate on "is it a constructor/factory" -- NOT on the structural
        # `isinstance(candidate, ComputeBackend)` check, because a class with class-level `api_version`
        # + methods structurally *is* a ComputeBackend, so that guard would store the class
        # uninstantiated and later bind the first call arg to `self`.
        if isinstance(candidate, type) or inspect.isroutine(candidate):
            candidate = candidate()
    except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
        warnings.warn(
            f"Compute backend {entry_point.name!r} ({entry_point.value}) could not be loaded and "
            f"was skipped: {error!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        if isinstance(candidate, ComputeBackend):
            backend = candidate
        else:
            warnings.warn(
                f"Compute backend {entry_point.name!r} ({entry_point.value}) does not implement the "
                f"ComputeBackend protocol; it was skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
    return backend


def _fill(resolved: dict[str, ComputeBackend]) -> None:
    """Load each entry point in the group into ``resolved`` (first-wins on a name collision)."""
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        backend = _load(entry_point)
        if backend is None:
            continue
        if entry_point.name in resolved:
            warnings.warn(
                f"Compute backend {entry_point.name!r} is registered twice; keeping the first and "
                f"skipping {entry_point.value}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        resolved[entry_point.name] = backend


def _discover() -> dict[str, ComputeBackend]:
    """Resolve the entry-point group into backends, keyed by name, in discovery order."""
    resolved: dict[str, ComputeBackend] = {}
    _IN_PROGRESS.resolved = resolved
    try:
        _fill(resolved)
    finally:
        _IN_PROGRESS.resolved = None
    return resolved


def compute_backends() -> dict[str, ComputeBackend]:
    """Return the installed compute backends, discovered once and cached.

    Returns:
        A copy of the registry: backends keyed by entry-point name, in discovery order. A bare
        pyramids install registers none, so this is ``{}`` and :func:`require_compute_backend` falls
        back to importing ``pyramids_lazy``.

    Examples:
        - A bare install registers no compute backend:
            ```python
            >>> from pyramids.base import _compute_backend
            >>> _compute_backend.clear_compute_backend_cache()
            >>> _compute_backend.compute_backends()
            {}

            ```
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    partial_map = getattr(_IN_PROGRESS, "resolved", None)
    if partial_map is not None:
        return dict(partial_map)
    cached = _CACHE
    if cached is None:
        resolved = _discover()
        with _PUBLISH_LOCK:
            if _CACHE is None:
                _CACHE = resolved
            cached = _CACHE
    return dict(cached)


def _fallback_backend() -> ComputeBackend | None:
    """Return the first-party ``pyramids_lazy`` backend, or ``None`` when it is not installed.

    Presence is probed with :func:`extra_missing` (``find_spec``, no import), so a bare pyramids
    install never imports ``pyramids_lazy`` — pyramids declares no dependency on it.

    Returns:
        The backend from ``pyramids_lazy.get_compute_backend()``, or ``None`` when absent.

    Raises:
        OptionalPackageDoesNotExist: ``pyramids_lazy`` is installed but too old to expose
            ``get_compute_backend``.
    """
    backend: ComputeBackend | None = None
    if not extra_missing(_FALLBACK_DISTRIBUTION):
        module = import_module(_FALLBACK_DISTRIBUTION)
        factory = getattr(module, "get_compute_backend", None)
        if factory is None:
            raise OptionalPackageDoesNotExist(
                "The installed pyramids-lazy is too old to serve as a compute backend "
                "(no `get_compute_backend`). Upgrade with `pip install -U pyramids-lazy`."
            )
        backend = factory()
    return backend


def require_compute_backend() -> ComputeBackend:
    """Resolve the one compute backend core should delegate lazy operations to.

    Precedence: the first backend registered in the ``pyramids.compute_backend`` entry-point group;
    else the first-party ``pyramids_lazy`` (optional import — no declared dependency); else a
    branded error. A resolved backend whose ``api_version`` is older than
    :data:`COMPUTE_BACKEND_API_VERSION` is rejected with an "upgrade" message.

    Returns:
        A :class:`ComputeBackend`.

    Raises:
        OptionalPackageDoesNotExist: No backend is registered and ``pyramids_lazy`` is not installed,
            or the resolved backend is too old for this core.

    Examples:
        - A registered backend in the cache is returned:
            ```python
            >>> from pyramids.base import _compute_backend
            >>> class _Fake:
            ...     api_version = _compute_backend.COMPUTE_BACKEND_API_VERSION
            ...     def __getattr__(self, name): return lambda *a, **k: None
            >>> _compute_backend._CACHE = {"pyramids-lazy": _Fake()}
            >>> _compute_backend.require_compute_backend().api_version
            1
            >>> _compute_backend.clear_compute_backend_cache()

            ```
        - With nothing registered and no pyramids-lazy installed, a branded error names the install:
            ```python
            >>> from pyramids.base import _compute_backend
            >>> _compute_backend._CACHE = {}
            >>> _restore = _compute_backend.extra_missing
            >>> _compute_backend.extra_missing = lambda name: True
            >>> try:
            ...     _compute_backend.require_compute_backend()
            ... except Exception as exc:
            ...     print(type(exc).__name__, "pyramids-lazy" in str(exc))
            OptionalPackageDoesNotExist True
            >>> _compute_backend.extra_missing = _restore
            >>> _compute_backend.clear_compute_backend_cache()

            ```
    """
    registered = compute_backends()
    backend = next(iter(registered.values())) if registered else _fallback_backend()
    if backend is None:
        raise OptionalPackageDoesNotExist(
            "pyramids' lazy/dask operations require the lazy backend, which is not installed. "
            "Install it with `pip install pyramids-lazy` (or the `pyramids-core[lazy]` / "
            "`[parquet]` extras)."
        )
    declared = getattr(backend, "api_version", None)
    if not isinstance(declared, int) or declared < COMPUTE_BACKEND_API_VERSION:
        raise OptionalPackageDoesNotExist(
            f"The installed lazy backend declares api_version {declared!r}, but this pyramids needs "
            f">= {COMPUTE_BACKEND_API_VERSION}. Upgrade with `pip install -U pyramids-lazy`."
        )
    return backend
