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

**Status:** the Phase 2 cutover has landed. Core's cube (``NetCDF``), vector
(``FeatureCollection``) and Zarr/kerchunk lazy bodies now dispatch through
:func:`require_compute_backend`, and ``pyramids-lazy`` ships the backend that serves them. On a
bare install those operations raise the branded error.

One lazy surface has **not** moved: ``Dataset.read_array(chunks=)`` and ``DatasetCollection.data``
still build their dask arrays inside core (``dataset/engines/io.py``, ``dataset/collection.py``), so
a bare core with dask installed keeps a working lazy raster path that never reaches this seam. See
:class:`ComputeBackend` for the exact split.
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
    "backend_lazy_type",
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
#:
#: v2 added the required ``lazy_netcdf_type`` / ``lazy_feature_collection_type`` accessors (the
#: public-class forwarding seam). A v1 backend lacks them, so this bump turns that mismatch into the
#: up-front "upgrade" message instead of a later structural rejection naming individual members.
COMPUTE_BACKEND_API_VERSION = 2

#: The import name of the first-party lazy backend, probed as the optional-import fallback (the
#: ``digitalearth`` analogue of the plot-backend seam) when no entry point is registered — e.g. an
#: editable ``pyramids-lazy`` checkout on ``PYTHONPATH`` with no installed metadata.
_FALLBACK_DISTRIBUTION = "pyramids_lazy"


@runtime_checkable
class ComputeBackend(Protocol):
    """The lazy/dask operations core delegates to once the lazy layer is extracted.

    A backend is a single object (not a class discovered per-source like a driver) exposing one
    method per lazy operation plus an :attr:`api_version`. The method set is exactly the forwarders
    Phase 2 moves out of core; signatures mirror today's in-core implementations so the cutover is a
    straight delegation. ``@runtime_checkable`` so :func:`require_compute_backend` can verify the
    object structurally.

    Scope: the v1 contract covers the raster-cube (``NetCDF``) and vector (``FeatureCollection``)
    lazy ops, plus the Zarr/kerchunk writers and readers.

    What is **not** dispatched yet, despite being lazy: ``Dataset.read_array(chunks=)`` and
    ``DatasetCollection.data`` still construct dask arrays inside core
    (``dataset/engines/io.py``, ``dataset/collection.py``), guarded by a local ``import
    dask.array`` and a plain ``ImportError`` rather than this seam's branded error. So a bare core
    with dask installed still has a working lazy raster path, and :meth:`read_array_lazy` is
    reached only from the ``NetCDF`` read and the xarray/UGRID interop readers. Completing that
    move is follow-up work; this docstring previously claimed the raster surface was covered, which
    it is not.

    ``MeshDataset``'s own lazy surface (``open_mfdataset`` / ``to_dask_dataframe`` / ``to_zarr`` /
    ``from_zarr``) is **intentionally out of v1** — mesh lazy parity is deferred to FR-B (issue
    #1458); its methods are added (additively, no api_version bump) when the mesh lazy bodies move
    to the backend.

    Attributes:
        api_version: The contract version this backend implements; must be ``>=``
            :data:`COMPUTE_BACKEND_API_VERSION` for core to use it.
    """

    api_version: int

    def lazy_netcdf_type(self) -> type:
        """Return the class ``build_lazy_cube`` produces (backs ``pyramids.netcdf.LazyNetCDF``)."""
        ...  # pragma: no cover - protocol

    def lazy_feature_collection_type(self) -> type:
        """Return the class the lazy vector reads produce.

        Backs ``pyramids.feature.LazyFeatureCollection`` and ``is_lazy_fc``. A **method**, not a
        property or a plain attribute, and that is load-bearing: ``runtime_checkable`` isinstance
        evaluates data members with ``getattr``, so a property here would be executed by the mere
        act of type-checking the backend. Resolving this class needs the backend's vector
        dependency, so that turned a type check into an import error in every environment without
        it -- and core, which treats any backend failure as "skip this backend", then silently fell
        back to eager execution. A method is only checked for presence, so the import happens when
        core actually asks for the class.
        """
        ...  # pragma: no cover - protocol

    def read_array_lazy(
        self, dataset: Any, *, chunks: Any, lock: Any = None, **kwargs: Any
    ) -> ArrayLike:
        """Lazy dask-backed read of a cube variable (``NetCDF.read_array(chunks=)``).

        Reached from the ``NetCDF`` read path and the xarray/UGRID interop readers.
        ``Dataset.read_array(chunks=)`` does **not** route here — it still builds its dask array in
        core (see the class docstring).
        """
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
        entry_point: The entry point to resolve. Its value may be an already-constructed backend
            object, the backend *class*, or a zero-argument factory *function* returning it (a class
            or a function/routine is called; an already-constructed instance is used as-is).

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
            missing = _missing_protocol_members(candidate)
            detail = f" (missing: {', '.join(missing)})" if missing else ""
            warnings.warn(
                f"Compute backend {entry_point.name!r} ({entry_point.value}) does not implement the "
                f"ComputeBackend protocol{detail}; it was skipped. If it targets an older contract, "
                f"upgrade it to one declaring api_version >= {COMPUTE_BACKEND_API_VERSION}.",
                RuntimeWarning,
                stacklevel=2,
            )
    return backend


def _missing_protocol_members(candidate: object) -> list[str]:
    """Return the :class:`ComputeBackend` members ``candidate`` does not expose, for diagnostics.

    ``runtime_checkable`` isinstance only answers yes/no, so this re-derives which names were absent
    to make the rejection actionable instead of "it was skipped".

    Args:
        candidate: The object that failed the structural check.

    Returns:
        The Protocol member names missing from ``candidate``, in declaration order.
    """
    # Derive the Protocol's member names without a version-specific typing internal
    # (``get_protocol_members`` is 3.13+, ``__protocol_attrs__`` is unpopulated on 3.11): the
    # members are exactly this class's own annotations plus the methods it declares. Guarded so a
    # diagnostic can never raise over the real rejection.
    members = set(getattr(ComputeBackend, "__annotations__", {}))
    for name, value in vars(ComputeBackend).items():
        if callable(value) and not name.startswith("__"):
            members.add(name)
    return sorted(name for name in members if not hasattr(candidate, name))


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
        - The registry maps each installed backend's entry-point name to its object. What it
          holds depends on what is installed — empty on a bare install, `{'lazy': ...}` once
          `pyramids-lazy` is present — so the example pins the shape rather than the contents.
          An example asserting `{}` passes only where no backend happens to be installed, and
          fails the moment one is (as it did once core's own test environments gained the
          plugin):
            ```python
            >>> from pyramids.base import _compute_backend
            >>> _compute_backend.clear_compute_backend_cache()
            >>> backends = _compute_backend.compute_backends()
            >>> isinstance(backends, dict)
            True
            >>> all(isinstance(name, str) for name in backends)
            True

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
            ``get_compute_backend``, or its backend does not implement :class:`ComputeBackend`.
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
        # Enforce the same structural check the entry-point path applies in `_load`, so a malformed
        # first-party backend fails here with a clear message rather than AttributeError-ing deep in
        # a later lazy call.
        if not isinstance(backend, ComputeBackend):
            raise OptionalPackageDoesNotExist(
                "The installed pyramids-lazy returned a backend that does not implement the "
                "ComputeBackend protocol (missing required members). Upgrade with "
                "`pip install -U pyramids-lazy`."
            )
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
            >>> got = _compute_backend.require_compute_backend()
            >>> got.api_version == _compute_backend.COMPUTE_BACKEND_API_VERSION
            True
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
            "pyramids' lazy/dask operations require the pyramids-lazy backend, which is not "
            "installed. Install it with `pip install pyramids-lazy`. The `pyramids-core[lazy]` / "
            "`[parquet]` extras install the dask/zarr stack the backend builds on, but not the "
            "backend itself, so they alone do not resolve this error."
        )
    declared = getattr(backend, "api_version", None)
    # `bool` subclasses `int`, so exclude it explicitly: `api_version = True` must not pass as 1.
    if (
        not isinstance(declared, int)
        or isinstance(declared, bool)
        or declared < COMPUTE_BACKEND_API_VERSION
    ):
        raise OptionalPackageDoesNotExist(
            f"The installed lazy backend declares api_version {declared!r}, but this pyramids needs "
            f">= {COMPUTE_BACKEND_API_VERSION}. Upgrade with `pip install -U pyramids-lazy`."
        )
    return backend


def backend_lazy_type(attribute: str, public_name: str) -> type:
    """Resolve a public lazy class from the *installed* backend, whichever one that is.

    Core's public lazy names (``pyramids.netcdf.LazyNetCDF``,
    ``pyramids.feature.LazyFeatureCollection``) are provided by the backend, not by core. Resolving
    them by importing ``pyramids_lazy`` directly would defeat the entry-point seam: a conforming
    third-party backend would be discovered by :func:`require_compute_backend` and then ignored,
    failing with a raw ``ModuleNotFoundError`` naming a distribution the user never installed.

    Args:
        attribute: The backend accessor returning the class (e.g. ``"lazy_netcdf_type"``). Called
            here rather than read, because these are Protocol *methods* -- see
            :meth:`ComputeBackend.lazy_feature_collection_type` for why they must not be
            attributes.
        public_name: The core-public name being resolved, for the error message.

    Returns:
        The class the installed backend publishes under `attribute`.

    Raises:
        OptionalPackageDoesNotExist: No backend is installed (from
            :func:`require_compute_backend`), or the installed backend does not publish
            `attribute`. The second case is normally caught earlier — the attribute is part of the
            Protocol, and ``runtime_checkable`` isinstance does check data members — so this guard
            is the backstop for a backend that bypassed discovery's structural check.
    """
    backend = require_compute_backend()
    accessor = getattr(backend, attribute, None)
    resolved: object
    if isinstance(accessor, type):
        # A backend that publishes the class directly rather than via the accessor. Taken as-is:
        # a class is callable, so calling it would hand back an *instance* and the mistake would
        # be silent.
        resolved = accessor
    elif callable(accessor):
        resolved = accessor()
    else:
        resolved = accessor
    if not isinstance(resolved, type):
        raise OptionalPackageDoesNotExist(
            f"The installed lazy backend ({type(backend).__module__}) does not publish a usable "
            f"`{attribute}` (got {resolved!r}), so `{public_name}` cannot be resolved. Upgrade the "
            f"backend (for the first-party one: `pip install -U pyramids-lazy`)."
        )
    return resolved
