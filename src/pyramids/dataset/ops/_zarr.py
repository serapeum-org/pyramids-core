"""Core Zarr-store resolution and routing predicates, shared by the eager and lazy paths.

The ``Dataset`` Zarr read/write bodies moved to the ``pyramids-lazy`` compute backend
(``Dataset.to_zarr`` / ``from_zarr`` dispatch through ``require_compute_backend``). What stays here is
small and backend-free:

* :func:`_resolve_store` — the store-target resolver that both the eager STAC loader
  (``stac._loader``) and the lazy backend use; the backend reaches it through the core re-export
  facade rather than importing this private module directly.
* :func:`is_zarr_store` — the cheap suffix check :meth:`Dataset.read_file` uses to decide whether a
  path is a Zarr store at all (no store is opened, so it needs neither ``zarr`` nor the backend).
* :func:`classify_zarr_store` / :func:`is_cube_store` — look *inside* a store to tell a single raster
  from a temporal cube from a foreign store. Opening the store needs the optional ``zarr`` package,
  which ships with the ``pyramids-lazy`` backend, so these delegate to the backend and report
  ``"other"`` whenever no backend is installed, the backend predates the classify capability, or the
  store cannot be confirmed. That makes ``read_file`` fall back to GDAL for a ``.zarr`` on a bare-core
  install, exactly as before this routing existed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyramids.base._compute_backend import require_compute_backend
from pyramids.base._utils import lazy_extra_hint, require_optional

logger = logging.getLogger(__name__)


def _resolve_store(
    store: str | Path | Any,
    storage_options: dict[str, Any] | None,
) -> Any:
    """Return a zarr-v3-compatible store target for ``store``.

    Local paths and cloud URLs are passed through as strings — zarr v3 and dask resolve them directly
    (fsspec-backed for `s3://` / `gs://` / … URLs). When ``storage_options`` is supplied for a URL, a
    v3 :class:`zarr.storage.FsspecStore` is built so the options reach fsspec. Pre-built stores are
    returned unchanged.

    Raises:
        OptionalPackageDoesNotExist: When ``storage_options`` is given but ``zarr`` (the ``[lazy]`` /
            ``pyramids-lazy`` stack) is not installed, so the branded install hint is surfaced rather
            than a bare ``ModuleNotFoundError``.
    """
    if isinstance(store, (str, Path)):
        if storage_options:
            zarr_storage = require_optional(
                "zarr.storage",
                lazy_extra_hint(
                    "reading a cloud Zarr store with `storage_options` requires the "
                    "optional 'zarr' dependency."
                ),
                return_module=True,
            )
            return zarr_storage.FsspecStore.from_url(
                str(store), storage_options=storage_options
            )
        return str(store)
    return store


def is_zarr_store(source: Any) -> bool:
    """Whether ``source`` names a ``.zarr`` store.

    The cheap suffix check :meth:`pyramids.dataset.Dataset.read_file` uses to route a Zarr store to the
    Zarr reader instead of GDAL's generic driver. No store is opened — only the name is inspected — so
    it needs neither the ``zarr`` package nor the compute backend.

    Args:
        source: A local path, cloud URL (``str`` / :class:`~pathlib.Path`), or a pre-built store /
            mapping object.

    Returns:
        ``True`` when ``source`` is a store: a path/URL whose final component ends in ``.zarr``
        (case-insensitive, trailing separators ignored), or any non-``str`` / non-``Path`` object
        (taken to be an already-opened store). ``False`` otherwise.

    Examples:
        A local or remote ``.zarr`` path is a store; the match ignores case and a trailing separator,
        and a GeoTIFF is not a store:

        >>> is_zarr_store("elevation.zarr")
        True
        >>> is_zarr_store("s3://bucket/cube.zarr/")
        True
        >>> is_zarr_store("elevation.tif")
        False
        >>> is_zarr_store("s3://bucket/x.zarr?versionId=9")
        True
    """
    if not isinstance(source, (str, Path)):
        return True
    text = str(source)
    # Strip a URL query string / fragment before the suffix test so a versioned or fragment-tagged
    # cloud store (`s3://bucket/x.zarr?versionId=...`, `...x.zarr#frag`) still matches — but ONLY for
    # URL-like sources (a scheme is present). `#` is a legal filename character on POSIX and Windows
    # and `?` is legal on POSIX, so a local path must be matched verbatim, never truncated.
    if "://" in text:
        text = text.split("?", 1)[0].split("#", 1)[0]
    return text.rstrip("/\\").lower().endswith(".zarr")


def classify_zarr_store(
    store: Any, storage_options: dict[str, Any] | None = None
) -> str:
    """Classify a ``.zarr`` store for :meth:`~pyramids.dataset.Dataset.read_file` routing.

    Returns one of:

    - ``"single"`` — a pyramids-written single-raster GeoZarr store; route it to
      :meth:`~pyramids.dataset.Dataset.from_zarr`.
    - ``"cube"`` — a pyramids-written temporal cube; it is not a single raster.
    - ``"other"`` — anything else: a foreign / xarray-written store, a plain file that merely ends in
      ``.zarr``, a store that cannot be opened here, or an install without a classify-capable backend.
      The caller should let GDAL handle it.

    Opening the store to inspect it needs the ``zarr`` package, which ships with the ``pyramids-lazy``
    compute backend, so the real work is delegated to the backend's own ``classify_zarr_store``. When
    no usable backend is present (absent, or installed but too old / malformed), the backend predates
    the capability, or classification raises for any reason, ``"other"`` is returned so ``read_file``
    falls back to GDAL — a bare-core install reads a ``.zarr`` through GDAL exactly as before this
    routing existed. This routing never raises; an explicit
    :meth:`~pyramids.dataset.Dataset.from_zarr` still surfaces ``require_compute_backend``'s branded
    "install / upgrade pyramids-lazy" error.

    Args:
        store: A ``.zarr`` store — path / fsspec URL / store object.
        storage_options: fsspec options forwarded to the backend for a cloud URL.

    Returns:
        ``"single"``, ``"cube"``, or ``"other"``.

    Examples:
        A pyramids single-raster store is ``"single"`` and a temporal cube is ``"cube"``; a foreign or
        unreadable store -- and a bare-core install with no classify-capable backend -- is ``"other"``
        (each needs a real store / backend, so these are illustrative):

        >>> classify_zarr_store("elevation.zarr")          # doctest: +SKIP
        'single'
        >>> classify_zarr_store("timeseries_cube.zarr")    # doctest: +SKIP
        'cube'
        >>> classify_zarr_store("xarray_export.zarr")      # doctest: +SKIP
        'other'

    See Also:
        is_zarr_store: the cheap suffix check that decides whether to classify at all.
        is_cube_store: the thin cube-only predicate over this classification.
    """
    try:
        backend = require_compute_backend()
    except Exception as exc:  # noqa: BLE001
        # No usable lazy backend — absent, or installed but too old / malformed (a `pyramids_lazy`
        # that fails at import, or one `require_compute_backend` rejects for its api_version /
        # structural contract, both raised as ImportError subclasses; plus any other resolution
        # failure). read_file's routing is deliberately lenient: core cannot open a Zarr store
        # itself, so it falls back to GDAL rather than crash the generic read path over an optional
        # backend. The actionable "install / upgrade pyramids-lazy" error is NOT lost — an explicit
        # `Dataset.from_zarr(...)` calls `require_compute_backend()` unguarded and still surfaces it.
        # Logged at debug so a silently-disabled or broken backend stays discoverable.
        logger.debug(
            "classify_zarr_store: no usable compute backend for %r -> 'other' (%s)",
            store,
            exc,
        )
        return "other"
    classify = getattr(backend, "classify_zarr_store", None)
    if classify is None:
        # A backend that predates the classify capability: fall back to GDAL rather than guess.
        logger.debug(
            "classify_zarr_store: backend %r exposes no classify_zarr_store -> 'other'",
            backend,
        )
        return "other"
    try:
        result = classify(store, storage_options)
    except Exception as exc:  # noqa: BLE001 - any backend failure means "let GDAL try", never raise.
        # A present-but-broken backend (e.g. a drifted classify signature raising TypeError) degrades
        # here; the debug log distinguishes it from the absent-backend case above.
        logger.debug(
            "classify_zarr_store: backend classify failed for %r -> 'other' (%s)",
            store,
            exc,
        )
        return "other"
    return result if result in ("single", "cube") else "other"


def is_cube_store(store: Any, storage_options: dict[str, Any] | None = None) -> bool:
    """Whether a pyramids Zarr ``store`` holds a temporal cube rather than a single raster.

    Thin predicate over :func:`classify_zarr_store`: ``True`` only for a pyramids-written
    ``(time, band, y, x)`` cube (as :meth:`pyramids.dataset.DatasetCollection.to_zarr` writes). A
    single raster, a foreign store, an unreadable path, or a backend-less install is ``False``.

    Args:
        store: A ``.zarr`` store — path / fsspec URL / store object.
        storage_options: fsspec options forwarded to the backend for a cloud URL.

    Returns:
        ``True`` for a pyramids cube store; ``False`` otherwise (including a backend-less install,
        which cannot open the store and so cannot call it a cube).

    Examples:
        A single-raster store is not a cube, while a cube store is (both need a real on-disk store and
        the lazy backend, so these are illustrative):

        >>> is_cube_store("single_raster.zarr")      # doctest: +SKIP
        False
        >>> is_cube_store("time_series_cube.zarr")   # doctest: +SKIP
        True

    See Also:
        classify_zarr_store: the three-way classification this predicate reads.
    """
    return classify_zarr_store(store, storage_options) == "cube"
