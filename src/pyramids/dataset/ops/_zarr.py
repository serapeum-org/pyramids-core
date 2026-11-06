"""Core Zarr-store resolution, shared by the eager and lazy paths.

The ``Dataset`` Zarr read/write bodies moved to the ``pyramids-lazy`` compute backend
(``Dataset.to_zarr`` / ``from_zarr`` dispatch through ``require_compute_backend``). What stays here is
:func:`_resolve_store`, the small store-target resolver that both the eager STAC loader
(``stac._loader``) and the lazy backend use; the backend reaches it through the core re-export facade
rather than importing this private module directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_store(
    store: str | Path | Any,
    storage_options: dict[str, Any] | None,
) -> Any:
    """Return a zarr-v3-compatible store target for ``store``.

    Local paths and cloud URLs are passed through as strings — zarr v3 and dask resolve them directly
    (fsspec-backed for `s3://` / `gs://` / … URLs). When ``storage_options`` is supplied for a URL, a
    v3 :class:`zarr.storage.FsspecStore` is built so the options reach fsspec. Pre-built stores are
    returned unchanged.
    """
    if isinstance(store, (str, Path)):
        if storage_options:
            from zarr.storage import FsspecStore

            return FsspecStore.from_url(str(store), storage_options=storage_options)
        return str(store)
    return store
