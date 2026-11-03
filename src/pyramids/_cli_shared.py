"""Shared helpers for the ``pyramids`` CLI command groups.

The top-level CLI (:mod:`pyramids.cli`) and the per-domain command groups
(:mod:`pyramids.cli_vector`, :mod:`pyramids.cli_netcdf`, …) all need the same
few primitives: the overwrite guard, the JSON non-finite scrubber, and the
quiet-aware informational printer. They live here — in a module that imports
nothing from the CLI package — so the groups can share them without creating an
import cycle with :mod:`pyramids.cli`.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
from collections.abc import Iterator


@contextlib.contextmanager
def map_lookup_errors() -> Iterator[None]:
    """Remap a ``KeyError``/``IndexError`` raised in the block to ``ValueError``.

    :func:`pyramids.cli.main` maps only ``ValueError``/``TypeError``/``OSError``/
    ``RuntimeError``/``_PyramidsError`` to the clean one-line ``error:`` exit; a
    ``KeyError`` or ``IndexError`` from a typo'd variable, dimension, label, or
    index would otherwise fall through to the ``unexpected failure`` branch that is
    reserved for genuine internal bugs. Wrap a ``NetCDF`` lookup that forwards a
    user token in this manager so such a typo is reported cleanly instead.
    """
    try:
        yield
    except (KeyError, IndexError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise ValueError(message) from exc


def json_safe(value: float | None) -> float | None:
    """Map non-finite floats to ``None`` so JSON output stays parseable.

    ``json.dumps`` serialises ``NaN``/``Infinity`` as bare ``NaN`` / ``Infinity``,
    which is not valid JSON and breaks strict consumers (e.g. ``jq``).

    Args:
        value: A float (possibly ``NaN``/infinite) or ``None``.

    Returns:
        float | None: ``value`` unchanged, or ``None`` when it is not finite.

    Examples:
        - A finite float passes through unchanged:
            ```python
            >>> from pyramids._cli_shared import json_safe
            >>> json_safe(3.5)
            3.5

            ```
        - Non-finite floats become ``None`` so ``json.dumps`` stays valid:
            ```python
            >>> json_safe(float("inf")) is None
            True
            >>> json_safe(float("nan")) is None
            True

            ```
        - ``None`` passes straight through:
            ```python
            >>> json_safe(None) is None
            True

            ```
    """
    result: float | None = value
    if isinstance(value, float) and not math.isfinite(value):
        result = None
    return result


def refuse_existing(path: str, overwrite: bool) -> None:
    """Raise if ``path`` exists and ``--overwrite`` was not passed.

    Args:
        path: Destination path the command is about to write.
        overwrite: Whether the user passed ``--overwrite``.

    Raises:
        ValueError: ``path`` already exists and ``overwrite`` is ``False``.

    Examples:
        - A path that does not exist is allowed (returns ``None``):
            ```python
            >>> from pyramids._cli_shared import refuse_existing
            >>> refuse_existing("no_such_file_8f3a1c.tif", overwrite=False) is None
            True

            ```
        - ``overwrite=True`` never refuses, even for a path that exists:
            ```python
            >>> import os, tempfile
            >>> handle = tempfile.NamedTemporaryFile(delete=False)
            >>> _ = handle.close()
            >>> refuse_existing(handle.name, overwrite=True) is None
            True
            >>> os.unlink(handle.name)

            ```
    """
    if not overwrite and os.path.exists(path):
        raise ValueError(
            f"output {path!r} already exists; pass --overwrite to replace it."
        )


def info_print(args: argparse.Namespace, message: str) -> None:
    """Print an informational line unless the global ``--quiet`` flag was passed.

    Errors always print (handlers raise, and :func:`pyramids.cli.main` reports
    them on stderr); this only gates the success chatter (``wrote …`` etc.).

    Args:
        args: The parsed arguments (carries the top-level ``quiet`` flag).
        message: The line to print to stdout.
    """
    if not getattr(args, "quiet", False):
        print(message)


# Output-extension -> OGR driver (alias or literal GDAL name, both accepted by
# FeatureCollection.to_file via its Catalog). FeatureCollection.to_file defaults
# to GeoJSON regardless of extension, so the CLI resolves the driver itself.
# Shared by the vector, fetch (OGC), and mesh groups.
VECTOR_DRIVER_BY_EXT = {
    ".geojson": "geojson",
    ".json": "geojson",
    ".shp": "ESRI Shapefile",
    ".gpkg": "gpkg",
    ".fgb": "FlatGeobuf",
    ".gml": "GML",
    ".kml": "KML",
}


def resolve_vector_driver(path: str, explicit: str | None) -> str:
    """Return the OGR driver for ``path``: ``explicit`` if given, else by extension.

    Args:
        path: The destination vector path.
        explicit: The value of ``--driver`` (``None`` if not passed).

    Returns:
        str: A driver alias or literal GDAL name for ``FeatureCollection.to_file``.

    Raises:
        ValueError: No ``--driver`` and the extension is not a known vector one.

    Examples:
        - An explicit driver wins and is returned unchanged:
            ```python
            >>> from pyramids._cli_shared import resolve_vector_driver
            >>> resolve_vector_driver("out.weird", "GPKG")
            'GPKG'

            ```
        - A known extension maps to its OGR driver:
            ```python
            >>> resolve_vector_driver("roads.gpkg", None)
            'gpkg'
            >>> resolve_vector_driver("pts.shp", None)
            'ESRI Shapefile'

            ```
        - An unknown extension with no explicit driver is refused:
            ```python
            >>> resolve_vector_driver("out.weird", None)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: cannot infer a vector driver from 'out.weird'; pass --driver ...

            ```
    """
    if explicit:
        return explicit
    ext = os.path.splitext(path)[1].lower()
    driver = VECTOR_DRIVER_BY_EXT.get(ext)
    if driver is None:
        raise ValueError(
            f"cannot infer a vector driver from {path!r}; pass --driver "
            f"(known extensions: {sorted(VECTOR_DRIVER_BY_EXT)})"
        )
    return driver
