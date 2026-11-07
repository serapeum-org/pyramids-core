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
from typing import Any

from pyramids.base._errors import OptionalPackageDoesNotExist


@contextlib.contextmanager
def needs_lazy_extra() -> Iterator[None]:
    """Remap a missing-``[lazy]`` ``ImportError`` to a clean ``ValueError``.

    The kerchunk / multi-file / Zarr readers and writers import ``zarr`` / ``kerchunk``
    from the optional ``[lazy]`` extra; its absence raises a raw ``ImportError``, which
    :func:`pyramids.cli.main` does not catch — it would surface as the internal-bug
    "unexpected failure" branch. Wrapping the call turns that into a clean user error
    naming the extra to install.

    Yields:
        None: Control returns to the wrapped block.

    Raises:
        ValueError: A dependency from the ``[lazy]`` extra (``zarr`` / ``kerchunk``) is
            not installed.
    """
    try:
        yield
    except OptionalPackageDoesNotExist as exc:
        # Pass the seam's own message through. It already names the distribution that provides the
        # backend and how to install it; replacing it with an extras hint hands the user advice
        # that cannot fix the problem (the extras install the stack, not the backend).
        raise ValueError(str(exc)) from exc
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [lazy] extra (zarr/kerchunk); install it "
            "with `pip install pyramids-core[lazy]`"
        ) from exc


@contextlib.contextmanager
def needs_parquet_extra() -> Iterator[None]:
    """Remap a missing-``[parquet]`` ``ImportError`` to a clean ``ValueError``.

    Writing Parquet imports ``pyarrow`` from the optional ``[parquet]`` extra; its
    absence raises a raw ``ImportError`` that :func:`pyramids.cli.main` does not catch.
    Wrapping the call turns it into a clean user error naming the extra to install.

    Yields:
        None: Control returns to the wrapped block.

    Raises:
        ValueError: ``pyarrow`` (the ``[parquet]`` extra) is not installed.
    """
    try:
        yield
    except OptionalPackageDoesNotExist as exc:
        # Pass the seam's own message through. It already names the distribution that provides the
        # backend and how to install it; replacing it with an extras hint hands the user advice
        # that cannot fix the problem (the extras install the stack, not the backend).
        raise ValueError(str(exc)) from exc
    except ImportError as exc:
        raise ValueError(
            "this command needs the optional [parquet] extra (pyarrow); install it "
            "with `pip install pyramids-core[parquet]`"
        ) from exc


def validate_index_pattern(pattern: str, **sample: object) -> None:
    """Validate a per-item filename ``--pattern`` before the write loop runs.

    A per-slice/-window writer formats ``pattern`` with a ``{i}`` index (plus any
    ``sample`` placeholders, e.g. ``{name}``). Two failure modes are caught here, up
    front, so neither surfaces mid-loop:

    - an unknown placeholder would raise a raw ``KeyError`` / ``IndexError`` that
      :func:`pyramids.cli.main` does not catch (it reports "unexpected failure");
    - a pattern with no ``{i}`` formats identically for every item, so the loop would
      silently overwrite every item onto one path and exit 0 with only the last kept.

    Args:
        pattern: The user's ``--pattern`` template.
        **sample: Sample values for the non-index placeholders the writer also binds
            (e.g. ``name="slice"``); ``i`` is always probed and need not be passed.

    Raises:
        ValueError: ``pattern`` uses a placeholder outside ``{i}`` / ``sample``, or it
            does not include ``{i}`` (so items would collide on one path).
    """
    probe = {"i": 0, **sample}
    allowed = ", ".join("{" + key + "}" for key in sorted(probe))
    try:
        first = pattern.format(**probe)
        second = pattern.format(**{**probe, "i": 1})
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"--pattern {pattern!r} uses an unknown placeholder {exc}; "
            f"available placeholders: {allowed}"
        ) from exc
    if first == second:
        raise ValueError(
            f"--pattern {pattern!r} must include the {{i}} index placeholder so each "
            "item is written to a distinct path"
        )


def require_matching_crs(left: Any, right: Any) -> None:
    """Raise if two layers carry a different CRS.

    Both operands of a two-layer vector op (overlay/set op, spatial join, planar
    distance, ...) must share a coordinate system; combining mismatched ones
    silently produces geometrically meaningless output. Raise a clean ``ValueError``
    telling the user to reproject first. A layer whose CRS is unset on either side
    is left alone (there is nothing to compare).

    Args:
        left: The primary layer (anything exposing a ``crs`` attribute).
        right: The second layer read from ``--other`` / ``--right`` / ``--mask``.

    Raises:
        ValueError: Both layers carry a CRS and the two differ.
    """
    left_crs = getattr(left, "crs", None)
    right_crs = getattr(right, "crs", None)
    if left_crs is not None and right_crs is not None and left_crs != right_crs:
        raise ValueError(
            f"the two layers have different CRS ({left_crs} vs {right_crs}); reproject "
            f"one to match first, e.g. `pyramids vector warp <src> <out> --crs <epsg>`"
        )


def json_sanitize(obj: Any) -> Any:
    """Recursively convert a structure to strict-JSON-safe values.

    Walks dicts, lists, and tuples; maps non-finite floats (``NaN``/``Infinity``) to
    ``None`` via :func:`json_safe`; and converts numpy scalars/arrays to Python via
    ``tolist``. This must run **before** :func:`json.dumps`, because ``dumps`` never
    calls its ``default=`` hook for floats (``numpy.float64`` is a ``float`` subclass
    the C encoder serialises directly), so a non-finite float would otherwise be
    emitted as a bare ``NaN``/``Infinity`` token that strict consumers (e.g. ``jq``)
    reject.
    """
    if isinstance(obj, dict):
        return {key: json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(value) for value in obj]
    if isinstance(obj, bool):
        return obj  # bool is an int subclass; keep it as-is, never scrub
    if isinstance(obj, float):
        return json_safe(obj)
    if type(obj).__module__ == "numpy" and hasattr(obj, "tolist"):
        return json_sanitize(obj.tolist())
    return obj


def force_agg_backend() -> None:
    """Force matplotlib's non-interactive Agg backend before any plot import.

    Assigned (not ``setdefault``) so a pre-exported interactive backend such as
    TkAgg cannot win and block on a GUI popup in a headless CLI/CI run. Call it in
    any plotting command before cleopatra/matplotlib is imported.
    """
    os.environ["MPLBACKEND"] = "Agg"


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
