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
import math
import os


def json_safe(value: float | None) -> float | None:
    """Map non-finite floats to ``None`` so JSON output stays parseable.

    ``json.dumps`` serialises ``NaN``/``Infinity`` as bare ``NaN`` / ``Infinity``,
    which is not valid JSON and breaks strict consumers (e.g. ``jq``).

    Args:
        value: A float (possibly ``NaN``/infinite) or ``None``.

    Returns:
        float | None: ``value`` unchanged, or ``None`` when it is not finite.
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
