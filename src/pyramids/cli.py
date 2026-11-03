"""Command-line interface for pyramids.

Thin, scriptable wrappers over the library's primitives, built on the
standard-library :mod:`argparse` (no extra dependency). Each command group lives
in its own ``cli_<domain>.py`` module and registers itself on the top-level
parser via an ``add_<domain>_commands(sub)`` function; this module only wires the
groups together, provides the global flags, and runs the dispatch loop.

Command groups:

- raster / COG (`pyramids.cli_dataset`) — `info`, `bounds`, `clip`, `warp`,
  `merge`, `overview`, `sample`, `convert`, `georeference`, `orthorectify`,
  `edit-info`, `calc`, `shapes`, `rasterize`, `render`, and `cog create|validate|info`
- vector (`pyramids.cli_vector`) — `vector info|bounds|convert|warp|clip`
- NetCDF (`pyramids.cli_netcdf` + `cli_netcdf_{inspect,vars,select,cftime,io,plot}`) —
  the full `nc` surface across six sub-modules: inspection + global attributes, variable & dimension
  management, positional/label selection & subsetting, CF/time/calendar/calculus, I/O export, and headless
  plotting (each sub-module documents its own commands; `pyramids nc --help` lists them)
- collection (`pyramids.cli_collection`) — `collection info|stats`
- STAC (`pyramids.cli_stac`) — `stac search|download` (`[stac]` extra)
- OGC (`pyramids.cli_ogc`) — `fetch wms|wcs|wfs`
- UGRID mesh (`pyramids.cli_ugrid`) — `mesh info|extract|convert`
- processing (`pyramids.processing.cli`) — `run`, `apply`, `tools`, `tool`

The top-level `--version`/`-V` prints the version, and global `--quiet`/`-q`
(before the command) suppresses informational output. Every command maps 1:1 onto
an existing library call — no business logic lives here. Expected user errors
(missing file, bad CRS, unknown driver) exit non-zero with a one-line message
instead of a traceback. The entry point is registered as the `pyramids` console
script; the functions here are also callable in-process (`main([...])`) for testing.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from pyramids import __version__
from pyramids.base._errors import _PyramidsError
from pyramids.cli_collection import add_collection_commands
from pyramids.cli_dataset import add_dataset_commands
from pyramids.cli_netcdf import add_netcdf_commands
from pyramids.cli_ogc import add_ogc_commands
from pyramids.cli_stac import add_stac_commands
from pyramids.cli_ugrid import add_ugrid_commands
from pyramids.cli_vector import add_vector_commands
from pyramids.processing.cli import add_processing_commands

try:
    # Optional shell-completion hook (E-5): enabled only if the user installs
    # argcomplete themselves. pyramids declares no dependency on it, and its
    # absence is a silent no-op — so this adds a convenience without a new
    # required package.
    import argcomplete
except ImportError:  # pragma: no cover - argcomplete is an optional convenience
    argcomplete = None  # type: ignore[assignment]


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        argparse.ArgumentParser: The parser with the global ``--version`` /
        ``--quiet`` flags and every command group registered via its
        ``add_<domain>_commands`` function.
    """
    parser = argparse.ArgumentParser(
        prog="pyramids", description="pyramids GIS toolkit"
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"pyramids {__version__}",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress informational output, e.g. 'wrote ...' (errors still print); "
        "place it before the command, e.g. `pyramids --quiet warp ...`",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    add_dataset_commands(sub)
    add_vector_commands(sub)
    add_netcdf_commands(sub)
    add_collection_commands(sub)
    add_stac_commands(sub)
    add_ogc_commands(sub)
    add_ugrid_commands(sub)
    add_processing_commands(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `pyramids` console script.

    Args:
        argv: Argument list (excluding the program name). Defaults to
            :data:`sys.argv` when `None`.

    Returns:
        int: Process exit code (`0` success, non-zero failure).

    Examples:
        - Inspect a COG from the shell:
            ```python
            >>> main(["cog", "info", "scene.tif"])  # doctest: +SKIP
            0

            ```
    """
    parser = _build_parser()
    if argcomplete is not None:  # pragma: no cover - optional convenience
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        _PyramidsError,
    ) as exc:
        # Expected user errors (missing file, bad CRS, unknown driver, ...)
        # exit non-zero with a one-line message instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        result = 1
    except Exception as exc:  # noqa: BLE001
        # An unexpected internal error (a bug, or a GDAL error type not listed
        # above) still gets a one-line message for the CLI user rather than a raw
        # traceback — unless PYRAMIDS_DEBUG is set, which re-raises the full stack.
        if os.environ.get("PYRAMIDS_DEBUG"):
            raise
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        result = 1
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
