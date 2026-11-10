"""Command-line interface for pyramids.

Thin, scriptable wrappers over the library's primitives, built on the
standard-library :mod:`argparse` (no extra dependency). Each command group lives
in its own ``cli_<domain>.py`` module and registers itself on the top-level
parser via an ``add_<domain>_commands(sub)`` function; this module only wires the
groups together, provides the global flags, and runs the dispatch loop.

Command groups (all now `pyramids <domain> [<subsystem>] <verb>`):

- raster (`pyramids.cli_dataset`) — the full single-raster (`Dataset`) surface under
  `pyramids raster …`: `info`, `bounds`, `clip`, `warp`, `merge`, `sample`, `convert`,
  `georeference`, `orthorectify`, `edit-info`, `calc`, `shapes`, `rasterize`, `render`, the
  band/inspect/overview/read/io/export/plot verbs, and `raster cog create|validate|info`.
  (The old flat spellings — `pyramids info`, `pyramids warp`, … — were removed.)
- vector (`pyramids.cli_vector` + `cli_vector_{inspect,geom,overlay,tess,io}`) — the full
  `FeatureCollection` surface: core (`info`/`bounds`/`convert`/`warp`/`clip`), inspection, unary geometry ops,
  overlay/set ops + predicate `filter`, tessellation/binning/sampling, measurements, I/O export, and remote
  readers (each sub-module documents its own commands; `pyramids vector --help` lists them)
- NetCDF (`pyramids.cli_netcdf` + `cli_netcdf_{inspect,vars,select,cftime,io,plot}`) —
  the full `nc` surface across six sub-modules: inspection + global attributes, variable & dimension
  management, positional/label selection & subsetting, CF/time/calendar/calculus, I/O export, and headless
  plotting (each sub-module documents its own commands; `pyramids nc --help` lists them)
- collection (`pyramids.cli_collection`) — the full `DatasetCollection` surface:
  `collection info|reduce|resample|merge|export {file,cog,netcdf,zarr,kerchunk}|reproject|crop|align`
  (each verb opens via `--source files|zarr|archive`)
- STAC (`pyramids.cli_stac`) — `stac search|download` (`[stac]` extra)
- OGC (`pyramids.cli_ogc`) — `fetch wms|wcs|wfs`
- UGRID mesh (`pyramids.cli_ugrid`) — `mesh info|extract|convert`
- processing (`pyramids.processing.cli`) — `process run|apply` and `process tool list|show`

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

from shapely.errors import GEOSException

from pyramids import __version__
from pyramids._cli_config import apply_config_defaults, find_config_file, load_config
from pyramids._cli_groups import SuggestingArgumentParser
from pyramids._cli_shared import OUTPUT_FORMATS, resolved_format
from pyramids.base._errors import _PyramidsError
from pyramids.cli_collection import add_collection_commands
from pyramids.cli_completion import add_completion_commands
from pyramids.cli_dataset import add_dataset_commands
from pyramids.cli_netcdf import add_netcdf_commands
from pyramids.cli_ogc import add_ogc_commands
from pyramids.cli_plugins import add_plugins_commands, load_plugin_commands
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
    parser = SuggestingArgumentParser(
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
        "place it before the command, e.g. `pyramids --quiet raster warp ...`",
    )
    parser.add_argument(
        "--format",
        # A distinct dest so the global flag never collides with a verb-level --format (e.g.
        # `nc time-values --format <strftime>`), whose own dest stays `format`.
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default=None,
        help="output format for data-emitting verbs (info/bounds/stats/list/...): "
        "'json' is honored by every verb that supports --json; 'csv'/'table' by the "
        "tabular verbs. Place it before the command, e.g. `pyramids --format json vector info ...`",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="read per-command defaults from this pyramids.yaml (overrides discovery of a "
        "nearby pyramids.yaml / PYRAMIDS_CONFIG / ~/.config/pyramids/config.yaml)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="ignore any pyramids.yaml config file and use built-in defaults only",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="show a progress indicator on stderr for multi-item write verbs (a no-op "
        "when output is not an interactive terminal or when --quiet is set)",
    )
    sub = parser.add_subparsers(
        dest="group", required=True, parser_class=SuggestingArgumentParser
    )

    add_dataset_commands(sub)
    add_vector_commands(sub)
    add_netcdf_commands(sub)
    add_collection_commands(sub)
    add_stac_commands(sub)
    add_ogc_commands(sub)
    add_ugrid_commands(sub)
    add_processing_commands(sub)
    add_completion_commands(sub)
    add_plugins_commands(sub)
    # Installed third-party plugins contribute their own verbs; a broken one is skipped,
    # never fatal (its error is surfaced by `pyramids plugins list`).
    load_plugin_commands(sub)

    return parser


def _apply_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> None:
    """Load a ``pyramids.yaml`` and set its per-command defaults on ``parser``.

    Peeks at ``--config`` / ``--no-config`` with a throwaway parser (the real parse needs a
    subcommand, which may be absent here), resolves the config file, and applies its
    defaults so an explicit CLI flag still overrides them. ``--no-config`` skips it.

    Args:
        parser: The fully built top-level parser to apply defaults onto.
        argv: The raw argument list being parsed.
    """
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default=None)
    peek.add_argument("--no-config", action="store_true")
    known, _ = peek.parse_known_args(argv)
    if known.no_config:
        return
    config = load_config(find_config_file(known.config))
    apply_config_defaults(parser, config)


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
            >>> main(["raster", "cog", "info", "scene.tif"])  # doctest: +SKIP
            0

            ```
    """
    parser = _build_parser()
    if argcomplete is not None:  # pragma: no cover - optional convenience
        argcomplete.autocomplete(parser)
    try:
        # Resolve config defaults inside the try so a bad --config / PYRAMIDS_CONFIG / invalid
        # TOML becomes the CLI's clean `error:` line, not a traceback. Defer any such error
        # until AFTER parse_args so `--help`/`--version` still work with a broken config.
        config_error: Exception | None = None
        try:
            _apply_config(parser, argv)
        except (ValueError, OSError) as exc:
            config_error = exc
        args = parser.parse_args(argv)
        if config_error is not None:
            raise config_error
        # Normalize --json to the resolved output format so verbs routed through emit() and
        # verbs that still read args.json directly agree. --format wins over the legacy --json
        # alias; csv/table (and text) resolve json to False.
        output_format = resolved_format(args)
        args.json = output_format == "json"
        # csv/table are rendered only by verbs routed through emit() (they call
        # mark_format_support). Reject them on any other verb with a clear error instead of
        # silently printing the text form. json is unaffected — it bridges to every --json verb.
        if output_format in ("csv", "table") and not getattr(
            args, "supports_format", False
        ):
            raise ValueError(
                f"this command does not support --format {output_format}; csv/table are "
                "available on the tabular verbs (e.g. vector info/bounds, nc groups/list-vars, "
                "mesh mesh-names). Use --format json, or omit --format for text."
            )
        result: int = args.func(args)
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        GEOSException,
        _PyramidsError,
    ) as exc:
        # Expected user errors (missing file, bad CRS, unknown driver, a GEOS
        # geometry error on the user's input, ...)
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
