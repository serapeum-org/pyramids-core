"""Config-file support for the ``pyramids`` CLI (E-2 / #1490).

A ``pyramids.toml`` file lets a user set per-command defaults once instead of repeating
flags (CRS, driver, ``--overwrite``, ...). Defaults are applied to each subcommand's
parser via :meth:`argparse.ArgumentParser.set_defaults` **before** parsing, so an explicit
flag on the command line always wins. Precedence, highest first:

1. the command line (an explicit flag),
2. the config file (``--config PATH`` > ``PYRAMIDS_CONFIG`` env > nearest ``pyramids.toml``
   searched upward from the working directory > ``~/.config/pyramids/config.toml``),
3. the built-in argparse default.

Schema — a top-level ``[defaults]`` table applies to every verb that has the option, and a
``[group.command]`` table applies to that one verb (a key is only applied when the target
verb actually defines that option, so an unrelated key is silently ignored):

```toml
[defaults]
overwrite = true

[vector.warp]
crs = "EPSG:4326"
driver = "gpkg"
```

The module reads nothing from the CLI package, so it adds no import cycle. ``tomllib`` is
in the standard library (Python 3.11+), so this needs no new dependency.
"""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

CONFIG_BASENAME = "pyramids.toml"
CONFIG_ENV_VAR = "PYRAMIDS_CONFIG"


def find_config_file(
    explicit: str | None = None, start: str | None = None
) -> Path | None:
    """Locate the config file to load, honoring the discovery precedence.

    Args:
        explicit: The value of ``--config`` (an explicit path), if given. Must exist.
        start: The directory to begin the upward ``pyramids.toml`` search from; defaults to
            the current working directory.

    Returns:
        Path | None: The resolved config path, or ``None`` when no config applies.

    Raises:
        ValueError: ``explicit`` (or ``PYRAMIDS_CONFIG``) names a path that does not exist.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f"--config {explicit!r} does not exist")
        return path
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        path = Path(env)
        if not path.is_file():
            raise ValueError(f"{CONFIG_ENV_VAR}={env!r} does not exist")
        return path
    here = Path(start) if start is not None else Path.cwd()
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_BASENAME
        if candidate.is_file():
            return candidate
    user = Path.home() / ".config" / "pyramids" / "config.toml"
    return user if user.is_file() else None


def load_config(path: Path | None) -> dict[str, Any]:
    """Parse a config file into a nested dict, or ``{}`` when ``path`` is ``None``.

    Args:
        path: The config file to read, or ``None``.

    Returns:
        dict[str, Any]: The parsed tables, or an empty dict.

    Raises:
        ValueError: The file is not valid TOML.
    """
    if path is None:
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid config file {str(path)!r}: {exc}") from exc


def _iter_subparsers(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    """Return every (dotted-path, subparser) pair reachable from ``parser``.

    Args:
        parser: The parser to walk.
        prefix: The command path accumulated so far.

    Returns:
        list: ``((group, command, ...), subparser)`` for each nested subparser.
    """
    found: list[tuple[tuple[str, ...], argparse.ArgumentParser]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                path = (*prefix, name)
                found.append((path, subparser))
                found.extend(_iter_subparsers(subparser, path))
    return found


def _section_for(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """Return the ``[group.command]`` table for ``path``, or ``{}`` if absent."""
    node: Any = config
    for part in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(part, {})
    return node if isinstance(node, dict) else {}


def apply_config_defaults(
    parser: argparse.ArgumentParser, config: dict[str, Any]
) -> None:
    """Apply a parsed config's defaults onto the matching subcommand parsers.

    A key is applied only when the target verb actually defines that option (so a typo or
    an option meant for another verb is ignored rather than injected as a stray attribute).
    The global ``[defaults]`` table applies to every verb that has the option; a
    ``[group.command]`` table applies to that one verb and overrides ``[defaults]``.

    Args:
        parser: The fully built top-level parser.
        config: The parsed config (as returned by :func:`load_config`).
    """
    if not config:
        return
    global_defaults = config.get("defaults", {})
    global_defaults = global_defaults if isinstance(global_defaults, dict) else {}
    for path, subparser in _iter_subparsers(parser):
        dests = {action.dest for action in subparser._actions}
        overrides: dict[str, Any] = {
            key: value for key, value in global_defaults.items() if key in dests
        }
        section = _section_for(config, path)
        for key, value in section.items():
            if key in dests:
                overrides[key] = value
        if overrides:
            subparser.set_defaults(**overrides)
