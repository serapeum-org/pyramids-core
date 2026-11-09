"""Config-file support for the ``pyramids`` CLI (E-2 / #1490).

A ``pyramids.yaml`` file lets a user set per-command defaults once instead of repeating
flags (CRS, driver, output format, ...). YAML is the package's config format (see
``pyramids/base/config.yaml`` and the processing-pipeline ``.yaml`` files), so this stays
consistent with the rest of the codebase. Defaults are applied to each subcommand's parser
via :meth:`argparse.ArgumentParser.set_defaults` **before** parsing, so an explicit flag on
the command line always wins. Precedence, highest first:

1. the command line (an explicit flag),
2. the config file (``--config PATH`` > ``PYRAMIDS_CONFIG`` env > nearest ``pyramids.yaml``
   searched upward from the working directory > ``~/.config/pyramids/config.yaml``),
3. the built-in argparse default.

Only plain **value-replacing** options (argparse ``store``) are configurable. Options the CLI
cannot fully override are intentionally not seeded, so "an explicit flag always wins" stays
true: zero-arg boolean flags (``store_true`` such as ``--overwrite``) have no ``--no-<flag>``
to flip a configured ``true`` back, and ``append``/``extend`` options would *merge* the CLI
value onto the configured default instead of replacing it. Positionals are never configured.

A configured value is validated the same way argparse would validate the CLI value: it must
satisfy the option's ``choices`` and, for a multi-value ``nargs``, be a list — otherwise a
clean, config-naming ``ValueError`` is raised.

Schema — a top-level ``defaults`` mapping applies to every verb that has the option, and a
``<group>.<command>`` mapping applies to that one verb (a key is only applied when the target
verb actually defines that option as a value option, so an unrelated or boolean key is
silently ignored):

```yaml
defaults:
  driver: gpkg
vector:
  warp:
    crs: "EPSG:4326"
    driver: gpkg
```

The module reads nothing from the CLI package, so it adds no import cycle, and ``PyYAML`` is
already a core dependency, so this needs no new one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_BASENAME = "pyramids.yaml"
CONFIG_ENV_VAR = "PYRAMIDS_CONFIG"


def find_config_file(
    explicit: str | None = None, start: str | None = None
) -> Path | None:
    """Locate the config file to load, honoring the discovery precedence.

    Args:
        explicit: The value of ``--config`` (an explicit path), if given. Must exist.
        start: The directory to begin the upward ``pyramids.yaml`` search from; defaults to
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
    user = Path.home() / ".config" / "pyramids" / "config.yaml"
    return user if user.is_file() else None


def load_config(path: Path | None) -> dict[str, Any]:
    """Parse a YAML config file into a nested dict, or ``{}`` when ``path`` is ``None``.

    An empty file (``yaml.safe_load`` returns ``None``) is treated as no config. A top-level
    document that is not a mapping is rejected, since the schema is a mapping of tables.

    Args:
        path: The config file to read, or ``None``.

    Returns:
        dict[str, Any]: The parsed mapping, or an empty dict.

    Raises:
        ValueError: The file is not valid YAML, or its top level is not a mapping.
    """
    if path is None:
        return {}
    try:
        with path.open("rb") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid config file {str(path)!r}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"invalid config file {str(path)!r}: top level must be a mapping, got "
            f"{type(data).__name__}"
        )
    return data


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
        # Only plain value-replacing optionals are configurable. We allowlist the ``store``
        # action and exclude everything else, because "an explicit flag always wins" only
        # holds for options the CLI can fully override:
        #   - zero-arg flags (``store_true``/``store_false``/``count`` like ``--overwrite``)
        #     have no ``--no-<flag>`` token, so a configured value would be sticky;
        #   - ``append``/``extend`` options MERGE the CLI value onto the configured default
        #     instead of replacing it, so the configured entry is unremovable.
        # Positionals (no option strings) are likewise never seeded from config.
        actions = {
            action.dest: action
            for action in subparser._actions
            if action.option_strings and isinstance(action, argparse._StoreAction)
        }
        overrides: dict[str, Any] = {}
        for source in (global_defaults, _section_for(config, path)):
            for key, value in source.items():
                if key in actions:
                    _validate_config_value(actions[key], key, value, path)
                    overrides[key] = value
        if overrides:
            subparser.set_defaults(**overrides)


def _validate_config_value(
    action: argparse.Action, key: str, value: Any, command: tuple[str, ...]
) -> None:
    """Validate a config value the way argparse would validate the equivalent CLI value.

    ``set_defaults`` bypasses the action's ``choices``/``nargs`` checks, so a config value
    could reach a handler in a shape the CLI would have rejected. Enforce the same two rules
    here and raise a clean, config-naming :class:`ValueError` on a mismatch.

    Args:
        action: The option's argparse action.
        key: The config key (the option's dest).
        value: The configured value.
        command: The ``(group, command, ...)`` path, for the error message.

    Raises:
        ValueError: ``value`` violates the option's ``nargs`` arity or ``choices``.
    """
    where = f"[{'.'.join(command)}] {key}" if command else key
    expects_list = action.nargs in ("+", "*") or (
        isinstance(action.nargs, int) and action.nargs > 1
    )
    if expects_list and not isinstance(value, list):
        raise ValueError(
            f"config {where} expects a list of values, got {type(value).__name__}"
        )
    if action.choices is not None:
        for item in value if isinstance(value, list) else [value]:
            if item not in action.choices:
                raise ValueError(
                    f"config {where}={item!r} is not one of {list(action.choices)}"
                )
