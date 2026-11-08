"""Plugin discovery for the ``pyramids`` CLI (E-4 / #1492).

A third-party package can contribute its own ``pyramids`` subcommands by advertising an
entry point in the :data:`PLUGIN_GROUP` group whose value is an
``add_<domain>_commands(sub)`` callable — the same registration shape the built-in groups
use. At startup the CLI discovers those entry points and calls each one with the top-level
subparsers action, so an installed plugin's verbs appear under ``pyramids`` automatically.

A plugin's ``pyproject.toml`` registers one like::

    [project.entry-points."pyramids.cli_commands"]
    citymodels = "pyramids_citymodels.cli:add_citymodels_commands"

Discovery is failure-isolated: a plugin that fails to import or raises while registering is
recorded and skipped, never crashing the whole CLI. ``pyramids plugins list`` reports what
is installed and whether each plugin loaded. Everything here is standard-library
(:mod:`importlib.metadata`), so it adds no dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from pyramids._cli_shared import emit

PLUGIN_GROUP = "pyramids.cli_commands"


@dataclass
class PluginRecord:
    """One discovered plugin entry point and the outcome of loading it.

    Attributes:
        name: The entry-point name (the plugin's short id).
        value: Its ``module:attr`` target string.
        distribution: The installed distribution that provides it, or ``"?"`` if unknown.
        version: The distribution version, or ``"?"`` if unknown.
        error: ``None`` if the callable imported cleanly, else the error message.
    """

    name: str
    value: str
    distribution: str
    version: str
    error: str | None


def _entry_points() -> list[Any]:
    """Return the entry points registered under :data:`PLUGIN_GROUP` (possibly empty)."""
    return list(entry_points(group=PLUGIN_GROUP))


def discover_plugins(eps: list[Any] | None = None) -> list[PluginRecord]:
    """Discover plugin entry points and whether each one imports cleanly.

    Loading here only *imports* the callable (to report status); it does not register any
    commands. Use :func:`load_plugin_commands` to actually register them.

    Args:
        eps: Entry points to inspect; defaults to the installed ones in :data:`PLUGIN_GROUP`
            (injectable for testing).

    Returns:
        list[PluginRecord]: One record per entry point, sorted by name.
    """
    records: list[PluginRecord] = []
    for ep in _entry_points() if eps is None else eps:
        dist = getattr(ep, "dist", None)
        error: str | None = None
        try:
            ep.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not break discovery
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            PluginRecord(
                name=getattr(ep, "name", "?"),
                value=getattr(ep, "value", "?"),
                distribution=getattr(dist, "name", "?") if dist is not None else "?",
                version=getattr(dist, "version", "?") if dist is not None else "?",
                error=error,
            )
        )
    return sorted(records, key=lambda record: record.name)


def load_plugin_commands(
    sub: argparse._SubParsersAction, eps: list[Any] | None = None
) -> list[str]:
    """Register every installed plugin's commands onto the subparsers, skipping broken ones.

    Args:
        sub: The top-level ``add_subparsers`` action to register plugin verbs onto.
        eps: Entry points to load; defaults to the installed ones (injectable for testing).

    Returns:
        list[str]: Error messages for plugins that failed to load/register (empty if all OK).
    """
    errors: list[str] = []
    for ep in _entry_points() if eps is None else eps:
        try:
            register = ep.load()
            register(sub)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break the CLI
            errors.append(f"{getattr(ep, 'name', '?')}: {type(exc).__name__}: {exc}")
    return errors


def _cmd_plugins_list(args: argparse.Namespace) -> int:
    """Handle ``pyramids plugins list`` — report installed CLI plugins and their status.

    Args:
        args: Parsed args (carries the global ``format`` / ``json`` for output).

    Returns:
        int: ``0`` on success (an empty list when no plugins are installed).
    """
    records = discover_plugins()
    payload = [
        {
            "name": record.name,
            "distribution": record.distribution,
            "version": record.version,
            "status": "ok" if record.error is None else "error",
            "target": record.value,
            "error": record.error,
        }
        for record in records
    ]
    if not records:
        text = "no pyramids CLI plugins installed"
    else:
        text = "\n".join(
            f"{record.name}  {record.version}  "
            f"{'ok' if record.error is None else 'ERROR: ' + record.error}"
            for record in records
        )
    emit(args, payload, text=text)
    return 0


def add_plugins_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``plugins`` command group on the top-level subparsers.

    Args:
        sub: The top-level ``add_subparsers`` action to register onto.
    """
    plugins = sub.add_parser("plugins", help="inspect installed pyramids CLI plugins")
    plugins_sub = plugins.add_subparsers(dest="plugins_command", required=True)
    listing = plugins_sub.add_parser(
        "list", help="list installed CLI plugins and their load status"
    )
    listing.add_argument("--json", action="store_true", help="emit JSON")
    listing.set_defaults(func=_cmd_plugins_list)
