"""Grouping foundations for the ``pyramids`` CLI restructure (Phase 1).

The CLI is moving to one consistent convention — ``pyramids <domain> [<subsystem>] <verb>`` —
and this module holds the shared plumbing every family uses to get there:

- :class:`SuggestingArgumentParser` — an :class:`argparse.ArgumentParser` that appends a
  ``did you mean …?`` hint (via :mod:`difflib`) when a subcommand is misspelled.
- :func:`add_group` — mount a nested command group on a subparsers action and return the
  group's own subparsers action, so a domain or subsystem registers its verbs uniformly.

The restructure is a **hard move**: when a verb moves to its nested path the old flat spelling
is removed outright (no deprecation aliases). Everything is standard-library only.
"""

from __future__ import annotations

import argparse
import difflib
import re
from typing import cast

_BAD_CHOICE = re.compile(r"invalid choice: '([^']*)'")
_CHOOSE_FROM = re.compile(r"choose from (.+)\)")


def _suggestion(message: str) -> str | None:
    """Return a ``did you mean: …?`` line for an invalid-choice ``message``, or ``None``.

    Returns ``None`` when the message is not an invalid-choice error, when it already carries
    a native suggestion ("maybe you meant"), or when nothing is close enough.

    Args:
        message: The argparse error message.

    Returns:
        str | None: The suggestion line, or ``None``.
    """
    bad = _BAD_CHOICE.search(message)
    choose = _CHOOSE_FROM.search(message)
    if bad is None or choose is None or "meant" in message:
        return None
    choices = re.findall(r"'([^']*)'", choose.group(1))
    close = difflib.get_close_matches(bad.group(1), choices, n=3)
    return f"did you mean: {', '.join(close)}?" if close else None


class SuggestingArgumentParser(argparse.ArgumentParser):
    """An argument parser that suggests near-matches for a misspelled subcommand.

    On an ``invalid choice`` error it appends ``did you mean: …?`` with the closest valid
    subcommand names, so ``pyramids raster wrap`` points the user at ``warp`` instead of just
    failing. Python 3.14+ argparse already suggests natively ("maybe you meant …?"); this is a
    backport for older interpreters and stays silent when a native suggestion is present, so
    the two never double up. Behaviour is otherwise identical to :class:`argparse.ArgumentParser`.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        """Append a difflib suggestion to an invalid-choice error, then defer to argparse.

        Args:
            message: The argparse error message.
        """
        hint = _suggestion(message)
        if hint is not None:
            message = f"{message}\n\n{hint}"
        super().error(message)


def add_group(
    sub: argparse._SubParsersAction,
    name: str,
    help: str,
    *,
    dest: str | None = None,
) -> argparse._SubParsersAction:
    """Mount a nested command group and return its subparsers action.

    The group's own subcommands inherit :class:`SuggestingArgumentParser`, so a misspelled
    verb inside the group is suggested too. ``required=True`` makes ``pyramids <group>`` with
    no verb print the group's help instead of dispatching to nothing.

    Args:
        sub: The parent subparsers action (the top level, or another group).
        name: The group name (e.g. ``"raster"``, ``"bands"``).
        help: One-line help shown in the parent's listing.
        dest: The attribute the chosen subcommand is stored under; defaults to
            ``"<name>_command"``.

    Returns:
        argparse._SubParsersAction: The group's subparsers action to register verbs on.
    """
    group = sub.add_parser(name, help=help)
    inner = group.add_subparsers(
        dest=dest or f"{name}_command",
        required=True,
        parser_class=SuggestingArgumentParser,
    )
    return cast("argparse._SubParsersAction", inner)
