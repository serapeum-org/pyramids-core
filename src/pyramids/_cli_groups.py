"""Grouping foundations for the ``pyramids`` CLI restructure (Phase 1).

The CLI is moving to one consistent convention — ``pyramids <domain> [<subsystem>] <verb>`` —
and this module holds the shared plumbing every family uses to get there:

- :class:`SuggestingArgumentParser` — an :class:`argparse.ArgumentParser` that appends a
  ``did you mean …?`` hint (via :mod:`difflib`) when a subcommand is misspelled.
- :func:`add_group` — mount a nested command group on a subparsers action and return the
  group's own subparsers action, so a domain or subsystem registers its verbs uniformly.
- :data:`DEPRECATED` + :func:`rewrite_deprecated` — the compatibility shim. As verbs move
  from their old flat path to a nested one, an entry ``"old-verb": ("domain", "verb")`` is
  added here; the shim rewrites ``pyramids old-verb …`` to the new path and prints a one-line
  deprecation note. The old spellings are **removed in 1.0.0** (no permanent aliases).

Everything is standard-library only.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from typing import TextIO, cast

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


# old flat verb -> new nested path. Populated as each phase moves a family. Empty until the
# raster/process groups land; the shim is a no-op while it is empty.
DEPRECATED: dict[str, tuple[str, ...]] = {}

_REMOVED_IN = "1.0.0"
# Global options that must be skipped when locating the verb token; the value-taking ones
# consume the following token too.
_GLOBAL_VALUE_OPTS = {"--format", "--config"}
_GLOBAL_FLAG_OPTS = {
    "--quiet",
    "-q",
    "--no-config",
    "--progress",
    "--version",
    "-V",
    "--help",
    "-h",
}


def _verb_index(argv: list[str]) -> int | None:
    """Return the index of the first subcommand token in ``argv``, skipping global options."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _GLOBAL_VALUE_OPTS:
            i += 2
            continue
        if token in _GLOBAL_FLAG_OPTS or token.startswith("-"):
            i += 1
            continue
        return i
    return None


def rewrite_deprecated(argv: list[str], *, stderr: TextIO | None = None) -> list[str]:
    """Rewrite a deprecated flat verb to its new nested path, warning once.

    If the subcommand in ``argv`` is a key of :data:`DEPRECATED`, its token is replaced by the
    new path (e.g. ``["info", "x"]`` -> ``["raster", "info", "x"]``) and a one-line
    deprecation note is printed to ``stderr``. Global options before the verb are preserved.
    Anything not deprecated is returned unchanged.

    Args:
        argv: The raw argument list (without the program name).
        stderr: Stream for the deprecation note; defaults to :data:`sys.stderr`.

    Returns:
        list[str]: The possibly-rewritten argument list.
    """
    if not DEPRECATED:
        return argv
    index = _verb_index(argv)
    if index is None:
        return argv
    new = DEPRECATED.get(argv[index])
    if new is None:
        return argv
    stream = stderr if stderr is not None else sys.stderr
    old_path = " ".join(argv[: index + 1][index:])
    print(
        f"warning: `pyramids {old_path}` is deprecated; use `pyramids {' '.join(new)}` "
        f"(removed in {_REMOVED_IN})",
        file=stream,
    )
    return [*argv[:index], *new, *argv[index + 1 :]]
