"""The ``pyramids completion`` command group (E-5 / #1493).

Shell tab-completion for ``pyramids`` is driven by `argcomplete
<https://pypi.org/project/argcomplete/>`_: the top-level CLI already calls
``argcomplete.autocomplete`` when the package is importable, so completion needs no change
to pyramids itself — only a one-time shell hook. This verb prints that hook for the user's
shell so they do not have to remember the exact snippet.

``argcomplete`` stays an *optional* convenience: pyramids declares no dependency on it, and
the generator here imports nothing — it only prints text. The emitted snippet calls
``register-python-argcomplete`` (shipped by argcomplete), so the user installs argcomplete
once (``pip install argcomplete``) for completion to take effect.
"""

from __future__ import annotations

import argparse

_SHELLS = ("bash", "zsh", "fish")

_SNIPPETS = {
    "bash": 'eval "$(register-python-argcomplete pyramids)"',
    "zsh": "autoload -U bashcompinit && bashcompinit\n"
    'eval "$(register-python-argcomplete pyramids)"',
    "fish": "register-python-argcomplete --shell fish pyramids | source",
}

_NOTE = (
    "# pyramids shell completion ({0}).\n"
    "# Requires argcomplete (`pip install argcomplete`). Add the line(s) below to your\n"
    "# shell startup file, or `eval`/`source` them in the current session:"
)


def _cmd_completion(args: argparse.Namespace) -> int:
    """Handle ``pyramids completion <shell>`` — print the completion hook for a shell.

    Args:
        args: Parsed args with ``shell`` (one of ``bash``/``zsh``/``fish``).

    Returns:
        int: ``0`` on success.
    """
    print(_NOTE.format(args.shell))
    print(_SNIPPETS[args.shell])
    return 0


def add_completion_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``completion`` command on the top-level subparsers.

    Args:
        sub: The top-level ``add_subparsers`` action to register onto.
    """
    parser = sub.add_parser(
        "completion",
        help="print the shell tab-completion hook for bash/zsh/fish (needs argcomplete)",
    )
    parser.add_argument(
        "shell", choices=_SHELLS, help="the shell to emit a completion hook for"
    )
    parser.set_defaults(func=_cmd_completion)
