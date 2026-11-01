"""Enable ``python -m pyramids`` — delegates to the CLI entry point.

Mirrors the ``pyramids`` console script (``pyramids.cli:main``) so the CLI is
reachable via the module form in environments where the script shim is awkward
or absent (CI steps, ``uvx``/``pipx run``, frozen environments).
"""

from __future__ import annotations

from pyramids.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
