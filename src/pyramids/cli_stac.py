"""STAC CLI command group — ``pyramids stac …`` (M-3).

Thin wrappers over :func:`pyramids.stac.search` and
:func:`pyramids.stac.download_item_collection`. ``pystac-client`` / ``stac-asset``
are the optional ``[stac]`` extra and are imported lazily by those functions, so
importing this module (and the whole CLI) never requires the extra — the
commands only raise a clear install hint when actually run without it.
"""

from __future__ import annotations

import argparse
import json
import os

from pyramids._cli_shared import info_print
from pyramids.stac import download_item_collection, search


def _cmd_stac_search(args: argparse.Namespace) -> int:
    """Handle ``pyramids stac search`` — print matched STAC item ids.

    Args:
        args: Parsed args with ``url``, ``collections``, ``bbox``, ``datetime``,
            ``max_items``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    items = search(
        args.url,
        args.collections,
        bbox=tuple(args.bbox) if args.bbox else None,
        datetime=args.datetime,
        max_items=args.max_items,
    )
    ids = [item.id for item in items]
    if args.json:
        print(json.dumps({"items": ids}))
    else:
        for item_id in ids:
            print(item_id)
    return 0


def _cmd_stac_download(args: argparse.Namespace) -> int:
    """Handle ``pyramids stac download`` — search, then pull assets to ``--out``.

    Args:
        args: Parsed args with ``url``, ``collections``, ``bbox``, ``datetime``,
            ``assets``, ``max_items``, ``out``.

    Returns:
        int: ``0`` on success.
    """
    os.makedirs(args.out, exist_ok=True)
    items = search(
        args.url,
        args.collections,
        bbox=tuple(args.bbox) if args.bbox else None,
        datetime=args.datetime,
        max_items=args.max_items,
    )
    download_item_collection(items, args.out, include=args.assets)
    info_print(args, f"downloaded assets to {args.out}")
    return 0


def add_stac_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``stac`` command group on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    stac = sub.add_parser("stac", help="STAC search / download")
    ssub = stac.add_subparsers(dest="command", required=True)

    s_search = ssub.add_parser("search", help="search a STAC API")
    s_search.add_argument("url", help="STAC API root URL")
    s_search.add_argument(
        "--collections", nargs="+", required=True, help="collection id(s)"
    )
    s_search.add_argument(
        "--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY")
    )
    s_search.add_argument(
        "--datetime", help="RFC3339 instant or range (e.g. 2020-01-01/2020-12-31)"
    )
    s_search.add_argument("--max-items", dest="max_items", type=int)
    s_search.add_argument("--json", action="store_true", help="emit JSON")
    s_search.set_defaults(func=_cmd_stac_search)

    s_dl = ssub.add_parser("download", help="download assets from a STAC search")
    s_dl.add_argument("url", help="STAC API root URL")
    s_dl.add_argument(
        "--collections", nargs="+", required=True, help="collection id(s)"
    )
    s_dl.add_argument(
        "--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY")
    )
    s_dl.add_argument("--datetime", help="RFC3339 instant or range")
    s_dl.add_argument(
        "--assets", nargs="+", help="asset keys to include (default: all)"
    )
    s_dl.add_argument("--max-items", dest="max_items", type=int)
    s_dl.add_argument("--out", required=True, help="output directory")
    s_dl.set_defaults(func=_cmd_stac_download)
