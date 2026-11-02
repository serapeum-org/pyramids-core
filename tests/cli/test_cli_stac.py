"""Tests for the ``pyramids stac`` command group (M-3).

Offline: the STAC calls are mocked at the names imported into ``pyramids.cli_stac``,
so these never touch the network and need no ``[stac]`` extra. A live smoke test
against a real endpoint belongs behind the ``live`` marker (not included here).
"""

from __future__ import annotations

import json

import pytest

from pyramids.cli import main

pytestmark = pytest.mark.core


class _Item:
    """A minimal stand-in for a STAC item (only ``.id`` is used by the CLI)."""

    def __init__(self, item_id: str):
        self.id = item_id


class TestStacSearch:
    """`pyramids stac search`."""

    def test_prints_ids_json(self, monkeypatch, capsys):
        """search prints the matched item ids as JSON."""
        monkeypatch.setattr(
            "pyramids.cli_stac.search", lambda *a, **k: [_Item("x"), _Item("y")]
        )
        rc = main(["stac", "search", "https://example", "--collections", "c", "--json"])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["items"] == ["x", "y"]

    def test_forwards_kwargs(self, monkeypatch, capsys):
        """search receives the collections, bbox, and max-items the CLI parsed."""
        captured: dict = {}

        def fake(url, collections, **kwargs):
            captured["url"] = url
            captured["collections"] = collections
            captured.update(kwargs)
            return []

        monkeypatch.setattr("pyramids.cli_stac.search", fake)
        main(
            [
                "stac",
                "search",
                "https://ex",
                "--collections",
                "a",
                "b",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--max-items",
                "5",
            ]
        )
        assert captured["collections"] == ["a", "b"]
        assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
        assert captured["max_items"] == 5


class TestStacDownload:
    """`pyramids stac download`."""

    def test_calls_download_with_dir_and_assets(self, monkeypatch, tmp_path):
        """download searches then forwards the out dir and asset keys."""
        monkeypatch.setattr("pyramids.cli_stac.search", lambda *a, **k: ["item"])
        captured: dict = {}

        def fake_download(items, directory, **kwargs):
            captured["directory"] = directory
            captured["include"] = kwargs.get("include")

        monkeypatch.setattr("pyramids.cli_stac.download_item_collection", fake_download)
        out = str(tmp_path / "dl")
        rc = main(
            [
                "stac",
                "download",
                "https://ex",
                "--collections",
                "c",
                "--assets",
                "B01",
                "--out",
                out,
            ]
        )
        assert rc == 0
        assert captured["directory"] == out
        assert captured["include"] == ["B01"]
