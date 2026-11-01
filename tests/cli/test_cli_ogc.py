"""Tests for the ``pyramids fetch`` (OGC) command group (M-4).

Offline: the ``Dataset.from_*`` / ``FeatureCollection.from_*`` readers are mocked,
so these assert the CLI's argument wiring (kwarg forwarding + write) without
hitting the network. Live smoke tests belong behind the ``live`` marker.
"""

from __future__ import annotations

import os

import pytest

from pyramids.cli import main

pytestmark = pytest.mark.core


class _FakeDataset:
    """A stand-in raster whose ``to_file`` just touches the output path."""

    def to_file(self, path):
        with open(path, "wb"):
            pass


class _FakeFeatureCollection:
    """A stand-in vector whose ``to_file`` just touches the output path."""

    def to_file(self, path, driver=None):
        with open(path, "wb"):
            pass


class TestFetchWcs:
    """`pyramids fetch wcs`."""

    def test_forwards_kwargs(self, monkeypatch, tmp_path):
        """The coverage id and bbox reach Dataset.from_wcs; the output is written."""
        captured: dict = {}

        def fake(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr("pyramids.cli_ogc.Dataset.from_wcs", staticmethod(fake))
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "fetch",
                "wcs",
                "https://x",
                out,
                "--coverage",
                "dem",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
            ]
        )
        assert rc == 0
        assert captured["coverage"] == "dem"
        assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
        assert os.path.exists(out)


class TestFetchWms:
    """`pyramids fetch wms`."""

    def test_forwards_kwargs(self, monkeypatch, tmp_path):
        """The layer and bbox reach Dataset.from_wms."""
        captured: dict = {}

        def fake(url, **kwargs):
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr("pyramids.cli_ogc.Dataset.from_wms", staticmethod(fake))
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "fetch",
                "wms",
                "https://x",
                out,
                "--layers",
                "L",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
            ]
        )
        assert rc == 0
        assert captured["layers"] == "L"

    def test_size_and_crs_forwarded(self, monkeypatch, tmp_path):
        """--size and --crs are forwarded to Dataset.from_wms when given."""
        captured: dict = {}

        def fake(url, **kwargs):
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr("pyramids.cli_ogc.Dataset.from_wms", staticmethod(fake))
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "fetch",
                "wms",
                "https://x",
                out,
                "--layers",
                "L",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--crs",
                "4326",
                "--size",
                "256",
                "128",
            ]
        )
        assert rc == 0
        assert captured["crs"] == "4326"
        assert captured["size"] == (256, 128)


class TestFetchWfs:
    """`pyramids fetch wfs`."""

    def test_forwards_kwargs(self, monkeypatch, tmp_path):
        """The typename and bbox reach FeatureCollection.from_wfs."""
        captured: dict = {}

        def fake(url, **kwargs):
            captured.update(kwargs)
            return _FakeFeatureCollection()

        monkeypatch.setattr(
            "pyramids.cli_ogc.FeatureCollection.from_wfs", staticmethod(fake)
        )
        out = str(tmp_path / "o.geojson")
        rc = main(
            [
                "fetch",
                "wfs",
                "https://x",
                out,
                "--typename",
                "T",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
            ]
        )
        assert rc == 0
        assert captured["typename"] == "T"
        assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
