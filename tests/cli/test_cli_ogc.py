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
    """A stand-in vector that records the driver and touches the output path."""

    def __init__(self):
        self.received_driver = None

    def to_file(self, path, driver=None):
        self.received_driver = driver
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
        """The layer, bbox, and size reach Dataset.from_wms."""
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
                "--size",
                "256",
                "256",
            ]
        )
        assert rc == 0
        assert captured["layers"] == "L"

    def test_requires_size_or_resolution(self, monkeypatch, tmp_path):
        """Neither --size nor --resolution is rejected by argparse (M1).

        from_wms needs exactly one of size/resolution, so the CLI must not let a
        no-grid call through to a runtime ValueError.
        """
        monkeypatch.setattr(
            "pyramids.cli_ogc.Dataset.from_wms",
            staticmethod(lambda *a, **k: _FakeDataset()),
        )
        with pytest.raises(SystemExit):
            main(
                [
                    "fetch",
                    "wms",
                    "https://x",
                    str(tmp_path / "o.tif"),
                    "--layers",
                    "L",
                    "--bbox",
                    "0",
                    "0",
                    "1",
                    "1",
                ]
            )

    def test_resolution_forwarded(self, monkeypatch, tmp_path):
        """--resolution is forwarded to Dataset.from_wms as an alternative to --size."""
        captured: dict = {}

        def fake(url, **kwargs):
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr("pyramids.cli_ogc.Dataset.from_wms", staticmethod(fake))
        rc = main(
            [
                "fetch",
                "wms",
                "https://x",
                str(tmp_path / "o.tif"),
                "--layers",
                "L",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--resolution",
                "10",
            ]
        )
        assert rc == 0
        assert captured["resolution"] == 10.0
        assert "size" not in captured

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


class TestFetchWmts:
    """`pyramids fetch wmts`."""

    def test_forwards_kwargs(self, monkeypatch, tmp_path):
        """The layer, bbox, resolution and output-crs reach Dataset.from_wmts."""
        captured: dict = {}

        def fake(endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr("pyramids.cli_ogc.Dataset.from_wmts", staticmethod(fake))
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "fetch",
                "wmts",
                "https://x/WMTSCapabilities.xml",
                out,
                "--layer",
                "truecolor",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--resolution",
                "0.01",
                "--output-crs",
                "3857",
            ]
        )
        assert rc == 0
        assert captured["layer"] == "truecolor"
        assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
        assert captured["resolution"] == 0.01
        assert captured["output_crs"] == "3857"
        assert os.path.exists(out)

    def test_missing_layer_is_rejected(self, monkeypatch, tmp_path):
        """--layer is required, so argparse exits before the handler runs."""
        monkeypatch.setattr(
            "pyramids.cli_ogc.Dataset.from_wmts",
            staticmethod(lambda *a, **k: _FakeDataset()),
        )
        argv = [
            "fetch",
            "wmts",
            "https://x",
            str(tmp_path / "o.tif"),
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
        with pytest.raises(SystemExit):
            main(argv)

    @pytest.mark.live
    def test_live_smoke(self, tmp_path):
        """Live GIBS crop (deselected by default; run with `-m live`)."""
        out = str(tmp_path / "gibs.tif")
        rc = main(
            [
                "fetch",
                "wmts",
                "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml",
                out,
                "--layer",
                "MODIS_Terra_CorrectedReflectance_TrueColor",
                "--bbox",
                "5",
                "51",
                "6",
                "52",
                "--resolution",
                "0.01",
            ]
        )
        assert rc == 0
        assert os.path.exists(out)


class TestFetchOgcCoverages:
    """`pyramids fetch ogc-coverages`."""

    def test_forwards_kwargs(self, monkeypatch, tmp_path):
        """The coverage id, bbox and resolution reach Dataset.from_ogc_coverages."""
        captured: dict = {}

        def fake(endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured.update(kwargs)
            return _FakeDataset()

        monkeypatch.setattr(
            "pyramids.cli_ogc.Dataset.from_ogc_coverages", staticmethod(fake)
        )
        out = str(tmp_path / "o.tif")
        rc = main(
            [
                "fetch",
                "ogc-coverages",
                "https://maps.example/ogcapi",
                out,
                "--coverage",
                "dem",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--resolution",
                "0.5",
                "--coverage-crs",
                "4326",
            ]
        )
        assert rc == 0
        assert captured["coverage"] == "dem"
        assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
        assert captured["resolution"] == 0.5
        assert captured["coverage_crs"] == "4326"
        assert os.path.exists(out)

    def test_missing_bbox_is_rejected(self, monkeypatch, tmp_path):
        """--bbox is required, so argparse exits before the handler runs."""
        monkeypatch.setattr(
            "pyramids.cli_ogc.Dataset.from_ogc_coverages",
            staticmethod(lambda *a, **k: _FakeDataset()),
        )
        argv = [
            "fetch",
            "ogc-coverages",
            "https://x",
            str(tmp_path / "o.tif"),
            "--coverage",
            "dem",
        ]
        with pytest.raises(SystemExit):
            main(argv)


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

    def test_driver_inferred_from_extension(self, monkeypatch, tmp_path):
        """fetch wfs infers the OGR driver from the output extension (L1)."""
        sink = _FakeFeatureCollection()
        monkeypatch.setattr(
            "pyramids.cli_ogc.FeatureCollection.from_wfs",
            staticmethod(lambda url, **kwargs: sink),
        )
        rc = main(
            [
                "fetch",
                "wfs",
                "https://x",
                str(tmp_path / "roads.gpkg"),
                "--typename",
                "T",
            ]
        )
        assert rc == 0
        assert sink.received_driver == "gpkg", (
            "a .gpkg output must not silently default to GeoJSON"
        )

    def test_unknown_extension_without_driver_errors(
        self, monkeypatch, tmp_path, capsys
    ):
        """An unknown output extension with no --driver exits 1 (L1)."""
        monkeypatch.setattr(
            "pyramids.cli_ogc.FeatureCollection.from_wfs",
            staticmethod(lambda url, **kwargs: _FakeFeatureCollection()),
        )
        rc = main(
            [
                "fetch",
                "wfs",
                "https://x",
                str(tmp_path / "roads.weird"),
                "--typename",
                "T",
            ]
        )
        assert rc == 1
        assert "cannot infer a vector driver" in capsys.readouterr().err
