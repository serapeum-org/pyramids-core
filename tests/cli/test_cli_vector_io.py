"""Tests for the ``pyramids vector`` I/O, export, and remote-reader commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids import cli_vector_io
from pyramids.cli import main
from pyramids.feature import FeatureCollection

_DATA = Path(__file__).parents[1] / "data"
_POLY = _DATA / "geometries" / "polygons.geojson"


def _run(argv):
    """Dispatch argv through the real `pyramids vector` CLI (so errors exit 1)."""
    return main(["vector", *argv])


class TestFromBbox:
    """`vector from-bbox`."""

    def test_writes_bbox_polygon(self, tmp_path):
        """from-bbox writes a one-feature polygon in the given CRS."""
        out = tmp_path / "bbox.geojson"
        rc = _run(
            ["from-bbox", str(out), "--bbox", "0", "0", "1", "1", "--epsg", "4326"]
        )
        assert rc == 0
        fc = FeatureCollection.read_file(str(out))
        assert len(fc) == 1

    def test_refuses_existing(self, tmp_path):
        """A second from-bbox without --overwrite is refused."""
        out = str(tmp_path / "bbox.geojson")
        assert (
            _run(["from-bbox", out, "--bbox", "0", "0", "1", "1", "--epsg", "4326"])
            == 0
        )
        assert (
            _run(["from-bbox", out, "--bbox", "0", "0", "1", "1", "--epsg", "4326"])
            == 1
        )


class TestGeoJsonExports:
    """`vector to-json` / `to-geo-dict`."""

    def test_to_json_stdout(self, capsys):
        """to-json prints a GeoJSON FeatureCollection string to stdout."""
        assert _run(["to-json", str(_POLY)]) == 0
        assert '"FeatureCollection"' in capsys.readouterr().out

    def test_to_json_out_file(self, tmp_path):
        """--out writes the GeoJSON to a file."""
        out = tmp_path / "o.json"
        assert _run(["to-json", str(_POLY), "--out", str(out)]) == 0
        assert out.stat().st_size > 0

    def test_to_geo_dict_json(self, capsys):
        """to-geo-dict --json emits a strict-valid mapping with features."""
        assert _run(["to-geo-dict", str(_POLY), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) > 0


class TestWktWkbExports:
    """`vector to-wkt` / `to-wkb`."""

    def test_to_wkt_csv(self, tmp_path):
        """to-wkt writes a non-empty CSV."""
        out = tmp_path / "w.csv"
        assert _run(["to-wkt", str(_POLY), str(out)]) == 0
        assert out.stat().st_size > 0

    def test_to_wkb_hex_csv(self, tmp_path):
        """to-wkb --hex writes a non-empty CSV."""
        out = tmp_path / "wb.csv"
        assert _run(["to-wkb", str(_POLY), str(out), "--hex"]) == 0
        assert out.stat().st_size > 0


class TestTileExports:
    """`vector to-mvt` / `to-pmtiles` (GDAL drivers, no extra)."""

    def test_to_mvt(self, tmp_path):
        """to-mvt writes an MVT archive."""
        out = tmp_path / "t.mvt"
        assert _run(["to-mvt", str(_POLY), str(out)]) == 0
        assert out.exists()

    def test_to_pmtiles(self, tmp_path):
        """to-pmtiles writes a PMTiles archive."""
        out = tmp_path / "t.pmtiles"
        assert _run(["to-pmtiles", str(_POLY), str(out)]) == 0
        assert out.exists()


class TestParquetRoundTrip:
    """`vector to-parquet` / `read-parquet` / `to-feather` (need the [parquet] extra)."""

    @pytest.mark.parquet
    def test_to_and_read_parquet(self, tmp_path):
        """to-parquet then read-parquet round-trips the features to a vector file."""
        pq = tmp_path / "x.parquet"
        assert _run(["to-parquet", str(_POLY), str(pq)]) == 0
        out = tmp_path / "rp.geojson"
        assert _run(["read-parquet", str(pq), str(out)]) == 0
        assert len(FeatureCollection.read_file(str(out))) == len(
            FeatureCollection.read_file(str(_POLY))
        )

    @pytest.mark.parquet
    def test_to_feather(self, tmp_path):
        """to-feather writes a non-empty GeoFeather file."""
        out = tmp_path / "x.feather"
        assert _run(["to-feather", str(_POLY), str(out)]) == 0
        assert out.stat().st_size > 0


class TestRemoteReaders:
    """Remote readers forward their arguments and write the result (offline-mocked)."""

    def test_from_ogc_features_mocked(self, tmp_path, monkeypatch):
        """from-ogc-features writes whatever the service reader returns."""
        fixture = FeatureCollection.read_file(str(_POLY))
        seen = {}

        def fake(cls, endpoint, *, collection, bbox=None):
            seen["endpoint"] = endpoint
            seen["collection"] = collection
            return fixture

        monkeypatch.setattr(
            cli_vector_io.FeatureCollection, "from_ogc_features", classmethod(fake)
        )
        out = tmp_path / "ogc.geojson"
        rc = _run(
            [
                "from-ogc-features",
                str(out),
                "--endpoint",
                "http://x",
                "--collection",
                "c",
            ]
        )
        assert rc == 0
        assert out.exists()
        assert seen["collection"] == "c"

    def test_from_featureserver_mocked(self, tmp_path, monkeypatch):
        """from-featureserver writes whatever the service reader returns."""
        fixture = FeatureCollection.read_file(str(_POLY))
        monkeypatch.setattr(
            cli_vector_io.FeatureCollection,
            "from_featureserver",
            classmethod(lambda cls, url, *, where="1=1", out_fields="*": fixture),
        )
        out = tmp_path / "fs.geojson"
        assert _run(["from-featureserver", str(out), "--url", "http://x/0"]) == 0
        assert out.exists()

    def test_from_vectortileserver_mocked(self, tmp_path, monkeypatch):
        """from-vectortileserver writes whatever the service reader returns."""
        fixture = FeatureCollection.read_file(str(_POLY))
        monkeypatch.setattr(
            cli_vector_io.FeatureCollection,
            "from_vectortileserver",
            classmethod(lambda cls, url, *, layer=None, bbox=None: fixture),
        )
        out = tmp_path / "vts.geojson"
        assert _run(["from-vectortileserver", str(out), "--url", "http://x"]) == 0
        assert out.exists()
