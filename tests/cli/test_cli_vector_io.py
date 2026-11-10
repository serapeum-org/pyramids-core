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
            [
                "io",
                "from-bbox",
                str(out),
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--epsg",
                "4326",
            ]
        )
        assert rc == 0
        fc = FeatureCollection.read_file(str(out))
        assert len(fc) == 1

    def test_refuses_existing(self, tmp_path):
        """A second from-bbox without --overwrite is refused."""
        out = str(tmp_path / "bbox.geojson")
        assert (
            _run(
                ["io", "from-bbox", out, "--bbox", "0", "0", "1", "1", "--epsg", "4326"]
            )
            == 0
        )
        assert (
            _run(
                ["io", "from-bbox", out, "--bbox", "0", "0", "1", "1", "--epsg", "4326"]
            )
            == 1
        )


class TestGeoJsonExports:
    """`vector to-json` / `to-geo-dict`."""

    def test_to_json_stdout(self, capsys):
        """to-json prints a GeoJSON FeatureCollection string to stdout."""
        assert _run(["io", "to-json", str(_POLY)]) == 0
        assert '"FeatureCollection"' in capsys.readouterr().out

    def test_to_json_out_file(self, tmp_path):
        """--out writes the GeoJSON to a file."""
        out = tmp_path / "o.json"
        assert _run(["io", "to-json", str(_POLY), "--out", str(out)]) == 0
        assert out.stat().st_size > 0

    def test_to_geo_dict_json(self, capsys):
        """to-geo-dict --json emits a strict-valid mapping with features."""
        assert _run(["io", "to-geo-dict", str(_POLY), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) > 0


class TestWktWkbExports:
    """`vector to-wkt` / `to-wkb`."""

    def test_to_wkt_csv(self, tmp_path):
        """to-wkt writes a non-empty CSV."""
        out = tmp_path / "w.csv"
        assert _run(["io", "to-wkt", str(_POLY), str(out)]) == 0
        assert out.stat().st_size > 0

    def test_to_wkb_hex_csv(self, tmp_path):
        """to-wkb --hex writes a non-empty CSV."""
        out = tmp_path / "wb.csv"
        assert _run(["io", "to-wkb", str(_POLY), str(out), "--hex"]) == 0
        assert out.stat().st_size > 0


class TestTileExports:
    """`vector to-mvt` / `to-pmtiles` (GDAL drivers, no extra)."""

    def test_to_mvt(self, tmp_path):
        """to-mvt writes an MVT archive."""
        out = tmp_path / "t.mvt"
        assert _run(["io", "to-mvt", str(_POLY), str(out)]) == 0
        assert out.exists()

    def test_to_pmtiles(self, tmp_path):
        """to-pmtiles writes a PMTiles archive."""
        out = tmp_path / "t.pmtiles"
        assert _run(["io", "to-pmtiles", str(_POLY), str(out)]) == 0
        assert out.exists()


class TestParquetRoundTrip:
    """`vector to-parquet` / `read-parquet` / `to-feather` (need the [parquet] extra)."""

    @pytest.mark.parquet
    def test_to_and_read_parquet(self, tmp_path):
        """to-parquet then read-parquet round-trips the features to a vector file."""
        pq = tmp_path / "x.parquet"
        assert _run(["io", "to-parquet", str(_POLY), str(pq)]) == 0
        out = tmp_path / "rp.geojson"
        assert _run(["io", "read-parquet", str(pq), str(out)]) == 0
        assert len(FeatureCollection.read_file(str(out))) == len(
            FeatureCollection.read_file(str(_POLY))
        )

    @pytest.mark.parquet
    def test_to_feather(self, tmp_path):
        """to-feather writes a non-empty GeoFeather file."""
        out = tmp_path / "x.feather"
        assert _run(["io", "to-feather", str(_POLY), str(out)]) == 0
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
                "io",
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
        assert _run(["io", "from-featureserver", str(out), "--url", "http://x/0"]) == 0
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
        assert _run(["io", "from-vectortileserver", str(out), "--url", "http://x"]) == 0
        assert out.exists()


class TestGeoDictPlain:
    """`vector to-geo-dict` plain (non-JSON) output."""

    def test_plain_output(self, capsys):
        """The plain form prints the type and feature count."""
        assert _run(["io", "to-geo-dict", str(_POLY)]) == 0
        out = capsys.readouterr().out
        assert "type:" in out
        assert "features:" in out


class TestWriteFormat:
    """Writers honor the output extension (H1)."""

    def test_from_bbox_shapefile_sidecars(self, tmp_path):
        """from-bbox to .shp produces a real shapefile (sidecars), not GeoJSON-in-.shp."""
        out = tmp_path / "b.shp"
        rc = _run(
            [
                "io",
                "from-bbox",
                str(out),
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--epsg",
                "4326",
            ]
        )
        assert rc == 0
        assert (tmp_path / "b.shx").exists()
        assert (tmp_path / "b.dbf").exists()


class TestParquetExtraMissing:
    """to-parquet/to-feather report a clean error when pyarrow is absent (M2)."""

    def test_to_parquet_missing_pyarrow(self, tmp_path, monkeypatch, capsys):
        """A missing-pyarrow ImportError becomes a clean [parquet]-extra error."""

        def boom(self, *args, **kwargs):
            raise ImportError("Missing optional dependency 'pyarrow'.")

        monkeypatch.setattr(cli_vector_io.FeatureCollection, "to_parquet", boom)
        rc = _run(["io", "to-parquet", str(_POLY), str(tmp_path / "o.parquet")])
        err = capsys.readouterr().err
        assert rc == 1
        assert "parquet" in err
        assert "unexpected failure" not in err


_GPX = (
    '<?xml version="1.0"?>\n'
    '<gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1">'
    '<wpt lat="1.0" lon="2.0"><name>wp1</name></wpt>'
    "<trk><name>t1</name><trkseg>"
    '<trkpt lat="1.0" lon="2.0"/><trkpt lat="1.1" lon="2.1"/>'
    "</trkseg></trk></gpx>"
)


class TestFromXyz:
    """`vector from-xyz` — issue #1468."""

    def test_builds_point_layer_with_value(self, tmp_path):
        """The x/y/z columns of a CSV become a point layer carrying the z column."""
        csv = tmp_path / "xyz.csv"
        csv.write_text("x,y,z\n0,0,10\n1,1,20\n2,2,30\n", encoding="utf-8")
        out = tmp_path / "pts.geojson"
        argv = [
            "io",
            "from-xyz",
            str(csv),
            str(out),
            "--x",
            "x",
            "--y",
            "y",
            "--z",
            "z",
        ]
        assert _run([*argv, "--crs", "EPSG:4326"]) == 0
        fc = FeatureCollection.read_file(str(out))
        assert len(fc) == 3
        assert "z" in fc.columns

    def test_without_z_is_bare_point_layer(self, tmp_path):
        """Omitting --z builds a point layer with no value column."""
        csv = tmp_path / "xy.csv"
        csv.write_text("lon,lat\n0,0\n1,1\n", encoding="utf-8")
        out = tmp_path / "bare.geojson"
        assert (
            _run(["io", "from-xyz", str(csv), str(out), "--x", "lon", "--y", "lat"])
            == 0
        )
        assert len(FeatureCollection.read_file(str(out))) == 2

    def test_missing_column_errors(self, tmp_path):
        """A --x naming an absent column exits 1 cleanly (lookup error remapped)."""
        csv = tmp_path / "xy.csv"
        csv.write_text("a,b\n0,0\n", encoding="utf-8")
        out = tmp_path / "o.geojson"
        assert (
            _run(["io", "from-xyz", str(csv), str(out), "--x", "nope", "--y", "b"]) == 1
        )


class TestFromFeatures:
    """`vector from-features` — issue #1468."""

    def test_feature_collection_mapping(self, tmp_path):
        """A GeoJSON FeatureCollection mapping is read into an FC."""
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"name": "a"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"name": "b"},
                },
            ],
        }
        src = tmp_path / "fc.json"
        src.write_text(json.dumps(data), encoding="utf-8")
        out = tmp_path / "ff.geojson"
        assert (
            _run(["io", "from-features", str(src), str(out), "--crs", "EPSG:4326"]) == 0
        )
        assert len(FeatureCollection.read_file(str(out))) == 2

    def test_bare_feature_list(self, tmp_path):
        """A bare list of feature dicts is also accepted (the else branch)."""
        feats = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"name": "a"},
            }
        ]
        src = tmp_path / "list.json"
        src.write_text(json.dumps(feats), encoding="utf-8")
        out = tmp_path / "ffl.geojson"
        assert _run(["io", "from-features", str(src), str(out)]) == 0
        assert len(FeatureCollection.read_file(str(out))) == 1


class TestFromRecords:
    """`vector from-records` — issue #1468."""

    def test_builds_from_wkt_csv(self, tmp_path):
        """A CSV whose geometry column holds WKT becomes a point FC with attributes."""
        csv = tmp_path / "rec.csv"
        csv.write_text("id,geometry\n1,POINT (0 0)\n2,POINT (1 1)\n", encoding="utf-8")
        out = tmp_path / "rec.geojson"
        assert (
            _run(["io", "from-records", str(csv), str(out), "--crs", "EPSG:4326"]) == 0
        )
        fc = FeatureCollection.read_file(str(out))
        assert len(fc) == 2
        assert "id" in fc.columns

    def test_missing_geometry_column_errors(self, tmp_path):
        """A CSV with no geometry column exits 1 cleanly (lookup error remapped)."""
        csv = tmp_path / "norec.csv"
        csv.write_text("id,value\n1,5\n", encoding="utf-8")
        out = tmp_path / "o.geojson"
        assert _run(["io", "from-records", str(csv), str(out)]) == 1


class TestReadGpx:
    """`vector read-gpx` — issue #1468."""

    def test_writes_sublayers_to_gpkg(self, tmp_path):
        """Each non-empty GPX sub-layer becomes a named layer in the output GPKG."""
        src = tmp_path / "t.gpx"
        src.write_text(_GPX, encoding="utf-8")
        out = tmp_path / "g.gpkg"
        assert _run(["io", "read-gpx", str(src), str(out)]) == 0
        waypoints = FeatureCollection.read_file(str(out), layer="waypoints")
        assert len(waypoints) == 1

    def test_refuses_existing(self, tmp_path):
        """An existing output without --overwrite is refused."""
        src = tmp_path / "t.gpx"
        src.write_text(_GPX, encoding="utf-8")
        out = tmp_path / "g.gpkg"
        out.write_text("placeholder", encoding="utf-8")
        assert _run(["io", "read-gpx", str(src), str(out)]) == 1

    def test_no_sublayers_errors(self, tmp_path, monkeypatch, capsys):
        """A GPX with no non-empty sub-layers exits 1 cleanly, not the catch-all."""
        src = tmp_path / "t.gpx"
        src.write_text(_GPX, encoding="utf-8")
        monkeypatch.setattr(
            cli_vector_io.FeatureCollection,
            "read_gpx_layers",
            classmethod(lambda cls, path: {}),
        )
        rc = _run(["io", "read-gpx", str(src), str(tmp_path / "g.gpkg")])
        err = capsys.readouterr().err
        assert rc == 1
        assert "unexpected failure" not in err
