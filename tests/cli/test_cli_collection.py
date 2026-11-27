"""Tests for the ``pyramids collection`` command group (M-5, #1469)."""

from __future__ import annotations

import glob as _glob
import json
import os
import zipfile

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset
from pyramids.dataset.collection import DatasetCollection


def _stack_with(tmp_path, *, epsg: int, no_data: float) -> str:
    """Write three co-registered 3x3 rasters and return a glob matching them.

    Args:
        tmp_path: pytest temp dir.
        epsg: CRS of the stack.
        no_data: no-data sentinel stamped on each raster.

    Returns:
        str: A glob matching the three written rasters.
    """
    for index in range(3):
        Dataset.from_array(
            np.full((3, 3), float(index), dtype="float32"),
            no_data_value=no_data,
            geo_ref=GeoReference(top_left_corner=(0, 3), cell_size=1.0, epsg=epsg),
        ).to_file(str(tmp_path / f"s{index}.tif"))
    return str(tmp_path / "s*.tif")


def _dated_stack(tmp_path) -> str:
    """Write three dated 4x4 rasters whose names carry a parseable timestamp.

    Args:
        tmp_path: pytest temp dir.

    Returns:
        str: A glob matching the three ``YYYY.MM.DD.tif`` rasters.
    """
    for day in (1, 2, 3):
        Dataset.from_array(
            np.full((4, 4), float(day), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
        ).to_file(str(tmp_path / f"2022.01.0{day}.tif"))
    return str(tmp_path / "2022.01.0*.tif")


class TestCollectionInfo:
    """`pyramids collection info` — metadata only, so it runs in the core wheel env."""

    pytestmark = pytest.mark.core

    def test_json_summary(self, cli_stack, capsys):
        """--json summarises the stack (count, CRS, grid)."""
        assert main(["collection", "info", cli_stack, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["rasters"] == 3
        assert payload["epsg"] == 4326
        assert (payload["rows"], payload["columns"]) == (4, 4)

    def test_text_summary(self, cli_stack, capsys):
        """The default (non-JSON) form prints one key per line."""
        assert main(["collection", "info", cli_stack]) == 0
        out = capsys.readouterr().out
        assert "rasters: 3" in out
        assert "epsg: 4326" in out

    def test_empty_glob_errors(self, tmp_path, capsys):
        """A glob matching nothing exits 1 with a clear message."""
        rc = main(["collection", "info", str(tmp_path / "none*.tif")])
        assert rc == 1
        assert "no rasters matched" in capsys.readouterr().err


class TestCollectionCropRequiresOneTarget:
    """`collection crop` requires exactly one of --mask / --bbox up front (N1)."""

    def test_neither_is_an_argparse_error(self):
        """Passing neither --mask nor --bbox fails at argparse (SystemExit 2)."""
        with pytest.raises(SystemExit):
            main(["collection", "crop", "stack/*.tif", "out"])

    def test_both_is_an_argparse_error(self):
        """Passing both --mask and --bbox is rejected as mutually exclusive."""
        with pytest.raises(SystemExit):
            main(
                [
                    "collection",
                    "crop",
                    "stack/*.tif",
                    "out",
                    "--mask",
                    "m.tif",
                    "--bbox",
                    "0",
                    "0",
                    "2",
                    "2",
                ]
            )


class TestCollectionResamplePattern:
    """`collection resample` validates --pattern before touching the stack (N2)."""

    def test_unknown_placeholder_is_clean(self, tmp_path, capsys):
        """A --pattern placeholder other than {i} fails fast with a clean error."""
        out = str(tmp_path / "rs")
        rc = main(
            [
                "collection",
                "resample",
                "no/such/*.tif",
                out,
                "--freq",
                "1D",
                "--pattern",
                "{name}.tif",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, "a bad pattern must exit 1"
        assert "unexpected failure" not in err, f"leaked internal-bug branch: {err!r}"
        assert "placeholder" in err, f"message unclear: {err!r}"
        assert "{i}" in err, f"message unclear: {err!r}"

    def test_fixed_pattern_without_index_rejected(self, tmp_path, capsys):
        """A --pattern lacking {i} is rejected so windows cannot collide (L1)."""
        out = str(tmp_path / "rs")
        rc = main(
            [
                "collection",
                "resample",
                "no/such/*.tif",
                out,
                "--freq",
                "1D",
                "--pattern",
                "fixed.tif",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, "a fixed pattern must exit 1"
        assert "unexpected failure" not in err, f"leaked internal-bug branch: {err!r}"
        assert "{i}" in err, f"message should require {{i}}, got: {err!r}"


class TestCollectionToCogPattern:
    """`collection export cog` validates --pattern up front, like resample (M1/L1)."""

    def test_unknown_placeholder_is_clean(self, tmp_path, capsys):
        """An unknown --pattern placeholder fails fast with a clean error (M1)."""
        out = str(tmp_path / "cog")
        rc = main(
            [
                "collection",
                "export",
                "cog",
                "no/such/*.tif",
                out,
                "--pattern",
                "{bad}.tif",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, "a bad pattern must exit 1"
        assert "unexpected failure" not in err, f"leaked internal-bug branch: {err!r}"
        assert "placeholder" in err, f"message unclear: {err!r}"

    def test_fixed_pattern_without_index_rejected(self, tmp_path, capsys):
        """A --pattern lacking {i} is rejected so slices cannot collide (L1)."""
        out = str(tmp_path / "cog")
        rc = main(
            [
                "collection",
                "export",
                "cog",
                "no/such/*.tif",
                out,
                "--pattern",
                "{name}.tif",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, "a fixed pattern must exit 1"
        assert "unexpected failure" not in err, f"leaked internal-bug branch: {err!r}"
        assert "{i}" in err, f"message should require {{i}}, got: {err!r}"

    def test_nonempty_output_dir_refused(self, tmp_path, capsys):
        """A non-empty output dir is refused up front, for parity with siblings (L2)."""
        out = tmp_path / "cog"
        out.mkdir()
        (out / "stale.txt").write_text("x", encoding="utf-8")
        rc = main(["collection", "export", "cog", "no/such/*.tif", str(out)])
        err = capsys.readouterr().err
        assert rc == 1, "a non-empty output dir must be refused"
        assert "not empty" in err, f"should be the output-dir guard, got: {err!r}"


class TestLazyExtraContract:
    """A missing backend exits 1 cleanly, never 'unexpected failure' (M1)."""

    def test_source_zarr_missing_extra_is_clean(self, monkeypatch, capsys):
        """A raw ImportError from from_zarr becomes a clean one-line error."""

        def _raise_import(*_a, **_k):
            raise ImportError("No module named 'zarr'")

        monkeypatch.setattr(DatasetCollection, "from_zarr", _raise_import)
        rc = main(["collection", "info", "store.zarr", "--source", "zarr"])
        err = capsys.readouterr().err
        assert rc == 1, "a missing extra must exit 1"
        assert "unexpected failure" not in err, f"leaked internal-bug branch: {err!r}"
        assert "pip install pyramids-lazy" in err, (
            f"error should name the backend install command, got: {err!r}"
        )


def _local_items(tmp_path) -> list[dict]:
    """Build three STAC item dicts over real local 3x3 rasters, in datetime order.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        list[dict]: Three item dicts whose single ``data`` asset points at a local GeoTIFF.
    """
    feats = []
    for index in range(3):
        path = str(tmp_path / f"scene{index}.tif")
        Dataset.from_array(
            np.full((3, 3), float(index + 1), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        ).io.to_file(path)
        feats.append(
            {
                "type": "Feature",
                "id": f"scene{index}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]],
                },
                "bbox": [0.0, 0.0, 3.0, 3.0],
                "properties": {"datetime": f"2023-06-0{index + 1}T00:00:00Z"},
                "assets": {"data": {"href": path, "type": "image/tiff"}},
                "stac_extensions": [],
            }
        )
    return feats


def _items_json(tmp_path, feats, *, as_list: bool = False) -> str:
    """Write ``feats`` to a JSON file, as a FeatureCollection or a bare list.

    Args:
        tmp_path: pytest temp directory.
        feats: The item dicts.
        as_list: Write a bare JSON list instead of a FeatureCollection wrapper.

    Returns:
        str: Path to the written JSON file.
    """
    path = str(tmp_path / "items.json")
    payload = feats if as_list else {"type": "FeatureCollection", "features": feats}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


class TestCollectionImportStac:
    """`collection import stac` — build a cube from a saved ItemCollection JSON (#1542)."""

    pytestmark = pytest.mark.core

    def test_writes_a_cube_from_a_feature_collection(self, tmp_path):
        """A FeatureCollection of local-asset items writes a non-empty NetCDF cube."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        assert (
            main(["collection", "import", "stac", items, out, "--asset", "data"]) == 0
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_accepts_a_bare_item_list(self, tmp_path):
        """A bare JSON list of items (no FeatureCollection wrapper) is also accepted."""
        items = _items_json(tmp_path, _local_items(tmp_path), as_list=True)
        out = str(tmp_path / "cube.nc")
        assert (
            main(["collection", "import", "stac", items, out, "--asset", "data"]) == 0
        )
        assert os.path.exists(out)

    def test_accepts_a_single_feature(self, tmp_path):
        """A single GeoJSON ``Feature`` (no ``features`` key) is wrapped and read (L2)."""
        feature = _local_items(tmp_path)[0]
        path = str(tmp_path / "one.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(feature, handle)
        out = str(tmp_path / "cube.nc")
        assert main(["collection", "import", "stac", path, out, "--asset", "data"]) == 0
        assert os.path.exists(out)

    def test_rejects_an_unknown_json_shape(self, tmp_path, capsys):
        """A JSON object that is neither a FeatureCollection, list, nor Feature errors cleanly (L2)."""
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"not": "stac"}, handle)
        rc = main(
            [
                "collection",
                "import",
                "stac",
                path,
                str(tmp_path / "x.nc"),
                "--asset",
                "data",
            ]
        )
        assert rc == 1
        assert "not a STAC" in capsys.readouterr().err

    def test_rejects_a_non_list_features_value(self, tmp_path, capsys):
        """A `features` key whose value is not a list falls through to the clean error (R2 N1)."""
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", "features": "oops"}, handle)
        rc = main(
            [
                "collection",
                "import",
                "stac",
                path,
                str(tmp_path / "x.nc"),
                "--asset",
                "data",
            ]
        )
        assert rc == 1
        assert "not a STAC" in capsys.readouterr().err

    def test_max_items_runs_end_to_end(self, tmp_path):
        """`--max-items 1` runs end-to-end and writes a cube (offline integration)."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        argv = [
            "collection",
            "import",
            "stac",
            items,
            out,
            "--asset",
            "data",
            "--max-items",
            "1",
        ]
        assert main(argv) == 0
        assert os.path.exists(out)

    def test_bbox_and_groupby_run_end_to_end(self, tmp_path):
        """`--bbox` (covering all items) and `--groupby id` run end-to-end and write a cube."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        argv = [
            "collection",
            "import",
            "stac",
            items,
            out,
            "--asset",
            "data",
            "--bbox",
            "0",
            "0",
            "3",
            "3",
            "--groupby",
            "id",
        ]
        assert main(argv) == 0
        assert os.path.exists(out)

    def test_optional_flags_are_forwarded_to_from_stac(self, tmp_path, monkeypatch):
        """`--asset`/`--bbox`/`--max-items`/`--groupby` reach `from_stac` as the right types (M1).

        The end-to-end tests above would still pass if a flag were silently dropped, so this
        stubs `from_stac` and asserts each option is actually forwarded, with `--bbox` becoming a
        4-tuple, `--max-items` an int, and a multi-key `--asset` a list.
        """
        captured: dict = {}

        class _FakeCube:
            def to_netcdf(self, path):
                captured["out"] = path

        def _fake_from_stac(items, asset, **kwargs):
            captured["items"] = items
            captured["asset"] = asset
            captured.update(kwargs)
            return _FakeCube()

        monkeypatch.setattr(
            DatasetCollection, "from_stac", staticmethod(_fake_from_stac)
        )
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        argv = [
            "collection",
            "import",
            "stac",
            items,
            out,
            "--asset",
            "B04,B03",
            "--bbox",
            "1",
            "2",
            "3",
            "4",
            "--max-items",
            "2",
            "--groupby",
            "id",
        ]
        assert main(argv) == 0
        assert isinstance(captured["items"], list)
        assert len(captured["items"]) == 3
        assert captured["asset"] == ["B04", "B03"]
        assert captured["bbox"] == (1.0, 2.0, 3.0, 4.0)
        assert captured["max_items"] == 2
        assert captured["groupby"] == "id"
        assert captured["out"] == out

    def test_bbox_defaults_to_none_when_omitted(self, tmp_path, monkeypatch):
        """Without `--bbox`/`--max-items`/`--groupby`, `from_stac` receives None for each (M1)."""
        captured: dict = {}

        def _fake_from_stac(items, asset, **kwargs):
            captured.update(kwargs)

            class _FakeCube:
                def to_netcdf(self, path):
                    captured["out"] = path

            return _FakeCube()

        monkeypatch.setattr(
            DatasetCollection, "from_stac", staticmethod(_fake_from_stac)
        )
        items = _items_json(tmp_path, _local_items(tmp_path))
        argv = [
            "collection",
            "import",
            "stac",
            items,
            str(tmp_path / "c.nc"),
            "--asset",
            "data",
        ]
        assert main(argv) == 0
        assert captured["bbox"] is None
        assert captured["max_items"] is None
        assert captured["groupby"] is None

    def test_requires_an_asset(self, tmp_path):
        """`--asset` is required; argparse rejects the call without it."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        with pytest.raises(SystemExit):
            main(["collection", "import", "stac", items, str(tmp_path / "x.nc")])

    def test_refuses_an_existing_output(self, tmp_path, capsys):
        """Without `--overwrite`, writing over an existing file exits 1 with a clear error."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        assert (
            main(["collection", "import", "stac", items, out, "--asset", "data"]) == 0
        )
        capsys.readouterr()
        rc = main(["collection", "import", "stac", items, out, "--asset", "data"])
        assert rc == 1
        assert "already exists" in capsys.readouterr().err


class TestCollectionImportPoint:
    """`collection import point` — forwards args to `DatasetCollection.from_point` (#1542).

    The real call needs the ``[stac]`` extra and a live endpoint, so these tests stub
    ``from_point`` and assert the CLI marshals the arguments and writes the cube.
    """

    pytestmark = pytest.mark.core

    @staticmethod
    def _patch(monkeypatch, captured):
        """Replace `from_point` with a stub recording its arguments (returns a fake cube)."""

        class _FakeCube:
            def to_netcdf(self, path):
                captured["out"] = path

        def _fake_from_point(lat, lon, **kwargs):
            captured["lat"] = lat
            captured["lon"] = lon
            captured.update(kwargs)
            return _FakeCube()

        monkeypatch.setattr(
            DatasetCollection, "from_point", staticmethod(_fake_from_point)
        )

    def _base_argv(self, out, *extra):
        """The minimal required `import point` argv, writing to `out`."""
        return [
            "collection",
            "import",
            "point",
            out,
            "--lat",
            "40",
            "--lon",
            "-3",
            "--collection",
            "sentinel-2-l2a",
            "--bands",
            "B04,B03",
            "--start",
            "2023-01-01",
            "--end",
            "2023-03-01",
            "--edge-size",
            "128",
            "--resolution",
            "10",
            *extra,
        ]

    def test_forwards_all_arguments_and_writes(self, tmp_path, monkeypatch):
        """Each scalar flag reaches `from_point`, and the cube is written to `output`."""
        captured: dict = {}
        self._patch(monkeypatch, captured)
        out = str(tmp_path / "pt.nc")
        assert main(self._base_argv(out)) == 0
        assert (captured["lat"], captured["lon"]) == (40.0, -3.0)
        assert captured["collection"] == "sentinel-2-l2a"
        assert captured["bands"] == ["B04", "B03"]
        assert (captured["start_date"], captured["end_date"]) == (
            "2023-01-01",
            "2023-03-01",
        )
        assert (captured["edge_size"], captured["resolution"]) == (128, 10.0)
        assert captured["units"] == "px"
        assert captured["align"] is True
        assert captured["stac"] is None, "the default --stac (None) must be forwarded"
        assert captured["out"] == out

    def test_single_band_is_passed_as_a_string(self, tmp_path, monkeypatch):
        """A single `--bands` key forwards as a bare string, not a one-element list."""
        captured: dict = {}
        self._patch(monkeypatch, captured)
        argv = self._base_argv(str(tmp_path / "pt.nc"))
        argv[argv.index("B04,B03")] = "B04"
        assert main(argv) == 0
        assert captured["bands"] == "B04"

    def test_band_whitespace_and_trailing_comma_are_stripped(
        self, tmp_path, monkeypatch
    ):
        """Spaces around keys and a trailing comma are stripped, not forwarded (L1)."""
        captured: dict = {}
        self._patch(monkeypatch, captured)
        argv = self._base_argv(str(tmp_path / "pt.nc"))
        argv[argv.index("B04,B03")] = "B04, B03,"
        assert main(argv) == 0
        assert captured["bands"] == ["B04", "B03"]

    def test_no_align_and_query_are_forwarded(self, tmp_path, monkeypatch):
        """`--no-align` sets align False and `--query` is parsed from JSON to a dict."""
        captured: dict = {}
        self._patch(monkeypatch, captured)
        out = str(tmp_path / "pt.nc")
        argv = self._base_argv(
            out, "--no-align", "--query", '{"eo:cloud_cover": {"lt": 10}}'
        )
        assert main(argv) == 0
        assert captured["align"] is False
        assert captured["query"] == {"eo:cloud_cover": {"lt": 10}}

    def test_requires_the_core_flags(self, tmp_path, monkeypatch):
        """Omitting a required flag (here `--collection`) is rejected by argparse."""
        captured: dict = {}
        self._patch(monkeypatch, captured)
        with pytest.raises(SystemExit):
            main(
                [
                    "collection",
                    "import",
                    "point",
                    str(tmp_path / "pt.nc"),
                    "--lat",
                    "40",
                    "--lon",
                    "-3",
                    "--bands",
                    "B04",
                    "--start",
                    "2023-01-01",
                    "--end",
                    "2023-03-01",
                    "--edge-size",
                    "128",
                    "--resolution",
                    "10",
                ]
            )

    def test_refuses_an_existing_output(self, tmp_path, monkeypatch, capsys):
        """Without `--overwrite`, an existing output exits 1 before any network call."""
        captured: dict = {}
        self._patch(monkeypatch, captured)
        out = str(tmp_path / "pt.nc")
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("x")
        rc = main(self._base_argv(out))
        assert rc == 1
        assert "already exists" in capsys.readouterr().err
        assert "out" not in captured, "from_point must not run when the output exists"
