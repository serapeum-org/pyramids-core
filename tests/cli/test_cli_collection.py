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


class TestCollectionInfoSources:
    """`collection info --source {zarr,archive}` — the alternate offline openers."""

    pytestmark = pytest.mark.lazy

    def test_source_zarr(self, cli_stack, tmp_path, capsys):
        """A Zarr store written from the stack reads back through --source zarr."""
        store = str(tmp_path / "cube.zarr")
        DatasetCollection.from_files(sorted(_glob.glob(cli_stack))).to_zarr(store)
        assert main(["collection", "info", store, "--source", "zarr", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["rasters"] == 3

    def test_source_archive(self, cli_stack, tmp_path, capsys):
        """A zip of the stack reads back through --source archive."""
        archive = str(tmp_path / "stack.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            for path in sorted(_glob.glob(cli_stack)):
                zf.write(path, os.path.basename(path))
        rc = main(
            [
                "collection",
                "info",
                archive,
                "--source",
                "archive",
                "--member-glob",
                "*.tif",
                "--json",
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["rasters"] == 3


class TestCollectionStats:
    """`pyramids collection reduce`.

    Marked ``lazy``, not ``core``: ``stats --reduce`` calls the DatasetCollection reducers, which
    route through ``pyramids.dataset._reduce_ops.resolve_dask_op`` and require the optional ``dask``
    dependency (the ``[lazy]`` extra). The core wheel has no dask, so these must not run in the
    ``-m "core ..."`` wheel jobs — only in the dask-equipped envs (dev / lazy), where the
    Tests - Suite selection ``-m 'not plot ...'`` still picks them up.
    """

    pytestmark = pytest.mark.lazy

    def test_mean_reduction(self, cli_stack, tmp_path):
        """--reduce mean writes the element-wise mean across the stack."""
        out = str(tmp_path / "mean.tif")
        assert main(["collection", "reduce", cli_stack, out, "--reduce", "mean"]) == 0
        # mean of constant rasters 0, 1, 2 is 1.0 everywhere.
        assert np.allclose(np.asarray(Dataset.read_file(out).read_array()), 1.0)

    def test_refuses_existing(self, cli_stack, tmp_path):
        """A second stats write without --overwrite is refused."""
        out = str(tmp_path / "m.tif")
        assert main(["collection", "reduce", cli_stack, out, "--reduce", "max"]) == 0
        assert main(["collection", "reduce", cli_stack, out, "--reduce", "max"]) == 1

    def test_carries_base_no_data_value(self, tmp_path):
        """The reduction output carries the base stack's no-data, not -9999 (L2)."""
        glob = _stack_with(tmp_path, epsg=4326, no_data=-1.0)
        out = str(tmp_path / "nd.tif")
        assert main(["collection", "reduce", glob, out, "--reduce", "mean"]) == 0
        assert Dataset.read_file(out).no_data_value[0] == -1.0, (
            "stats must carry the base no-data sentinel, not from_array's -9999 default"
        )

    def test_preserves_non_4326_crs(self, tmp_path):
        """A non-4326 stack keeps its CRS on the reduction output (M2 via crs_spec)."""
        glob = _stack_with(tmp_path, epsg=32633, no_data=-9999.0)
        out = str(tmp_path / "utm.tif")
        assert main(["collection", "reduce", glob, out, "--reduce", "sum"]) == 0
        assert Dataset.read_file(out).epsg == 32633, "output CRS must match the stack"


class TestCollectionResample:
    """`pyramids collection resample` — calendar-frequency reduce_time (needs dask)."""

    pytestmark = pytest.mark.lazy

    def test_writes_one_raster_per_window(self, cli_stack, tmp_path):
        """Daily windows over three days write three per-window rasters."""
        out = str(tmp_path / "windows")
        rc = main(
            [
                "collection",
                "resample",
                cli_stack,
                out,
                "--freq",
                "1D",
                "--op",
                "mean",
                "--times",
                "2022-01-01,2022-01-02,2022-01-03",
            ]
        )
        assert rc == 0
        assert sorted(os.listdir(out)) == [
            "window_0000.tif",
            "window_0001.tif",
            "window_0002.tif",
        ]

    @staticmethod
    def _force_progress_on(monkeypatch):
        """Replace ProgressReporter with one forced enabled, drawing to a throwaway buffer."""
        import io

        from pyramids import cli_collection

        class _AlwaysOn(cli_collection.ProgressReporter):
            def __init__(self, args, total, **kwargs):
                kwargs.pop("stream", None)
                super().__init__(args, total, stream=io.StringIO(), **kwargs)
                self.enabled = True

        monkeypatch.setattr(cli_collection, "ProgressReporter", _AlwaysOn)

    def _resample_argv(self, cli_stack, tmp_path):
        """Return argv for a three-window daily resample into a temp dir."""
        return [
            "--progress",
            "collection",
            "resample",
            cli_stack,
            str(tmp_path / "windows"),
            "--freq",
            "1D",
            "--op",
            "mean",
            "--times",
            "2022-01-01,2022-01-02,2022-01-03",
        ]

    def test_progress_suppresses_log_on_tty_stdout(
        self, cli_stack, tmp_path, monkeypatch
    ):
        """With progress drawing AND stdout a tty, the per-window log is suppressed (L4)."""
        import contextlib
        import io

        self._force_progress_on(monkeypatch)

        class _TTYOut(io.StringIO):
            def isatty(self):
                return True

        out = _TTYOut()
        with contextlib.redirect_stdout(out):
            rc = main(self._resample_argv(cli_stack, tmp_path))
        assert rc == 0, "resample under --progress should succeed"
        assert "window" not in out.getvalue(), (
            "log suppressed when it would collide with the bar"
        )

    def test_progress_keeps_log_when_stdout_redirected(
        self, cli_stack, tmp_path, monkeypatch
    ):
        """With progress drawing but stdout redirected (non-tty), the log is KEPT (round-2 L4)."""
        import contextlib
        import io

        self._force_progress_on(monkeypatch)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(self._resample_argv(cli_stack, tmp_path))
        assert rc == 0, "resample under --progress should succeed"
        assert "window" in out.getvalue(), (
            "redirected stdout must still capture the per-window log"
        )

    def test_percentile_needs_q(self, cli_stack, tmp_path):
        """--op percentile --q 90 reduces one monthly window without error."""
        out = str(tmp_path / "p90")
        rc = main(
            [
                "collection",
                "resample",
                cli_stack,
                out,
                "--freq",
                "1MS",
                "--op",
                "percentile",
                "--q",
                "90",
                "--times",
                "2022-01-01,2022-01-02,2022-01-03",
            ]
        )
        assert rc == 0
        assert os.listdir(out) == ["window_0000.tif"]

    def test_falls_back_to_stack_time_axis(self, tmp_path):
        """With --date-format the stack carries a time axis, so --times is optional."""
        glob = _dated_stack(tmp_path)
        out = str(tmp_path / "auto")
        rc = main(
            [
                "collection",
                "resample",
                glob,
                out,
                "--source",
                "files",
                "--date-format",
                "%Y.%m.%d",
                "--freq",
                "1D",
            ]
        )
        assert rc == 0
        assert len(os.listdir(out)) == 3

    def test_no_time_axis_errors(self, cli_stack, tmp_path, capsys):
        """Without --times and without a parsed time axis, resample refuses cleanly."""
        out = str(tmp_path / "none")
        rc = main(["collection", "resample", cli_stack, out, "--freq", "1D"])
        assert rc == 1
        assert "no time axis" in capsys.readouterr().err

    def test_refuses_non_empty_dir(self, cli_stack, tmp_path):
        """A second write into a populated dir is refused unless --overwrite."""
        out = str(tmp_path / "win")
        argv = [
            "collection",
            "resample",
            cli_stack,
            out,
            "--freq",
            "1D",
            "--times",
            "2022-01-01,2022-01-02,2022-01-03",
        ]
        assert main(argv) == 0
        assert main(argv) == 1
        assert main([*argv, "--overwrite"]) == 0


class TestCollectionMerge:
    """`pyramids collection merge` — mosaic every timestep into one raster."""

    pytestmark = pytest.mark.lazy

    def test_merge_mean(self, cli_stack, tmp_path):
        """--method mean mosaics the aligned stack into one raster."""
        out = str(tmp_path / "mosaic.tif")
        assert main(["collection", "merge", cli_stack, out, "--method", "mean"]) == 0
        assert np.allclose(np.asarray(Dataset.read_file(out).read_array()), 1.0)

    def test_merge_bbox_and_nodata(self, cli_stack, tmp_path):
        """--bbox and --nodata are forwarded to the mosaic."""
        out = str(tmp_path / "win_mosaic.tif")
        rc = main(
            [
                "collection",
                "merge",
                cli_stack,
                out,
                "--method",
                "last",
                "--nodata",
                "-5",
                "--bbox",
                "0",
                "0",
                "2",
                "2",
            ]
        )
        assert rc == 0
        assert os.path.exists(out)

    def test_refuses_existing(self, cli_stack, tmp_path):
        """A second merge without --overwrite is refused."""
        out = str(tmp_path / "m.tif")
        assert main(["collection", "merge", cli_stack, out]) == 0
        assert main(["collection", "merge", cli_stack, out]) == 1


class TestCollectionToFile:
    """`pyramids collection export file` — write every timestep to a directory."""

    pytestmark = pytest.mark.lazy

    def test_writes_each_timestep(self, cli_stack, tmp_path):
        """Each timestep lands as its own file in the output directory."""
        out = str(tmp_path / "steps")
        assert main(["collection", "export", "file", cli_stack, out]) == 0
        assert sorted(os.listdir(out)) == ["0.tif", "1.tif", "2.tif"]

    def test_refuses_non_empty_dir(self, cli_stack, tmp_path):
        """A second write into a populated dir is refused unless --overwrite."""
        out = str(tmp_path / "steps")
        assert main(["collection", "export", "file", cli_stack, out]) == 0
        assert main(["collection", "export", "file", cli_stack, out]) == 1
        assert (
            main(["collection", "export", "file", cli_stack, out, "--overwrite"]) == 0
        )


class TestCollectionToCog:
    """`pyramids collection export cog` — each timestep as its own COG."""

    pytestmark = pytest.mark.lazy

    def test_writes_cogs(self, cli_stack, tmp_path):
        """Each timestep is written as a COG under the slice-named pattern."""
        out = str(tmp_path / "cogs")
        assert main(["collection", "export", "cog", cli_stack, out]) == 0
        assert sorted(os.listdir(out)) == [
            "slice_0000.tif",
            "slice_0001.tif",
            "slice_0002.tif",
        ]

    def test_overwrite_guard(self, cli_stack, tmp_path):
        """A re-run is refused unless --overwrite, which the method enforces."""
        out = str(tmp_path / "cogs")
        assert main(["collection", "export", "cog", cli_stack, out]) == 0
        assert main(["collection", "export", "cog", cli_stack, out]) == 1
        assert main(["collection", "export", "cog", cli_stack, out, "--overwrite"]) == 0


class TestCollectionToNetcdf:
    """`pyramids collection export netcdf` — write the (T,B,R,C) cube to NetCDF."""

    pytestmark = pytest.mark.lazy

    def test_writes_netcdf(self, cli_stack, tmp_path):
        """The cube is written to a single NetCDF file."""
        out = str(tmp_path / "cube.nc")
        assert main(["collection", "export", "netcdf", cli_stack, out]) == 0
        assert os.path.exists(out)

    def test_single_variable(self, cli_stack, tmp_path):
        """--no-var-per-band writes one variable for all bands."""
        out = str(tmp_path / "single.nc")
        assert (
            main(
                ["collection", "export", "netcdf", cli_stack, out, "--no-var-per-band"]
            )
            == 0
        )
        assert os.path.exists(out)

    def test_refuses_existing(self, cli_stack, tmp_path):
        """A second write without --overwrite is refused."""
        out = str(tmp_path / "c.nc")
        assert main(["collection", "export", "netcdf", cli_stack, out]) == 0
        assert main(["collection", "export", "netcdf", cli_stack, out]) == 1


class TestCollectionToZarr:
    """`pyramids collection export zarr` — write the (T,B,R,C) cube to a Zarr store."""

    pytestmark = pytest.mark.lazy

    def test_writes_zarr(self, cli_stack, tmp_path):
        """The cube is written to a Zarr store readable back through --source zarr."""
        store = str(tmp_path / "cube.zarr")
        assert main(["collection", "export", "zarr", cli_stack, store]) == 0
        assert os.path.isdir(store)

    def test_refuses_existing(self, cli_stack, tmp_path):
        """A second write without --overwrite is refused."""
        store = str(tmp_path / "c.zarr")
        assert main(["collection", "export", "zarr", cli_stack, store]) == 0
        assert main(["collection", "export", "zarr", cli_stack, store]) == 1


class TestCollectionToKerchunk:
    """`pyramids collection export kerchunk` — write a kerchunk reference JSON."""

    pytestmark = pytest.mark.lazy

    def test_forwards_arguments(self, cli_stack, tmp_path, monkeypatch):
        """The handler forwards the output path and --concat-dim to the method.

        The GeoTIFF fixtures cannot be kerchunked (that path needs NetCDF/HDF5
        sources and the kerchunk+h5py stack), so the real call is replaced by a
        stub that records its arguments and writes a placeholder reference — this
        exercises the thin wrapper's success path deterministically offline.
        """
        out = str(tmp_path / "ref.json")
        captured: dict = {}

        def _fake_to_kerchunk(self, output_path, *, concat_dim="time"):
            captured["output_path"] = str(output_path)
            captured["concat_dim"] = concat_dim
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("{}")
            return {}

        monkeypatch.setattr(DatasetCollection, "to_kerchunk", _fake_to_kerchunk)
        rc = main(
            ["collection", "export", "kerchunk", cli_stack, out, "--concat-dim", "t"]
        )
        assert rc == 0
        assert captured == {"output_path": out, "concat_dim": "t"}
        assert os.path.exists(out)

    def test_geotiff_sources_error(self, cli_stack, tmp_path, capsys):
        """A GeoTIFF-backed stack reports the NetCDF/HDF5-only limit cleanly."""
        out = str(tmp_path / "ref.json")
        rc = main(["collection", "export", "kerchunk", cli_stack, out])
        assert rc == 1
        assert "kerchunk" in capsys.readouterr().err.lower()


class TestCollectionReproject:
    """`pyramids collection reproject` — reproject every timestep."""

    pytestmark = pytest.mark.lazy

    def test_reprojects_each_timestep(self, cli_stack, tmp_path):
        """--crs reprojects every timestep onto the target CRS."""
        out = str(tmp_path / "rep")
        assert main(["collection", "reproject", cli_stack, out, "--crs", "3857"]) == 0
        files = sorted(os.listdir(out))
        assert len(files) == 3
        assert Dataset.read_file(os.path.join(out, files[0])).epsg == 3857

    def test_maintain_alignment(self, cli_stack, tmp_path):
        """--maintain-alignment is accepted and still writes a file per timestep."""
        out = str(tmp_path / "rep2")
        rc = main(
            [
                "collection",
                "reproject",
                cli_stack,
                out,
                "--crs",
                "3857",
                "--maintain-alignment",
            ]
        )
        assert rc == 0
        assert len(os.listdir(out)) == 3


class TestCollectionCrop:
    """`pyramids collection crop` — crop every timestep to a mask or bbox."""

    pytestmark = pytest.mark.lazy

    def test_crop_bbox(self, cli_stack, tmp_path):
        """--bbox crops every timestep to the window."""
        out = str(tmp_path / "crp")
        rc = main(["collection", "crop", cli_stack, out, "--bbox", "0", "0", "2", "2"])
        assert rc == 0
        assert len(os.listdir(out)) == 3
        assert Dataset.read_file(os.path.join(out, "0.tif")).rows <= 4

    def test_crop_mask_raster(self, cli_stack, tmp_path):
        """--mask crops every timestep to a mask raster's footprint."""
        mask = sorted(_glob.glob(cli_stack))[0]
        out = str(tmp_path / "crp_mask")
        rc = main(["collection", "crop", cli_stack, out, "--mask", mask, "--no-touch"])
        assert rc == 0
        assert len(os.listdir(out)) == 3


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


class TestCollectionAlign:
    """`pyramids collection align` — align every timestep to a reference grid."""

    pytestmark = pytest.mark.lazy

    def test_aligns_to_reference(self, cli_stack, tmp_path):
        """--reference aligns every timestep onto the reference raster's grid."""
        reference = sorted(_glob.glob(cli_stack))[0]
        out = str(tmp_path / "ali")
        rc = main(["collection", "align", cli_stack, out, "--reference", reference])
        assert rc == 0
        assert len(os.listdir(out)) == 3


class TestLazyExtraContract:
    """A missing [lazy] extra exits 1 cleanly, never 'unexpected failure' (M1)."""

    def test_source_zarr_missing_extra_is_clean(self, monkeypatch, capsys):
        """A raw ImportError from from_zarr becomes a clean one-line error."""

        def _raise_import(*_a, **_k):
            raise ImportError("No module named 'zarr'")

        monkeypatch.setattr(DatasetCollection, "from_zarr", _raise_import)
        rc = main(["collection", "info", "store.zarr", "--source", "zarr"])
        err = capsys.readouterr().err
        assert rc == 1, "a missing extra must exit 1"
        assert "unexpected failure" not in err, f"leaked internal-bug branch: {err!r}"
        assert "[lazy]" in err, f"error should name the [lazy] extra, got: {err!r}"


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
        ).to_file(path)
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
        assert os.path.exists(out) and os.path.getsize(out) > 0

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

    def test_max_items_caps_the_cube(self, tmp_path):
        """`--max-items 1` consumes a single item and still writes a cube."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        rc = main(
            [
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
        )
        assert rc == 0 and os.path.exists(out)

    def test_bbox_filter_is_forwarded(self, tmp_path):
        """A `--bbox` covering every item keeps them all and writes a cube."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        rc = main(
            [
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
            ]
        )
        assert rc == 0 and os.path.exists(out)

    def test_groupby_collapses_items(self, tmp_path):
        """`--groupby id` buckets items by id and still writes a cube."""
        items = _items_json(tmp_path, _local_items(tmp_path))
        out = str(tmp_path / "cube.nc")
        rc = main(
            [
                "collection",
                "import",
                "stac",
                items,
                out,
                "--asset",
                "data",
                "--groupby",
                "id",
            ]
        )
        assert rc == 0 and os.path.exists(out)

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
        assert captured["units"] == "px" and captured["align"] is True
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
