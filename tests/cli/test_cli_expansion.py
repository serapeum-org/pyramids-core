"""Tests for the CLI ergonomics and bug-fix tasks.

Covers E-1 (`--version`), E-2 (`python -m pyramids`), T-1 (`--help` on every
subcommand), T-2 (top-level help / no-args), E-3 (shared resampling vocabulary),
E-4 (`--quiet`), B-1 (calc note ordering), B-3 (`edit-info --nodata` tokens),
B-4 (`edit-info` driver guard), and M-7 (`render`).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from osgeo import gdal

import pyramids
from pyramids import __version__, cli_dataset
from pyramids.base.georeference import GeoReference
from pyramids.cli import main
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestVersion:
    """`pyramids --version` (E-1)."""

    def test_version_exits_zero(self, capsys):
        """--version prints 'pyramids <ver>' and exits 0."""
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_short_flag_exits_zero(self, capsys):
        """-V is an alias for --version."""
        with pytest.raises(SystemExit) as exc:
            main(["-V"])
        assert exc.value.code == 0


class TestPythonMEntrypoint:
    """`python -m pyramids` (E-2)."""

    def test_module_form_runs(self, cli_raster):
        """`python -m pyramids info FILE --json` behaves like the console script."""
        src_dir = os.path.dirname(os.path.dirname(pyramids.__file__))
        env = {**os.environ, "PYTHONPATH": src_dir}
        proc = subprocess.run(
            [sys.executable, "-m", "pyramids", "raster", "info", cli_raster, "--json"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert '"epsg": 4326' in proc.stdout


class TestTopLevelHelp:
    """Top-level help surface (T-2)."""

    def test_help_lists_groups(self, capsys):
        """`pyramids --help` names the raster, cog, and new command groups."""
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for token in (
            "cog",
            "vector",
            "nc",
            "collection",
            "stac",
            "fetch",
            "mesh",
            "apply",
        ):
            assert token in out, f"top-level help should list {token!r}"

    def test_no_args_errors_cleanly(self, capsys):
        """`pyramids` with no args exits non-zero with usage, not a traceback."""
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert "usage" in err.lower()
        assert "Traceback" not in err


_HELP_ARGVS = [
    ["raster", "info", "--help"],
    ["raster", "bounds", "--help"],
    ["raster", "clip", "--help"],
    ["raster", "warp", "--help"],
    ["raster", "merge", "--help"],
    ["raster", "overview", "--help"],
    ["raster", "sample", "--help"],
    ["raster", "convert", "--help"],
    ["raster", "georeference", "--help"],
    ["raster", "orthorectify", "--help"],
    ["raster", "edit-info", "--help"],
    ["raster", "calc", "--help"],
    ["raster", "shapes", "--help"],
    ["raster", "rasterize", "--help"],
    ["raster", "render", "--help"],
    ["raster", "cog", "--help"],
    ["raster", "cog", "create", "--help"],
    ["raster", "cog", "validate", "--help"],
    ["raster", "cog", "info", "--help"],
    ["process", "run", "--help"],
    ["process", "apply", "--help"],
    ["process", "tool", "list", "--help"],
    ["process", "tool", "show", "--help"],
    # issue #1436: full Dataset surface — read / construct
    ["raster", "import", "bytes", "--help"],
    ["raster", "import", "dataframe", "--help"],
    ["raster", "import", "points", "--help"],
    ["raster", "import", "point-arrays", "--help"],
    ["raster", "import", "band-files", "--help"],
    ["raster", "import", "archive", "--help"],
    ["raster", "import", "zarr", "--help"],
    ["raster", "create", "--help"],
    ["raster", "create-empty", "--help"],
    ["raster", "empty-like", "--help"],
    ["raster", "dataset-like", "--help"],
    ["fetch", "wmts", "--help"],
    ["fetch", "ogc-coverages", "--help"],
    # issue #1436: inspect / metadata
    ["raster", "stats", "--help"],
    ["raster", "histogram", "--help"],
    ["raster", "get-meta-data", "--help"],
    ["raster", "get-attribute-table", "--help"],
    ["raster", "set-attribute-table", "--help"],
    ["raster", "mask-flags", "--help"],
    ["raster", "footprint", "--help"],
    ["raster", "cell-area", "--help"],
    ["raster", "domain-area", "--help"],
    ["raster", "count-domain-cells", "--help"],
    ["raster", "same-grid", "--help"],
    # issue #1436: bands / georef edit
    ["raster", "add-band", "--help"],
    ["raster", "select-bands", "--help"],
    ["raster", "get-band-by-color", "--help"],
    ["raster", "create-mask-band", "--help"],
    ["raster", "read-masks", "--help"],
    ["raster", "get-mask", "--help"],
    ["raster", "set-color-ramp", "--help"],
    ["raster", "set-rpcs", "--help"],
    ["raster", "geolocate", "--help"],
    # issue #1436: spatial ops
    ["raster", "align", "--help"],
    ["raster", "overlay", "--help"],
    ["raster", "combine", "--help"],
    ["raster", "warped-view", "--help"],
    ["raster", "wrap-longitude", "--help"],
    ["raster", "translate", "--help"],
    ["raster", "transect", "--help"],
    ["raster", "extract", "--help"],
    # issue #1436: analysis / terrain
    ["raster", "proximity", "--help"],
    ["raster", "cluster", "--help"],
    ["raster", "zonal-stats", "--help"],
    ["raster", "match-histogram", "--help"],
    ["raster", "percentile-stretch", "--help"],
    ["raster", "pan-sharpen", "--help"],
    ["raster", "fillna", "--help"],
    ["raster", "where", "--help"],
    ["raster", "convert-units", "--help"],
    ["raster", "contour", "--help"],
    # issue #1436: overviews / windowed reads
    ["raster", "recreate-overviews", "--help"],
    ["raster", "read-overview-array", "--help"],
    ["raster", "read-part", "--help"],
    ["raster", "preview", "--help"],
    ["raster", "point", "--help"],
    ["raster", "read-tile", "--help"],
    # issue #1436: export / convert
    ["raster", "export", "bytes", "--help"],
    ["raster", "export", "cog-bytes", "--help"],
    ["raster", "export", "dataframe", "--help"],
    ["raster", "export", "polygons", "--help"],
    ["raster", "export", "raster", "--help"],
    ["raster", "export", "image", "--help"],
    ["raster", "export", "xyz", "--help"],
    ["raster", "export", "zarr", "--help"],
    ["raster", "export", "stac-item", "--help"],
    ["raster", "export", "terrain-rgb", "--help"],
    # issue #1436: plot (headless)
    ["raster", "plot-histogram", "--help"],
    ["raster", "plot-vector-field", "--help"],
    ["raster", "animate", "--help"],
    # issue #1471: finish the Dataset surface — checksum + subdataset
    ["raster", "checksum", "--help"],
    ["raster", "subdataset", "--help"],
    ["vector", "--help"],
    ["vector", "info", "--help"],
    ["vector", "warp", "--help"],
    ["vector", "clip", "--help"],
    ["vector", "affine-transform", "--help"],
    ["vector", "bounds", "--help"],
    ["vector", "buffer", "--help"],
    ["vector", "build-area", "--help"],
    ["vector", "clip-by-rect", "--help"],
    ["vector", "concave-hull", "--help"],
    ["vector", "constrained-delaunay-triangles", "--help"],
    ["vector", "convert", "--help"],
    ["vector", "count-coordinates", "--help"],
    ["vector", "count-geometries", "--help"],
    ["vector", "count-interior-rings", "--help"],
    ["vector", "delaunay-triangles", "--help"],
    ["vector", "difference", "--help"],
    ["vector", "dissolve", "--help"],
    ["vector", "distance", "--help"],
    ["vector", "estimate-utm", "--help"],
    ["vector", "explode", "--help"],
    ["vector", "extract-unique-points", "--help"],
    ["vector", "feature-count", "--help"],
    ["vector", "feature-info", "--help"],
    ["vector", "filter", "--help"],
    ["vector", "fishnet", "--help"],
    ["vector", "force-2d", "--help"],
    ["vector", "force-3d", "--help"],
    ["vector", "frechet-distance", "--help"],
    ["vector", "from-bbox", "--help"],
    ["vector", "from-featureserver", "--help"],
    ["vector", "from-ogc-features", "--help"],
    ["vector", "from-vectortileserver", "--help"],
    ["vector", "geodesic-area", "--help"],
    ["vector", "geodesic-distance", "--help"],
    ["vector", "geodesic-length", "--help"],
    ["vector", "get-coordinates", "--help"],
    ["vector", "h3-bin", "--help"],
    ["vector", "hausdorff-distance", "--help"],
    ["vector", "hilbert-distance", "--help"],
    ["vector", "intersection", "--help"],
    ["vector", "intersection-all", "--help"],
    ["vector", "invalid-coverage-edges", "--help"],
    ["vector", "is-valid-coverage", "--help"],
    ["vector", "is-valid-reason", "--help"],
    ["vector", "list-layers", "--help"],
    ["vector", "make-valid", "--help"],
    ["vector", "maximum-inscribed-circle", "--help"],
    ["vector", "minimum-bounding-circle", "--help"],
    ["vector", "minimum-clearance", "--help"],
    ["vector", "minimum-rotated-rectangle", "--help"],
    ["vector", "normalize", "--help"],
    ["vector", "offset-curve", "--help"],
    ["vector", "overlay", "--help"],
    ["vector", "polygonize", "--help"],
    ["vector", "read-parquet", "--help"],
    ["vector", "remove-repeated-points", "--help"],
    ["vector", "representative-point", "--help"],
    ["vector", "reverse", "--help"],
    ["vector", "rotate", "--help"],
    ["vector", "sample-points", "--help"],
    ["vector", "scale", "--help"],
    ["vector", "segmentize", "--help"],
    ["vector", "set-precision", "--help"],
    ["vector", "shared-paths", "--help"],
    ["vector", "shortest-line", "--help"],
    ["vector", "simplify", "--help"],
    ["vector", "simplify-coverage", "--help"],
    ["vector", "sjoin", "--help"],
    ["vector", "sjoin-nearest", "--help"],
    ["vector", "snap", "--help"],
    ["vector", "symmetric-difference", "--help"],
    ["vector", "to-feather", "--help"],
    ["vector", "to-geo-dict", "--help"],
    ["vector", "to-h3", "--help"],
    ["vector", "to-json", "--help"],
    ["vector", "to-mvt", "--help"],
    ["vector", "to-parquet", "--help"],
    ["vector", "to-pmtiles", "--help"],
    ["vector", "to-wkb", "--help"],
    ["vector", "to-wkt", "--help"],
    ["vector", "translate", "--help"],
    ["vector", "union", "--help"],
    ["vector", "union-all", "--help"],
    ["vector", "voronoi-polygons", "--help"],
    # issue #1468: remaining FeatureCollection-own verbs
    ["vector", "voronoi", "--help"],
    ["vector", "quadtree", "--help"],
    ["vector", "interpolate-to-raster", "--help"],
    ["vector", "from-xyz", "--help"],
    ["vector", "from-features", "--help"],
    ["vector", "from-records", "--help"],
    ["vector", "read-gpx", "--help"],
    ["vector", "with-coordinates", "--help"],
    ["vector", "with-centroid", "--help"],
    ["nc", "--help"],
    ["nc", "info", "--help"],
    ["nc", "extract", "--help"],
    ["nc", "sel", "--help"],
    ["nc", "crop", "--help"],
    ["nc", "summary", "--help"],
    ["nc", "metadata", "--help"],
    ["nc", "dim-values", "--help"],
    ["nc", "time-values", "--help"],
    ["nc", "set-attr", "--help"],
    ["nc", "del-attr", "--help"],
    ["nc", "rename-var", "--help"],
    ["nc", "remove-var", "--help"],
    ["nc", "reproject-var", "--help"],
    ["nc", "resample-var", "--help"],
    ["nc", "rename-dims", "--help"],
    ["nc", "drop-dims", "--help"],
    ["nc", "set-coords", "--help"],
    ["nc", "reset-coords", "--help"],
    ["nc", "squeeze", "--help"],
    ["nc", "isel", "--help"],
    ["nc", "thin", "--help"],
    ["nc", "head", "--help"],
    ["nc", "tail", "--help"],
    ["nc", "drop-isel", "--help"],
    ["nc", "drop-sel", "--help"],
    ["nc", "drop-duplicates", "--help"],
    ["nc", "dropna", "--help"],
    ["nc", "pad", "--help"],
    ["nc", "convert-calendar", "--help"],
    ["nc", "interp-calendar", "--help"],
    ["nc", "groupby-bins", "--help"],
    ["nc", "rolling", "--help"],
    ["nc", "rolling-exp", "--help"],
    ["nc", "shift", "--help"],
    ["nc", "integrate", "--help"],
    ["nc", "cumulative-integrate", "--help"],
    ["nc", "differentiate", "--help"],
    ["nc", "polyfit", "--help"],
    ["nc", "interp", "--help"],
    ["nc", "interpolate-na", "--help"],
    ["nc", "reduce", "--help"],
    ["nc", "convert", "--help"],
    ["nc", "to-dataframe", "--help"],
    ["nc", "to-dict", "--help"],
    ["nc", "to-zarr", "--help"],
    ["nc", "plot", "--help"],
    ["nc", "plot-histogram", "--help"],
    ["nc", "animate", "--help"],
    # issue #1467: finish the nc group — remaining NetCDF-specific verbs
    ["nc", "groups", "--help"],
    ["nc", "get-group", "--help"],
    ["nc", "add-var", "--help"],
    ["nc", "set-var", "--help"],
    ["nc", "crop-var", "--help"],
    ["nc", "expand-dims", "--help"],
    ["nc", "assign-coords", "--help"],
    ["nc", "subset", "--help"],
    ["nc", "interp-like", "--help"],
    ["nc", "coarsen", "--help"],
    ["nc", "cumulative", "--help"],
    ["nc", "weighted", "--help"],
    ["nc", "to-kerchunk", "--help"],
    ["nc", "combine-kerchunk", "--help"],
    ["nc", "open-mf", "--help"],
    ["nc", "from-dataframe", "--help"],
    ["nc", "from-dict", "--help"],
    ["nc", "concat", "--help"],
    ["nc", "merge", "--help"],
    ["collection", "--help"],
    ["collection", "info", "--help"],
    ["collection", "stats", "--help"],
    # issue #1469: full DatasetCollection surface
    ["collection", "resample", "--help"],
    ["collection", "merge", "--help"],
    ["collection", "to-file", "--help"],
    ["collection", "to-cog", "--help"],
    ["collection", "to-netcdf", "--help"],
    ["collection", "to-zarr", "--help"],
    ["collection", "to-kerchunk", "--help"],
    ["collection", "reproject", "--help"],
    ["collection", "crop", "--help"],
    ["collection", "align", "--help"],
    ["stac", "--help"],
    ["stac", "search", "--help"],
    ["fetch", "--help"],
    ["fetch", "wcs", "--help"],
    ["fetch", "wfs", "--help"],
    ["mesh", "--help"],
    ["mesh", "info", "--help"],
    ["mesh", "extract", "--help"],
    ["mesh", "convert", "--help"],
    # issue #1470: mesh group Phase 2+ — full MeshDataset surface
    ["mesh", 'active-mask', "--help"],
    ["mesh", 'animate', "--help"],
    ["mesh", 'aspect', "--help"],
    ["mesh", 'attach-data', "--help"],
    ["mesh", 'bfill', "--help"],
    ["mesh", 'binary-dilation', "--help"],
    ["mesh", 'binary-erosion', "--help"],
    ["mesh", 'bounding-polygon', "--help"],
    ["mesh", 'burn', "--help"],
    ["mesh", 'circumcenters', "--help"],
    ["mesh", 'clip', "--help"],
    ["mesh", 'clip-values', "--help"],
    ["mesh", 'cluster', "--help"],
    ["mesh", 'coarsen', "--help"],
    ["mesh", 'collapse-layer', "--help"],
    ["mesh", 'concat', "--help"],
    ["mesh", 'connected-components', "--help"],
    ["mesh", 'contour', "--help"],
    ["mesh", 'convert-calendar', "--help"],
    ["mesh", 'convert-units', "--help"],
    ["mesh", 'count-domain-cells', "--help"],
    ["mesh", 'crop', "--help"],
    ["mesh", 'cumulative-integrate', "--help"],
    ["mesh", 'del-attr', "--help"],
    ["mesh", 'differentiate', "--help"],
    ["mesh", 'domain-area', "--help"],
    ["mesh", 'drop-duplicates', "--help"],
    ["mesh", 'drop-isel', "--help"],
    ["mesh", 'drop-sel', "--help"],
    ["mesh", 'drop-variables', "--help"],
    ["mesh", 'dropna', "--help"],
    ["mesh", 'exterior-edges', "--help"],
    ["mesh", 'exterior-faces', "--help"],
    ["mesh", 'ffill', "--help"],
    ["mesh", 'fill-na-spatial', "--help"],
    ["mesh", 'fillna', "--help"],
    ["mesh", 'focal-mean', "--help"],
    ["mesh", 'focal-std', "--help"],
    ["mesh", 'footprint', "--help"],
    ["mesh", 'from-arrays', "--help"],
    ["mesh", 'from-bytes', "--help"],
    ["mesh", 'from-geodataframe', "--help"],
    ["mesh", 'from-structured', "--help"],
    ["mesh", 'from-zarr', "--help"],
    ["mesh", 'get', "--help"],
    ["mesh", 'head', "--help"],
    ["mesh", 'hillshade', "--help"],
    ["mesh", 'histogram', "--help"],
    ["mesh", 'integrate', "--help"],
    ["mesh", 'interp-calendar', "--help"],
    ["mesh", 'interp-like', "--help"],
    ["mesh", 'interp-time', "--help"],
    ["mesh", 'interpolate-na', "--help"],
    ["mesh", 'isel', "--help"],
    ["mesh", 'isel-layer', "--help"],
    ["mesh", 'items', "--help"],
    ["mesh", 'keys', "--help"],
    ["mesh", 'laplace-interpolate', "--help"],
    ["mesh", 'masked-values', "--help"],
    ["mesh", 'merge', "--help"],
    ["mesh", 'mesh-names', "--help"],
    ["mesh", 'open-mf', "--help"],
    ["mesh", 'plot', "--help"],
    ["mesh", 'plot-histogram', "--help"],
    ["mesh", 'plot-outline', "--help"],
    ["mesh", 'plot-vector-field', "--help"],
    ["mesh", 'polygonize', "--help"],
    ["mesh", 'proximity', "--help"],
    ["mesh", 'read-all', "--help"],
    ["mesh", 'regrid', "--help"],
    ["mesh", 'rename-variable', "--help"],
    ["mesh", 'resample', "--help"],
    ["mesh", 'rolling', "--help"],
    ["mesh", 'sample', "--help"],
    ["mesh", 'sample-line', "--help"],
    ["mesh", 'sample-point', "--help"],
    ["mesh", 'sample-random', "--help"],
    ["mesh", 'sel', "--help"],
    ["mesh", 'sel-layer', "--help"],
    ["mesh", 'sel-time-range', "--help"],
    ["mesh", 'set-attr', "--help"],
    ["mesh", 'set-crs', "--help"],
    ["mesh", 'shift', "--help"],
    ["mesh", 'sieve', "--help"],
    ["mesh", 'slope', "--help"],
    ["mesh", 'sparse-adjacency', "--help"],
    ["mesh", 'squeeze', "--help"],
    ["mesh", 'subset-by-bounds', "--help"],
    ["mesh", 'summary', "--help"],
    ["mesh", 'tail', "--help"],
    ["mesh", 'thin', "--help"],
    ["mesh", 'to-bytes', "--help"],
    ["mesh", 'to-crs', "--help"],
    ["mesh", 'to-dataframe', "--help"],
    ["mesh", 'to-dataset', "--help"],
    ["mesh", 'to-file', "--help"],
    ["mesh", 'to-geodataframe', "--help"],
    ["mesh", 'to-image', "--help"],
    ["mesh", 'to-location', "--help"],
    ["mesh", 'to-zarr', "--help"],
    ["mesh", 'values', "--help"],
    ["mesh", 'where', "--help"],
    ["mesh", 'with-variable', "--help"],
    ["mesh", 'zonal-stats', "--help"],
]


class TestHelpSurfaceAll:
    """Every subcommand exposes --help and exits 0 (T-1)."""

    @pytest.mark.parametrize("argv", _HELP_ARGVS)
    def test_help_exits_zero(self, argv, capsys):
        """Each subcommand (and subgroup) prints usage and exits 0."""
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 0, f"{argv} --help must exit 0"
        assert "usage" in capsys.readouterr().out.lower()


class TestQuiet:
    """Global --quiet flag (E-4)."""

    def test_quiet_suppresses_wrote_line(self, cli_raster, tmp_path, capsys):
        """--quiet suppresses the 'wrote' chatter but still writes the output."""
        out = str(tmp_path / "q.tif")
        assert (
            main(["--quiet", "raster", "warp", cli_raster, out, "--crs", "3857"]) == 0
        )
        assert capsys.readouterr().out == "", "no stdout under --quiet"
        assert os.path.exists(out)


class TestResamplingVocab:
    """A shared resampling vocabulary across warp and overview (E-3)."""

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average"])
    def test_name_accepted_by_warp_and_overview(self, cli_raster, tmp_path, method):
        """The same name works on both warp and overview."""
        assert (
            main(
                [
                    "raster",
                    "warp",
                    cli_raster,
                    str(tmp_path / f"w_{method}.tif"),
                    "--crs",
                    "3857",
                    "--resampling",
                    method,
                ]
            )
            == 0
        )
        assert (
            main(
                [
                    "raster",
                    "overview",
                    cli_raster,
                    "--levels",
                    "2",
                    "--resampling",
                    method,
                ]
            )
            == 0
        )

    def test_cubic_spline_alias_on_overview(self, cli_raster):
        """The warp spelling 'cubic_spline' maps to the overview 'cubicspline'."""
        assert (
            main(
                [
                    "raster",
                    "overview",
                    cli_raster,
                    "--levels",
                    "2",
                    "--resampling",
                    "cubic_spline",
                ]
            )
            == 0
        )


class TestCalcNote:
    """calc CRS-note ordering (B-1)."""

    def test_untagged_companion_grid_mismatch_has_no_note(self, tmp_path, capsys):
        """A CRS-less companion on a different grid raises without the 'assuming' note."""
        a = str(tmp_path / "a.tif")
        Dataset.from_array(
            np.full((2, 2), 4.0, "float32"),
            geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
        ).to_file(a)
        b = str(tmp_path / "b_nocrs.tif")
        out = gdal.GetDriverByName("GTiff").Create(b, 2, 2, 1, gdal.GDT_Float32)
        out.SetGeoTransform(
            (50.0, 1.0, 0.0, 2.0, 0.0, -1.0)
        )  # different origin, no CRS
        out.GetRasterBand(1).WriteArray(np.full((2, 2), 2.0, "float32"))
        out.FlushCache()
        out = None
        res = str(tmp_path / "diff.tif")

        rc = main(["raster", "calc", "A - B", a, b, res])
        err = capsys.readouterr().err

        assert rc == 1, "a grid mismatch must exit 1"
        assert "does not share the grid" in err
        assert "assuming it is on" not in err


class TestEditInfoNodata:
    """edit-info --nodata tokens (B-3)."""

    def test_unset_removes_sentinel(self, cli_raster):
        """--nodata none clears the no-data value."""
        assert main(["raster", "edit-info", cli_raster, "--nodata", "none"]) == 0
        assert Dataset.read_file(cli_raster).no_data_value[0] is None

    def test_nan_token(self, cli_raster):
        """--nodata nan sets a NaN sentinel."""
        assert main(["raster", "edit-info", cli_raster, "--nodata", "nan"]) == 0
        assert math.isnan(Dataset.read_file(cli_raster).no_data_value[0])

    def test_numeric_still_works(self, cli_raster):
        """A numeric --nodata still sets the value."""
        assert main(["raster", "edit-info", cli_raster, "--nodata", "0"]) == 0
        assert Dataset.read_file(cli_raster).no_data_value[0] == 0

    def test_bad_token_errors(self, cli_raster, capsys):
        """A non-numeric, non-token --nodata exits 1 cleanly."""
        assert main(["raster", "edit-info", cli_raster, "--nodata", "banana"]) == 1
        assert "not a number" in capsys.readouterr().err


class TestEditInfoDriverGuard:
    """edit-info on a non-updatable driver (B-4)."""

    def test_non_updatable_driver_clear_message(self, cli_raster, monkeypatch, capsys):
        """An update-only open failure yields the CLI-level message (read-only works)."""

        def fake(_path, *args, read_only=True, **kwargs):
            if not read_only:
                raise RuntimeError("driver X does not support update access")
            return object()  # a read-only open succeeds -> genuine update-mode limit

        monkeypatch.setattr(cli_dataset.Dataset, "read_file", staticmethod(fake))
        rc = main(["raster", "edit-info", cli_raster, "--nodata", "0"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "update-mode" in err
        assert "pyramids convert" in err

    def test_corrupt_file_keeps_original_error(self, cli_raster, monkeypatch, capsys):
        """When a read-only open also fails, the original error is not mislabeled (L1)."""

        def boom(_path, *args, **kwargs):
            raise RuntimeError("TIFFReadDirectory: Failed to read directory")

        monkeypatch.setattr(cli_dataset.Dataset, "read_file", staticmethod(boom))
        rc = main(["raster", "edit-info", cli_raster, "--nodata", "0"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "Failed to read directory" in err, "the real error must surface"
        assert "update-mode" not in err, "a corrupt file must not be mislabeled"


class TestCogJson:
    """`cog create --json` / `cog validate --json` (E-4)."""

    def test_cog_create_json_result(self, cli_raster, tmp_path, capsys):
        """cog create --json emits a parseable result with the validity flag."""
        out = str(tmp_path / "out.cog.tif")
        rc = main(["raster", "cog", "create", cli_raster, out, "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0, "a valid COG must exit 0"
        assert payload["is_valid"] is True
        assert payload["output"].endswith(".cog.tif")

    def test_cog_validate_json_result(self, cli_raster, tmp_path, capsys):
        """cog validate --json emits is_valid / errors / warnings as JSON."""
        out = str(tmp_path / "out.cog.tif")
        assert main(["raster", "cog", "create", cli_raster, out]) == 0
        capsys.readouterr()
        rc = main(["raster", "cog", "validate", out, "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["is_valid"] is True
        assert payload["errors"] == []


@pytest.mark.plot
class TestRender:
    """`pyramids render` quick-look PNG (M-7)."""

    def test_writes_png(self, cli_raster, tmp_path):
        """render saves a non-empty PNG headlessly."""
        out = str(tmp_path / "ql.png")
        assert main(["raster", "render", cli_raster, out]) == 0
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_refuses_existing(self, cli_raster, tmp_path):
        """render refuses an existing output without --overwrite."""
        out = str(tmp_path / "exists.png")
        main(["raster", "render", cli_raster, out])
        assert main(["raster", "render", cli_raster, out]) == 1

    def test_forces_agg_backend(self, cli_raster, tmp_path, monkeypatch):
        """render forces MPLBACKEND=Agg even when an interactive one is pre-set (L3).

        It both forces the env var over a pre-set TkAgg and still renders headlessly
        (a non-empty PNG), so the assertion is not satisfied by the env write alone.
        """
        monkeypatch.setenv("MPLBACKEND", "TkAgg")
        out = str(tmp_path / "forced.png")
        assert main(["raster", "render", cli_raster, out]) == 0
        assert os.environ["MPLBACKEND"] == "Agg", (
            "a pre-exported interactive backend must not win over the forced Agg"
        )
        assert os.path.exists(out), "render must still produce the PNG headlessly"
        assert os.path.getsize(out) > 0, "the rendered PNG must not be empty"
