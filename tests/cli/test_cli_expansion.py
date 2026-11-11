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
        """`pyramids --help` names the top-level command groups."""
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for token in (
            "raster",
            "vector",
            "nc",
            "collection",
            "stac",
            "fetch",
            "mesh",
            "process",
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
    ["raster", "crop", "--help"],
    ["raster", "warp", "--help"],
    ["raster", "merge", "--help"],
    ["raster", "overview", "create", "--help"],
    ["raster", "overview", "recreate", "--help"],
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
    ["raster", "get-metadata", "--help"],
    ["raster", "advanced", "get-attribute-table", "--help"],
    ["raster", "advanced", "set-attribute-table", "--help"],
    ["raster", "advanced", "mask-flags", "--help"],
    ["raster", "footprint", "--help"],
    ["raster", "cell-area", "--help"],
    ["raster", "domain-area", "--help"],
    ["raster", "count-domain-cells", "--help"],
    ["raster", "advanced", "same-grid", "--help"],
    # issue #1436: bands / georef edit
    ["raster", "add-band", "--help"],
    ["raster", "select-bands", "--help"],
    ["raster", "advanced", "get-band-by-color", "--help"],
    ["raster", "advanced", "create-mask-band", "--help"],
    ["raster", "advanced", "read-masks", "--help"],
    ["raster", "advanced", "get-mask", "--help"],
    ["raster", "set-color-ramp", "--help"],
    ["raster", "advanced", "set-rpcs", "--help"],
    ["raster", "advanced", "geolocate", "--help"],
    # issue #1436: spatial ops
    ["raster", "align", "--help"],
    ["raster", "combine", "--help"],
    ["raster", "advanced", "warped-view", "--help"],
    ["raster", "advanced", "wrap-longitude", "--help"],
    ["raster", "advanced", "translate", "--help"],
    ["raster", "advanced", "transect", "--help"],
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
    ["raster", "advanced", "read-overview-array", "--help"],
    ["raster", "advanced", "read-part", "--help"],
    ["raster", "advanced", "preview", "--help"],
    ["raster", "advanced", "point", "--help"],
    ["raster", "advanced", "read-tile", "--help"],
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
    ["raster", "advanced", "checksum", "--help"],
    ["raster", "subdataset", "--help"],
    ["vector", "--help"],
    ["vector", "info", "--help"],
    ["vector", "warp", "--help"],
    ["vector", "clip", "--help"],
    ["vector", "advanced", "affine-transform", "--help"],
    ["vector", "bounds", "--help"],
    ["vector", "geom", "buffer", "--help"],
    ["vector", "advanced", "build-area", "--help"],
    ["vector", "overlay", "clip-by-rect", "--help"],
    ["vector", "geom", "concave-hull", "--help"],
    ["vector", "tess", "constrained-delaunay-triangles", "--help"],
    ["vector", "convert", "--help"],
    ["vector", "inspect", "count-coordinates", "--help"],
    ["vector", "inspect", "count-geometries", "--help"],
    ["vector", "inspect", "count-interior-rings", "--help"],
    ["vector", "tess", "delaunay-triangles", "--help"],
    ["vector", "overlay", "difference", "--help"],
    ["vector", "overlay", "dissolve", "--help"],
    ["vector", "tess", "distance", "--help"],
    ["vector", "inspect", "estimate-utm", "--help"],
    ["vector", "overlay", "explode", "--help"],
    ["vector", "advanced", "extract-unique-points", "--help"],
    ["vector", "inspect", "feature-count", "--help"],
    ["vector", "inspect", "feature-info", "--help"],
    ["vector", "overlay", "filter", "--help"],
    ["vector", "tess", "fishnet", "--help"],
    ["vector", "advanced", "force-2d", "--help"],
    ["vector", "advanced", "force-3d", "--help"],
    ["vector", "advanced", "frechet-distance", "--help"],
    ["vector", "io", "from-bbox", "--help"],
    ["vector", "io", "from-featureserver", "--help"],
    ["vector", "io", "from-ogc-features", "--help"],
    ["vector", "io", "from-vectortileserver", "--help"],
    ["vector", "tess", "geodesic-area", "--help"],
    ["vector", "tess", "geodesic-distance", "--help"],
    ["vector", "tess", "geodesic-length", "--help"],
    ["vector", "tess", "get-coordinates", "--help"],
    ["vector", "tess", "h3-bin", "--help"],
    ["vector", "advanced", "hausdorff-distance", "--help"],
    ["vector", "advanced", "hilbert-distance", "--help"],
    ["vector", "overlay", "intersection", "--help"],
    ["vector", "overlay", "intersection-all", "--help"],
    ["vector", "inspect", "invalid-coverage-edges", "--help"],
    ["vector", "inspect", "is-valid-coverage", "--help"],
    ["vector", "inspect", "is-valid-reason", "--help"],
    ["vector", "inspect", "list-layers", "--help"],
    ["vector", "geom", "make-valid", "--help"],
    ["vector", "advanced", "maximum-inscribed-circle", "--help"],
    ["vector", "geom", "minimum-bounding-circle", "--help"],
    ["vector", "advanced", "minimum-clearance", "--help"],
    ["vector", "advanced", "minimum-rotated-rectangle", "--help"],
    ["vector", "advanced", "normalize", "--help"],
    ["vector", "advanced", "offset-curve", "--help"],
    ["vector", "overlay", "apply", "--help"],
    ["vector", "tess", "polygonize", "--help"],
    ["vector", "io", "read-parquet", "--help"],
    ["vector", "advanced", "remove-repeated-points", "--help"],
    ["vector", "advanced", "representative-point", "--help"],
    ["vector", "advanced", "reverse", "--help"],
    ["vector", "geom", "rotate", "--help"],
    ["vector", "tess", "sample-points", "--help"],
    ["vector", "geom", "scale", "--help"],
    ["vector", "advanced", "segmentize", "--help"],
    ["vector", "advanced", "set-precision", "--help"],
    ["vector", "advanced", "shared-paths", "--help"],
    ["vector", "advanced", "shortest-line", "--help"],
    ["vector", "geom", "simplify", "--help"],
    ["vector", "advanced", "simplify-coverage", "--help"],
    ["vector", "overlay", "sjoin", "--help"],
    ["vector", "overlay", "sjoin-nearest", "--help"],
    ["vector", "advanced", "snap", "--help"],
    ["vector", "overlay", "symmetric-difference", "--help"],
    ["vector", "io", "to-feather", "--help"],
    ["vector", "io", "to-geo-dict", "--help"],
    ["vector", "tess", "to-h3", "--help"],
    ["vector", "io", "to-json", "--help"],
    ["vector", "io", "to-mvt", "--help"],
    ["vector", "io", "to-parquet", "--help"],
    ["vector", "io", "to-pmtiles", "--help"],
    ["vector", "io", "to-wkb", "--help"],
    ["vector", "io", "to-wkt", "--help"],
    ["vector", "geom", "translate", "--help"],
    ["vector", "overlay", "union", "--help"],
    ["vector", "overlay", "union-all", "--help"],
    ["vector", "tess", "voronoi-polygons", "--help"],
    # issue #1468: remaining FeatureCollection-own verbs
    ["vector", "tess", "voronoi", "--help"],
    ["vector", "tess", "quadtree", "--help"],
    ["vector", "tess", "interpolate-to-raster", "--help"],
    ["vector", "io", "from-xyz", "--help"],
    ["vector", "io", "from-features", "--help"],
    ["vector", "io", "from-records", "--help"],
    ["vector", "io", "read-gpx", "--help"],
    ["vector", "advanced", "with-coordinates", "--help"],
    ["vector", "advanced", "with-centroid", "--help"],
    ["nc", "--help"],
    ["nc", "info", "--help"],
    ["nc", "extract", "--help"],
    ["nc", "sel", "--help"],
    ["nc", "crop", "--help"],
    ["nc", "inspect", "summary", "--help"],
    ["nc", "inspect", "metadata", "--help"],
    ["nc", "inspect", "dim-values", "--help"],
    ["nc", "inspect", "time-values", "--help"],
    ["nc", "inspect", "set-attr", "--help"],
    ["nc", "inspect", "del-attr", "--help"],
    ["nc", "vars", "rename-var", "--help"],
    ["nc", "vars", "remove-var", "--help"],
    ["nc", "vars", "reproject-var", "--help"],
    ["nc", "vars", "resample-var", "--help"],
    ["nc", "advanced", "rename-dims", "--help"],
    ["nc", "advanced", "drop-dims", "--help"],
    ["nc", "advanced", "set-coords", "--help"],
    ["nc", "advanced", "reset-coords", "--help"],
    ["nc", "advanced", "squeeze", "--help"],
    ["nc", "select", "isel", "--help"],
    ["nc", "select", "thin", "--help"],
    ["nc", "select", "head", "--help"],
    ["nc", "select", "tail", "--help"],
    ["nc", "advanced", "drop-isel", "--help"],
    ["nc", "advanced", "drop-sel", "--help"],
    ["nc", "advanced", "drop-duplicates", "--help"],
    ["nc", "select", "dropna", "--help"],
    ["nc", "advanced", "pad", "--help"],
    ["nc", "calc", "convert-calendar", "--help"],
    ["nc", "calc", "interp-calendar", "--help"],
    ["nc", "advanced", "groupby-bins", "--help"],
    ["nc", "advanced", "rolling", "--help"],
    ["nc", "advanced", "rolling-exp", "--help"],
    ["nc", "advanced", "shift", "--help"],
    ["nc", "advanced", "integrate", "--help"],
    ["nc", "advanced", "cumulative-integrate", "--help"],
    ["nc", "advanced", "differentiate", "--help"],
    ["nc", "calc", "polyfit", "--help"],
    ["nc", "calc", "interp", "--help"],
    ["nc", "calc", "interpolate-na", "--help"],
    ["nc", "calc", "reduce", "--help"],
    ["nc", "export", "convert", "--help"],
    ["nc", "export", "dataframe", "--help"],
    ["nc", "export", "dict", "--help"],
    ["nc", "export", "zarr", "--help"],
    ["nc", "plot", "render", "--help"],
    ["nc", "plot", "plot-histogram", "--help"],
    ["nc", "plot", "animate", "--help"],
    # issue #1467: finish the nc group — remaining NetCDF-specific verbs
    ["nc", "groups", "--help"],
    ["nc", "get-group", "--help"],
    ["nc", "vars", "add-var", "--help"],
    ["nc", "vars", "set-var", "--help"],
    ["nc", "vars", "crop-var", "--help"],
    ["nc", "advanced", "expand-dims", "--help"],
    ["nc", "advanced", "assign-coords", "--help"],
    ["nc", "select", "subset", "--help"],
    ["nc", "advanced", "interp-like", "--help"],
    ["nc", "advanced", "coarsen", "--help"],
    ["nc", "advanced", "cumulative", "--help"],
    ["nc", "advanced", "weighted", "--help"],
    ["nc", "export", "kerchunk", "--help"],
    ["nc", "export", "combine-kerchunk", "--help"],
    ["nc", "import", "open-mf", "--help"],
    ["nc", "import", "dataframe", "--help"],
    ["nc", "import", "dict", "--help"],
    ["nc", "import", "concat", "--help"],
    ["nc", "import", "merge", "--help"],
    ["collection", "--help"],
    ["collection", "info", "--help"],
    ["collection", "reduce", "--help"],
    # issue #1469: full DatasetCollection surface
    ["collection", "resample", "--help"],
    ["collection", "merge", "--help"],
    ["collection", "export", "file", "--help"],
    ["collection", "export", "cog", "--help"],
    ["collection", "export", "netcdf", "--help"],
    ["collection", "export", "zarr", "--help"],
    ["collection", "export", "kerchunk", "--help"],
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
    ["mesh", "advanced", "active-mask", "--help"],
    ["mesh", "plot", "animate", "--help"],
    ["mesh", "analysis", "aspect", "--help"],
    ["mesh", "export", "attach-data", "--help"],
    ["mesh", "advanced", "bfill", "--help"],
    ["mesh", "analysis", "binary-dilation", "--help"],
    ["mesh", "analysis", "binary-erosion", "--help"],
    ["mesh", "inspect", "bounding-polygon", "--help"],
    ["mesh", "export", "burn", "--help"],
    ["mesh", "inspect", "circumcenters", "--help"],
    ["mesh", "select", "clip", "--help"],
    ["mesh", "select", "clip-values", "--help"],
    ["mesh", "analysis", "cluster", "--help"],
    ["mesh", "advanced", "coarsen", "--help"],
    ["mesh", "advanced", "collapse-layer", "--help"],
    ["mesh", "join", "concat", "--help"],
    ["mesh", "advanced", "connected-components", "--help"],
    ["mesh", "analysis", "contour", "--help"],
    ["mesh", "interp", "convert-calendar", "--help"],
    ["mesh", "interp", "convert-units", "--help"],
    ["mesh", "analysis", "count-domain-cells", "--help"],
    ["mesh", "select", "crop", "--help"],
    ["mesh", "advanced", "cumulative-integrate", "--help"],
    ["mesh", "attrs", "del-attr", "--help"],
    ["mesh", "advanced", "differentiate", "--help"],
    ["mesh", "analysis", "domain-area", "--help"],
    ["mesh", "advanced", "drop-duplicates", "--help"],
    ["mesh", "advanced", "drop-isel", "--help"],
    ["mesh", "advanced", "drop-sel", "--help"],
    ["mesh", "advanced", "drop-variables", "--help"],
    ["mesh", "advanced", "dropna", "--help"],
    ["mesh", "inspect", "exterior-edges", "--help"],
    ["mesh", "inspect", "exterior-faces", "--help"],
    ["mesh", "advanced", "ffill", "--help"],
    ["mesh", "interp", "fill-na-spatial", "--help"],
    ["mesh", "interp", "fillna", "--help"],
    ["mesh", "analysis", "focal-mean", "--help"],
    ["mesh", "analysis", "focal-std", "--help"],
    ["mesh", "analysis", "footprint", "--help"],
    ["mesh", "read", "arrays", "--help"],
    ["mesh", "read", "bytes", "--help"],
    ["mesh", "read", "geodataframe", "--help"],
    ["mesh", "read", "structured", "--help"],
    ["mesh", "read", "zarr", "--help"],
    ["mesh", "advanced", "get", "--help"],
    ["mesh", "select", "head", "--help"],
    ["mesh", "analysis", "hillshade", "--help"],
    ["mesh", "analysis", "histogram", "--help"],
    ["mesh", "advanced", "integrate", "--help"],
    ["mesh", "interp", "interp-calendar", "--help"],
    ["mesh", "interp", "interp-like", "--help"],
    ["mesh", "interp", "interp-time", "--help"],
    ["mesh", "interp", "interpolate-na", "--help"],
    ["mesh", "select", "isel", "--help"],
    ["mesh", "select", "isel-layer", "--help"],
    ["mesh", "advanced", "items", "--help"],
    ["mesh", "advanced", "keys", "--help"],
    ["mesh", "interp", "laplace-interpolate", "--help"],
    ["mesh", "advanced", "masked-values", "--help"],
    ["mesh", "join", "merge", "--help"],
    ["mesh", "inspect", "mesh-names", "--help"],
    ["mesh", "read", "open-mf", "--help"],
    ["mesh", "plot", "plot", "--help"],
    ["mesh", "plot", "plot-histogram", "--help"],
    ["mesh", "plot", "plot-outline", "--help"],
    ["mesh", "plot", "plot-vector-field", "--help"],
    ["mesh", "export", "polygonize", "--help"],
    ["mesh", "analysis", "proximity", "--help"],
    ["mesh", "read", "read-all", "--help"],
    ["mesh", "export", "regrid", "--help"],
    ["mesh", "advanced", "rename-variable", "--help"],
    ["mesh", "interp", "resample", "--help"],
    ["mesh", "advanced", "rolling", "--help"],
    ["mesh", "analysis", "sample", "--help"],
    ["mesh", "analysis", "sample-line", "--help"],
    ["mesh", "analysis", "sample-point", "--help"],
    ["mesh", "analysis", "sample-random", "--help"],
    ["mesh", "select", "sel", "--help"],
    ["mesh", "select", "sel-layer", "--help"],
    ["mesh", "select", "isel-time-range", "--help"],
    ["mesh", "attrs", "set-attr", "--help"],
    ["mesh", "advanced", "set-crs", "--help"],
    ["mesh", "advanced", "shift", "--help"],
    ["mesh", "analysis", "sieve", "--help"],
    ["mesh", "analysis", "slope", "--help"],
    ["mesh", "advanced", "sparse-adjacency", "--help"],
    ["mesh", "advanced", "squeeze", "--help"],
    ["mesh", "select", "subset-by-bounds", "--help"],
    ["mesh", "inspect", "summary", "--help"],
    ["mesh", "select", "tail", "--help"],
    ["mesh", "advanced", "thin", "--help"],
    ["mesh", "export", "bytes", "--help"],
    ["mesh", "export", "crs", "--help"],
    ["mesh", "export", "dataframe", "--help"],
    ["mesh", "export", "dataset", "--help"],
    ["mesh", "export", "file", "--help"],
    ["mesh", "export", "geodataframe", "--help"],
    ["mesh", "export", "image", "--help"],
    ["mesh", "export", "location", "--help"],
    ["mesh", "export", "zarr", "--help"],
    ["mesh", "advanced", "values", "--help"],
    ["mesh", "select", "where", "--help"],
    ["mesh", "advanced", "with-variable", "--help"],
    ["mesh", "analysis", "zonal-stats", "--help"],
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
                    "create",
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
                    "create",
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
        assert "pyramids raster convert" in err

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
