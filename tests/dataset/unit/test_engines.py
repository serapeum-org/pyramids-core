"""Tests for :mod:`pyramids.dataset.engines`.

These tests pin down the Stage 1 contract documented in the module's
docstring and in the L-2 plan:

* The seven collaborator classes are accessible as attributes on a
  ``Dataset`` (``ds.io``, ``ds.spatial``, ``ds.bands``, ``ds.analysis``,
  ``ds.cell``, ``ds.vector``, ``ds.cog``).
* Every public method on every collaborator is a forwarder — calling
  ``ds.<collab>.<method>(...)`` invokes ``ds.<method>(...)`` with the same
  positional and keyword arguments and returns the same value.
* Read-only and read/write properties on the collaborators forward in
  both directions.
* The back-reference is a :class:`weakref.proxy` so the parent ``Dataset``
  can be garbage-collected while a collaborator instance is still
  referenced (otherwise the GDAL handle leaks and Windows file-unlink
  fails in tests).
* ``Dataset`` survives a pickle round-trip (covered more broadly in
  :mod:`tests.dataset.test_pickle`); the round-tripped instance carries
  fresh collaborator instances of the right type.
* Pickling a collaborator *directly* yields a ``_Placeholder`` on
  unpickle rather than crashing or producing a circular pickle through
  the parent ``Dataset``.
* :meth:`Analysis.normalize` (the only collaborator method with a real
  body in Stage 1) min-max scales arrays into the [0, 1] range.
"""

from __future__ import annotations

import gc
import pickle
import warnings
import weakref

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base.crs import sr_from_epsg
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines import (
    _COG,
    _IO,
    _Analysis,
    _Bands,
    _Cell,
    _Spatial,
    _Vector,
)
from pyramids.dataset.engines._base import (
    _Engine,
    _Placeholder,
    _recreate_placeholder,
)
from pyramids.dataset.engines.bands import _is_read_only_error
from pyramids.dataset.engines.cog import _cached_transformer


@pytest.fixture
def in_memory_dataset() -> Dataset:
    """Build a small in-memory ``Dataset`` for collaborator method tests.

    Returns:
        Dataset: A 4x4 float32 dataset in EPSG:4326 backed by GDAL's MEM
        driver. Suitable for any test that does not need to round-trip
        through pickle (in-memory datasets cannot pickle by design).
    """
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return Dataset.from_array(
        arr=arr,
        geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
    )


@pytest.fixture
def file_backed_dataset(tmp_path) -> Dataset:
    """Write a tiny GeoTIFF to ``tmp_path`` and return it as a ``Dataset``.

    Args:
        tmp_path: pytest-provided per-test temporary directory.

    Returns:
        Dataset: A file-backed dataset suitable for pickle round-trips
        (``RasterBase.__reduce__`` re-opens via ``cls.read_file(path)``,
        which only works for paths that exist on disk).
    """
    path = str(tmp_path / "tiny.tif")
    drv = gdal.GetDriverByName("GTiff")
    raster = drv.Create(path, 3, 4, 1, gdal.GDT_Float32)
    raster.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    raster.SetProjection(srs.ExportToWkt())
    raster.GetRasterBand(1).WriteArray(np.arange(12, dtype=np.float32).reshape(4, 3))
    raster.FlushCache()
    raster = None
    return Dataset.read_file(path)


class TestPlaceholder:
    """Tests for :class:`_Placeholder` and :func:`_recreate_placeholder`."""

    def test_recreate_placeholder_returns_placeholder(self):
        """The factory must return a fresh ``_Placeholder`` instance.

        Test scenario:
            ``_recreate_placeholder()`` is the unpickle target referenced
            from ``_Engine.__reduce__``; calling it directly should
            yield a usable placeholder.
        """
        result = _recreate_placeholder()
        assert isinstance(result, _Placeholder), (
            f"Expected _Placeholder instance, got {type(result).__name__}"
        )

    def test_each_call_returns_distinct_instance(self):
        """Successive calls must produce distinct objects, not a singleton.

        Test scenario:
            Two separate calls to the factory should yield two distinct
            placeholders so a buggy callsite cannot accidentally share
            state across collaborator instances.
        """
        first = _recreate_placeholder()
        second = _recreate_placeholder()
        assert first is not second, (
            "Successive calls to _recreate_placeholder returned the same instance"
        )


class TestCollaboratorBase:
    """Tests for :class:`_Engine` (base class shared by all collaborators)."""

    def test_init_stores_weak_proxy_to_dataset(self, in_memory_dataset):
        """Constructor must wrap the Dataset in a ``weakref.proxy``.

        Test scenario:
            After construction, attribute access through ``self._ds`` should
            transparently resolve to the wrapped Dataset's attributes.
        """
        collab = _Engine(in_memory_dataset)
        assert collab._ds.epsg == in_memory_dataset.epsg, (
            f"Proxy did not resolve epsg: {collab._ds.epsg} != {in_memory_dataset.epsg}"
        )
        assert collab._ds.rows == in_memory_dataset.rows, (
            "Proxy did not resolve rows attribute"
        )

    def test_a_dataset_with_no_external_refs_is_collected(self):
        """No stored ds<->engine cycle: the engine accessors are properties, so a Dataset
        with no external reference (and no engine held) is collected promptly, releasing its
        GDAL handle -- which keeps Windows file-unlink reliable.
        """
        arr = np.zeros((4, 4), dtype=np.float32)
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
        )
        ref = weakref.ref(ds)
        del ds
        gc.collect()
        assert ref() is None, (
            "Dataset must be collected once no external reference remains; a stored engine "
            "would create a cycle and delay GDAL handle release"
        )

    def test_a_held_engine_keeps_its_dataset_alive(self):
        """The engine holds a strong back-reference, so a chain on a temporary
        (``Dataset.read_file(p).io.read_array()``) stays valid for the call; the Dataset is
        released once the engine itself is dropped.
        """
        arr = np.zeros((4, 4), dtype=np.float32)
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
        )
        ref = weakref.ref(ds)
        engine = ds.io
        del ds
        gc.collect()
        assert ref() is not None, "a held engine keeps its Dataset alive for the call"
        assert engine._ds.epsg == 4326
        del engine
        gc.collect()
        assert ref() is None, "dropping the engine releases the Dataset"

    def test_reduce_returns_placeholder_recipe(self, in_memory_dataset):
        """``__reduce__`` must return ``(_recreate_placeholder, ())``.

        Test scenario:
            Direct collaborator pickle should not attempt to serialize the
            parent Dataset — it should reduce to the placeholder factory
            with no arguments so unpickle yields a ``_Placeholder``.
        """
        collab = _Engine(in_memory_dataset)
        recipe = collab.__reduce__()
        assert recipe == (
            _recreate_placeholder,
            (),
        ), f"Unexpected reduce recipe: {recipe}"

    def test_slots_only_contains_ds(self):
        """``__slots__`` must be exactly ``('_ds',)``.

        Test scenario:
            The slots declaration prevents accidental attribute attachment
            (e.g., a future bug that sets ``self.foo = bar`` on a
            collaborator) and keeps the back-reference the only state.
        """
        assert _Engine.__slots__ == ("_ds",), (
            f"Expected __slots__ == ('_ds',), got {_Engine.__slots__!r}"
        )


class TestCollaboratorAttachment:
    """Tests that ``Dataset.__init__`` attaches every collaborator."""

    @pytest.mark.parametrize(
        "attr_name, expected_type",
        [
            ("io", _IO),
            ("spatial", _Spatial),
            ("bands", _Bands),
            ("analysis", _Analysis),
            ("cell", _Cell),
            ("vector", _Vector),
            ("cog", _COG),
        ],
    )
    def test_dataset_exposes_collaborator(
        self, in_memory_dataset, attr_name, expected_type
    ):
        """Each collaborator must be accessible as a Dataset attribute.

        Args:
            attr_name: The attribute name on the Dataset (e.g. ``"io"``).
            expected_type: The collaborator class that attribute should hold.

        Test scenario:
            After ``Dataset.__init__`` runs, the seven collaborators are
            wired in and reachable by attribute access.
        """
        collab = getattr(in_memory_dataset, attr_name)
        assert isinstance(collab, expected_type), (
            f"ds.{attr_name} should be {expected_type.__name__}, got {type(collab).__name__}"
        )


# Stage 2 facades: Dataset method delegates to the collaborator method
# (the mixin has been removed from Dataset's MRO). PR 2.1 — cell, PR 2.2 —
# cog, PR 2.3 — vector, PR 2.4 — analysis, PR 2.5 — spatial.
FACADE_METHODS = [
    ("bands", "get_attribute_table"),
    ("bands", "set_attribute_table"),
    ("bands", "add_band"),
    ("bands", "get_band_by_color"),
    ("bands", "change_no_data_value"),
    ("io", "read_array"),
    ("io", "write_array"),
    ("io", "to_file"),
    # NB: ``("io", "to_raster")`` is intentionally excluded — ``Dataset.to_raster`` is
    # now a deprecated alias that warns and delegates to ``io.to_file`` (NOT ``io.to_raster``),
    # so it is no longer a pure same-named pass-through facade. Its deprecation + delegation
    # behaviour is covered by ``tests/dataset/test_dataset_naming_deprecations.py``.
    ("io", "get_block_arrangement"),
    ("io", "get_tile"),
    ("io", "map_blocks"),
    ("io", "to_xyz"),
    ("io", "create_overviews"),
    ("io", "recreate_overviews"),
    ("io", "get_overview"),
    ("io", "get_overview_dataset"),
    ("io", "read_overview_array"),
    ("spatial", "crop"),
    ("spatial", "to_crs"),
    ("spatial", "set_crs"),
    ("spatial", "wrap_longitude"),
    ("spatial", "resample"),
    ("spatial", "align"),
    ("spatial", "fill_gaps"),
    ("cell", "cell_area"),
    ("cell", "get_cell_coords"),
    ("cell", "nearest_cell"),
    ("cog", "to_cog"),
    ("cog", "validate_cog"),
    ("vector", "to_geodataframe"),
    ("vector", "translate"),
    ("vector", "cluster"),
    ("vector", "to_polygons"),
    ("analysis", "stats"),
    ("analysis", "count_domain_cells"),
    ("analysis", "domain_area"),
    ("analysis", "apply"),
    ("analysis", "fill_domain"),
    ("analysis", "extract"),
    # NB: ``("analysis", "overlay")`` is intentionally excluded (#1496) — ``Dataset.overlay`` is
    # now a deprecated alias that warns and delegates to ``zonal_stats`` (not ``analysis.overlay``),
    # so it is no longer a pure same-named pass-through facade. Its deprecation + parity with
    # ``zonal_stats(raw_values=True)`` is covered by ``tests/dataset/ops/test_zonal_stats_raster.py``.
    ("analysis", "get_mask"),
    ("analysis", "footprint"),
    ("analysis", "get_histogram"),
    # NB: ``("analysis", "plot")`` is intentionally excluded — ``Dataset.plot`` is
    # no longer a pure pass-through facade since PR-1 (D-0). It now applies the
    # GeoTIFF/Sentinel band-resolution policy (``_resolve_plot_band``) before
    # delegating, so its signature and call shape differ from the engine's. The
    # band-resolution behaviour is covered by ``TestResolvePlotBand`` in
    # ``tests/dataset/test_plot.py``.
]


class TestFacadeDelegation:
    """Each migrated Dataset facade method delegates to the collaborator method."""

    @pytest.mark.parametrize("collab_attr, method_name", FACADE_METHODS)
    def test_facade_calls_collaborator(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        method_name,
    ):
        """``ds.<method>(...)`` should invoke ``ds.<collab>.<method>(...)``.

        Args:
            collab_attr: Collaborator attribute name on the Dataset.
            method_name: Public method that has been migrated onto that
                collaborator.

        Test scenario:
            For methods migrated to a collaborator (Stage 2), the
            Dataset method is now a thin facade. Patch the collaborator
            method to return a sentinel; calling ``ds.<method>(...)``
            should invoke the patched collaborator method with the same
            args and return the sentinel.
        """
        sentinel = object()
        collab = getattr(in_memory_dataset, collab_attr)
        mock = mocker.patch.object(type(collab), method_name, return_value=sentinel)

        facade = getattr(in_memory_dataset, method_name)
        result = facade(1, 2, foo="bar")

        assert result is sentinel, (
            f"Dataset.{method_name} facade did not return the collaborator's value"
        )
        mock.assert_called_once_with(1, 2, foo="bar")


# Stage 2 facade properties: Dataset property delegates to a same-named
# property on the collaborator. PR 2.2 — cog.is_cog. PR 2.6 — io.overview_count.
FACADE_PROPERTIES = [
    ("cog", "is_cog"),
    ("io", "overview_count"),
    ("bands", "band_color"),
    ("bands", "color_table"),
]


class TestPropertyForwarding:
    """Properties on collaborators forward to the same-named Dataset property."""

    @pytest.mark.parametrize("collab_attr, prop_name", FACADE_PROPERTIES)
    def test_facade_property_reads_from_collaborator(
        self,
        in_memory_dataset,
        mocker,
        collab_attr,
        prop_name,
    ):
        """``ds.<prop>`` should read from ``ds.<collab>.<prop>``.

        Args:
            collab_attr: Collaborator attribute name (e.g. ``"cog"``).
            prop_name: Property migrated onto that collaborator.

        Test scenario:
            For Stage 2 properties, the Dataset property is now a thin
            facade reading from the collaborator. Patch the collaborator
            class property with a sentinel value; reading ``ds.<prop>``
            should produce that sentinel.
        """
        sentinel = object()
        collab = getattr(in_memory_dataset, collab_attr)
        mocker.patch.object(
            type(collab),
            prop_name,
            new_callable=mocker.PropertyMock,
            return_value=sentinel,
        )
        result = getattr(in_memory_dataset, prop_name)
        assert result is sentinel, (
            f"Dataset.{prop_name} facade did not read from {collab_attr}.{prop_name} "
            f"(got {result!r})"
        )


class TestAnalysisNormalize:
    """Direct tests for ``Analysis.normalize`` — the only Stage 1 method with a real body."""

    def test_normalize_1d_simple(self):
        """1D array maps min->0 and max->1 with linear scaling between.

        Test scenario:
            Input ``[0, 5, 10]`` with min=0 and max=10 should produce
            ``[0.0, 0.5, 1.0]``.
        """
        out = _Analysis.normalize(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_array_equal(
            out,
            np.array([0.0, 0.5, 1.0]),
            err_msg=f"Unexpected normalize output: {out.tolist()}",
        )

    def test_normalize_2d_extrema_and_shape(self):
        """2D array preserves shape and hits 0 and 1 at the extrema.

        Test scenario:
            A 2x2 array ``[[2, 4], [6, 8]]`` should normalize to a 2x2 array
            with min=0.0 and max=1.0 (linear scaling preserves rank).
        """
        out = _Analysis.normalize(np.array([[2.0, 4.0], [6.0, 8.0]]))
        assert float(out.min()) == pytest.approx(0.0), (
            f"Expected min 0.0, got {out.min()}"
        )
        assert float(out.max()) == pytest.approx(1.0), (
            f"Expected max 1.0, got {out.max()}"
        )
        assert out.shape == (2, 2), f"Shape mismatch: {out.shape}"

    def test_normalize_signed_values(self):
        """Negative values are normalized into [0, 1] alongside positives.

        Test scenario:
            Input ``[-10, 0, 10]`` (range 20, min -10) should map to
            ``[0.0, 0.5, 1.0]``.
        """
        out = _Analysis.normalize(np.array([-10.0, 0.0, 10.0]))
        np.testing.assert_array_equal(
            out,
            np.array([0.0, 0.5, 1.0]),
            err_msg=f"Unexpected normalize output for signed input: {out.tolist()}",
        )

    def test_normalize_returns_ndarray(self):
        """Return value is a NumPy ndarray regardless of input dtype.

        Test scenario:
            Integer input should not be returned as a Python list or
            Python scalar; the staticmethod always returns ``np.ndarray``.
        """
        out = _Analysis.normalize(np.array([1, 2, 3, 4]))
        assert isinstance(out, np.ndarray), (
            f"Expected numpy ndarray, got {type(out).__name__}"
        )


class TestDatasetNormalizeFacade:
    """`Dataset.normalize` is the public facade over `Analysis.normalize`."""

    def test_facade_matches_engine_and_is_unit_range(self):
        """`ds.normalize()` returns the engine's 0-1 rescaling of the raster.

        Test scenario:
            The documented public `Dataset.normalize` must exist and forward to
            `Analysis.normalize` over the dataset's own data, mapping the global
            min to 0.0 and max to 1.0.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.array([[0.0, 5.0], [10.0, 2.5]], dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=geo)

        out = ds.analysis.normalize()

        assert float(np.nanmin(out)) == pytest.approx(0.0)
        assert float(np.nanmax(out)) == pytest.approx(1.0)

    def test_the_facade_warns_on_a_multiband_raster_too(self):
        """`ds.normalize()` emits its deprecation on multi-band input, not only single-band.

        Test scenario:
            The warning must fire unconditionally (it used to sit inside the single-band
            branch), so a `-W error::DeprecationWarning` sweep catches every `ds.normalize()`.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.arange(18, dtype="float32").reshape(2, 3, 3)
        ds = Dataset.from_array(arr=arr, geo_ref=geo)

        with pytest.warns(DeprecationWarning, match="ds.normalize"):
            ds.normalize()

    def test_nan_no_data_is_excluded_not_propagated(self):
        """A NaN no-data cell is excluded from the range, not spread to every cell.

        Test scenario:
            A float raster with one NaN no-data cell must still rescale its real
            values to 0-1 (the no-data cell stays NaN), rather than returning
            all-NaN because the global min/max saw the NaN.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.array([[0.0, 5.0], [10.0, np.nan]], dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=geo, no_data_value=np.nan)

        out = ds.analysis.normalize()

        assert np.isnan(out[1, 1]), "the no-data cell must stay NaN"
        np.testing.assert_allclose(
            np.sort(out[~np.isnan(out)]), np.array([0.0, 0.5, 1.0])
        )

    def test_sentinel_no_data_does_not_pollute_the_scale(self):
        """A sentinel no-data value is excluded, not treated as the global minimum.

        Test scenario:
            With ``no_data_value=-9999`` the sentinel must not become the range
            floor (which would squash every real value into ~1.0); it is excluded
            and the real values span 0-1.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.array([[0.0, 5.0], [10.0, -9999.0]], dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=geo, no_data_value=-9999.0)

        out = ds.analysis.normalize()

        assert np.isnan(out[1, 1]), "the no-data cell must be excluded (NaN)"
        np.testing.assert_allclose(
            np.sort(out[~np.isnan(out)]), np.array([0.0, 0.5, 1.0])
        )

    def test_a_2d_raster_without_no_data_is_normalized_unmasked(self):
        """A 2-D raster whose band declares no sentinel is rescaled without masking.

        Test scenario:
            With ``no_data_value=None`` the physical sentinel is ``None``, so the
            2-D facade path skips the masking step (the ``sentinel is None``
            branch) and normalises every cell. No cell becomes NaN and the real
            values still span 0-1.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.array([[0.0, 5.0], [10.0, 2.5]], dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=geo, no_data_value=None)

        out = ds.analysis.normalize()

        assert not np.isnan(out).any(), "no cell should be NaN without a sentinel"
        assert float(np.min(out)) == pytest.approx(0.0)
        assert float(np.max(out)) == pytest.approx(1.0)

    def test_multiband_masks_each_bands_no_data_against_a_global_range(self):
        """A multi-band raster masks each band's sentinel and scales on one global range.

        Test scenario:
            The 3-D facade path loops over every band, masks that band's physical
            no-data cell to NaN, then normalises the whole stack against a single
            global min/max. Each band's sentinel cell must come back NaN while the
            global minimum maps to 0.0 and the global maximum to 1.0.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        stack = np.array(
            [
                [[0.0, 5.0], [10.0, -9999.0]],
                [[1.0, 6.0], [11.0, -9999.0]],
                [[2.0, 7.0], [12.0, -9999.0]],
            ],
            dtype="float32",
        )
        ds = Dataset.from_array(arr=stack, geo_ref=geo, no_data_value=-9999.0)

        out = ds.normalize()

        assert out.shape == (3, 2, 2), f"expected a 3-band result, got {out.shape}"
        assert np.isnan(out[:, 1, 1]).all(), "every band's sentinel cell must stay NaN"
        assert float(np.nanmin(out)) == pytest.approx(0.0), "global min must map to 0.0"
        assert float(np.nanmax(out)) == pytest.approx(1.0), "global max must map to 1.0"
        assert float(out[0, 0, 0]) == pytest.approx(0.0), "the smallest value is 0.0"
        assert float(out[2, 1, 0]) == pytest.approx(1.0), "the largest value is 1.0"

    def test_multiband_without_no_data_is_normalized_unmasked(self):
        """A multi-band raster with no sentinel runs the loop but masks nothing.

        Test scenario:
            With ``no_data_value=None`` every band's physical sentinel is ``None``,
            so the 3-D loop iterates without masking (the per-band
            ``sentinel is None`` branch). No cell becomes NaN and the stack is
            rescaled to 0-1 against its global range.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        stack = np.array(
            [
                [[0.0, 3.0], [6.0, 9.0]],
                [[1.0, 4.0], [7.0, 10.0]],
                [[2.0, 5.0], [8.0, 11.0]],
            ],
            dtype="float32",
        )
        ds = Dataset.from_array(arr=stack, geo_ref=geo, no_data_value=None)

        out = ds.normalize()

        assert not np.isnan(out).any(), "no cell should be NaN without a sentinel"
        assert float(np.min(out)) == pytest.approx(0.0), "global min must map to 0.0"
        assert float(np.max(out)) == pytest.approx(1.0), "global max must map to 1.0"

    def test_entirely_no_data_returns_all_nan_without_warning(self):
        """A fully-no-data raster yields all-NaN and emits no numpy range warning.

        Test scenario:
            Every cell is the sentinel, so after masking the whole array is NaN;
            `normalize` must short-circuit to all-NaN rather than let
            ``np.nanmin`` emit ``All-NaN slice encountered``.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.full((2, 2), -9999.0, dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=geo, no_data_value=-9999.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = ds.analysis.normalize()

        assert np.isnan(out).all()

    def test_constant_raster_maps_to_zero_without_warning(self):
        """A constant (zero-range) raster maps every finite cell to 0.0, no warning.

        Test scenario:
            ``max == min`` would make numpy's ``(x-min)/(max-min)`` a ``0/0`` and
            warn; `normalize` must return a deterministic 0.0 for the finite
            cells instead.
        """
        geo = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        arr = np.full((2, 2), 7.0, dtype="float32")
        ds = Dataset.from_array(arr=arr, geo_ref=geo, no_data_value=None)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = ds.analysis.normalize()

        assert np.array_equal(out, np.zeros((2, 2)))


class TestPickleRoundTrip:
    """Pickle behaviour of collaborators and of the parent Dataset."""

    def test_dataset_round_trip_yields_fresh_collaborators(self, file_backed_dataset):
        """A round-tripped Dataset has fresh collaborators of the right types.

        Test scenario:
            ``RasterBase.__reduce__`` reduces a Dataset to a recipe
            ``(reconstruct_fn, (cls, path, access))`` and re-opens it via
            ``cls.read_file(...)``, which calls ``Dataset.__init__``, which
            instantiates fresh collaborators. The unpickled instance must
            therefore expose all seven collaborators with the correct
            types — never ``_Placeholder`` instances.
        """
        roundtripped = pickle.loads(pickle.dumps(file_backed_dataset))
        assert isinstance(roundtripped.io, _IO), (
            f"Roundtripped ds.io is wrong type: {type(roundtripped.io).__name__}"
        )
        assert isinstance(roundtripped.spatial, _Spatial), (
            f"Roundtripped ds.spatial is wrong type: {type(roundtripped.spatial).__name__}"
        )
        assert isinstance(roundtripped.bands, _Bands), (
            f"Roundtripped ds.bands is wrong type: {type(roundtripped.bands).__name__}"
        )
        assert isinstance(roundtripped.analysis, _Analysis), (
            f"Roundtripped ds.analysis is wrong type: {type(roundtripped.analysis).__name__}"
        )
        assert isinstance(roundtripped.cell, _Cell), (
            f"Roundtripped ds.cell is wrong type: {type(roundtripped.cell).__name__}"
        )
        assert isinstance(roundtripped.vector, _Vector), (
            f"Roundtripped ds.vector is wrong type: {type(roundtripped.vector).__name__}"
        )
        assert isinstance(roundtripped.cog, _COG), (
            f"Roundtripped ds.cog is wrong type: {type(roundtripped.cog).__name__}"
        )

    def test_round_tripped_collaborators_are_functional(self, file_backed_dataset):
        """After round-trip, calling a forwarder still works end-to-end.

        Test scenario:
            ``ds.io.read_array()`` on the unpickled Dataset should return
            the same array data as the original. This confirms that the
            collaborators on the unpickled Dataset are wired to a working
            GDAL handle (i.e., the weakref proxy points at the
            reconstructed Dataset, not at a stale reference).
        """
        original_array = file_backed_dataset.io.read_array()
        roundtripped = pickle.loads(pickle.dumps(file_backed_dataset))
        roundtripped_array = roundtripped.io.read_array()
        np.testing.assert_array_equal(
            roundtripped_array,
            original_array,
            err_msg="Roundtripped collaborator did not read identical array data",
        )

    @pytest.mark.parametrize(
        "collab_attr",
        ["io", "spatial", "bands", "analysis", "cell", "vector", "cog"],
    )
    def test_directly_pickled_collaborator_yields_placeholder(
        self,
        in_memory_dataset,
        collab_attr,
    ):
        """Pickling a collaborator directly produces a ``_Placeholder``.

        Args:
            collab_attr: Collaborator attribute name on the Dataset.

        Test scenario:
            ``pickle.dumps(ds.io)`` must not attempt to pickle the parent
            Dataset (that would either explode the payload size or fail
            on the GDAL handle). Instead, ``_Engine.__reduce__``
            short-circuits to ``_recreate_placeholder``, so the unpickled
            object is a ``_Placeholder``.
        """
        collab = getattr(in_memory_dataset, collab_attr)
        unpickled = pickle.loads(pickle.dumps(collab))
        assert isinstance(unpickled, _Placeholder), (
            f"Direct pickle of {collab_attr} collaborator yielded "
            f"{type(unpickled).__name__}, expected _Placeholder"
        )


class TestIsReadOnlyError:
    """ARC-68: the read-only classification lives in one place now."""

    @pytest.mark.parametrize(
        "refusing_call",
        ["GDALRasterBand::Fill()", "GDALRasterBand::RasterIO()"],
    )
    def test_a_real_gdal_refusal_is_recognised(self, refusing_call, tmp_path):
        """The message GDAL actually emits classifies as read-only.

        Args:
            refusing_call: The GDAL entry point named in the message.
            tmp_path: pytest's temporary directory fixture.

        Test scenario:
            Reproduces the exact string GDAL 3.13 produces -- file, band and
            refusing call, then the reason -- for both the Fill and RasterIO
            paths. The check was repeated at three no-data setters, each testing
            a different subset of the wordings, so two of them never recognised
            a refusal and surfaced a raw RuntimeError instead.
        """
        message = (
            f"ro.tif, band 1: {refusing_call}: attempt to write to dataset "
            "opened in read-only mode."
        )
        assert _is_read_only_error(RuntimeError(message)), (
            f"{message!r} is a read-only refusal and must be classified as one"
        )

    def test_the_classification_matches_a_live_gdal_error(self, tmp_path):
        """The pattern is checked against GDAL itself, not a remembered string.

        Test scenario:
            The wording is GDAL's, so a hard-coded copy drifts silently on a
            version bump. Opens a real raster read-only, provokes the refusal,
            and asserts the classifier recognises whatever GDAL raised.
        """
        path = tmp_path / "ro.tif"
        Dataset.from_array(
            np.zeros((4, 4), "float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0),
        ).io.to_file(str(path))
        # `read_only` has to stay bound: a gdal.Band does not keep its dataset
        # alive, so chaining the two calls invalidates the band immediately.
        read_only = gdal.Open(str(path), gdal.GA_ReadOnly)
        read_only_band = read_only.GetRasterBand(1)
        with pytest.raises(RuntimeError) as excinfo:
            read_only_band.Fill(1.0)
        assert _is_read_only_error(excinfo.value), (
            f"GDAL's own refusal must classify as read-only: {excinfo.value}"
        )

    def test_an_unrelated_error_is_not_classified(self):
        """A different GDAL failure must not be swallowed as read-only.

        Test scenario:
            Misclassifying would turn a genuine I/O failure into a ReadOnlyError
            and send the caller looking at the access mode instead of the disk.
        """
        assert not _is_read_only_error(RuntimeError("Cannot allocate memory")), (
            "an unrelated RuntimeError must not classify as a read-only refusal"
        )


class TestCachedTransformer:
    """ARC-60: the pyproj transformer is built once per CRS pair."""

    def test_the_same_crs_pair_returns_the_same_object(self):
        """A repeat call is a cache hit, not a rebuild.

        Test scenario:
            Building a transformer parses both CRS definitions and resolves a
            pipeline. The COG read paths rebuilt one per tile and per point, so
            a read_tile loop paid that cost on every call.
        """
        _cached_transformer.cache_clear()
        first = _cached_transformer(4326, 3857)
        second = _cached_transformer(4326, 3857)
        assert first is second, "a repeated CRS pair must return the cached object"
        assert _cached_transformer.cache_info().hits == 1, (
            f"expected one cache hit, got {_cached_transformer.cache_info()}"
        )

    def test_a_different_pair_builds_a_new_transformer(self):
        """The cache is keyed on both CRSes, not just the source."""
        _cached_transformer.cache_clear()
        to_mercator = _cached_transformer(4326, 3857)
        to_utm = _cached_transformer(4326, 32636)
        assert to_mercator is not to_utm, "distinct destinations must not share a key"

    def test_a_wkt_string_key_works_and_transforms_correctly(self):
        """A WKT string is a valid, hashable key and yields a working transformer.

        Test scenario:
            Datasets without an EPSG code pass their WKT through instead, so the
            cache has to accept it. Round-trips a point through the transformer
            and back to confirm the cached object is functional, not just
            identical.
        """
        _cached_transformer.cache_clear()
        wkt = sr_from_epsg(4326).ExportToWkt()
        forward = _cached_transformer(wkt, 3857)
        back = _cached_transformer(3857, wkt)
        x, y = forward.transform(12.0, 55.0)
        lon, lat = back.transform(x, y)
        assert round(lon, 6) == 12.0 and round(lat, 6) == 55.0, (
            f"round trip through the cached transformers drifted: {lon}, {lat}"
        )
