"""`Dataset.read_file` single-raster ``.zarr`` routing helpers (core side).

pyramids-core's ``read_file`` applies the cheap ``is_zarr_store`` suffix gate and, for a ``.zarr``,
asks the compute backend (through ``classify_zarr_store``) whether the store is a single raster it
should hand to :meth:`Dataset.from_zarr`. The two helpers that implement that decision live in
:mod:`pyramids.dataset.dataset`:

* ``_route_zarr_store`` -- returns a routed :class:`Dataset` for a single-raster store, raises for a
  cube store, and returns ``None`` (fall through to GDAL) for an opt-out / non-store / foreign store.
* ``_warn_ignored_zarr_args`` -- warns that GDAL-only ``read_file`` arguments are dropped when a store
  routes to ``from_zarr``.

These tests drive every branch by stubbing ``classify_zarr_store`` (so they are independent of whether
the installed backend can classify) and ``Dataset.from_zarr`` (so no real Zarr store is needed).
"""

from __future__ import annotations

import warnings

import pytest

from pyramids.dataset import dataset as dataset_module
from pyramids.dataset.dataset import (
    Dataset,
    _route_zarr_store,
    _warn_ignored_zarr_args,
)

pytestmark = pytest.mark.core

_DEFAULTS = {
    "read_only": True,
    "file_i": 0,
    "gdal_env": None,
    "warn_on_container": True,
    "open_options": None,
}


class TestWarnIgnoredZarrArgs:
    """``_warn_ignored_zarr_args`` flags each non-default GDAL-only argument, and only those."""

    def test_no_warning_when_all_defaults(self):
        """All-default GDAL arguments route cleanly, so nothing is warned.

        Test scenario:
            With ``read_only=True``, ``file_i=0``, no ``gdal_env``, ``warn_on_container=True`` and no
            ``open_options``, no argument is dropped, so no ``UserWarning`` is emitted.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_ignored_zarr_args(True, 0, None, warn_on_container=True, options=None)

    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({"read_only": False}, "read_only=False"),
            ({"file_i": 2}, "file_i=2"),
            ({"gdal_env": object()}, "gdal_env"),
            ({"warn_on_container": False}, "warn_on_container=False"),
            ({"open_options": ["A=1"]}, "open_options"),
        ],
    )
    def test_each_non_default_is_flagged(self, kwargs, needle):
        """Each individually non-default GDAL argument is named in a single ``UserWarning``.

        Args:
            kwargs: One overridden argument versus the all-default baseline.
            needle: The substring that argument must contribute to the warning text.

        Test scenario:
            A lone non-default argument produces exactly one ``UserWarning`` whose message points at
            ``from_zarr`` and names the dropped argument.
        """
        call = {**_DEFAULTS, **kwargs}
        with pytest.warns(UserWarning, match="from_zarr") as record:
            _warn_ignored_zarr_args(
                call["read_only"],
                call["file_i"],
                call["gdal_env"],
                warn_on_container=call["warn_on_container"],
                options=call["open_options"],
            )
        assert len(record) == 1, f"expected exactly one warning, got {len(record)}"
        assert needle in str(record[0].message), (
            f"warning should name {needle!r}, got: {record[0].message}"
        )

    def test_multiple_non_defaults_listed_together(self):
        """Several dropped arguments are reported in one warning, each named.

        Test scenario:
            ``read_only=False`` together with ``file_i=3`` yields a single warning naming both.
        """
        with pytest.warns(UserWarning, match="from_zarr") as record:
            _warn_ignored_zarr_args(
                False, 3, None, warn_on_container=True, options=None
            )
        message = str(record[0].message)
        assert "read_only=False" in message, f"missing read_only in: {message}"
        assert "file_i=3" in message, f"missing file_i in: {message}"


class TestRouteZarrStore:
    """``_route_zarr_store`` routes a single store, raises for a cube, else returns ``None``."""

    @pytest.fixture
    def sentinel_from_zarr(self, monkeypatch):
        """Replace ``Dataset.from_zarr`` with a stub returning a sentinel.

        Returns:
            object: The sentinel object the stubbed ``from_zarr`` returns, so a test can assert the
            router forwarded to it without constructing a real Zarr store.
        """
        sentinel = object()
        monkeypatch.setattr(
            Dataset, "from_zarr", classmethod(lambda cls, path: sentinel)
        )
        return sentinel

    def test_vsi_opt_out_returns_none_without_classifying(self, monkeypatch):
        """A non-``None`` ``vsi`` short-circuits before the store is ever classified.

        Test scenario:
            With ``vsi`` set, the router returns ``None`` (GDAL path) and never consults
            ``classify_zarr_store`` -- proven by a tracker that must stay empty.
        """
        consulted: list[str] = []
        monkeypatch.setattr(
            dataset_module,
            "classify_zarr_store",
            lambda *a, **k: consulted.append("called") or "single",
        )
        result = _route_zarr_store(Dataset, "x.zarr", "/vsimem/x", **_DEFAULTS)
        assert result is None, f"vsi opt-out must return None, got {result!r}"
        assert consulted == [], "classify must not be consulted when vsi is set"

    def test_non_zarr_path_returns_none(self, monkeypatch):
        """A path that is not a ``.zarr`` store returns ``None`` before classification.

        Test scenario:
            ``is_zarr_store`` is ``False`` for a GeoTIFF, so the router falls through to GDAL without
            classifying.
        """
        monkeypatch.setattr(
            dataset_module,
            "classify_zarr_store",
            lambda *a, **k: pytest.fail("classify should not run for a non-zarr path"),
        )
        result = _route_zarr_store(Dataset, "raster.tif", None, **_DEFAULTS)
        assert result is None, f"a .tif must return None, got {result!r}"

    def test_single_store_routes_to_from_zarr(self, monkeypatch, sentinel_from_zarr):
        """A ``"single"`` classification routes to ``from_zarr`` and returns its result.

        Test scenario:
            With ``classify_zarr_store`` reporting ``"single"`` and all-default GDAL args, the router
            returns exactly what ``Dataset.from_zarr`` returns and emits no warning.
        """
        monkeypatch.setattr(
            dataset_module, "classify_zarr_store", lambda *a, **k: "single"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _route_zarr_store(Dataset, "x.zarr", None, **_DEFAULTS)
        assert result is sentinel_from_zarr, (
            f"a single store must return from_zarr's result, got {result!r}"
        )

    def test_single_store_warns_when_gdal_args_dropped(
        self, monkeypatch, sentinel_from_zarr
    ):
        """Routing a single store while a GDAL-only argument is set warns about the drop.

        Test scenario:
            ``read_only=False`` cannot apply on the ``from_zarr`` path, so the router still routes
            (returns the sentinel) but emits a ``UserWarning`` naming ``from_zarr``.
        """
        monkeypatch.setattr(
            dataset_module, "classify_zarr_store", lambda *a, **k: "single"
        )
        call = {**_DEFAULTS, "read_only": False}
        with pytest.warns(UserWarning, match="from_zarr"):
            result = _route_zarr_store(Dataset, "x.zarr", None, **call)
        assert result is sentinel_from_zarr, (
            "a single store must still route when args are dropped"
        )

    def test_cube_store_raises_pointing_at_collection(self, monkeypatch):
        """A ``"cube"`` classification raises ``ValueError`` pointing at ``DatasetCollection``.

        Test scenario:
            A cube store is not a single raster, so the router refuses it with a message naming
            ``DatasetCollection.from_zarr``.
        """
        monkeypatch.setattr(
            dataset_module, "classify_zarr_store", lambda *a, **k: "cube"
        )
        with pytest.raises(ValueError, match="DatasetCollection.from_zarr"):
            _route_zarr_store(Dataset, "cube.zarr", None, **_DEFAULTS)

    def test_other_store_returns_none(self, monkeypatch):
        """An ``"other"`` classification falls through to GDAL (returns ``None``).

        Test scenario:
            A foreign / unreadable / backend-less store classifies as ``"other"``, so the router
            returns ``None`` and ``read_file`` lets GDAL handle it.
        """
        monkeypatch.setattr(
            dataset_module, "classify_zarr_store", lambda *a, **k: "other"
        )
        result = _route_zarr_store(Dataset, "foreign.zarr", None, **_DEFAULTS)
        assert result is None, f"an 'other' store must return None, got {result!r}"
