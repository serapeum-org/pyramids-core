"""Core-side Zarr routing predicates — the backend-free part of the read_file ``.zarr`` routing.

The heavy classify / read / write bodies live in the ``pyramids-lazy`` compute backend and are
exercised by its own suite. What pyramids-core owns, and what is tested here, is only:

* :func:`pyramids.dataset.ops._zarr.is_zarr_store` — the pure suffix check that decides whether a path
  is a Zarr store at all (no store opened, no backend needed).
* :func:`pyramids.dataset.ops._zarr.classify_zarr_store` / ``is_cube_store`` — thin dispatchers that
  delegate to the backend and must degrade to ``"other"`` / ``False`` (so ``read_file`` falls back to
  GDAL) whenever no backend is installed, the backend predates the classify capability, or the backend
  raises. The real single/cube/foreign classification is the backend's responsibility.
"""

from pathlib import Path

from pyramids.dataset.ops import _zarr


class TestIsZarrStore:
    """The cheap, backend-free suffix check."""

    def test_detects_by_suffix_case_and_separators(self):
        """A ``.zarr`` path is a store regardless of case or a trailing separator."""
        assert _zarr.is_zarr_store("elevation.zarr")
        assert _zarr.is_zarr_store("X.ZARR")
        assert _zarr.is_zarr_store("store.zarr/")
        assert _zarr.is_zarr_store("store.zarr\\")
        assert _zarr.is_zarr_store("s3://bucket/path/x.zarr")
        assert _zarr.is_zarr_store(Path("dir") / "x.zarr")

    def test_non_path_is_taken_to_be_an_opened_store(self):
        """A non-str / non-Path object is treated as an already-opened store."""
        assert _zarr.is_zarr_store({"a mapping": "stands in for a zarr.Store"})

    def test_rejects_non_zarr_paths(self):
        """A GeoTIFF (local or remote) is not a store."""
        assert not _zarr.is_zarr_store("x.tif")
        assert not _zarr.is_zarr_store("s3://bucket/x.tif")
        assert not _zarr.is_zarr_store(Path("plain") / "raster.tif")


class TestClassifyDispatch:
    """``classify_zarr_store`` delegates to the backend and degrades to ``"other"``."""

    def test_other_when_no_backend_installed(self, monkeypatch):
        """A missing backend raises ``ImportError`` from ``require_compute_backend`` -> ``"other"``."""

        def _raise():
            raise ImportError("no compute backend installed")

        monkeypatch.setattr(_zarr, "require_compute_backend", _raise)
        assert _zarr.classify_zarr_store("x.zarr") == "other"
        assert _zarr.is_cube_store("x.zarr") is False

    def test_other_when_backend_predates_classify(self, monkeypatch):
        """A backend with no ``classify_zarr_store`` attribute -> ``"other"`` (never raises)."""
        monkeypatch.setattr(_zarr, "require_compute_backend", lambda: object())
        assert _zarr.classify_zarr_store("x.zarr") == "other"
        assert _zarr.is_cube_store("x.zarr") is False

    def test_delegates_single_and_cube(self, monkeypatch):
        """A classify-capable backend's verdict is passed through for the known labels."""

        class _Backend:
            def __init__(self, verdict):
                self._verdict = verdict

            def classify_zarr_store(self, store, storage_options=None):
                return self._verdict

        monkeypatch.setattr(
            _zarr, "require_compute_backend", lambda: _Backend("single")
        )
        assert _zarr.classify_zarr_store("x.zarr") == "single"
        assert _zarr.is_cube_store("x.zarr") is False

        monkeypatch.setattr(_zarr, "require_compute_backend", lambda: _Backend("cube"))
        assert _zarr.classify_zarr_store("x.zarr") == "cube"
        assert _zarr.is_cube_store("x.zarr") is True

    def test_unknown_label_normalized_to_other(self, monkeypatch):
        """A backend returning anything outside single/cube is normalized to ``"other"``."""

        class _Backend:
            def classify_zarr_store(self, store, storage_options=None):
                return "weird"

        monkeypatch.setattr(_zarr, "require_compute_backend", lambda: _Backend())
        assert _zarr.classify_zarr_store("x.zarr") == "other"

    def test_backend_failure_is_swallowed_to_other(self, monkeypatch):
        """Any exception from the backend's classify means "let GDAL try", not a surfaced error."""

        class _Backend:
            def classify_zarr_store(self, store, storage_options=None):
                raise RuntimeError("store could not be opened")

        monkeypatch.setattr(_zarr, "require_compute_backend", lambda: _Backend())
        assert _zarr.classify_zarr_store("x.zarr") == "other"


class TestResolveStore:
    """``_resolve_store`` normalizes a store target for zarr v3 / dask."""

    def test_plain_string_passed_through(self):
        """A plain path string is returned unchanged for zarr/dask to resolve directly."""
        assert _zarr._resolve_store("local/x.zarr", None) == "local/x.zarr"

    def test_path_is_stringified(self):
        """A :class:`~pathlib.Path` is converted to ``str`` (zarr v3 takes a string target)."""
        result = _zarr._resolve_store(Path("dir") / "x.zarr", None)
        assert isinstance(result, str), f"expected str, got {type(result)}"
        assert result.endswith("x.zarr"), f"unexpected store target: {result}"

    def test_prebuilt_store_returned_unchanged(self):
        """A non-str / non-Path (an already-built store / mapping) is returned as-is."""
        store = {"already": "a store"}
        assert _zarr._resolve_store(store, None) is store

    def test_storage_options_build_fsspec_store(self, monkeypatch):
        """A URL with ``storage_options`` is wrapped in a v3 ``FsspecStore`` so options reach fsspec.

        Test scenario:
            ``FsspecStore.from_url`` is stubbed (no real network) and must receive the URL string and
            the storage options, and its result is returned verbatim.
        """
        captured = {}
        sentinel = object()

        class _FakeFsspecStore:
            @staticmethod
            def from_url(url, storage_options):
                captured["url"] = url
                captured["storage_options"] = storage_options
                return sentinel

        monkeypatch.setattr("zarr.storage.FsspecStore", _FakeFsspecStore)
        opts = {"key": "secret"}
        result = _zarr._resolve_store("s3://bucket/x.zarr", opts)
        assert result is sentinel, f"expected the FsspecStore, got {result!r}"
        assert captured["url"] == "s3://bucket/x.zarr", (
            f"wrong url forwarded: {captured['url']}"
        )
        assert captured["storage_options"] is opts, (
            "storage_options must reach fsspec unchanged"
        )
