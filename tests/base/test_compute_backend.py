"""Tests for the lazy compute-backend seam (plan Phase 1, tracker #4).

The seam is additive scaffolding — core does not call `require_compute_backend` until the Phase 2
cutover — so these tests exercise the resolver, the entry-point discovery, the `ComputeBackend`
protocol check, and the branded errors directly, with in-memory fakes.
"""

from __future__ import annotations

import pytest

from pyramids.base import _compute_backend as cb
from pyramids.base._errors import OptionalPackageDoesNotExist

_METHODS = (
    "read_array_lazy",
    "build_lazy_cube",
    "to_dask_dataframe",
    "to_kerchunk",
    "combine_kerchunk",
    "open_mfdataset",
    "write_dataset_to_zarr",
    "read_dataset_from_zarr",
    "write_collection_to_zarr",
    "read_collection_from_zarr",
    "read_vector_lazy",
    "read_vector_parquet_lazy",
)


def _make_backend(api_version: int = cb.COMPUTE_BACKEND_API_VERSION):
    """Build a minimal object that structurally satisfies the ComputeBackend protocol."""

    class _Backend:
        pass

    backend = _Backend()
    backend.api_version = api_version
    for name in _METHODS:
        setattr(backend, name, lambda *a, **k: None)
    return backend


class _FakeEP:
    """A stand-in for `importlib.metadata.EntryPoint` with a controllable `load()`."""

    def __init__(self, name, value, target):
        self.name = name
        self.value = value
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the discovery cache around every test so entry-point fakes don't leak between cases."""
    cb.clear_compute_backend_cache()
    yield
    cb.clear_compute_backend_cache()


class TestContract:
    """The published contract constants and the protocol check."""

    def test_group_and_version_constants(self):
        """The entry-point group and api-version are the published, stable contract values."""
        assert cb.ENTRY_POINT_GROUP == "pyramids.compute_backend"
        assert (
            isinstance(cb.COMPUTE_BACKEND_API_VERSION, int)
            and cb.COMPUTE_BACKEND_API_VERSION >= 1
        )

    def test_conforming_object_is_a_computebackend(self):
        """An object exposing api_version + every op method satisfies the runtime_checkable protocol."""
        assert isinstance(_make_backend(), cb.ComputeBackend)

    def test_object_missing_a_method_is_not_a_computebackend(self):
        """Dropping any op method makes the object fail the protocol check."""
        backend = _make_backend()
        del backend.to_kerchunk
        assert not isinstance(backend, cb.ComputeBackend)


class TestComputeBackends:
    """Discovery via the entry-point group."""

    def test_bare_install_registers_nothing(self, monkeypatch):
        """With no entry points, discovery yields an empty registry (and imports no plugin)."""
        monkeypatch.setattr(cb, "entry_points", lambda group=None: [])
        assert cb.compute_backends() == {}

    def test_conforming_entry_point_is_registered(self, monkeypatch):
        """A conforming backend published under the group is discovered and cached by name."""
        backend = _make_backend()
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("lazy", "m:b", backend)]
        )
        assert cb.compute_backends() == {"lazy": backend}

    def test_factory_entry_point_is_called(self, monkeypatch):
        """An entry point resolving to a zero-arg factory is called to obtain the backend."""
        backend = _make_backend()
        monkeypatch.setattr(
            cb,
            "entry_points",
            lambda group=None: [_FakeEP("lazy", "m:make", lambda: backend)],
        )
        assert cb.compute_backends()["lazy"] is backend

    def test_non_conforming_entry_point_is_skipped_with_warning(self, monkeypatch):
        """A loaded object that is not a ComputeBackend is skipped with a RuntimeWarning."""
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("bad", "m:x", object())]
        )
        with pytest.warns(
            RuntimeWarning, match="does not implement the ComputeBackend"
        ):
            assert cb.compute_backends() == {}

    def test_raising_entry_point_is_skipped_with_warning(self, monkeypatch):
        """An entry point whose load() raises is skipped with a RuntimeWarning, not propagated."""
        ep = _FakeEP("boom", "m:x", RuntimeError("nope"))
        monkeypatch.setattr(cb, "entry_points", lambda group=None: [ep])
        with pytest.warns(RuntimeWarning, match="could not be loaded"):
            assert cb.compute_backends() == {}

    def test_class_entry_point_is_instantiated(self, monkeypatch):
        """A backend registered as a class is instantiated, not stored as the class object.

        A class is structurally a ComputeBackend (class-level api_version + methods), so the
        factory guard must instantiate it — otherwise a later `backend.read_array_lazy(ds, ...)`
        binds `ds` to `self`.
        """
        members = {name: (lambda self, *a, **k: None) for name in _METHODS}
        members["api_version"] = cb.COMPUTE_BACKEND_API_VERSION
        backend_cls = type("ClassBackend", (), members)
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("lazy", "m:C", backend_cls)]
        )
        resolved = cb.compute_backends()["lazy"]
        assert not isinstance(resolved, type), (
            "a class backend must be instantiated, not stored as the class"
        )
        assert isinstance(resolved, backend_cls), (
            f"expected a ClassBackend instance, got {resolved!r}"
        )

    def test_reentrant_discovery_sees_partial_map(self, monkeypatch):
        """A re-entrant call during discovery (plugin importing itself) sees the partial mapping."""
        partial = {"lazy": _make_backend()}
        monkeypatch.setattr(cb._IN_PROGRESS, "resolved", partial, raising=False)
        assert cb.compute_backends() == partial

    def test_duplicate_name_keeps_first(self, monkeypatch):
        """Two entry points with the same name keep the first and warn on the second."""
        first, second = _make_backend(), _make_backend()
        monkeypatch.setattr(
            cb,
            "entry_points",
            lambda group=None: [
                _FakeEP("lazy", "a", first),
                _FakeEP("lazy", "b", second),
            ],
        )
        with pytest.warns(RuntimeWarning, match="registered twice"):
            assert cb.compute_backends()["lazy"] is first


class TestRequireComputeBackend:
    """The single-backend resolver and its branded errors."""

    def test_registered_backend_is_returned(self, monkeypatch):
        """A registered, current backend is returned by the resolver."""
        backend = _make_backend()
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("lazy", "m:b", backend)]
        )
        assert cb.require_compute_backend() is backend

    def test_fallback_import_when_nothing_registered(self, monkeypatch):
        """With no entry point, the resolver imports pyramids_lazy and calls get_compute_backend()."""
        backend = _make_backend()
        monkeypatch.setattr(cb, "entry_points", lambda group=None: [])
        monkeypatch.setattr(cb, "extra_missing", lambda name: False)
        monkeypatch.setattr(
            cb,
            "import_module",
            lambda name: type(
                "M", (), {"get_compute_backend": staticmethod(lambda: backend)}
            ),
        )
        assert cb.require_compute_backend() is backend

    def test_missing_backend_raises_branded_error(self, monkeypatch):
        """No entry point and no pyramids_lazy -> branded error naming the install command."""
        monkeypatch.setattr(cb, "entry_points", lambda group=None: [])
        monkeypatch.setattr(cb, "extra_missing", lambda name: True)
        with pytest.raises(OptionalPackageDoesNotExist, match="pyramids-lazy"):
            cb.require_compute_backend()

    def test_non_conforming_fallback_backend_is_rejected(self, monkeypatch):
        """A pyramids_lazy fallback that doesn't implement the protocol is rejected, not returned.

        The entry-point path validates structurally in `_load`; the fallback path must enforce the
        same check, or a malformed first-party backend AttributeErrors deep in a Phase-2 call.
        """

        class _Partial:  # has api_version but none of the op methods -> not a ComputeBackend
            api_version = cb.COMPUTE_BACKEND_API_VERSION

        monkeypatch.setattr(cb, "entry_points", lambda group=None: [])
        monkeypatch.setattr(cb, "extra_missing", lambda name: False)
        monkeypatch.setattr(
            cb,
            "import_module",
            lambda name: type(
                "M", (), {"get_compute_backend": staticmethod(lambda: _Partial())}
            ),
        )
        with pytest.raises(
            OptionalPackageDoesNotExist, match="does not implement the ComputeBackend"
        ):
            cb.require_compute_backend()

    def test_fallback_too_old_without_factory_raises(self, monkeypatch):
        """pyramids_lazy present but lacking get_compute_backend -> branded 'too old' error."""
        monkeypatch.setattr(cb, "entry_points", lambda group=None: [])
        monkeypatch.setattr(cb, "extra_missing", lambda name: False)
        monkeypatch.setattr(cb, "import_module", lambda name: type("M", (), {}))
        with pytest.raises(OptionalPackageDoesNotExist, match="too old"):
            cb.require_compute_backend()

    def test_backend_with_old_api_version_is_rejected(self, monkeypatch):
        """A resolved backend declaring an older api_version is rejected with an upgrade message."""
        backend = _make_backend(api_version=cb.COMPUTE_BACKEND_API_VERSION - 1)
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("lazy", "m:b", backend)]
        )
        with pytest.raises(OptionalPackageDoesNotExist, match="needs >="):
            cb.require_compute_backend()

    def test_backend_with_non_int_api_version_is_rejected(self, monkeypatch):
        """A backend whose api_version is not an int (garbage/missing) is rejected, not trusted."""
        backend = _make_backend()
        backend.api_version = "oops"
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("lazy", "m:b", backend)]
        )
        with pytest.raises(OptionalPackageDoesNotExist, match="needs >="):
            cb.require_compute_backend()

    def test_backend_with_bool_api_version_is_rejected(self, monkeypatch):
        """A bool api_version is rejected: `True == 1` must not be silently taken as version 1."""
        backend = _make_backend()
        backend.api_version = True
        monkeypatch.setattr(
            cb, "entry_points", lambda group=None: [_FakeEP("lazy", "m:b", backend)]
        )
        with pytest.raises(OptionalPackageDoesNotExist, match="needs >="):
            cb.require_compute_backend()
