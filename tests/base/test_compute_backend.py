"""Tests for the lazy compute-backend seam (plan Phase 1, tracker #4).

The seam is additive scaffolding — core does not call `require_compute_backend` until the Phase 2
cutover — so these tests exercise the resolver, the entry-point discovery, the `ComputeBackend`
protocol check, and the branded errors directly, with in-memory fakes.
"""

from __future__ import annotations

import typing
from typing import Protocol, runtime_checkable

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
    # The protocol's two data members (the lazy class types) are part of structural conformance:
    # runtime_checkable isinstance checks attribute presence, not just methods.
    # Accessors, matching the Protocol: isinstance only checks that the method exists, which is
    # exactly why they are methods (a property would be executed by the type check itself).
    _fake_nc = type("_FakeLazyNetCDF", (), {})
    _fake_fc = type("_FakeLazyFrame", (), {})
    backend.lazy_netcdf_type = lambda _cls=_fake_nc: _cls
    backend.lazy_feature_collection_type = lambda _cls=_fake_fc: _cls
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


@runtime_checkable
class _BaseProtocol(Protocol):
    """A Protocol that contributes a member to its subclasses by inheritance."""

    inherited: int


@runtime_checkable
class _DerivedProtocol(_BaseProtocol, Protocol):
    """A Protocol declaring one member of every shape `isinstance` enforces.

    Each member is a shape the old annotations-plus-`vars` scan handled differently: `annotated`
    is in the class's own annotations, `plain` is in its `vars` as a function, `read_only` is a
    property (present but not callable), `built` is a classmethod (also not callable), and
    `inherited` lives on the base class only.
    """

    annotated: str

    @property
    def read_only(self) -> int: ...

    @classmethod
    def built(cls) -> None: ...

    def plain(self) -> None: ...


_DERIVED_MEMBERS = frozenset({"annotated", "built", "inherited", "plain", "read_only"})
# What the last-resort scan can see: the class's own annotation and its one plain method. The
# inherited member, the property and the classmethod are invisible to it -- the documented gap that
# makes the helper prefer typing.
_SCANNABLE_MEMBERS = frozenset({"annotated", "plain"})


def _typing_reports_members(protocol: type) -> bool:
    """Whether this interpreter can report `protocol`'s full enforced member set.

    Args:
        protocol: The Protocol to ask about.

    Returns:
        True when `typing.get_protocol_members` exists (3.13+) or the Protocol carries a populated
        `__protocol_attrs__` (3.12+); False on 3.11, where neither is available and the helper
        falls back to the weaker scan.
    """
    return hasattr(typing, "get_protocol_members") or bool(
        getattr(protocol, "__protocol_attrs__", None)
    )


def _make_derived():
    """Build an object that structurally satisfies `_DerivedProtocol`.

    Returns:
        An instance exposing every member of `_DERIVED_MEMBERS`.
    """

    class _Candidate:
        pass

    candidate = _Candidate()
    for name in _DERIVED_MEMBERS:
        setattr(candidate, name, lambda *a, **k: None)
    return candidate


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
        assert isinstance(cb.COMPUTE_BACKEND_API_VERSION, int), (
            "api_version must be an int"
        )
        assert cb.COMPUTE_BACKEND_API_VERSION >= 1, "api_version must be >= 1"

    def test_conforming_object_is_a_computebackend(self):
        """An object exposing api_version + every op method satisfies the runtime_checkable protocol."""
        assert isinstance(_make_backend(), cb.ComputeBackend)

    def test_object_missing_a_method_is_not_a_computebackend(self):
        """Dropping any op method makes the object fail the protocol check."""
        backend = _make_backend()
        del backend.to_kerchunk
        assert not isinstance(backend, cb.ComputeBackend)

    def test_the_protocol_member_set_is_frozen(self):
        """Adding a member to the protocol silently disables every released backend.

        Test scenario:
            The protocol is ``runtime_checkable`` and :func:`_load` drops any backend failing
            its ``isinstance`` check, so a new member makes every already-published plugin fail
            structurally -- not just for the new call, but for *every* lazy operation, with only
            a RuntimeWarning. Core cannot require a newer plugin either: the ``[lazy]`` extra was
            removed (#1565), so users install ``pyramids-lazy`` themselves.

            New backend capabilities must therefore be probed with ``getattr`` at the call site
            (see ``MeshDataset.open_mfdataset``'s ``chunks`` branch, issue #1458), never added
            here. This pins the member set so that choice cannot be undone by accident.

            The set is read with ``cb.protocol_members()``, which asks typing what ``isinstance``
            enforces, and compared against ``_METHODS`` -- the list this file already builds its
            fake from -- so a new *method* breaks the guard and the fake together instead of being
            described in a third copy. The three data members are the exception: they are spelled
            here and again in ``_make_backend``, so adding a fourth would break this guard but not
            the fake.
        """
        members = cb.protocol_members()

        assert members == frozenset(_METHODS) | {
            "api_version",
            "lazy_netcdf_type",
            "lazy_feature_collection_type",
        }, (
            "the ComputeBackend protocol gained or lost a member. Adding one drops every "
            "released plugin at isinstance; probe new capabilities with getattr instead. "
            f"Current members: {sorted(members)}"
        )

    def test_protocol_members_accepts_an_explicit_protocol(self):
        """`protocol_members` inspects whichever Protocol it is handed, not only ComputeBackend.

        Test scenario:
            The parameter exists so the helper can be pointed at any Protocol -- including the
            inheritance/property/classmethod shapes that motivated deriving the set from typing.
            Passing `_DerivedProtocol` must report its members, while the default still answers
            for `ComputeBackend`. Which members those are is interpreter-dependent: 3.12+ reports
            all five, while 3.11 has neither typing hook and reaches the weaker scan, so the
            expectation follows the interpreter rather than assuming the richest one. The branch
            was previously asserted unconditionally and failed every 3.11 job in the CI matrix.
        """
        members = cb.protocol_members(_DerivedProtocol)

        if _typing_reports_members(_DerivedProtocol):
            expected = _DERIVED_MEMBERS
        else:
            expected = _SCANNABLE_MEMBERS
        assert members == expected, (
            f"expected {sorted(expected)} for the derived protocol on this interpreter, got "
            f"{sorted(members)}"
        )
        assert members != cb.protocol_members(), (
            "an explicit protocol must not be ignored in favour of ComputeBackend"
        )

    @pytest.mark.parametrize("member", sorted(_DERIVED_MEMBERS))
    def test_every_reported_member_is_one_isinstance_enforces(self, member):
        """Each name the helper reports is a name whose absence fails the structural check.

        Args:
            member: One member name reported for `_DerivedProtocol`.

        Test scenario:
            The member set is pinned by a guard test, so a name the helper invents would make
            that guard assert on something `isinstance` does not care about. Removing each
            reported name from a conforming object must therefore break the check -- which also
            proves the inherited, property and classmethod members are genuinely enforced and
            not merely listed.
        """
        candidate = _make_derived()
        assert isinstance(candidate, _DerivedProtocol), (
            "the fixture object should satisfy the protocol before anything is removed"
        )

        delattr(candidate, member)

        assert not isinstance(candidate, _DerivedProtocol), (
            f"{member!r} is reported as a protocol member but isinstance ignores its absence"
        )

    @pytest.mark.skipif(
        not getattr(_DerivedProtocol, "__protocol_attrs__", None),
        reason="no __protocol_attrs__ to fall back to on this interpreter (3.11 and older)",
    )
    def test_protocol_members_falls_back_to_the_typing_attribute(self, monkeypatch):
        """Without `typing.get_protocol_members`, the Protocol's own attribute set is used.

        Test scenario:
            `get_protocol_members` is 3.13+, so on 3.12 the helper reads `__protocol_attrs__`
            instead. That branch is unreachable on this interpreter, so the getter is removed to
            reach it; it must still report the full enforced set, inherited and property members
            included. Skipped on 3.11, where there is no such attribute to fall back *to* -- the
            test would otherwise land in the last-resort scan and assert the wrong set.
        """
        monkeypatch.delattr(typing, "get_protocol_members", raising=False)

        members = cb.protocol_members(_DerivedProtocol)

        assert members == _DERIVED_MEMBERS, (
            "the __protocol_attrs__ fallback must report the same members as typing's getter, "
            f"got {sorted(members)}"
        )

    def test_protocol_members_last_resort_scans_annotations_and_methods(
        self, monkeypatch
    ):
        """With neither typing hook available, the helper scans annotations plus methods.

        Test scenario:
            On 3.11 and older `get_protocol_members` is absent *and* `__protocol_attrs__` is
            unpopulated, which is simulated by removing both. The last-resort scan must still
            return a usable set, and it is knowably weaker: it sees the own annotation and the
            plain method but cannot see the inherited member, the property or the classmethod.
            Pinning that gap is the point -- it is the reason the helper prefers typing, and a
            silent change to it would alter what the guard test enforces.
        """
        monkeypatch.delattr(typing, "get_protocol_members", raising=False)
        # raising=False: on 3.11 the attribute does not exist at all, and setting it with the
        # default raising=True errored in setup rather than running the test.
        monkeypatch.setattr(
            _DerivedProtocol, "__protocol_attrs__", frozenset(), raising=False
        )

        members = cb.protocol_members(_DerivedProtocol)

        assert members == _SCANNABLE_MEMBERS, (
            "the last-resort scan should find the own annotation and the declared method, got "
            f"{sorted(members)}"
        )
        assert members < _DERIVED_MEMBERS, (
            "the scan is documented as weaker than what isinstance enforces; it must not be "
            f"reported as complete: {sorted(members)}"
        )

    def test_a_missing_member_is_named_in_the_diagnostic(self):
        """The rejection diagnostic lists exactly the protocol members a candidate lacks.

        Test scenario:
            `isinstance` only answers yes/no, so `_missing_protocol_members` re-derives the
            absent names -- now through `protocol_members`. A backend missing two methods should
            have both named, sorted, and nothing else.
        """
        backend = _make_backend()
        del backend.to_kerchunk
        del backend.read_array_lazy

        missing = cb._missing_protocol_members(backend)

        assert missing == ["read_array_lazy", "to_kerchunk"], (
            f"expected both absent members named in sorted order, got {missing}"
        )


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
        _fake_nc = type("_FakeLazyNetCDF", (), {})
        _fake_fc = type("_FakeLazyFrame", (), {})
        members["lazy_netcdf_type"] = lambda self, _cls=_fake_nc: _cls
        members["lazy_feature_collection_type"] = lambda self, _cls=_fake_fc: _cls
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


class TestAThirdPartyBackendIsHonoured:
    """Core's public lazy names resolve through the seam, not through a hard-coded distribution."""

    @pytest.fixture
    def third_party_backend(self, monkeypatch):
        """Register a conforming backend that is not pyramids-lazy."""

        class _OtherLazyNetCDF:
            pass

        class _OtherLazyFrame:
            pass

        class _OtherBackend:
            api_version = cb.COMPUTE_BACKEND_API_VERSION

            def lazy_netcdf_type(self):
                return _OtherLazyNetCDF

            def lazy_feature_collection_type(self):
                return _OtherLazyFrame

        backend = _OtherBackend()
        monkeypatch.setattr(cb, "_discover", lambda: {"other": backend})
        # These assertions are about which backend the seam resolves, not about whether the
        # [parquet] extra is installed. Core short-circuits both the LazyFeatureCollection
        # forwarding and is_lazy_fc on this flag before consulting any backend, so without it the
        # tests would only pass in an env that happens to carry dask-geopandas — and the fake
        # backend's frame is a plain class that needs none of it.
        import pyramids.feature as feature_pkg

        monkeypatch.setattr(feature_pkg, "_HAS_DASK_GEOPANDAS", True)
        cb.clear_compute_backend_cache()
        yield backend
        cb.clear_compute_backend_cache()

    def test_netcdf_public_name_comes_from_the_backend(self, third_party_backend):
        """``pyramids.netcdf.LazyNetCDF`` is the registered backend's class."""
        import pyramids.netcdf as netcdf_pkg

        assert netcdf_pkg.LazyNetCDF is third_party_backend.lazy_netcdf_type(), (
            "forwarding ignored the registered backend"
        )

    def test_feature_public_name_comes_from_the_backend(self, third_party_backend):
        """``pyramids.feature.LazyFeatureCollection`` is the registered backend's class."""
        import pyramids.feature as feature_pkg

        assert (
            feature_pkg.LazyFeatureCollection
            is third_party_backend.lazy_feature_collection_type()
        ), "forwarding ignored the registered backend"

    def test_is_lazy_fc_recognises_the_backends_frame(self, third_party_backend):
        """``is_lazy_fc`` answers True for the registered backend's own lazy frame.

        Test scenario:
            A dispatch predicate that answered False here would be silently wrong — the worst
            failure mode, since callers would route a lazy frame down the eager path.
        """
        from pyramids.feature import is_lazy_fc

        frame = third_party_backend.lazy_feature_collection_type()()
        assert is_lazy_fc(frame) is True, (
            "is_lazy_fc did not recognise the backend's frame"
        )

    def test_a_backend_missing_the_type_is_branded(self, monkeypatch):
        """A backend that omits the attribute gets a branded error naming it.

        Test scenario:
            ``lazy_netcdf_type`` is a Protocol *data* member, and ``runtime_checkable`` isinstance
            checks only methods, so its absence cannot be caught structurally and must be reported
            at resolution.
        """

        backend = _make_backend()
        del backend.lazy_netcdf_type
        assert not isinstance(backend, cb.ComputeBackend), (
            "a backend without the type must fail the structural check"
        )
        monkeypatch.setattr(cb, "_discover", lambda: {"other": backend})
        cb.clear_compute_backend_cache()
        import pyramids.netcdf as netcdf_pkg

        with pytest.raises(OptionalPackageDoesNotExist, match="lazy_netcdf_type"):
            netcdf_pkg.LazyNetCDF
        cb.clear_compute_backend_cache()


class TestMultipleBackendsWarn:
    """Two backends under distinct names must not resolve silently (review L14)."""

    def test_two_backends_warn_and_first_wins(self, monkeypatch):
        """With two registered backends, resolution warns and uses the first by discovery order."""
        first = _make_backend()
        second = _make_backend()
        monkeypatch.setattr(cb, "_discover", lambda: {"first": first, "second": second})
        cb.clear_compute_backend_cache()
        with pytest.warns(RuntimeWarning, match="Multiple compute backends"):
            resolved = cb.require_compute_backend()
        assert resolved is first, "first-by-discovery-order must win"
        cb.clear_compute_backend_cache()
