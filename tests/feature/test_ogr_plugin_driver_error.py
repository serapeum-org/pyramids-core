"""`iter_features` names the remedy when a format's OGR driver ships as an absent plugin.

Streaming a `.parquet` needs GDAL's OGR Parquet driver, which `libgdal-arrow-parquet` supplies and
which none of the pyramids wheels bundle (#457 measures the Arrow runtime's cost). Before this, the
attempt surfaced GDAL's own text — `not recognized as being in a supported file format`, or on
conda-Windows `Can't load requested DLL: duckdb.dll 126` — which tells the user neither what to
install nor that `read_parquet()` would have worked.

`ogr.GetDriverByName("Parquet")` returns `None` in both of those cases, so the condition is
detectable and the error can carry the remedy. These tests pin that it does, that it fires only for
a genuinely missing plugin driver, and that every other failure still propagates untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyramids.base._errors import VectorDriverError
from pyramids.feature import _read
from pyramids.feature._read import (
    _local_source_absent,
    _missing_ogr_plugin,
    _read_info_branded,
)
from pyramids.feature.collection import FeatureCollection

pytestmark = pytest.mark.core


class TestMissingPluginDetection:
    """Which paths are candidates for re-branding."""

    @pytest.mark.parametrize("suffix", [".parquet", ".geoparquet", ".PARQUET"])
    def test_a_parquet_suffix_maps_to_the_parquet_driver(self, monkeypatch, suffix):
        """Case-insensitive, because a path's suffix is the user's spelling."""
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)

        missing = _missing_ogr_plugin(f"points{suffix}")

        assert missing is not None
        assert missing[0] == "Parquet"
        assert missing[1] == "libgdal-arrow-parquet"

    @pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".shp", ""])
    def test_a_format_with_an_in_libgdal_driver_is_never_a_candidate(
        self, monkeypatch, suffix
    ):
        """Only the optional-plugin formats are in the map; everything else is untouched."""
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)

        assert _missing_ogr_plugin(f"points{suffix}") is None

    def test_a_present_driver_is_not_reported_missing(self, monkeypatch):
        """Once the driver is installable the branding must switch itself off."""
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: object())

        assert _missing_ogr_plugin("points.parquet") is None


class TestBrandedError:
    """What the user reads, and what they can act on."""

    @pytest.fixture
    def failing_read_info(self, monkeypatch):
        """Make `pyogrio.read_info` fail the way a missing driver makes it fail."""

        def boom(path, **kwargs):
            raise RuntimeError("not recognized as being in a supported file format")

        monkeypatch.setattr(_read.pyogrio, "read_info", boom)
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)

    @pytest.fixture
    def present(self, tmp_path: Path):
        """Create the source files on disk, so branding is not suppressed by their absence.

        `_read_info_branded` refuses to blame the driver for a path that is not there (M3), so a
        test about the *message* must use a file that exists.
        """

        def _make(name: str) -> str:
            path = tmp_path / name
            path.write_bytes(b"PAR1")
            return str(path)

        return _make

    def test_the_remedy_is_named(self, failing_read_info, present):
        """The driver, the conda package, and the no-driver alternative all appear."""
        with pytest.raises(VectorDriverError) as caught:
            _read_info_branded(present("points.parquet"))

        message = str(caught.value)
        assert 'no OGR "Parquet" driver' in message
        assert "conda install -c conda-forge libgdal-arrow-parquet" in message
        assert "FeatureCollection.read_parquet()" in message
        assert "GDAL_DRIVER_PATH" in message
        assert "pyramids.gdal_plugin_path" in message

    def test_it_never_points_at_a_core_extra(self, failing_read_info, present):
        """No `pip install pyramids-core[...]` can supply a native GDAL driver."""
        with pytest.raises(VectorDriverError) as caught:
            _read_info_branded(present("points.parquet"))

        assert "pyramids-core[" not in str(caught.value)

    def test_the_original_failure_is_chained(self, failing_read_info, present):
        """GDAL's own text stays reachable on `__cause__` for anyone debugging."""
        with pytest.raises(VectorDriverError) as caught:
            _read_info_branded(present("points.parquet"))

        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "supported file format" in str(caught.value.__cause__)

    def test_it_is_catchable_as_a_runtime_error(self, failing_read_info, present):
        """`VectorDriverError` multi-inherits `RuntimeError`, like every driver failure."""
        with pytest.raises(RuntimeError):
            _read_info_branded(present("points.parquet"))

    def test_the_arrow_formats_offer_no_read_parquet_alternative(
        self, failing_read_info, present
    ):
        """`.feather` has no in-memory FeatureCollection reader, so none is suggested."""
        with pytest.raises(VectorDriverError) as caught:
            _read_info_branded(present("points.feather"))

        message = str(caught.value)
        assert 'no OGR "Arrow" driver' in message
        assert "read_parquet()" not in message


class TestAMissingSourceIsNotADriverProblem:
    """A path that is not there must surface as itself, not as a packaging complaint."""

    @pytest.fixture
    def failing_read_info(self, monkeypatch):
        """Fail the way a missing local file fails, with no driver registered."""

        def boom(path, **kwargs):
            raise RuntimeError(f"{path}: No such file or directory")

        monkeypatch.setattr(_read.pyogrio, "read_info", boom)
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)

    def test_a_typod_parquet_path_is_not_branded(self, failing_read_info, tmp_path):
        """The regression M3 names: a typo sent the user to install a conda package.

        Test scenario:
            Every pyramids wheel lacks the OGR Parquet driver, so suffix + absent-driver is true
            for any `.parquet` path. Before this, a path that simply did not exist was reported as
            "The active GDAL has no OGR Parquet driver", advising a conda install and
            `read_parquet()` -- which fails the same way. The original error must propagate.
        """
        with pytest.raises(RuntimeError, match="No such file") as caught:
            _read_info_branded(str(tmp_path / "typo.parquet"))

        assert not isinstance(caught.value, VectorDriverError), (
            f"a missing file was branded as a missing driver: {caught.value}"
        )

    def test_an_existing_parquet_is_still_branded(self, failing_read_info, tmp_path):
        """The guard must not disable the branding for files that are really there.

        Test scenario:
            With the file present, suffix + absent-driver is a sound diagnosis again, so the
            actionable error is still raised.
        """
        present = tmp_path / "points.parquet"
        present.write_bytes(b"PAR1")

        with pytest.raises(VectorDriverError, match="Parquet"):
            _read_info_branded(str(present))

    def test_a_remote_path_is_never_probed(self, failing_read_info):
        """Confirming a remote object costs a round trip, so it counts as present.

        Test scenario:
            An `https://` source cannot be stat'ed locally; branding still applies, and a genuinely
            missing remote object fails with GDAL's own message instead.
        """
        with pytest.raises(VectorDriverError, match="Parquet"):
            _read_info_branded("https://example.com/points.parquet")


class TestTheBrandedMessageCarriesGdalsText:
    """The user should not have to inspect `__cause__` to see what GDAL actually said."""

    def test_the_original_text_is_in_the_message(self, monkeypatch, tmp_path):
        """GDAL's own line appears in the branded message, trimmed.

        Test scenario:
            On conda-Windows the real cause is a `duckdb.dll` load failure. Chaining it on
            `__cause__` alone hides it from anyone reading the error text, so it is appended.
        """
        present = tmp_path / "points.parquet"
        present.write_bytes(b"PAR1")

        def boom(path, **kwargs):
            raise RuntimeError("Can't load requested DLL: duckdb.dll 126" + os.linesep)

        monkeypatch.setattr(_read.pyogrio, "read_info", boom)
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)

        with pytest.raises(VectorDriverError) as caught:
            _read_info_branded(str(present))

        message = str(caught.value)
        assert "GDAL reported: Can't load requested DLL: duckdb.dll 126" in message, (
            f"GDAL's own text is missing from the branded message: {message}"
        )
        assert message == message.rstrip(), "GDAL's trailing newline should be trimmed"


class TestEverythingElsePropagates:
    """Re-branding is narrow: it must not swallow unrelated failures."""

    def test_a_failure_on_a_non_plugin_format_is_untouched(self, monkeypatch, tmp_path):
        """A missing `.geojson` still raises whatever pyogrio raised."""

        def boom(path, **kwargs):
            raise RuntimeError("No such file or directory")

        monkeypatch.setattr(_read.pyogrio, "read_info", boom)
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)

        with pytest.raises(RuntimeError, match="No such file"):
            _read_info_branded(str(tmp_path / "missing.geojson"))

    def test_a_parquet_failure_with_the_driver_present_is_untouched(
        self, monkeypatch, tmp_path
    ):
        """A corrupt Parquet is a data problem, not a packaging one."""

        def boom(path, **kwargs):
            raise RuntimeError("Parquet magic bytes not found in footer")

        monkeypatch.setattr(_read.pyogrio, "read_info", boom)
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: object())

        with pytest.raises(RuntimeError, match="magic bytes"):
            _read_info_branded(str(tmp_path / "corrupt.parquet"))

    def test_a_successful_read_passes_the_mapping_through(self, monkeypatch, tmp_path):
        """The happy path is a plain delegation, kwargs included."""
        seen: dict[str, object] = {}

        def record(path, **kwargs):
            seen.update(path=path, **kwargs)
            return {"features": 7}

        monkeypatch.setattr(_read.pyogrio, "read_info", record)

        info = _read_info_branded(str(tmp_path / "points.parquet"), layer="pts")

        assert info == {"features": 7}
        assert seen["layer"] == "pts"


class TestThroughThePublicApi:
    """The branding reached through `FeatureCollection.iter_features`, not the private helper.

    Every other test here calls `_read_info_branded` directly, which leaves the wiring at the
    single call site uncovered — and the public method is a generator, so how the error surfaces
    is itself part of the contract.
    """

    @pytest.fixture
    def parquet_without_a_driver(self, monkeypatch, tmp_path: Path):
        """A real file with a `.parquet` suffix, and no OGR Parquet driver registered."""
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: None)
        source = tmp_path / "reaches.parquet"
        source.write_bytes(b"PAR1 not really a parquet file")
        return source

    def test_iter_features_raises_the_branded_error(self, parquet_without_a_driver):
        """The remedy reaches a caller of the public API.

        Test scenario:
            `list()` drives the generator to its first `next()`, which is where the read happens.
        """
        with pytest.raises(VectorDriverError) as caught:
            list(FeatureCollection.iter_features(parquet_without_a_driver))

        message = str(caught.value)
        assert 'no OGR "Parquet" driver' in message, f"unexpected message: {message}"
        assert "FeatureCollection.read_parquet()" in message, (
            "the working alternative must be named"
        )

    def test_calling_it_does_not_raise_until_iterated(self, parquet_without_a_driver):
        """`iter_features` is a generator, so the call itself cannot raise.

        Test scenario:
            Guidance that shows the error at call time is wrong, and a caller wrapping only the
            call in `try` catches nothing. Binding the generator must be silent.
        """
        features = FeatureCollection.iter_features(parquet_without_a_driver)

        with pytest.raises(VectorDriverError):
            next(iter(features))

    def test_it_is_catchable_as_a_runtime_error_end_to_end(
        self, parquet_without_a_driver
    ):
        """Callers who widen from `DataSourceError` to `RuntimeError` keep working.

        Test scenario:
            `pyogrio.errors.DataSourceError` is itself a `RuntimeError` subclass, so an
            `except RuntimeError` handler written against the old behaviour still catches the
            new exception. This is the migration path the docstring points at.
        """
        with pytest.raises(RuntimeError):
            list(FeatureCollection.iter_features(parquet_without_a_driver))


class TestADeferredPluginProxyStillBrands:
    """A driver that exists on paper but cannot load is as unusable as an absent one.

    GDAL >= 3.9 registers a *deferred* proxy as soon as it finds a file named
    `ogr_Parquet.<ext>` on `GDAL_DRIVER_PATH`, and only dlopens it on first use. So
    `GetDriverByName` returns a driver in exactly the "installed but broken" case this branding
    exists for -- conda-Windows' `duckdb.dll 126` -- and keying on the lookup alone switched the
    remedy off there. Every other test here stubs `GetDriverByName` to `None`, so none of them
    can see it; these drive the predicate directly instead.
    """

    def test_a_registered_driver_with_a_load_failure_is_branded(self, tmp_path: Path):
        """The conda-Windows case: proxy registered, shared library unloadable."""
        source = tmp_path / "points.parquet"
        source.write_bytes(b"PAR1")

        missing = _missing_ogr_plugin(
            str(source), RuntimeError("Can't load requested DLL: duckdb.dll 126")
        )

        assert missing is not None, (
            "a registered-but-unloadable plugin must still produce the remedy"
        )
        assert missing[0] == "Parquet", f"wrong driver reported: {missing}"

    def test_the_gdal_not_available_wording_is_recognised(self, tmp_path: Path):
        """GDAL's other phrasing for the same condition, seen on Linux/macOS."""
        missing = _missing_ogr_plugin(
            str(tmp_path / "points.parquet"),
            RuntimeError(
                "It could have been recognized by driver Parquet, but plugin "
                "ogr_Parquet.so is not available in your installation."
            ),
        )

        assert missing is not None, (
            "the 'not available in your installation' text must brand"
        )

    def test_a_corrupt_file_is_not_a_plugin_problem(self, monkeypatch, tmp_path: Path):
        """A working driver failing on bad bytes must not be branded.

        Test scenario:
            With the driver registered and the failure text carrying no plugin marker, the
            diagnosis is a data problem, so the original error propagates.
        """
        monkeypatch.setattr(_read.ogr, "GetDriverByName", lambda name: object())

        missing = _missing_ogr_plugin(
            str(tmp_path / "points.parquet"),
            RuntimeError("Parquet magic bytes not found in footer"),
        )

        assert missing is None, (
            f"a corrupt file was branded as a plugin failure: {missing}"
        )


class TestALocalArchiveIsStatted:
    """`/vsizip` and friends address the local disk, so a typo there is still a typo.

    `_io._parse_path` turns `reaches.zip` into `/vsizip/<abs>/reaches.zip/inner.parquet` before
    the read helper sees it. Treating every `/vsi` form as present left a mistyped archive
    reported as a missing OGR driver -- the exact mis-diagnosis the guard was added to kill.
    """

    def test_a_missing_archive_is_absent(self, tmp_path: Path):
        """The container does not exist, so this is not a driver problem."""
        path = f"/vsizip/{tmp_path / 'nope.zip'}/inner.parquet"

        assert _local_source_absent(path) is True, (
            "a mistyped local archive must be reported absent"
        )

    def test_a_real_archive_is_present(self, tmp_path: Path):
        """The container exists, so branding remains available for it."""
        archive = tmp_path / "ok.zip"
        archive.write_bytes(b"PK")
        path = f"/vsizip/{archive}/inner.parquet"

        assert _local_source_absent(path) is False, (
            "an existing archive must not be reported absent"
        )

    def test_a_network_vsi_path_is_never_statted(self):
        """Confirming a remote object costs a round trip, so it counts as present."""
        assert (
            _local_source_absent("/vsicurl/https://example.com/x.parquet") is False
        ), "a network /vsi path must not be probed"
