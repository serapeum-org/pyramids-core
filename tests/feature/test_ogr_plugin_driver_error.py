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
from pyramids.feature._read import _missing_ogr_plugin, _read_info_branded

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
