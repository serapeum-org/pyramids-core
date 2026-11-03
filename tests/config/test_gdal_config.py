"""Tests for the scoped GDAL-config context manager `pyramids.gdal_config`."""

from osgeo import gdal

from pyramids import gdal_config


class TestGdalConfig:
    """`gdal_config` applies options process-wide and restores them on exit."""

    def test_sets_and_restores(self):
        """An option is visible inside the block and the prior value is restored after."""
        prior = gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
        with gdal_config(GDAL_HTTP_TIMEOUT="123") as applied:
            assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "123"
            assert applied["GDAL_HTTP_TIMEOUT"] == "123"
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == prior

    def test_process_wide_option_accepted(self):
        """A process-wide-only option (GDAL_CACHEMAX) is applied without raising.

        GDAL consumes GDAL_CACHEMAX specially so GetConfigOption does not echo it;
        the point is that the scoped manager accepts it (thread_local=False) rather
        than rejecting it the way a thread-local scope would. A companion echo-able
        option proves the manager body actually mutates GDAL state (and restores
        it), so the test is sensitive to the manager body, not just the built dict.
        """
        prior = gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
        with gdal_config(GDAL_CACHEMAX="512", GDAL_HTTP_TIMEOUT="77") as applied:
            assert applied["GDAL_CACHEMAX"] == "512"
            assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "77"
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == prior

    def test_nesting_restores_outer(self):
        """An inner block overrides a key; the outer value returns when it exits."""
        with gdal_config(CPL_DEBUG="ON"):
            with gdal_config(CPL_DEBUG="OFF"):
                assert gdal.GetConfigOption("CPL_DEBUG") == "OFF"
            assert gdal.GetConfigOption("CPL_DEBUG") == "ON"

    def test_bool_coercion(self):
        """Booleans map to GDAL's YES/NO."""
        with gdal_config(GDAL_HTTP_UNSAFESSL=True) as applied:
            assert applied["GDAL_HTTP_UNSAFESSL"] == "YES"
            assert gdal.GetConfigOption("GDAL_HTTP_UNSAFESSL") == "YES"
        with gdal_config(GDAL_HTTP_UNSAFESSL=False) as applied:
            assert applied["GDAL_HTTP_UNSAFESSL"] == "NO"

    def test_non_str_coercion(self):
        """Non-bool, non-str values are str()-ified."""
        with gdal_config(GDAL_HTTP_TIMEOUT=256) as applied:
            assert applied["GDAL_HTTP_TIMEOUT"] == "256"
            assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "256"

    def test_none_unsets_for_block(self):
        """A None value unsets the option for the block, then restores it."""
        with gdal_config(CPL_DEBUG="ON"):
            with gdal_config(CPL_DEBUG=None) as applied:
                assert "CPL_DEBUG" not in applied
                assert gdal.GetConfigOption("CPL_DEBUG") in (None, "")
            assert gdal.GetConfigOption("CPL_DEBUG") == "ON"

    def test_empty_call_is_noop_block(self):
        """gdal_config() with no options is a valid empty block."""
        with gdal_config() as applied:
            assert applied == {}
