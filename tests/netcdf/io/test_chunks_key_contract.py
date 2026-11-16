"""Contract tests for the keys a ``chunks=`` dict accepts on the NetCDF read paths.

Issue #1582: twelve documentation snippets spelled ``chunks={"time": ...}`` — a dimension-name
key no backend has ever accepted — and nothing executed them, so neither the test suite, the
nbval run nor the docs build noticed. These tests execute the specs instead of describing them:
they pin the accepted key vocabulary (integer axis index plus the ``rows`` / ``cols`` /
``columns`` / ``bands`` aliases), pin the rejection of everything else, and pin the three places
where a bad key fails *silently* rather than loudly. A backend change that widened or narrowed
the vocabulary, or that turned one of the silent paths loud, would fail here instead of being
discovered in the docs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.lazy

da = pytest.importorskip("dask.array")

NC_3D = "tests/data/netcdf/cf__4v__1d3-3d1__proj__y-desc.nc"
NC_2D = "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"
NC_4D = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
NC_MF = "tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"

UNKNOWN_KEY = r"Unknown chunks dict key"


@pytest.fixture(scope="function")
def var_3d():
    """Yield the 3-D ``values`` variable, whose eager read is ``(3, 13, 14)``.

    Yields:
        The variable handle; the owning NetCDF is closed on teardown.
    """
    nc = NetCDF.read_file(NC_3D)
    try:
        yield nc.get_variable("values")
    finally:
        nc.close()


@pytest.fixture(scope="function")
def var_2d():
    """Yield the 2-D ``Band1`` variable, whose eager read has no band axis.

    Yields:
        The variable handle; the owning NetCDF is closed on teardown.
    """
    nc = NetCDF.read_file(NC_2D)
    try:
        yield nc.get_variable("Band1")
    finally:
        nc.close()


@pytest.fixture(scope="function")
def var_4d():
    """Yield the 4-D ``temperature`` variable, which has two non-spatial axes.

    Yields:
        The variable handle; the owning NetCDF is closed on teardown.
    """
    nc = NetCDF.read_file(NC_4D)
    try:
        yield nc.get_variable("temperature")
    finally:
        nc.close()


class TestChunksDictAcceptedKeys:
    """Tests for the keys ``NetCDF`` variable ``read_array(chunks=...)`` resolves to an axis."""

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ({0: 1}, ((1, 1, 1), (13,), (14,))),
            ({1: 5}, ((3,), (5, 5, 3), (14,))),
            ({2: 5}, ((3,), (13,), (5, 5, 4))),
            ({"bands": 1}, ((1, 1, 1), (13,), (14,))),
            ({"BANDS": 1}, ((1, 1, 1), (13,), (14,))),
            ({"rows": 5}, ((3,), (5, 5, 3), (14,))),
            ({"ROWS": 5}, ((3,), (5, 5, 3), (14,))),
            ({"cols": 5}, ((3,), (13,), (5, 5, 4))),
            ({"columns": 5}, ((3,), (13,), (5, 5, 4))),
            ({"COLS": 5}, ((3,), (13,), (5, 5, 4))),
        ],
    )
    def test_accepted_key_chunks_the_axis_it_names(self, var_3d, spec, expected):
        """Each accepted key chunks exactly the axis it resolves to on a ``(3, 13, 14)`` read.

        Args:
            var_3d: The 3-D ``values`` variable fixture.
            spec: The ``chunks=`` dict under test.
            expected: The dask ``.chunks`` tuple the spec must produce.

        Test scenario:
            Asserting the resolved chunk layout rather than merely "did not raise" is what makes
            this non-vacuous: ``rows`` must land on axis ``ndim - 2`` and ``cols`` on ``ndim - 1``,
            so a backend that swapped them, or that silently ignored the spec and returned the
            default layout, fails here. Name keys are case-insensitive.
        """
        lazy = var_3d.read_array(chunks=spec)
        assert isinstance(lazy, da.Array), (
            f"chunks={spec} must return a dask array, got {type(lazy).__name__}"
        )
        assert lazy.chunks == expected, (
            f"chunks={spec} produced layout {lazy.chunks}, expected {expected}"
        )

    def test_a_chunked_read_preserves_the_eager_values(self, var_3d):
        """A chunked read computes to the same values as the eager read.

        Args:
            var_3d: The 3-D ``values`` variable fixture.

        Test scenario:
            The key vocabulary is only useful if honouring it still returns the right data, so a
            spec that chunks every axis must round-trip to the eager array element for element.
        """
        eager = np.asarray(var_3d.read_array())
        lazy = var_3d.read_array(chunks={"bands": 1, "rows": 5, "cols": 5})
        np.testing.assert_array_equal(
            np.asarray(lazy.compute()),
            eager,
            err_msg="a fully chunked read returned different values than the eager read",
        )


class TestChunksDictRejectedKeys:
    """Tests for the keys ``read_array(chunks=...)`` refuses with a ``ValueError``."""

    @pytest.mark.parametrize(
        "key, match",
        [
            ("time", UNKNOWN_KEY),
            ("lat", UNKNOWN_KEY),
            ("lon", UNKNOWN_KEY),
            ("x", UNKNOWN_KEY),
            ("y", UNKNOWN_KEY),
            ("banana", UNKNOWN_KEY),
            ((0,), UNKNOWN_KEY),
            (99, r"out of range"),
            (-1, r"out of range"),
            (True, r"is a bool"),
        ],
    )
    def test_unusable_key_raises_value_error(self, var_3d, key, match):
        """A key outside the accepted vocabulary raises ``ValueError`` on the static read path.

        Args:
            var_3d: The 3-D ``values`` variable fixture.
            key: The unusable ``chunks=`` dict key under test.
            match: Regex the raised message must contain.

        Test scenario:
            The dimension-name keys (``time`` / ``lat`` / ``lon`` / ``x`` / ``y``) are the #1582
            defect itself — they read as valid because the entry point is xarray-shaped, but the
            backend keys by axis. A negative index is rejected rather than wrapping from the end,
            and ``True`` is rejected rather than resolving as axis 1 through ``bool``'s ``int``
            inheritance.
        """
        with pytest.raises(ValueError, match=match) as exc_info:
            var_3d.read_array(chunks={key: 1})
        assert "chunks" in str(exc_info.value), (
            f"message should name the chunks spec, got: {exc_info.value}"
        )

    def test_a_valid_key_alongside_an_unusable_one_still_raises(self, var_3d):
        """One bad key poisons the whole dict — a partially valid spec is not salvaged.

        Args:
            var_3d: The 3-D ``values`` variable fixture.

        Test scenario:
            The documented snippets mixed a working alias with a dimension name. Resolution must
            not quietly drop the unusable entry and chunk by the rest, which would hide the typo.
        """
        with pytest.raises(ValueError, match=UNKNOWN_KEY):
            var_3d.read_array(chunks={"rows": 5, "time": 1})


class TestBandsAliasArity:
    """Tests for the ``bands`` alias, which depends on how many non-spatial axes exist."""

    def test_bands_resolves_the_sole_non_spatial_axis(self, var_3d):
        """On a 3-D read ``bands`` is unambiguous and chunks the leading axis.

        Args:
            var_3d: The 3-D ``values`` variable fixture.

        Test scenario:
            Exactly one non-spatial axis exists, so ``bands`` resolves to axis 0.
        """
        lazy = var_3d.read_array(chunks={"bands": 1})
        assert lazy.chunks[0] == (1, 1, 1), (
            f"bands must chunk the leading axis, got {lazy.chunks}"
        )

    def test_bands_is_not_meaningful_on_a_2d_read(self, var_2d):
        """A 2-D read has no band axis, so ``bands`` is refused rather than ignored.

        Args:
            var_2d: The 2-D ``Band1`` variable fixture.

        Test scenario:
            Silently dropping the key would make the same spec mean different things depending on
            the variable's rank.
        """
        with pytest.raises(ValueError, match=r"not meaningful"):
            var_2d.read_array(chunks={"bands": 1})

    def test_bands_is_ambiguous_with_two_non_spatial_axes(self, var_4d):
        """A 4-D read has two non-spatial axes, so ``bands`` is refused as ambiguous.

        Args:
            var_4d: The 4-D ``temperature`` variable fixture.

        Test scenario:
            Picking either candidate axis would be a guess, so the backend refuses and points the
            caller at integer axis indices, which stay unambiguous at any rank.
        """
        with pytest.raises(ValueError, match=r"ambiguous"):
            var_4d.read_array(chunks={"bands": 1})

    def test_an_integer_axis_still_works_at_4d(self, var_4d):
        """The integer-index escape hatch the ``bands`` error recommends actually works.

        Args:
            var_4d: The 4-D ``temperature`` variable fixture.

        Test scenario:
            An error message that recommends an unusable alternative is as bad as no message, so
            the recommended ``{0: 1}`` form is exercised on the same variable that refuses
            ``bands``.
        """
        lazy = var_4d.read_array(chunks={0: 1})
        assert isinstance(lazy, da.Array), (
            f"an integer axis key must chunk a 4-D read, got {type(lazy).__name__}"
        )
        assert lazy.chunks[0] == (1, 1, 1, 1), (
            f"axis 0 must be chunked one element per block, got {lazy.chunks}"
        )


class TestChunkReportIsNotRoundTrippable:
    """Tests that ``NetCDF.chunk(...).chunks`` reports a mapping ``chunks=`` cannot consume."""

    def test_reported_keys_are_dimension_names_not_chunks_keys(self, var_3d):
        """The reported mapping is keyed by dimension name, which ``chunks=`` rejects.

        Args:
            var_3d: The 3-D ``values`` variable fixture.

        Test scenario:
            This asymmetry is the trap that produced #1582: the obvious way to learn the key
            spelling is to chunk a dataset and read back ``.chunks``, but that mapping reports
            dimension names, so copying it into ``chunks=`` raises. The assertion is on the
            general property — the reported key set is not a subset of the accepted vocabulary —
            so it does not hard-code which dimension names a given fixture happens to carry.
        """
        nc = NetCDF.read_file(NC_3D)
        try:
            reported = set(dict(nc.chunk({"rows": 5, "cols": 5}).chunks))
        finally:
            nc.close()
        accepted = {"bands", "rows", "cols", "columns"}
        assert reported - accepted, (
            f"reported keys {reported} are all valid chunks= keys; the documented round-trip "
            "trap this test guards would no longer exist"
        )
        for key in sorted(reported - accepted):
            with pytest.raises(ValueError, match=UNKNOWN_KEY):
                var_3d.read_array(chunks={key: 1})


class TestToXarrayChunksFailsSilently:
    """Tests that ``to_xarray(chunks=...)`` degrades to an eager read instead of raising."""

    @pytest.mark.interop
    def test_a_dimension_name_key_returns_an_unchunked_dataset(self):
        """``chunks={"time": 1}`` yields a fully eager dataset — no error, no warning.

        Test scenario:
            The per-variable fallback catches the ``ValueError`` the key would otherwise raise and
            reads that variable eagerly, so the caller gets a silently unchunked result. Empty
            ``chunksizes`` is the only observable signal, which is why the docstring tells callers
            to check it. A future change that let this raise, or that chunked anyway, breaks here.
        """
        nc = NetCDF.read_file(NC_3D)
        try:
            result = nc.to_xarray(chunks={"time": 1})
        finally:
            nc.close()
        assert dict(result.chunksizes) == {}, (
            f"a dimension-name key must leave the dataset eager, got {dict(result.chunksizes)}"
        )

    @pytest.mark.interop
    def test_an_integer_axis_key_actually_chunks(self):
        """``chunks={0: 1}`` chunks the leading axis, proving the previous test is not vacuous.

        Test scenario:
            Without this companion case, an empty ``chunksizes`` could mean ``to_xarray`` never
            chunks at all. A working spec on the same file shows the eager result above is caused
            by the key, not by the entry point.
        """
        nc = NetCDF.read_file(NC_3D)
        try:
            result = nc.to_xarray(chunks={0: 1})
        finally:
            nc.close()
        sizes = dict(result.chunksizes)
        assert sizes, "an integer axis key must produce a chunked dataset, got an eager one"
        assert (1, 1, 1) in sizes.values(), (
            f"the leading axis must be chunked one element per block, got {sizes}"
        )


class TestOpenMfdatasetChunksArePerFile:
    """Tests that ``open_mfdataset(chunks=...)`` resolves keys against the per-file variable."""

    def test_an_axis_beyond_the_per_file_rank_is_rejected(self):
        """A 3-axis spec is refused for a 2-D per-file variable, despite the stacked result.

        Test scenario:
            Stacking two files along a new leading axis makes the *result* 3-D, so a caller may
            reasonably reach for axis 2. The spec is applied to the per-file variable instead,
            which is 2-D, so axis 2 is out of range. This is the one place the accepted axis range
            depends on something other than the object the caller holds.
        """
        with pytest.raises(ValueError, match=r"out of range"):
            NetCDF.open_mfdataset([NC_MF, NC_MF], variable="area", chunks={2: 1})

    @pytest.mark.parametrize("spec", [{0: 1}, {"rows": 8}, {"cols": 8}])
    def test_a_spec_valid_for_the_per_file_rank_is_accepted(self, spec):
        """Specs within the 2-D per-file rank are accepted, so the rejection above is about rank.

        Args:
            spec: A ``chunks=`` dict valid for a 2-D per-file variable.

        Test scenario:
            Axis 0 and both spatial aliases resolve against the 2-D per-file variable, which rules
            out "``open_mfdataset`` rejects every chunks spec" as the explanation for the failure
            in the previous test.
        """
        lazy = NetCDF.open_mfdataset([NC_MF, NC_MF], variable="area", chunks=spec)
        assert isinstance(lazy, da.Array), (
            f"chunks={spec} must return a dask array, got {type(lazy).__name__}"
        )
