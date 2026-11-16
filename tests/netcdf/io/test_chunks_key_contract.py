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
NC_PACKED = "tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc"

UNKNOWN_KEY = r"Unknown chunks dict key"


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def var_packed_2d():
    """Yield the CF-packed ``z`` variable, stored 2-D but reported as ``(1, 21, 21)``.

    Yields:
        The variable handle; the owning NetCDF is closed on teardown.
    """
    nc = NetCDF.read_file(NC_PACKED)
    try:
        yield nc.get_variable("z")
    finally:
        nc.close()


@pytest.fixture
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
        with pytest.raises(ValueError, match=match):
            var_3d.read_array(chunks={key: 1})

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
        assert sizes, (
            "an integer axis key must produce a chunked dataset, got an eager one"
        )
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
            which is 2-D, so axis 2 is out of range. The accepted axis range therefore depends on
            something other than the object the caller holds -- as it also does for a variable
            stored at a lower rank than ``Variable.shape`` reports, pinned by
            :class:`TestKeysResolveAgainstTheStoredRankNotTheReportedShape`.
        """
        with pytest.raises(ValueError, match=r"out of range"):
            NetCDF.open_mfdataset([NC_MF, NC_MF], variable="area", chunks={2: 1})

    @pytest.mark.parametrize(
        ("spec", "axis", "blocks"),
        [({0: 1}, 1, 128), ({"rows": 8}, 1, 16), ({"cols": 8}, 2, 32)],
    )
    def test_a_spec_valid_for_the_per_file_rank_resolves_that_axis(
        self, spec, axis, blocks
    ):
        """A valid per-file spec chunks the stack axis its per-file axis became.

        Args:
            spec: A ``chunks=`` dict valid for a 2-D per-file variable.
            axis: The stacked-result axis the spec is expected to land on.
            blocks: The number of blocks expected along that axis.

        Test scenario:
            Asserting the resolved layout rather than the return type is what makes this
            discriminating: ``open_mfdataset`` returns a dask array with no ``chunks=`` at all, and
            that baseline leaves one block on every axis. The per-file axis the spec names shifts by
            one in the result, because stacking prepends the file axis -- which stays one block per
            file, pinning that ``chunks=`` cannot address it.
        """
        lazy = NetCDF.open_mfdataset([NC_MF, NC_MF], variable="area", chunks=spec)
        assert len(lazy.chunks[axis]) == blocks, (
            f"chunks={spec} should split result axis {axis} into {blocks} blocks, "
            f"got {lazy.chunks[axis]}"
        )
        assert lazy.chunks[0] == (1, 1), (
            f"the file axis must stay one block per file, got {lazy.chunks[0]}"
        )


class TestTheFacadeFormTheDocsUseBehavesIdentically:
    """``NetCDF.read_array(variable, chunks=...)`` -- the spelling the fixed snippets use."""

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ({0: 1, 1: 5, 2: 5}, ((1, 1, 1), (5, 5, 3), (5, 5, 4))),
            ({"rows": 5}, ((3,), (5, 5, 3), (14,))),
            ({"BANDS": 1}, ((1, 1, 1), (13,), (14,))),
        ],
    )
    def test_the_facade_resolves_the_documented_vocabulary(self, spec, expected):
        """The container-level facade resolves int keys and aliases exactly as the variable does.

        Args:
            spec: A ``chunks=`` dict in the documented vocabulary.
            expected: The per-axis chunk layout the spec must produce.

        Test scenario:
            Nine of the twelve snippets corrected for #1582 call ``nc.read_array("var", chunks=...)``
            rather than the variable handle, so the contract has to be pinned on that signature too
            -- otherwise the tests guard a path the documentation never takes.
        """
        nc = NetCDF.read_file(NC_3D)
        try:
            assert nc.read_array("values", chunks=spec).chunks == expected, (
                f"facade chunks={spec} should resolve to {expected}"
            )
        finally:
            nc.close()

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ({"time": 1}, UNKNOWN_KEY),
            ({-1: 1}, r"out of range"),
            ({True: 1}, r"is a bool"),
        ],
    )
    def test_the_facade_rejects_what_the_variable_rejects(self, spec, match):
        """An unusable key raises through the facade as well, not only the variable handle.

        Args:
            spec: A ``chunks=`` dict carrying an unusable key.
            match: Pattern the raised message must match.

        Test scenario:
            The dimension-name key is the #1582 defect reached by the documented spelling.
        """
        nc = NetCDF.read_file(NC_3D)
        try:
            with pytest.raises(ValueError, match=match):
                nc.read_array("values", chunks=spec)
        finally:
            nc.close()

    def test_unpack_and_chunks_together_unpack_inside_the_graph(self):
        """``unpack=True`` with ``chunks=`` unpacks *and* chunks -- the combination a snippet shows.

        Test scenario:
            ``lazy-netcdf.md`` documents CF unpacking applied inside the dask graph, so the two
            arguments have to compose. Run on the ``scaleoffset`` fixture rather than on
            ``values``, which declares no ``scale_factor`` at all: against an unpacked variable
            ``unpack=True`` is inert and a layout-only assertion passes verbatim with
            ``unpack=False``, pinning the chunk spec and nothing about unpacking. Here the
            computed result must equal the eager *unpacked* read and differ from the raw stored
            one -- which fails if unpacking is dropped inside the graph -- while the layout must
            still honour the spec, which fails if unpacking forces an eager read.
        """
        nc = NetCDF.read_file(NC_PACKED)
        try:
            raw = np.asarray(nc.read_array("z", unpack=False))
            eager = np.asarray(nc.read_array("z", unpack=True))
            lazy = nc.read_array("z", unpack=True, chunks={"rows": 3})
            assert lazy.chunks == ((3, 3, 3, 3, 3, 3, 3), (21,)), (
                f"unpack=True must not defeat the chunk spec, got {lazy.chunks}"
            )
            computed = np.asarray(lazy.compute())
            np.testing.assert_array_equal(
                computed,
                eager,
                err_msg="the chunked unpacked read differs from the eager unpacked read",
            )
            assert not np.array_equal(computed, raw), (
                "the chunked read returned the raw stored values, so unpack=True was dropped "
                "inside the dask graph"
            )
        finally:
            nc.close()


class TestKeysResolveAgainstTheStoredRankNotTheReportedShape:
    """Tests that a key resolves against the variable's *stored* rank, not ``Variable.shape``.

    ``Variable.shape`` pads a 2-D stored variable to ``(1, rows, cols)`` so the eager read
    presents a bands axis, but the chunked read returns the stored rank and resolves ``chunks=``
    keys against *that*. The shape the caller can see therefore advertises an axis the spec
    vocabulary does not have -- the same "reads as valid, resolves elsewhere" shape as #1582, and
    the more dangerous half of it, because the index case resolves to the wrong axis in silence
    rather than raising.
    """

    def test_the_fixture_reports_a_rank_above_the_one_it_stores(self, var_packed_2d):
        """``Variable.shape`` is 3-D while the chunked read is 2-D -- the premise of this class.

        Args:
            var_packed_2d: The 2-D ``z`` variable fixture.

        Test scenario:
            Stated as its own assertion so the three tests below cannot be misread as covering a
            variable that is genuinely 3-D. If a future read path made the two ranks agree, this
            fails first and names the reason the rest of the class exists.
        """
        assert len(var_packed_2d.shape) == 3, (
            f"fixture must report a padded 3-D shape, got {var_packed_2d.shape}"
        )
        assert var_packed_2d.read_array(chunks={"rows": 3}).ndim == 2, (
            "fixture must store a 2-D variable for this class to mean anything"
        )

    def test_an_axis_the_reported_shape_offers_is_out_of_range(self, var_packed_2d):
        """Axis 2 is rejected although ``Variable.shape`` has three entries.

        Args:
            var_packed_2d: The 2-D ``z`` variable fixture.

        Test scenario:
            A caller sizing a spec off ``var.shape == (1, 21, 21)`` reads axis 2 as the columns
            axis. The spec is resolved against the stored ``ndim=2`` instead, so the index is
            refused and the message quotes a rank the caller never saw.
        """
        with pytest.raises(ValueError, match=r"out of range for ndim=2"):
            var_packed_2d.read_array(chunks={2: 1})

    def test_bands_is_refused_though_the_reported_shape_leads_with_one(
        self, var_packed_2d
    ):
        """``bands`` is refused as 2-D although ``Variable.shape`` leads with a length-1 axis.

        Args:
            var_packed_2d: The 2-D ``z`` variable fixture.

        Test scenario:
            ``(1, 21, 21)`` is exactly the shape a single-band 3-D variable has, so ``bands`` is
            the natural spelling for chunking it. The stored rank carries no band axis, so the
            alias raises -- the same refusal a genuinely 2-D variable gets, reached from a shape
            that does not look 2-D.
        """
        with pytest.raises(ValueError, match=r"not meaningful"):
            var_packed_2d.read_array(chunks={"bands": 1})

    def test_axis_zero_silently_chunks_rows_not_the_reported_leading_axis(
        self, var_packed_2d
    ):
        """``{0: 1}`` chunks rows, not the length-1 leading axis of the reported shape.

        Args:
            var_packed_2d: The 2-D ``z`` variable fixture.

        Test scenario:
            This is the silent half, and the reason the two raising cases above are not the whole
            story: ``{0: 1}`` on a reported ``(1, 21, 21)`` reads as "one band per block" and is
            resolved as "one *row* per block". Nothing raises and nothing warns, so the only
            signal is a graph with 21 blocks where the caller expected one. Pinned by equality
            with the ``rows`` spelling rather than by a literal layout, so it states which axis
            was chosen rather than merely how many blocks appeared.
        """
        by_index = var_packed_2d.read_array(chunks={0: 1})
        by_rows = var_packed_2d.read_array(chunks={"rows": 1})
        assert by_index.chunks == by_rows.chunks, (
            f"axis 0 must resolve to the rows axis; the index spec gave {by_index.chunks} and "
            f"the rows spec gave {by_rows.chunks}"
        )
        assert len(by_index.chunks[0]) == 21, (
            f"the rows axis must split into 21 blocks of one row, got {by_index.chunks[0]}"
        )


class TestToXarrayFallsBackOneVariableAtATime:
    """Tests that ``to_xarray(chunks=...)`` degrades per variable, so a result can be part eager."""

    @pytest.mark.interop
    def test_a_spec_only_the_deepest_variable_accepts_leaves_the_rest_eager(self):
        """``chunks={3: 1}`` chunks the one 4-D variable and silently reads the others eagerly.

        Test scenario:
            The per-variable fallback catches the ``ValueError`` for each variable separately, so
            a spec whose axis exists in only some of them splits the result: a store holding 2-D,
            3-D and 4-D variables yields chunked and eager variables from a single call, with no
            error and no warning. ``TestToXarrayChunksFailsSilently`` covers only all-bad and
            all-good specs, where the result is uniform and the degradation is invisible as a
            *per-variable* effect.

            The last assertion is what the fallback costs the caller: the docstring's remedy is
            to check ``result.chunksizes``, and that check does not survive a mixed result.
            ``chunksizes`` is keyed by dimension, not by variable, so the chunked variable alone
            populates it -- while eager variables sharing those dimensions sit in the same
            dataset. A non-empty ``chunksizes`` therefore does not mean the spec took effect for
            any particular variable, which is the limit of the advice as documented.
        """
        nc = NetCDF.read_file(NC_MF)
        try:
            result = nc.to_xarray(chunks={3: 1})
        finally:
            nc.close()
        chunked = sorted(
            name for name, var in result.data_vars.items() if var.chunks is not None
        )
        eager = sorted(
            name for name, var in result.data_vars.items() if var.chunks is None
        )
        assert chunked, (
            f"a mixed-rank store must chunk at least one variable under an axis-3 spec; "
            f"eager={eager}"
        )
        assert eager, (
            f"a mixed-rank store must leave at least one variable eager under an axis-3 "
            f"spec; chunked={chunked}"
        )
        assert all(len(result[name].dims) < 4 for name in eager), (
            f"only variables below 4-D may fall back for an axis-3 spec, got eager={eager}"
        )
        assert all(len(result[name].dims) == 4 for name in chunked), (
            f"only a 4-D variable can honour an axis-3 spec, got chunked={chunked}"
        )
        chunksizes = dict(result.chunksizes)
        shared = [name for name in eager if set(result[name].dims) & set(chunksizes)]
        assert shared, (
            "expected an eager variable sharing a dimension with the chunked one, so that a "
            f"non-empty chunksizes provably does not describe it; eager={eager}, "
            f"chunksizes={sorted(chunksizes)}"
        )


class TestChunksArgumentTypeScreen:
    """Tests the ``chunks=`` *type* screen, which runs before any key is resolved."""

    @pytest.mark.parametrize(
        "spec",
        [pytest.param(3.5, id="float"), pytest.param(object(), id="object")],
    )
    def test_an_unsupported_chunks_type_raises_type_error(self, var_3d, spec):
        """A type outside the accepted set raises ``TypeError``, not ``ValueError``.

        Args:
            var_3d: The 3-D ``values`` variable fixture.
            spec: The unsupported ``chunks=`` value under test.

        Test scenario:
            Every other rejection on this path is a ``ValueError``, so a caller writing
            ``except ValueError`` around a read lets this one through. Pinned by exception type
            as well as message because the two classes are caught differently -- notably by
            ``to_xarray``'s per-variable fallback, which swallows only ``ValueError``, so a bad
            *type* fails the whole conversion where a bad *key* degrades silently.
        """
        with pytest.raises(TypeError, match=r"Unsupported chunks type"):
            var_3d.read_array(chunks=spec)

    def test_a_list_spec_is_accepted_exactly_like_a_tuple(self, var_3d):
        """A ``list`` of per-axis sizes is accepted, though the documented type list omits it.

        Args:
            var_3d: The 3-D ``values`` variable fixture.

        Test scenario:
            ``read_array``'s ``Raises:`` section enumerates the accepted types as ``None`` /
            ``int`` / ``tuple`` / ``dict`` / ``"auto"``, so a reader concludes a list raises
            ``TypeError``. It does not -- the backend's own message names ``list`` among the
            accepted types and resolves one positionally, like a tuple. Pinned so the gap is
            closed by correcting that sentence rather than by narrowing the code to match it,
            which would break any caller already passing a list.
        """
        assert var_3d.read_array(chunks=[1, 5, 5]).chunks == (
            (1, 1, 1),
            (5, 5, 3),
            (5, 5, 4),
        ), "a list spec must resolve positionally, exactly as the same tuple does"
