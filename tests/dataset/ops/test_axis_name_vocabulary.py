"""One list of the names a coordinate axis goes by.

The GeoTIFF path knew eleven spellings of the x axis -- `x`, `lon`, `rlon` from
a rotated-pole grid, `nav_lon` from NEMO, `easting` from a projected one. The
Zarr reader kept its own list of six.

These are the backend-free vocabulary/ordering invariants. The real-zarr
``detect_data_var`` tests (which build stores to exercise the reader) moved to the
pyramids-lazy plugin — they need the zarr stack.
"""

from __future__ import annotations

import pytest

from pyramids.base._axes import (
    AXIS_NAMES,
    X_AXIS_NAMES,
    X_AXIS_NAMES_ORDERED,
    Y_AXIS_NAMES,
    Y_AXIS_NAMES_ORDERED,
)
from pyramids.dataset.dataset import (
    _AXIS_VARIABLE_NAMES,
    _X_AXIS_NAMES,
    _Y_AXIS_NAMES,
)
from pyramids.dataset.ops._geobox_zarr import (
    _ALWAYS_COORDS,
    _NEVER_DATA_ARRAYS,
    _NON_DATA_ARRAYS,
)

pytestmark = pytest.mark.core


class TestTheVocabularyIsOneList:
    """The three names in `dataset` are the shared ones, not copies."""

    def test_the_two_halves_are_the_shared_ones(self):
        """A copy would drift the moment either list gained a spelling."""
        assert _X_AXIS_NAMES is X_AXIS_NAMES
        assert _Y_AXIS_NAMES is Y_AXIS_NAMES

    def test_the_union_is_derived_rather_than_restated(self):
        """It used to be a third literal list that could disagree.

        Test scenario:
            The union was written out by hand next to the two halves it was
            meant to combine, with a comment asserting they partitioned it
            exactly. Deriving it makes that true by construction.
        """
        assert _AXIS_VARIABLE_NAMES == _X_AXIS_NAMES | _Y_AXIS_NAMES

    def test_the_halves_do_not_overlap(self):
        """A name cannot be both axes, and a pairing rule relies on that.

        Test scenario:
            The x/y split exists so a rule can require a *pair* of axes to
            agree before it fires. A spelling landing in both halves would let
            one array satisfy that pair on its own.
        """
        assert not (X_AXIS_NAMES & Y_AXIS_NAMES)

    @pytest.mark.parametrize(
        "name",
        ["nav_lon", "nav_lat", "rlon", "rlat", "easting", "northing", "xc", "yc"],
    )
    def test_the_less_common_spellings_are_all_there(self, name: str):
        """These are the ones the shorter list dropped.

        Args:
            name: A coordinate spelling from a real model convention.

        Test scenario:
            NEMO writes `nav_lon`/`nav_lat`, a rotated-pole grid writes
            `rlon`/`rlat`, a projected one `easting`/`northing`. Each has to be
            recognised as a coordinate by every reader, not just one.
        """
        assert name in AXIS_NAMES


class TestTheZarrReaderKnowsThemAll:
    """The reader's non-data vocabulary is derived from the shared list."""

    def test_its_non_data_set_covers_the_shared_vocabulary(self):
        """Derived from the shared list, so it cannot fall behind again."""
        assert AXIS_NAMES <= _NON_DATA_ARRAYS

    def test_it_still_excludes_the_non_spatial_arrays(self):
        """The reader's own additions must survive the derivation.

        Test scenario:
            `time`, `band` and the CRS arrays are not axes but are not data
            either. Composing the set from the axis vocabulary must add to
            them, not replace them.
        """
        assert {"time", "band", "crs"} <= _NON_DATA_ARRAYS


class TestTheTiersAreNested:
    """The exclusion tiers are nested, so the fallback can only ever widen."""

    def test_the_three_tiers_are_nested(self):
        """The tiers are nested, so the fallback can only ever widen.

        Test scenario:
            Names that are only ever coordinates stay excluded in both tiers;
            the ambiguous ones sit between, excluded first and allowed second.
        """
        assert _ALWAYS_COORDS <= _NEVER_DATA_ARRAYS <= _NON_DATA_ARRAYS


class TestTheOrderedSequences:
    """Order is part of the definition, so its shape is pinned too."""

    @pytest.mark.parametrize(
        ("ordered", "first"),
        [(X_AXIS_NAMES_ORDERED, "x"), (Y_AXIS_NAMES_ORDERED, "y")],
        ids=["x", "y"],
    )
    def test_the_grid_dimension_name_comes_first(self, ordered, first):
        """The whole point of the ordering, stated on the sequence itself.

        Args:
            ordered: One of the two preference sequences.
            first: The name that must lead it.

        Test scenario:
            A reader picking one axis from an array carrying several has to
            reach the grid's own dimension before any auxiliary alias, so `x`
            and `y` lead. Sorting alphabetically puts `east` and `lat` first,
            which is the regression this ordering exists to prevent.
        """
        assert ordered[0] == first

    @pytest.mark.parametrize(
        "ordered",
        [X_AXIS_NAMES_ORDERED, Y_AXIS_NAMES_ORDERED],
        ids=["x", "y"],
    )
    def test_the_geographic_aliases_come_last(self, ordered):
        """On a projected grid these describe the same cells in other units.

        Args:
            ordered: One of the two preference sequences.

        Test scenario:
            Every `lon`/`lat`-family spelling must sit after every
            dimension/projected one, or a projected grid carrying auxiliary
            degree coordinates picks the degrees.
        """
        geographic = {
            "lon",
            "long",
            "longitude",
            "nav_lon",
            "lat",
            "latitude",
            "nav_lat",
        }
        positions = [i for i, name in enumerate(ordered) if name in geographic]
        others = [i for i, name in enumerate(ordered) if name not in geographic]

        assert min(positions) > max(others)

    @pytest.mark.parametrize(
        "ordered",
        [X_AXIS_NAMES_ORDERED, Y_AXIS_NAMES_ORDERED],
        ids=["x", "y"],
    )
    def test_a_sequence_lists_each_name_once(self, ordered):
        """A repeat would make the preference order ambiguous.

        Args:
            ordered: One of the two preference sequences.
        """
        assert len(ordered) == len(set(ordered))

    @pytest.mark.parametrize(
        "ordered",
        [X_AXIS_NAMES_ORDERED, Y_AXIS_NAMES_ORDERED],
        ids=["x", "y"],
    )
    def test_every_name_is_lowercase(self, ordered):
        """Lookups compare raw coordinate names against these directly.

        Args:
            ordered: One of the two preference sequences.

        Test scenario:
            Nothing lowercases a store's coordinate names before the
            comparison, so a capitalised entry here would simply never match.
        """
        assert all(name == name.lower() for name in ordered)

    def test_the_sets_are_exactly_their_sequences(self):
        """Derived, so the two spellings of the vocabulary cannot diverge."""
        assert X_AXIS_NAMES == frozenset(X_AXIS_NAMES_ORDERED)
        assert Y_AXIS_NAMES == frozenset(Y_AXIS_NAMES_ORDERED)
        assert AXIS_NAMES == frozenset(X_AXIS_NAMES_ORDERED + Y_AXIS_NAMES_ORDERED)
