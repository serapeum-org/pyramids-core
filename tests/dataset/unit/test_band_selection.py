"""Unit tests for the shared band-selection normaliser.

`normalize_band_selection` is the single validator the three raster->table exporters
(`to_dataframe`, `to_xyz`, `to_geodataframe`) share, so a `bands=` subset is selected and
rejected identically everywhere. These tests pin that shared contract directly.
"""

import numpy as np
import pytest

from pyramids.base._utils import normalize_band_selection


class TestNormalizeBandSelection:
    """The shared `bands=` normaliser (`None`/int/sequence -> validated 0-based list)."""

    def test_none_expands_to_every_band(self):
        """`None` returns every 0-based index, so callers never special-case it."""
        assert normalize_band_selection(None, 3, caller="Dataset.to_xyz") == [0, 1, 2]

    def test_int_wraps_to_single_element_list(self):
        """A plain `int` selects exactly that one band."""
        assert normalize_band_selection(2, 3, caller="Dataset.to_xyz") == [2]

    def test_numpy_integer_is_accepted(self):
        """A NumPy integer is coerced like a Python `int` (array indexing hands these in)."""
        assert normalize_band_selection(np.int64(1), 3, caller="Dataset.to_xyz") == [1]

    def test_sequence_preserves_order(self):
        """A sequence keeps the caller's order -- `[2, 0]` is not sorted to `[0, 2]`."""
        assert normalize_band_selection([2, 0], 3, caller="Dataset.to_xyz") == [2, 0]

    def test_tuple_is_accepted(self):
        """Any sequence works, not only `list` (the old `to_xyz` accepted only `list`)."""
        assert normalize_band_selection((0, 1), 3, caller="Dataset.to_xyz") == [0, 1]

    def test_bool_is_rejected_as_typeerror(self):
        """`True`/`False` subclass `int`; reject them rather than select band 1/0."""
        with pytest.raises(TypeError, match="boolean"):
            normalize_band_selection(True, 3, caller="Dataset.to_dataframe")

    def test_str_is_rejected_as_typeerror(self):
        """A `str` is iterable; reject it rather than iterate characters into `int()`."""
        with pytest.raises(TypeError, match="string"):
            normalize_band_selection("invalid", 3, caller="Dataset.to_xyz")

    def test_empty_selection_is_rejected(self):
        """An empty sequence is a caller mistake -- `None` is how you ask for every band."""
        with pytest.raises(ValueError, match="empty bands="):
            normalize_band_selection([], 3, caller="Dataset.to_dataframe")

    def test_out_of_range_index_is_rejected(self):
        """An index outside `0..band_count - 1` names the offending value and the range."""
        with pytest.raises(ValueError, match=r"out of range.*3-band"):
            normalize_band_selection([0, 5], 3, caller="Dataset.to_geodataframe")

    def test_negative_index_is_rejected(self):
        """Negative indices are out of range -- band selection is not Python slicing."""
        with pytest.raises(ValueError, match="out of range"):
            normalize_band_selection([-1], 3, caller="Dataset.to_xyz")

    def test_caller_name_appears_in_message(self):
        """The `caller` is woven in so the error names the method the user actually called."""
        with pytest.raises(ValueError, match=r"Dataset\.to_geodataframe\(\)"):
            normalize_band_selection([9], 2, caller="Dataset.to_geodataframe")
