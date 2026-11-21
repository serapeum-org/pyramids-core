"""Direct unit tests for the shared exclude-list / frame-label helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset._exclude import build_exclude_list, require_one_label_per_frame


class TestBuildExcludeList:
    """`build_exclude_list` assembles the no-data-plus-extra exclusion list."""

    def test_sentinel_and_extra(self):
        """A sentinel and an extra value both appear, in order."""
        assert build_exclude_list(-9999, 5) == [-9999, 5]

    def test_sentinel_only(self):
        """No extra (missing or None) yields the sentinel alone."""
        assert build_exclude_list(-9999) == [-9999]
        assert build_exclude_list(-9999, None) == [-9999]

    def test_none_sentinel_is_dropped(self):
        """A None sentinel contributes nothing -- the animate path's `None -> []`."""
        assert build_exclude_list(None) == []
        assert build_exclude_list(None, 5) == [5]

    def test_nan_sentinel_is_kept(self):
        """A NaN sentinel is a value, not None, so it stays in the list."""
        out = build_exclude_list(float("nan"))
        assert len(out) == 1
        assert np.isnan(out[0])


class TestRequireOneLabelPerFrame:
    """`require_one_label_per_frame` guards the animate label count."""

    def test_matching_counts_pass(self):
        """Equal counts raise nothing."""
        require_one_label_per_frame(3, 3)

    def test_mismatch_raises_with_the_shared_message(self):
        """A count mismatch raises the reconciled frame-oriented message."""
        with pytest.raises(ValueError, match=r"got 2 frame label\(s\) for 3 frame"):
            require_one_label_per_frame(2, 3)
