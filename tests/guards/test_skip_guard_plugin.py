"""Unit tests for the ``--strict-extra-skips`` dependency-skip detector (``tests/_skip_guard.py``).

These pin the two gaps a review found in the gate: an ``xfail`` report (which pytest marks
``skipped=True`` with ``wasxfail`` set) must not be counted as a dependency-skip, and the detector
must recognise every dependency-skip idiom the suite uses — not just the literal "not installed".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._skip_guard import is_unexpected_dependency_skip

pytestmark = pytest.mark.core


def _report(reason: str, *, skipped: bool = True, wasxfail=None) -> SimpleNamespace:
    """A minimal stand-in for a pytest report: skip longrepr is a (path, lineno, text) tuple."""
    return SimpleNamespace(
        skipped=skipped, wasxfail=wasxfail, longrepr=("f.py", 1, f"Skipped: {reason}")
    )


@pytest.mark.parametrize(
    "reason",
    [
        "pyramids-core[viz] not installed",
        "the optional interop dependency is not installed",
        "could not import 'pyarrow'",
        "dask.dataframe needs dask + pyarrow (pyramids-core[parquet])",
    ],
)
def test_dependency_skip_idioms_are_flagged(reason: str):
    """Every missing-dependency skip phrasing the suite produces trips the gate."""
    assert is_unexpected_dependency_skip(_report(reason))


@pytest.mark.parametrize(
    "reason",
    [
        "live network test; run it with `-m live`",
        "intentional non-dependency skip",
        "known platform quirk on this runner",
    ],
)
def test_non_dependency_skips_are_ignored(reason: str):
    """A skip that is not about a missing dependency does not trip the gate."""
    assert not is_unexpected_dependency_skip(_report(reason))


def test_xfail_reports_are_not_counted():
    """An xfail (skipped=True with wasxfail) is not a dependency-skip even if its reason matches."""
    report = _report(
        "OGR Parquet driver (libgdal-arrow-parquet) not installed", wasxfail="absent"
    )
    assert not is_unexpected_dependency_skip(report)


def test_passed_reports_are_ignored():
    """A non-skip report is never flagged."""
    assert not is_unexpected_dependency_skip(_report("", skipped=False))
