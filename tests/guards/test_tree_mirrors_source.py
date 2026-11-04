"""Guard meta-test: the test tree keeps mirroring ``src/pyramids`` and stays collectable.

Locks in the structural cleanup (every ``src/pyramids`` package has a test home, no
uncategorised ``test_*.py`` floating at the ``tests/`` root, no file that *looks* like a test
but cannot be collected). A test that silently drops out of collection — a hyphenated name
that ``python_files = test_*.py`` never matches, or a stray repro at the root nobody runs —
is the same invisible-coverage failure the extras-marker guard defends against, one directory
level up.

Dependency-free (``pathlib`` only), so it runs in the bare-wheel ``-m core`` job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.core

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "pyramids"
_TESTS = _REPO / "tests"

# src/pyramids packages whose tests deliberately live somewhere other than tests/<name>/.
# Empty today — every immediate source package has its own tests/<name>/ directory. A future
# exception must be listed here *with a reason*, never granted by loosening the assertion, so
# the mirror stays auditable.
_TESTS_HOME_EXCEPTIONS: dict[str, str] = {}

# Files allowed to sit directly at the tests/ root: shared helpers (underscore-prefixed),
# the pytest bootstrap, and the HTTP mock. Anything matching test_*.py at the root is a
# mis-placed test and must move into the mirroring subdirectory.
_ALLOWED_ROOT_FILES = {
    "__init__.py",
    "conftest.py",
    "_marks.py",
    "_skip_guard.py",
    "_helpers.py",
    "_driver_discovery_fixtures.py",
    "http_mock.py",
}


def _source_packages() -> list[str]:
    """Immediate ``src/pyramids`` subpackages (a directory carrying ``__init__.py``)."""
    return sorted(
        child.name
        for child in _SRC.iterdir()
        if child.is_dir() and (child / "__init__.py").exists()
    )


def test_every_source_package_has_a_tests_directory():
    """Each ``src/pyramids/<pkg>/`` must have a mirroring ``tests/<pkg>/`` (or a listed exception)."""
    missing = [
        pkg
        for pkg in _source_packages()
        if pkg not in _TESTS_HOME_EXCEPTIONS and not (_TESTS / pkg).is_dir()
    ]
    assert not missing, (
        f"these src/pyramids packages have no mirroring tests/<pkg>/ directory: {missing}. "
        f"Add the directory (the tree mirrors the source), or record the package in "
        f"_TESTS_HOME_EXCEPTIONS with a reason if its tests genuinely live elsewhere."
    )


def test_no_stray_test_file_at_the_tests_root():
    """No ``test_*.py`` may sit at the tests/ root; every test lives in a mirroring subdir."""
    stray = sorted(p.name for p in _TESTS.glob("test_*.py"))
    assert not stray, (
        f"these test files sit at the tests/ root instead of a subdirectory mirroring the "
        f"source: {stray}. Move each into the tests/<area>/ that matches the module it covers."
    )


def test_no_unexpected_python_file_at_the_tests_root():
    """The tests/ root holds only shared helpers + conftest, never a loose module."""
    unexpected = sorted(
        p.name
        for p in _TESTS.glob("*.py")
        if p.name not in _ALLOWED_ROOT_FILES
    )
    assert not unexpected, (
        f"unexpected Python file(s) at the tests/ root: {unexpected}. Shared helpers belong in "
        f"the subdir that uses them; if a new cross-cutting helper is truly root-level, add it to "
        f"_ALLOWED_ROOT_FILES."
    )


def test_no_hyphenated_python_file_is_silently_uncollectable():
    """A hyphenated .py filename never matches ``test_*.py`` — it would be a test that never runs."""
    hyphenated = sorted(
        str(p.relative_to(_REPO)) for p in _TESTS.rglob("*-*.py")
    )
    assert not hyphenated, (
        f"these .py files under tests/ carry a hyphen, so pytest's `test_*.py` pattern cannot "
        f"collect them — a hyphenated repro or helper that silently never runs: {hyphenated}. "
        f"Rename with underscores (and give it a `test_` prefix if it is meant to be collected)."
    )


def test_ugrid_tests_mirror_the_nested_source_package():
    """UGRID source is src/pyramids/netcdf/ugrid/, so its tests live under tests/netcdf/ugrid/."""
    assert (_TESTS / "netcdf" / "ugrid").is_dir(), (
        "tests/netcdf/ugrid/ is missing — the UGRID tests must mirror src/pyramids/netcdf/ugrid/."
    )
    assert not (_TESTS / "ugrid").is_dir(), (
        "tests/ugrid/ exists again — the UGRID tests belong under tests/netcdf/ugrid/ to mirror "
        "src/pyramids/netcdf/ugrid/, not at the top level."
    )
