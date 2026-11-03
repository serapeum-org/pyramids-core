"""Guard: every extras marker is executed by at least one CI job (no marker runs nowhere).

This is the check that would have caught the ``stac`` suite running in zero CI jobs: an extras
marker that has tests but no CI job that selects it in an environment that installs its dependency
**skips silently in every run** — green, folded into the skip count, undiscovered for months. This
guard turns that into a loud, same-PR failure. See ``docs/how-to/testing.md`` for the mechanism.

Dependency-free (``re`` + ``tomllib`` only) so it runs in the bare-wheel ``-m core`` job too.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tests._marks import EXTRA_MARKERS

pytestmark = pytest.mark.core

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "tests.yml"
_PYPROJECT = _REPO / "pyproject.toml"

# Markers executed by the main-package job rather than the extras-package matrix:
#  - "lazy": the py3XX matrix envs carry the `lazy` feature and the main pytest step does not
#    exclude lazy, so lazy-marked tests run there for real;
#  - "interop" / "plot": run by the dedicated `interop-tests` / `plot` steps of the main-package job.
# Each entry is validated against the live config by test_main_package_marker_allowlist_is_not_stale.
_MAIN_PACKAGE_MARKERS = {"lazy", "interop", "plot"}


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _extras_target_names() -> set[str]:
    """Names of the extras-package matrix targets (lowercase identifiers on their own line).

    Step names ("Set up Python", "Run ... tests") contain spaces/capitals and are not matched, so
    this picks up only the ``- name: <marker>`` matrix entries.
    """
    return set(
        re.findall(r"- name:\s*([a-z][a-z0-9_]*)\s*$", _workflow_text(), re.MULTILINE)
    )


def _extras_markexpr_tokens() -> set[str]:
    """Marker tokens that appear positively in any extras-package ``markexpr``."""
    tokens: set[str] = set()
    for expr in re.findall(r'markexpr:\s*"([^"]*)"', _workflow_text()):
        tokens.update(
            tok
            for tok in re.findall(r"[a-z_]+", expr)
            if tok not in {"not", "and", "or"}
        )
    return tokens


def test_every_extras_marker_is_run_by_a_ci_job():
    """Each registered extras marker must be run by some CI job, not silently skipped everywhere."""
    covered = _MAIN_PACKAGE_MARKERS | _extras_target_names() | _extras_markexpr_tokens()
    missing = set(EXTRA_MARKERS) - covered
    assert not missing, (
        f"these extras markers run in NO CI job: {sorted(missing)}. A marker with tests but no job "
        f"skips silently in every run (what happened to `stac`). Add a target to the extras-package "
        f"matrix in .github/workflows/tests.yml named after the marker (or selecting it via "
        f"`markexpr`), or — if it runs in the main-package job — add it to _MAIN_PACKAGE_MARKERS with "
        f"a justification. See docs/how-to/testing.md."
    )


def test_main_package_marker_allowlist_is_not_stale():
    """The _MAIN_PACKAGE_MARKERS allow-list must still match how the workflow/envs are wired."""
    text = _workflow_text()
    assert "interop" in text and "plot" in text, (
        "the main-package job no longer references interop/plot; _MAIN_PACKAGE_MARKERS is stale"
    )
    envs = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["pixi"][
        "environments"
    ]
    py_envs = [name for name in envs if re.fullmatch(r"py3\d+", name)]
    assert py_envs, "no py3XX pixi environments found to carry the lazy feature"
    for name in py_envs:
        features = (
            envs[name].get("features", []) if isinstance(envs[name], dict) else []
        )
        assert "lazy" in features, (
            f"env {name} no longer carries the `lazy` feature; `lazy` main-matrix coverage is stale"
        )
