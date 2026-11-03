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


def _extras_targets() -> list[dict[str, str]]:
    """Parse the extras-package matrix targets into ``{name, env, paths, markexpr, strict}`` dicts.

    Dependency-free line parse scoped to the ``extras-package:`` job block (targets are block-style
    YAML). Only a lowercase ``- name:`` line starts a target, so the job's step names ("Set up
    Python", "Run ... tests") are not mistaken for targets.
    """
    block = re.search(
        r"^  extras-package:.*?(?=^\S|\Z)", _workflow_text(), re.MULTILINE | re.DOTALL
    )
    region = block.group(0) if block else _workflow_text()
    targets: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in region.splitlines():
        name = re.match(r"\s*- name:\s*([a-z][a-z0-9_]*)\s*$", line)
        if name:
            current = {"name": name.group(1)}
            targets.append(current)
            continue
        if current is not None:
            for key in ("env", "paths", "markexpr", "strict"):
                field = re.match(rf'\s*{key}:\s*"?([^"\n]*)"?\s*$', line)
                if field:
                    current[key] = field.group(1).strip()
    return targets


def _markexpr_markers(markexpr: str) -> set[str]:
    """Marker tokens a ``markexpr`` selects positively (drops the boolean operators)."""
    return {
        tok
        for tok in re.findall(r"[a-z_]+", markexpr)
        if tok not in {"not", "and", "or"}
    }


def _markers_under(paths: str) -> set[str]:
    """Extras markers carried by test files under a target's ``paths`` (what the job runs)."""
    found: set[str] = set()
    for part in paths.split():
        root = _REPO / part
        files = [root] if root.is_file() else root.rglob("test_*.py")
        for file in files:
            found.update(
                re.findall(r"pytest\.mark\.([a-z_]+)", file.read_text(encoding="utf-8"))
            )
    return found & set(EXTRA_MARKERS)


def _markers_run_in_ci() -> set[str]:
    """Extras markers a CI job actually SELECTS — via markexpr, or a test under the target's path."""
    covered = set(_MAIN_PACKAGE_MARKERS)
    for target in _extras_targets():
        covered |= _markexpr_markers(target.get("markexpr", ""))
        if target.get("paths"):
            covered |= _markers_under(target["paths"])
    return covered


def test_every_extras_marker_is_run_by_a_ci_job():
    """Each registered extras marker must be *selected* by some CI job, not silently skipped."""
    missing = set(EXTRA_MARKERS) - _markers_run_in_ci()
    assert not missing, (
        f"these extras markers run in NO CI job: {sorted(missing)}. A marker with tests but no job "
        f"that selects them skips silently in every run (what happened to `stac`). Extend an "
        f"extras-package target in .github/workflows/tests.yml so the marker is selected — via "
        f"`markexpr`, or a `paths` that contains the marker's tests — or, if it runs in the "
        f"main-package job, add it to _MAIN_PACKAGE_MARKERS with a justification. A target merely "
        f"*named* after the marker no longer counts; the job must select it. See docs/how-to/testing.md."
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
