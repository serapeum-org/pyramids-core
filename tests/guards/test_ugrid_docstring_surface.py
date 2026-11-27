"""Guard meta-test: every mesh public method documents the parameters it actually takes.

A parameter added to a signature without a matching `Args:` entry is invisible: the method
works, every behaviour test passes, and the rendered reference page simply never mentions
the knob.

Since #1650 the mesh public API lives on engine accessors (`mesh.io`, `mesh.spatial`,
`mesh.analysis`, ...), and the historical flat `MeshDataset.<name>` methods are thin
`DeprecationWarning` facades that just delegate. So the real documented surface this guard
must police is the **engine** methods; the flat facades are exempt (their one-line docstring
points at the engine, where the parameters are documented). `MeshDataset`'s own non-facade
public methods -- constructors, the `Mapping` protocol, `copy`/`summary`/attribute mutation --
are still checked directly.

Static reflection over the live classes, so it needs no docs build and no optional dependency.
"""

from __future__ import annotations

import inspect
import re

import pytest

from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid.engines.analysis import _MeshAnalysis
from pyramids.netcdf.ugrid.engines.glyph import _MeshGlyph
from pyramids.netcdf.ugrid.engines.interop import _MeshInterop
from pyramids.netcdf.ugrid.engines.io import _MeshIO
from pyramids.netcdf.ugrid.engines.selection import _MeshSelection
from pyramids.netcdf.ugrid.engines.spatial import _MeshSpatial
from pyramids.netcdf.ugrid.engines.terrain import _MeshTerrain
from pyramids.netcdf.ugrid.engines.topology import _MeshTopology
from pyramids.netcdf.ugrid.engines.variables import _MeshVariables
from pyramids.netcdf.ugrid.engines.vector import _MeshVector

pytestmark = pytest.mark.core

_ARG_LINE = re.compile(r"\s*\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:")

_ENGINES = [
    _MeshIO,
    _MeshInterop,
    _MeshSpatial,
    _MeshSelection,
    _MeshAnalysis,
    _MeshTerrain,
    _MeshVector,
    _MeshGlyph,
    _MeshVariables,
    _MeshTopology,
]


def _documented_parameters(doc: str) -> set[str]:
    """The parameter names a Google-style `Args:` section lists.

    Args:
        doc: The de-indented docstring to read.

    Returns:
        set[str]: The names at the `Args:` block's own indentation level; continuation lines
        of a description sit deeper and are ignored.
    """
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return set()
    indent = len(lines[start]) - len(lines[start].lstrip())
    names: set[str] = set()
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        here = len(line) - len(line.lstrip())
        if here <= indent:
            break
        match = _ARG_LINE.match(line)
        if match and here == indent + 4:
            names.add(match.group(1))
    return names


def _declared_parameters(member) -> set[str]:
    """The names a callable takes, minus `self`/`cls` and the `*args`/`**kwargs` catch-alls."""
    signature = inspect.signature(member)
    return {
        name
        for name, spec in signature.parameters.items()
        if name not in ("self", "cls")
        and spec.kind not in (spec.VAR_POSITIONAL, spec.VAR_KEYWORD)
    }


def _is_deprecated_facade(member) -> bool:
    """True when the callable is a flat deprecation facade that delegates to an engine.

    A facade's body emits a `DeprecationWarning`; its parameters are documented on the engine
    method it forwards to, so this guard checks the engine, not the facade.
    """
    try:
        source = inspect.getsource(member)
    except (OSError, TypeError):
        return False
    return "DeprecationWarning" in source


def _public_callables(cls: type, *, skip_facades: bool) -> list[tuple[str, object]]:
    """Public callables of `cls` (inherited included); optionally drop deprecation facades."""
    found = []
    for name, member in inspect.getmembers(cls, callable):
        if name.startswith("_"):
            continue
        try:
            inspect.signature(member)
        except (TypeError, ValueError):
            continue
        if skip_facades and _is_deprecated_facade(member):
            continue
        found.append((name, member))
    return found


def _surface() -> list[tuple[str, object]]:
    """The documented mesh surface: engine methods, plus non-facade MeshDataset methods."""
    items: list[tuple[str, object]] = [
        (f"MeshDataset.{name}", member)
        for name, member in _public_callables(MeshDataset, skip_facades=True)
    ]
    for engine in _ENGINES:
        label = engine.__name__.lstrip("_")
        items.extend(
            (f"{label}.{name}", member)
            for name, member in _public_callables(engine, skip_facades=False)
        )
    return items


_CALLABLES = _surface()


class TestUgridDatasetDocstringSurface:
    """Every documented mesh entry point against its own signature."""

    def test_the_sweep_sees_the_whole_surface(self):
        """The sweep below is only as good as the surface it walks.

        Test scenario:
            The mesh API spans `MeshDataset` (constructors, the `Mapping` protocol) plus ten
            engine classes. A sweep that forgot the engines, or forgot the mixin-supplied
            members, would silently check a fraction of the API and pass. Pin a floor and
            representative names from both the class and an engine.
        """
        names = {name for name, _ in _CALLABLES}

        assert len(names) >= 90, (
            f"expected at least 90 documented callables, got {len(names)}"
        )
        assert {
            "MeshDataset.from_arrays",
            "MeshSpatial.crop",
            "MeshAnalysis.where",
        } <= names, (
            "the sweep must reach both MeshDataset constructors and the engine methods"
        )

    @pytest.mark.parametrize("name,member", _CALLABLES, ids=[n for n, _ in _CALLABLES])
    def test_every_parameter_is_documented(self, name, member):
        """A parameter in the signature has an `Args:` entry on the documented method.

        Args:
            name: The `<owner>.<attr>` label of the callable.
            member: The callable itself.

        Test scenario:
            A parameter added to an engine method (or a non-facade MeshDataset method) must be
            listed in that method's own `Args:` block; the flat deprecation facades are exempt
            because their parameters are documented on the engine they delegate to.
        """
        doc = inspect.getdoc(member) or ""
        missing = sorted(_declared_parameters(member) - _documented_parameters(doc))

        assert not missing, (
            f"{name} takes {missing} but its docstring's Args section does not list them"
        )
