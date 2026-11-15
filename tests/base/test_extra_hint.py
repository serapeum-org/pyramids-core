"""The composers behind the optional-dependency install hints.

Every guard for an optional dependency raised the same two-line install block
with only the extra's name changing, and each had written it out. `extra_hint`
composes that block for one of core's own extras.

The lazy backend is not an extra -- it is the separate `pyramids-lazy`
distribution -- so its guards use `lazy_backend_hint`, which names the package.
"""

from __future__ import annotations

import builtins

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import extra_hint, import_xarray, lazy_backend_hint

pytestmark = pytest.mark.core


class TestExtraHint:
    """The lead is the caller's; the install commands are not."""

    def test_the_lead_sentence_is_preserved_verbatim(self):
        """The lead says which operation needed the dependency."""
        lead = "Zarr IO requires the optional 'zarr' dependency."

        assert extra_hint(lead, "lazy").startswith(lead)

    @pytest.mark.parametrize("extra", ["lazy", "stac", "parquet", "viz"])
    def test_both_commands_name_the_extra(self, extra: str):
        """PyPI and conda-forge instructions both track the extra."""
        hint = extra_hint("Needs something.", extra)

        assert f"pyramids-core[{extra}]" in hint
        assert f"conda-forge pyramids-{extra}" in hint

    def test_the_header_line_is_stable(self):
        """Call sites and their tests match on this wording."""
        assert extra_hint("X.", "lazy").splitlines()[0] == "X. Install with one of:"


# Needs the [interop] extra: `to_xarray` / `from_xarray` import xarray, which a
# bare install does not have. The module-level `core` mark still selects these
# under `-m core`, and `tests/conftest.py` turns this marker into the matching
# skip when xarray is absent -- which is what the pure-wheel job needs.
@pytest.mark.interop
class TestLazyBackendHint:
    """The hint core's own lazy guards raise: it names a package, not an extra."""

    def test_the_whole_message_is_pinned(self):
        """The full rendered string, since this is what users read on a missing backend.

        Test scenario:
            A lead sentence produces the lead, the header, and exactly one PyPI
            command naming the plugin distribution.
        """
        hint = lazy_backend_hint("Lazy reads need the compute backend.")

        assert hint == (
            "Lazy reads need the compute backend. Install it with:\n"
            "  - PyPI:        pip install pyramids-lazy"
        ), f"rendered hint changed: {hint!r}"

    def test_the_lead_sentence_is_preserved_verbatim(self):
        """The caller's lead opens the message untouched.

        Test scenario:
            The lead says which operation needed the backend, so it must not be
            reworded or truncated.
        """
        lead = "open_mfdataset requires the optional 'dask' dependency."
        hint = lazy_backend_hint(lead)

        assert hint.startswith(lead), f"lead not preserved: {hint!r}"

    def test_it_never_names_a_core_extra(self):
        """No core extra can install the backend, so none may be suggested.

        Test scenario:
            `[lazy]` was removed and the backend is a plugin, so a hint naming
            `pyramids-core[...]` would send the user somewhere that cannot help.
        """
        hint = lazy_backend_hint("X needs the backend.")

        assert "pyramids-core[" not in hint, f"hint names a core extra: {hint!r}"

    def test_it_offers_no_conda_line(self):
        """Deliberately PyPI-only, unlike `extra_hint`'s two-line block.

        Test scenario:
            conda-forge's `pyramids-lazy` output is core's old lazy extra -- the
            raw dask/zarr libraries with no backend in them -- so naming it would
            not resolve the error. The omission is the point, so it is pinned.
        """
        hint = lazy_backend_hint("X needs the backend.")

        assert "conda" not in hint, f"unexpected conda remediation: {hint!r}"
        assert len(hint.splitlines()) == 2, (
            f"expected a lead line and one command line, got {hint.splitlines()!r}"
        )

    @pytest.mark.parametrize(
        "lead",
        ["", "Trailing space. ", "Braces {not} a placeholder.", "Unicode — dash."],
        ids=["empty", "trailing-space", "braces", "unicode"],
    )
    def test_the_command_survives_any_lead(self, lead: str):
        """The install command is appended whatever the lead looks like.

        Args:
            lead: Lead sentence variant, including empty and format-special text.

        Test scenario:
            The lead is interpolated as a value, so braces must not be treated as
            a format placeholder, and an empty lead must still yield the command.
        """
        hint = lazy_backend_hint(lead)

        assert "pip install pyramids-lazy" in hint, f"command missing for {lead!r}"
        assert hint.startswith(lead), f"lead {lead!r} not preserved in {hint!r}"

    def test_it_is_pure(self):
        """Repeated calls return the same string and keep no state.

        Test scenario:
            The composer is used at module scope by several guards, so it must be
            a pure function of its argument.
        """
        first = lazy_backend_hint("X needs the backend.")
        second = lazy_backend_hint("X needs the backend.")

        assert first == second, f"not deterministic: {first!r} vs {second!r}"


class TestImportXarray:
    """The two xarray guards share one importer.

    Both call sites need the live module back -- one to build a `Dataset`, the
    other to type-check the argument it was handed -- so this importer returns
    it rather than only asserting the import worked.
    """

    # Needs the live module back, which a bare core install lacks. The interop
    # mark is what `tests/conftest.py` turns into a skip when xarray is absent,
    # so the pure-wheel core job (no extras) selects it under `-m core` but
    # skips rather than fails. The sibling `test_a_missing_xarray_raises...`
    # stays unmarked: it monkeypatches the import away and must run everywhere.
    @pytest.mark.interop
    def test_it_returns_the_module_itself(self):
        """A guard that returned None would force a second import.

        Test scenario:
            With xarray installed, the call yields the module, and the module
            is usable: constructing a `DataArray` from it works.
        """
        xr = import_xarray("unused: xarray is installed in this environment")

        assert xr.__name__ == "xarray"
        assert xr.DataArray([1, 2, 3]).shape == (3,)

    @pytest.mark.interop
    def test_calling_it_twice_returns_the_same_module(self):
        """Guards run per call; re-importing must not build a new module.

        Test scenario:
            Python caches modules in `sys.modules`, so two guarded imports have
            to yield the identical object -- otherwise an `isinstance` check
            against a type from one would fail for a value made by the other.
        """
        first = import_xarray("unused")
        second = import_xarray("unused")

        assert first is second

    def test_a_missing_xarray_raises_the_hint_it_was_given(self, monkeypatch):
        """The message is the caller's, so the guard must not rewrite it.

        Args:
            monkeypatch: Fixture used to make the import fail.

        Test scenario:
            `builtins.__import__` is patched to reject xarray, simulating an
            environment without the package. The raised error must carry the
            caller's own install hint verbatim -- that hint names the operation
            that needed xarray, which a generic message would lose.
        """
        hint = "to_xarray() needs xarray. Install with: pip install xarray"
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "xarray":
                raise ImportError("No module named 'xarray'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)

        with pytest.raises(OptionalPackageDoesNotExist) as exc_info:
            import_xarray(hint)

        assert hint in str(exc_info.value), f"hint not carried: {exc_info.value}"
