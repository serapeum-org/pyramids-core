"""Tests for pyramids.basemap.basemap.

``add_basemap`` is a thin wrapper over the cleopatra glyph ``add_tiles``
method (``cleopatra.basemap.geo.GeoMixin.add_tiles``) and ``get_provider``
over ``cleopatra.basemap.tiles.get_provider``. As of ``cleopatra >= 0.46``
the module-level ``cleopatra.basemap.tiles.add_tiles`` was made private
(``_add_tiles``); web tiles are added through the glyph method, which
``add_basemap`` drives from a bare ``GeoMixin`` with the axes passed
explicitly. These tests cover the delegation contract: the right cleopatra
entry point is called with the translated kwargs, its return value is
propagated, the missing-extra error path is wired up, and the public
signature has not drifted (downstream code patches against it).
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.plot

pytest.importorskip(
    "cleopatra.basemap.geo", reason="cleopatra[tiles] extra not installed"
)

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.basemap.basemap import add_basemap, get_provider

_ADD_TILES = "cleopatra.basemap.geo.GeoMixin.add_tiles"


class TestGetProvider:
    """``get_provider`` delegates to ``cleopatra.basemap.tiles.get_provider``."""

    def test_delegates_to_cleopatra_get_provider(self):
        """The ``name`` argument is forwarded and the result returned verbatim.

        Test scenario:
            Patch ``cleopatra.basemap.tiles.get_provider`` with a sentinel and
            confirm ``get_provider("CartoDB.Positron")`` calls it once
            with that name and hands back its return value.
        """
        with patch("cleopatra.basemap.tiles.get_provider") as mock_get:
            result = get_provider("CartoDB.Positron")

        mock_get.assert_called_once_with("CartoDB.Positron")
        assert result is mock_get.return_value, (
            "get_provider must return cleopatra.basemap.tiles.get_provider's result"
        )

    def test_default_provider_round_trip(self):
        """A no-arg call resolves the real default provider through cleopatra.

        Test scenario:
            With cleopatra installed, ``get_provider()`` should resolve
            the OpenStreetMap.Mapnik default — a smoke test that the
            wrapper does not mangle the call.
        """
        provider = get_provider()
        name = getattr(provider, "name", "") or str(provider)
        assert "openstreetmap" in name.lower(), (
            f"default provider should be OpenStreetMap, got {provider!r}"
        )

    def test_missing_extra_raises(self):
        """When the cleopatra ``[tiles]`` extra is absent the guard fires.

        Test scenario:
            Patch the ``import_basemap`` guard to raise
            ``OptionalPackageDoesNotExist`` and confirm ``get_provider``
            surfaces it (rather than an opaque ``ImportError``).
        """
        with patch(
            "pyramids.basemap.basemap.import_basemap",
            side_effect=OptionalPackageDoesNotExist("no cleopatra[tiles]"),
        ):
            with pytest.raises(OptionalPackageDoesNotExist):
                get_provider("CartoDB.Positron")


class TestAddBasemap:
    """``add_basemap`` delegates to the cleopatra glyph ``add_tiles`` method."""

    def test_delegates_with_default_kwargs(self):
        """A bare ``add_basemap(ax)`` forwards the documented defaults.

        Test scenario:
            Patch ``GeoMixin.add_tiles`` and call ``add_basemap`` with only
            ``ax``. Every wrapper-level default must reach the glyph method
            as a keyword (``ax=``, ``crs=3857``, ``zoom="auto"``, …).
        """
        sentinel_ax = object()
        with patch(_ADD_TILES) as mock_add:
            result = add_basemap(sentinel_ax)

        mock_add.assert_called_once_with(
            ax=sentinel_ax,
            source=None,
            crs=3857,
            zoom="auto",
            alpha=1.0,
            attribution=True,
            zorder=-1,
            interpolation="bilinear",
            timeout=10,
            retries=2,
        )
        assert result is mock_add.return_value, (
            "add_basemap must return GeoMixin.add_tiles's result"
        )

    def test_delegates_with_custom_kwargs(self):
        """Caller-supplied options are forwarded to cleopatra unchanged.

        Test scenario:
            Pass a non-default ``crs``, ``source``, ``alpha``, and
            ``zorder``; the patched ``add_tiles`` must receive exactly
            those values (and the untouched defaults for the rest).
        """
        sentinel_ax = object()
        with patch(_ADD_TILES) as mock_add:
            add_basemap(
                sentinel_ax,
                crs=4326,
                source="CartoDB.Positron",
                alpha=0.5,
                zorder=-2,
            )

        mock_add.assert_called_once_with(
            ax=sentinel_ax,
            source="CartoDB.Positron",
            crs=4326,
            zoom="auto",
            alpha=0.5,
            attribution=True,
            zorder=-2,
            interpolation="bilinear",
            timeout=10,
            retries=2,
        )

    def test_missing_extra_raises(self):
        """``add_basemap`` surfaces the missing-extra error before delegating.

        Test scenario:
            Patch the ``import_basemap`` guard to raise; ``add_basemap``
            must propagate ``OptionalPackageDoesNotExist`` and never reach
            the glyph ``add_tiles`` method.
        """
        with patch(
            "pyramids.basemap.basemap.import_basemap",
            side_effect=OptionalPackageDoesNotExist("no cleopatra[tiles]"),
        ):
            with patch(_ADD_TILES) as mock_add:
                with pytest.raises(OptionalPackageDoesNotExist):
                    add_basemap(object())
        mock_add.assert_not_called()


class TestCleopatraDelegation:
    """The ``add_basemap`` -> ``GeoMixin.add_tiles`` contract.

    ``add_basemap`` is a thin wrapper over the cleopatra glyph ``add_tiles``
    method (pinned via the ``[viz]`` extra as ``cleopatra[tiles]>=0.46.1``).
    These tests pin the contract that wrapper relies on: the glyph method is
    present, and the kwargs ``add_basemap`` forwards stay compatible with the
    underlying tile worker — cleopatra may add new *optional* params (pyramids
    just won't expose them), but a previously-shared param going *required*
    upstream, or a pyramids-only param, would break the delegation.
    """

    def test_cleopatra_glyph_add_tiles_exists(self):
        """``GeoMixin.add_tiles`` exists so ``add_basemap`` can delegate."""
        geo = pytest.importorskip(
            "cleopatra.basemap.geo",
            reason="cleopatra[tiles] extra not installed",
        )
        assert hasattr(geo.GeoMixin, "add_tiles"), (
            "cleopatra.basemap.geo.GeoMixin.add_tiles must exist so "
            "pyramids.basemap.add_basemap can delegate to it."
        )

    def test_add_basemap_kwargs_accepted_by_tile_worker(self):
        """``add_basemap``'s kwargs line up with cleopatra's tile worker.

        Test scenario:
            The glyph ``add_tiles`` forwards ``**kwargs`` to the module-level
            tile worker (``_add_tiles`` in cleopatra >= 0.46, ``add_tiles``
            before). Compare ``add_basemap``'s parameter names against that
            worker so a future signature drift is caught here rather than at a
            downstream call site.
        """
        tiles = pytest.importorskip(
            "cleopatra.basemap.tiles",
            reason="cleopatra[tiles] extra not installed",
        )
        worker = getattr(tiles, "_add_tiles", None) or getattr(tiles, "add_tiles", None)
        assert worker is not None, (
            "cleopatra.basemap.tiles must expose a tile worker "
            "(`_add_tiles` or `add_tiles`)."
        )
        cleo_sig = inspect.signature(worker)
        pyr_sig = inspect.signature(add_basemap)
        cleo_params = set(cleo_sig.parameters) - {"ax"}
        pyr_params = set(pyr_sig.parameters) - {"ax"}
        extra = pyr_params - cleo_params
        # Cleopatra is allowed to grow its surface: new *optional* params
        # don't break the `add_basemap(ax, ...) -> add_tiles(ax=ax, ...)`
        # delegation — pyramids just won't expose them. What *would* break it
        # is a pyramids-only param the worker cannot accept.
        assert not extra, (
            f"pyramids.add_basemap has params the cleopatra tile worker does "
            f"not accept: {extra}. The thin-wrapper delegation will not work "
            "until these are pushed upstream."
        )


class TestAddBasemapSignatureStability:
    """Assert ``add_basemap``'s public signature has not drifted.

    Tests across the suite (and ``pyramids.dataset._plot_helpers``) call
    ``add_basemap`` with positional / keyword combinations. Any signature
    change here breaks those silently — pin the parameter set so a future
    drift surfaces with a clear failure.
    """

    EXPECTED_PARAMS = (
        "ax",
        "crs",
        "source",
        "zoom",
        "alpha",
        "attribution",
        "zorder",
        "interpolation",
        "timeout",
        "retries",
    )

    def test_add_basemap_parameter_names(self):
        """Parameter names (and order) match the documented contract.

        Test scenario:
            Compare ``add_basemap``'s parameter names against the frozen
            ``EXPECTED_PARAMS``. Order matters because most call sites pass
            ``ax`` positionally — a rename or reorder must surface here.
        """
        params = tuple(inspect.signature(add_basemap).parameters.keys())
        assert params == self.EXPECTED_PARAMS, (
            f"add_basemap signature changed; expected {self.EXPECTED_PARAMS}, "
            f"got {params}."
        )
