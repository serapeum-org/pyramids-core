"""Guard the ``pyramids-lazy`` floor against a compute-backend seam bump.

Core publishes :data:`COMPUTE_BACKEND_API_VERSION` and rejects any backend declaring an
older ``api_version``, telling the user to upgrade the plugin. But what pip actually
resolves is the declared floor, and that floor is unbounded on both sides:
``pyramids-lazy>=0.1.0`` here, ``pyramids-core>=0.79.0`` there, with nothing linking
either to the seam version.

So if the seam is bumped without the floor moving, the dev/CI environments still resolve
the oldest plugin and core rejects it at import. These tests make that pairing explicit,
so the floor bump cannot be forgotten alongside a seam bump.

The floor lives in the ``lazy`` dependency-group, not in a published extra: the backend is
a plugin installed as ``pip install pyramids-lazy``, exactly like pyramids-eo and
pyramids-mesh. The group is dev/CI-only and the same-named pixi feature maps to it.
"""

import re
import tomllib
from pathlib import Path

import pytest

from pyramids.base._compute_backend import COMPUTE_BACKEND_API_VERSION

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The first pyramids-lazy release implementing each core compute-backend api_version.
# When COMPUTE_BACKEND_API_VERSION is bumped, add the release that implements the new
# version here and raise the lazy group's floor to match. Verified for 0.1.0 against the
# installed wheel: require_compute_backend().api_version == 2.
FIRST_PLUGIN_RELEASE: dict[int, str] = {2: "0.1.0"}


# The first pyramids-lazy release that composes its install hints locally and no longer imports
# `lazy_extra_hint` from `pyramids.lazy_support` (serapeum-org/pyramids-lazy#20 -- merged, not yet
# released; PyPI is still on 0.1.0, whose `_collection_zarr`, `_kerchunk_builder`,
# `_kerchunk_facade`, `_mfdataset` and `_zarr` all import it).
#
# Set this to that version when it ships. The test below then flips from "the shim must still
# exist" to "the shim must be gone AND the floor raised past the versions that need it", which is
# the pairing that must not be done by halves: deleting the shim while the environments can still
# resolve 0.1.0 breaks that plugin at import and takes core's lazy-marked suite red.
FIRST_HINT_FREE_PLUGIN_RELEASE: str | None = None


def _lazy_group_floor() -> str:
    """The `>=` floor of the `pyramids-lazy` requirement in the `lazy` group."""
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    groups = manifest["dependency-groups"]
    requirements = [
        req
        for req in groups["lazy"]
        if req.replace("_", "-").startswith("pyramids-lazy")
    ]
    assert len(requirements) == 1, (
        f"expected exactly one pyramids-lazy requirement in the lazy group, "
        f"got {requirements!r}"
    )
    match = re.search(r">=\s*([0-9][0-9.]*)", requirements[0])
    assert match is not None, f"no `>=` floor in {requirements[0]!r}"
    return match.group(1)


def _as_tuple(version: str) -> tuple[int, ...]:
    """`"0.1.0"` -> `(0, 1, 0)`, for ordering comparisons."""
    return tuple(int(part) for part in version.split("."))


def test_the_current_seam_version_has_a_recorded_plugin_release():
    """Bumping the seam must record which pyramids-lazy release first implements it."""
    assert COMPUTE_BACKEND_API_VERSION in FIRST_PLUGIN_RELEASE, (
        f"COMPUTE_BACKEND_API_VERSION is {COMPUTE_BACKEND_API_VERSION}, which has no "
        f"recorded pyramids-lazy release. Add it to FIRST_PLUGIN_RELEASE and raise the "
        f"lazy group's floor, or the envs will resolve a plugin core then rejects."
    )


def test_the_lazy_group_floor_covers_the_current_seam_version():
    """The declared floor must exclude plugins too old for the current seam."""
    required = FIRST_PLUGIN_RELEASE[COMPUTE_BACKEND_API_VERSION]
    floor = _lazy_group_floor()

    assert _as_tuple(floor) >= _as_tuple(required), (
        f"the lazy group floors pyramids-lazy at {floor}, but api_version "
        f"{COMPUTE_BACKEND_API_VERSION} needs >= {required}"
    )


def test_the_backend_is_not_published_as_an_extra():
    """The backend is a plugin, so no core extra may claim to install it.

    Reintroducing a `[lazy]` extra would put core back to advertising an install path for
    something it does not own, and would make every hint that names the plugin wrong again.
    """
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = manifest["project"]["optional-dependencies"]

    assert "lazy" not in extras, (
        "a `lazy` extra is back in [project.optional-dependencies]; the backend is a plugin "
        "installed as `pip install pyramids-lazy`, declared in the `lazy` dependency-group"
    )
    assert "lazy" in manifest["dependency-groups"], (
        "the `lazy` dependency-group is gone; the pixi `lazy` feature maps to it, so the "
        "dev/CI environments would stop installing the backend"
    )


def test_the_deprecated_hint_shim_tracks_the_floor():
    """`lazy_extra_hint` may only be deleted once the floor excludes plugins needing it.

    Test scenario:
        While FIRST_HINT_FREE_PLUGIN_RELEASE is unset, a released pyramids-lazy still
        imports the shim through `pyramids.lazy_support`, so core must keep exporting it.
        Once it is set, the floor must have been raised to that release and the shim must
        be gone -- doing either half alone is the failure this pins.
    """
    from pyramids import lazy_support

    exported = "lazy_extra_hint" in getattr(lazy_support, "__all__", ())

    if FIRST_HINT_FREE_PLUGIN_RELEASE is None:
        assert exported, (
            "lazy_extra_hint was removed from pyramids.lazy_support while "
            "FIRST_HINT_FREE_PLUGIN_RELEASE is still unset. A released pyramids-lazy "
            "imports it, so removing it breaks that plugin at import. Release the "
            "hint-free plugin first, record its version here, and raise the lazy "
            "dependency-group floor in the same change."
        )
    else:
        floor = _lazy_group_floor()
        assert _as_tuple(floor) >= _as_tuple(FIRST_HINT_FREE_PLUGIN_RELEASE), (
            f"the lazy group floors pyramids-lazy at {floor}, which still permits a "
            f"plugin that imports lazy_extra_hint; raise it to "
            f">= {FIRST_HINT_FREE_PLUGIN_RELEASE}"
        )
        assert not exported, (
            "the floor now excludes every plugin that imports lazy_extra_hint, so the "
            "deprecated shim should be deleted from pyramids.base._utils and dropped "
            "from pyramids.lazy_support"
        )
