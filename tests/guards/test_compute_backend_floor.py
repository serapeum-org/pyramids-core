"""Guard the ``[lazy]`` extra's floor against a compute-backend seam bump.

Core publishes :data:`COMPUTE_BACKEND_API_VERSION` and rejects any backend declaring an
older ``api_version``, telling the user to upgrade the plugin. But what pip actually
resolves is the floor in the ``[lazy]`` extra, and that floor is unbounded on both sides:
``pyramids-lazy>=0.1.0`` here, ``pyramids-core>=0.79.0`` there, with nothing linking
either to the seam version.

So if the seam is bumped without the floor moving, ``pip install pyramids-core[lazy]``
still happily resolves the oldest plugin, core rejects it at import, and the user is told
no backend is installed -- having just installed the extra whose only job is to provide
one. These tests make that pairing explicit, so the floor bump cannot be forgotten
alongside a seam bump.
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
# version here and raise the [lazy] extra's floor to match. Verified for 0.1.0 against the
# installed wheel: require_compute_backend().api_version == 2.
FIRST_PLUGIN_RELEASE: dict[int, str] = {2: "0.1.0"}


def _lazy_extra_floor() -> str:
    """The `>=` floor of the `pyramids-lazy` requirement in the `[lazy]` extra."""
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = manifest["project"]["optional-dependencies"]
    requirements = [
        req for req in extras["lazy"] if req.replace("_", "-").startswith("pyramids-lazy")
    ]
    assert len(requirements) == 1, (
        f"expected exactly one pyramids-lazy requirement in the [lazy] extra, "
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
        f"[lazy] extra's floor, or pip will resolve a plugin core then rejects."
    )


def test_the_lazy_extra_floor_covers_the_current_seam_version():
    """The published floor must exclude plugins too old for the current seam."""
    required = FIRST_PLUGIN_RELEASE[COMPUTE_BACKEND_API_VERSION]
    floor = _lazy_extra_floor()

    assert _as_tuple(floor) >= _as_tuple(required), (
        f"the [lazy] extra floors pyramids-lazy at {floor}, but api_version "
        f"{COMPUTE_BACKEND_API_VERSION} needs >= {required}"
    )
