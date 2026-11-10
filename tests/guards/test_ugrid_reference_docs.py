"""Guard meta-test: the UGRID reference docs do not present retired-with-no-alias names as public.

MESH-7 made ``Mesh2d.from_gdal_group`` / ``Connectivity.from_gdal_array`` private
(``_from_gdal_*``) with **no** deprecation alias, so a reader who copies the bare public
spelling from a reference page hits ``AttributeError``. The reflection guard on the NetCDF
``public-api.md`` has no UGRID equivalent, which is how the stale references slipped through;
this closes that gap for the names that are genuinely gone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

_DOCS = Path(__file__).parents[2] / "docs" / "reference" / "netcdf" / "ugrid"
# A public use is the bare name not preceded by an underscore (so ``_from_gdal_group`` is fine).
_RETIRED_PUBLIC = re.compile(r"(?<!_)\bfrom_gdal_(?:group|array)\b")


@pytest.mark.parametrize("page", sorted(_DOCS.glob("*.md")), ids=lambda p: p.name)
def test_no_public_from_gdal_names(page: Path):
    """No UGRID reference page may spell the retired GDAL constructors as public.

    They are private with no alias, so any public mention is a broken instruction.
    """
    hits = _RETIRED_PUBLIC.findall(page.read_text(encoding="utf-8"))
    assert not hits, (
        f"{page.name} references retired public GDAL constructor(s) {sorted(set(hits))}; "
        f"these are now private (_from_gdal_*) with no alias"
    )
