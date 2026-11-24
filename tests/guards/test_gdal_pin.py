"""Guard the shared GDAL pin and its wiring against silent breakage.

The conda GDAL stack is declared once, in ``[tool.pixi.feature.gdal.dependencies]``, and
shared into every pixi environment by listing the ``gdal`` feature. The SWIG ``osgeo``
bindings are ABI-locked to ``libgdal``, so that single pin must reach every env — dev / CI
and the lean ``wheel-build`` container alike (the latter is ``no-default-feature``, so it
cannot inherit the pin from ``[tool.pixi.dependencies]``).

Single-sourcing removes version drift, but introduces one new way to break it: an
environment that forgets to list the ``gdal`` feature resolves with no GDAL at all. These
tests enforce the invariants that keep the wiring sound — every environment references the
feature, the feature pins all four packages, and ``ci/gdal-pin.py`` (which CI uses to
install gdal in the conda-forge-shape wheel test) reads that same table.
"""

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GDAL_PIN_SCRIPT = REPO_ROOT / "ci" / "gdal-pin.py"

# The conda packages the shared gdal feature must pin. swig is build-only and lives in
# the wheel-build feature, so it is intentionally excluded here.
GDAL_PACKAGES = [
    "gdal",
    "libgdal-netcdf",
    "libgdal-hdf4",
    "libgdal-grib",
    "libgdal-jp2openjpeg",
]


def _load_pixi() -> dict:
    """Return the ``[tool.pixi]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["pixi"]


def _env_features(env_spec) -> list:
    """Return an environment's feature list (pixi allows a bare list or a table)."""
    return env_spec["features"] if isinstance(env_spec, dict) else env_spec


# The packages ci/setup-gdal-micromamba.sh asks ci/gdal-pin.py to resolve — the shared
# gdal libs plus build-only swig. gdal-pin.py must serve every one of them.
MICROMAMBA_PACKAGES = [
    "gdal",
    "libgdal-netcdf",
    "libgdal-hdf4",
    "libgdal-jp2openjpeg",
    "swig",
]


def _load_gdal_pin():
    """Import ``ci/gdal-pin.py`` as a module (its hyphenated path is not importable)."""
    spec = importlib.util.spec_from_file_location("gdal_pin", GDAL_PIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gdal_feature_pins_every_package():
    """The shared gdal feature must pin exactly the four conda GDAL packages."""
    deps = _load_pixi()["feature"]["gdal"]["dependencies"]
    assert sorted(deps) == sorted(GDAL_PACKAGES)


@pytest.mark.parametrize("env", sorted(_load_pixi()["environments"]))
def test_every_environment_references_gdal_feature(env: str):
    """Every pixi env must list the ``gdal`` feature, or it resolves with no GDAL at all."""
    features = _env_features(_load_pixi()["environments"][env])
    assert "gdal" in features, (
        f"environment {env!r} does not list the 'gdal' feature; it would resolve without "
        "the shared GDAL pin. Add 'gdal' to its features list in [tool.pixi.environments]."
    )


def test_gdal_pin_script_matches_pyproject():
    """``ci/gdal-pin.py`` must emit the shared gdal pin CI installs the wheel against."""
    assert (
        _load_gdal_pin().gdal_spec(PYPROJECT)
        == _load_pixi()["feature"]["gdal"]["dependencies"]["gdal"]
    )


def test_gdal_pin_script_resolves_micromamba_packages():
    """``ci/gdal-pin.py`` must resolve every pin ci/setup-gdal-micromamba.sh requests."""
    pins = _load_gdal_pin().feature_pins()
    missing = [pkg for pkg in MICROMAMBA_PACKAGES if pkg not in pins]
    assert not missing, (
        f"ci/gdal-pin.py cannot resolve {missing} that setup-gdal-micromamba.sh requests"
    )


LOCK = REPO_ROOT / "pixi.lock"
MICROMAMBA_SCRIPT = REPO_ROOT / "ci" / "setup-gdal-micromamba.sh"

# The libraries whose versions ci/native-stack-baseline.toml asserts in a built wheel. The
# cross-compile branch must pin these from the lock, or it re-solves them to whatever is newest.
LOCKED_FROM_PIXI_LOCK = ["proj", "geos", "hdf5"]


def _locked_platforms() -> list[str]:
    """The conda subdirs `pixi.lock` carries entries for in the wheel-build environment."""
    module = _load_gdal_pin()
    candidates = ("linux-64", "linux-aarch64", "osx-64", "osx-arm64", "win-64")
    present = []
    for platform in candidates:
        try:
            module.locked_versions(platform)
        except KeyError:
            continue
        present.append(platform)
    return present


class TestTheLockedPinsFeedTheCrossCompileSolve:
    """`--locked` must give the cross-compile branch the versions the lock already resolved.

    A range spec pins a *version*, not a *solve*. Re-resolving `gdal >=3.13.3,<3.13.4` for
    osx-64 picked the newest matching build, which links proj 9.9.0 and geos 3.15.0, while every
    platform installed from `pixi.lock` got the older build with proj 9.8.1 and geos 3.14.1. One
    release shipped two different native stacks under one version number, and only the osx-64
    wheel -- the single cross-compiled target -- was affected (#1627).
    """

    def test_the_locked_native_stack_agrees_across_every_platform(self):
        """The parity property itself: one native stack per release, whatever the platform."""
        module = _load_gdal_pin()
        platforms = _locked_platforms()
        resolved = {
            platform: module.locked_pins(LOCKED_FROM_PIXI_LOCK, platform)
            for platform in platforms
        }

        assert len(platforms) >= 2, (
            f"expected several locked platforms, found {platforms}"
        )
        distinct = {tuple(sorted(pins.items())) for pins in resolved.values()}
        assert len(distinct) == 1, (
            "pixi.lock resolves a different native stack per platform, so no single pin can make "
            f"the wheels agree: {resolved}"
        )

    def test_locked_pins_are_exact_equality_specs(self):
        """A floor would let the solver drift upward again, so the pin must be `==`."""
        pins = _load_gdal_pin().locked_pins(LOCKED_FROM_PIXI_LOCK, "osx-64")

        assert sorted(pins) == sorted(LOCKED_FROM_PIXI_LOCK), (
            f"every name must resolve: {pins}"
        )
        for name, spec in pins.items():
            assert spec.startswith("=="), f"{name} must be pinned exactly, got {spec!r}"

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (
                {"names": ["no-such-package"], "platform": "osx-64"},
                "pins no no-such-package",
            ),
            ({"names": ["proj"], "platform": "osx-99"}, "no platform"),
            (
                {"names": ["proj"], "platform": "osx-64", "environment": "nope"},
                "no environment",
            ),
        ],
        ids=["unknown-package", "unknown-platform", "unknown-environment"],
    )
    def test_an_unresolvable_pin_raises_rather_than_being_dropped(
        self, kwargs, expected
    ):
        """A pin that resolves to nothing must fail loudly, not hand the choice back to the solver.

        Args:
            kwargs: Arguments naming something absent from the lock.
            expected: Text the raised message must contain.
        """
        with pytest.raises(KeyError, match=expected):
            _load_gdal_pin().locked_pins(**kwargs)

    def test_the_range_mode_still_reads_pyproject(self):
        """Adding `--locked` must not change what the original mode emits."""
        module = _load_gdal_pin()

        assert module.gdal_spec(PYPROJECT).startswith(">="), (
            "the range mode must keep emitting the pyproject spec, not an exact pin"
        )


class TestTheCrossCompileScriptUsesTheLockedPins:
    """The wiring: `setup-gdal-micromamba.sh` must actually consume `--locked`."""

    def test_the_script_reads_the_locked_pins(self):
        """Without this call the branch re-solves and the divergence returns."""
        script = MICROMAMBA_SCRIPT.read_text(encoding="utf-8")

        assert "--locked" in script, (
            "ci/setup-gdal-micromamba.sh must call gdal-pin.py --locked, or it re-resolves the "
            "native stack to whatever conda-forge currently ships newest (#1627)"
        )
        assert '--platform "${TARGET_PLATFORM}"' in script, (
            "the locked lookup must use the target platform, not the host's"
        )

    @pytest.mark.parametrize("library", LOCKED_FROM_PIXI_LOCK)
    def test_every_baseline_library_is_passed_to_micromamba(self, library):
        """Each library the baseline asserts a version for must reach the solve as a pin.

        Args:
            library: The conda package name that must appear in the create command.
        """
        script = MICROMAMBA_SCRIPT.read_text(encoding="utf-8")
        variable = f"${{{library.upper()}_SPEC}}"

        assert f'"{library}{variable}"' in script, (
            f"{library} must be passed to micromamba create as {library}{variable}, or its "
            "version is left to the solver on the cross-compile branch"
        )
