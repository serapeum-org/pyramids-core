"""Read and validate `ci/native-stack-baseline.toml`, the one declaration of the native stack.

`pyramids-core` obtains its native stack three ways -- compiled from source for the Linux wheels, built
through vcpkg for `win_arm64`, taken as conda-forge binaries for `win_amd64` and macOS -- so something
has to compare the *built artifact* against a single declaration (#1627). Two scripts do that, from
opposite sides, and both read this module:

* `ci/check-bundled-stack.py` reads a repaired wheel's bundled binaries in the repair step.
* `ci/verify-wheel.py` asserts GDAL's version after installing the wheel, because GDAL's version is
  not reliably byte-readable.

The validation here is the point of the module rather than a nicety. A guard whose declaration can be
mistyped into vacuity is worse than no guard, because it reports `ok`: a baseline naming a library no
extractor knows, or an exception with no reason, or a platform key that matches nothing, would all have
passed silently. `validate` refuses each of them by name.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import re
import tomllib

BASELINE_PATH = pathlib.Path(__file__).resolve().parent / "native-stack-baseline.toml"

#: The libraries whose own build-settings string can be read out of a shipped binary. A baseline may
#: only pin these: anything else has no extractor, so it could never be compared.
SUPPORTED_LIBRARIES = ("hdf5", "proj", "geos")

#: The properties a `[libraries.*]` table may pin. `version` is compared exactly; `threadsafety` is the
#: build flag #1620 was about, and an exception may never waive it.
SUPPORTED_PROPERTIES = ("version", "threadsafety")

#: The platform tags this project actually publishes a wheel for, as PEP 425 spells them. An
#: exception key must match one of these, because a key matching no built wheel relaxes nothing: the
#: platform is then compared against the unrelaxed baseline and fails its own repair step against a
#: version it provably cannot reach. A loose prefix-family check was not enough -- it admitted
#: `winarm64` and bare `win`, both of which look right and resolve to no platform.
#:
#: Each family keeps its version segment open (`manylinux_2_28_`, `macosx_11_0_`) so an upstream bump
#: needs no edit here, while the arch suffix is required, which is what a typo drops.
_PLATFORM_TAG = re.compile(
    r"^(?:"
    r"manylinux(?:_\d+_\d+|\d+)_(?:x86_64|aarch64|i686)"
    r"|musllinux_\d+_\d+_(?:x86_64|aarch64)"
    r"|macosx_\d+_\d+_(?:x86_64|arm64|universal2)"
    r"|win_(?:amd64|arm64)"
    r"|win32"
    r")$"
)


def load(path: pathlib.Path = BASELINE_PATH) -> dict:
    """Read the baseline and refuse it if it is not usable.

    Args:
        path: The baseline file. Defaults to the one beside this module.

    Returns:
        The parsed, validated document.

    Raises:
        FileNotFoundError: If `path` does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        ValueError: If the document would compare nothing, or less than it appears to -- see
            `validate` for every case.
    """
    baseline = tomllib.loads(path.read_text(encoding="utf-8"))
    validate(baseline, name=path.name)
    return baseline


def validate(baseline: dict, name: str = "baseline") -> None:
    """Refuse a baseline that cannot do its job, naming what is wrong.

    Args:
        baseline: The parsed document.
        name: The file name, for the message.

    Raises:
        ValueError: If `libraries` is missing or empty; if it pins a library outside
            `SUPPORTED_LIBRARIES`, or a property outside `SUPPORTED_PROPERTIES`, or an empty table;
            if an exception is keyed by something that is not a wheel platform tag, names a library
            nothing compares, carries no non-empty `reason`, or relaxes no `version`; or if
            `installed.gdal` exists without a `version`. Each is a way to write a declaration that
            silently compares nothing. Every problem found is reported, not just the first.
    """
    problems = []
    libraries = baseline.get("libraries")
    if not libraries:
        problems.append("declares no [libraries] table, so it would compare nothing")
    elif not isinstance(libraries, dict):
        problems.append(f"[libraries] must be a table, not {type(libraries).__name__}")
        libraries = {}
    for library, properties in (libraries or {}).items():
        if not isinstance(properties, dict):
            problems.append(
                f"[libraries.{library}] must be a table of properties, not "
                f"{type(properties).__name__}"
            )
            continue
        if library not in SUPPORTED_LIBRARIES:
            problems.append(
                f"[libraries.{library}] has no extractor, so it could never be compared "
                f"(supported: {', '.join(SUPPORTED_LIBRARIES)})"
            )
        pinned = [key for key in properties if key != "reason"]
        if not pinned:
            problems.append(f"[libraries.{library}] pins no property")
        for prop in pinned:
            if prop not in SUPPORTED_PROPERTIES:
                problems.append(
                    f"[libraries.{library}] pins unknown property {prop!r} "
                    f"(supported: {', '.join(SUPPORTED_PROPERTIES)})"
                )
    exceptions = baseline.get("exceptions", {})
    if not isinstance(exceptions, dict):
        problems.append(
            f"[exceptions] must be a table, not {type(exceptions).__name__}"
        )
        exceptions = {}
    for tag, relaxations in exceptions.items():
        if not isinstance(relaxations, dict):
            problems.append(
                f"[exceptions.{tag}] must be a table, not {type(relaxations).__name__}"
            )
            continue
        if not _PLATFORM_TAG.match(tag):
            problems.append(
                f"[exceptions.{tag}] is not a platform tag this project publishes, so it would "
                "relax nothing (expected e.g. win_arm64, macosx_11_0_x86_64, "
                "manylinux_2_28_aarch64 -- note the required arch suffix, and that conda subdirs "
                "like osx-64 are not wheel tags)"
            )
        for library, relaxation in relaxations.items():
            if not isinstance(relaxation, dict):
                problems.append(
                    f"[exceptions.{tag}.{library}] must be a table, not "
                    f"{type(relaxation).__name__}"
                )
                continue
            if library not in (*SUPPORTED_LIBRARIES, "gdal"):
                problems.append(
                    f"[exceptions.{tag}.{library}] names a library nothing compares"
                )
            if not str(relaxation.get("reason", "")).strip():
                problems.append(
                    f"[exceptions.{tag}.{library}] carries no reason; an exception is a reviewed "
                    "decision, not a silent difference"
                )
            if "version" not in relaxation:
                problems.append(f"[exceptions.{tag}.{library}] relaxes no version")
    installed = baseline.get("installed")
    if not isinstance(installed, dict) or not installed.get("gdal", {}).get("version"):
        # `ci/verify-wheel.py` is a hard consumer of this value, so an absent, emptied or misspelled
        # table is not an "optional section" -- it is a document that compares nothing on the
        # installed side while looking complete.
        problems.append(
            "declares no [installed.gdal] version, so nothing would be compared on the installed side"
        )
    if problems:
        raise ValueError(f"{name} is unusable:\n  - " + "\n  - ".join(problems))


def expected_for(baseline: dict, tag: str) -> dict:
    """The properties each library must report on one platform, exceptions applied.

    Args:
        baseline: The parsed baseline.
        tag: The wheel's platform tag. An empty tag resolves no exception, so the caller should warn.

    Returns:
        A `{library: {property: value}}` mapping. An exception may override `version` only: every other
        key, `threadsafety` above all, comes from `libraries` and cannot be waived. An exception naming
        a library absent from `libraries` is ignored here -- `gdal` is the live case, relaxed on the
        installed side instead.
    """
    expected = {
        library: dict(properties)
        for library, properties in baseline["libraries"].items()
    }
    for library, relaxation in baseline.get("exceptions", {}).get(tag, {}).items():
        if library in expected and "version" in relaxation:
            expected[library]["version"] = relaxation["version"]
    return expected


def expected_gdal(baseline: dict, tag: str) -> tuple[str, str]:
    """The GDAL version one platform must report, and where that expectation came from.

    GDAL is resolved here rather than through `expected_for` because it is not a `[libraries]` entry:
    its version cannot be read from a wheel's bytes, so it is asserted from the installed package by
    `ci/verify-wheel.py`. That consumer used to hand-roll this lookup, which left the only evaluation
    of `[exceptions.<tag>.gdal]` -- a relaxation that exists for exactly one platform -- with no
    executable test (#1627).

    Args:
        baseline: The parsed baseline.
        tag: The installed wheel's platform tag. An empty tag resolves no exception.

    Returns:
        A `(version, source)` pair, where `source` names the table the value came from, for the
        caller's message: `exceptions.<tag>.gdal` when this platform declares a relaxation, otherwise
        `installed.gdal`. The version is an empty string when the baseline pins none, which the caller
        must refuse rather than treat as "anything goes".
    """
    pinned = baseline.get("installed", {}).get("gdal", {}).get("version", "")
    relaxation = baseline.get("exceptions", {}).get(tag, {}).get("gdal", {})
    relaxed = relaxation.get("version", "") if tag else ""
    if relaxed:
        resolved = (relaxed, f"exceptions.{tag}.gdal")
    else:
        resolved = (pinned, "installed.gdal")
    return resolved


def wheel_name_platform_tag(filename: str) -> str:
    """The platform tag of a wheel, from its PEP 427 filename.

    Args:
        filename: The wheel's file name, e.g. `pyramids_core-0.81.0-cp312-cp312-win_arm64.whl`.

    Returns:
        The platform tag, or an empty string when the name does not carry the five dash-separated
        parts PEP 427 requires, or when it is `any` -- neither resolves an exception, and the caller
        reports that rather than guessing at one.
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    tag = parts[-1] if len(parts) >= 5 else ""
    # `any` is not a platform, so it resolves no exception -- the same normalisation the installed
    # side applies, so both sides really do share one namespace.
    return "" if tag == "any" else tag


def installed_platform_tag(distribution: str = "pyramids-core") -> str:
    """The platform tag of the *installed* wheel, read from its own metadata.

    Taken from `.dist-info/WHEEL`'s `Tag:` lines rather than from `sysconfig.get_platform()`, so the
    installed side resolves exceptions in the same namespace as the bundled side. `sysconfig` reports
    `linux_x86_64` for both the manylinux and musllinux wheels and the interpreter's own deployment
    target on macOS, so a Linux or macOS exception would have needed two different keys, and whichever
    was missing would have silently relaxed nothing.

    Args:
        distribution: The installed distribution to ask.

    Returns:
        The single platform tag its wheel was built for, or an empty string when the distribution
        is not installed, is installed more than once, ships no `WHEEL` metadata, carries no `Tag:`
        line, names more than one platform at once (a fat install), or is platform-independent
        (`any`). Each of those the caller reports rather than guessing at.
    """
    wanted = distribution.replace("-", "_").lower()
    installed = [
        found
        for found in importlib.metadata.distributions()
        if (found.metadata.get("Name") or "").replace("-", "_").lower() == wanted
    ]
    platforms = set()
    # More than one install of the same name -- a botched upgrade leaving two .dist-info trees --
    # would otherwise be resolved by sys.path order, silently keying an exception to whichever
    # directory happened to come first. Anything short of one concrete platform yields "".
    if len(installed) == 1:
        wheel_metadata = installed[0].read_text("WHEEL") or ""
        platforms = {
            line.split(":", 1)[1].strip().rsplit("-", 1)[-1]
            for line in wheel_metadata.splitlines()
            if line.startswith("Tag:")
        }
    tag = ""
    if len(platforms) == 1:
        found = platforms.pop()
        tag = "" if found == "any" else found
    return tag
