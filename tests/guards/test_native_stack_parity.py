"""Guard the bundled native stack against per-route drift.

`pyramids-core` obtains its native stack three ways — compiled from source for the Linux wheels, built
through vcpkg for `win_arm64`, and taken as conda-forge binaries for `win_amd64`/macOS — and each route
silently inherits its upstream's defaults for anything not explicitly requested. That is how one release
came to ship HDF5 with thread-safety on three platforms and off on the other two (#1620), and how
`win_arm64` still ships a GDAL a minor version behind everyone else (#1627).

`ci/check-bundled-stack.py` reads what a built wheel actually contains and compares it against
`ci/native-stack-baseline.toml`. These tests drive that checker over synthetic wheels: a zip carrying a
member named like a bundled library whose bytes contain the build-settings strings HDF5, PROJ and GEOS
embed in themselves. Synthetic rather than real wheels because the point is the comparison logic, and a
real 33 MB wheel per case would make the suite network-bound.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "ci" / "check-bundled-stack.py"
_BASELINE = _REPO_ROOT / "ci" / "native-stack-baseline.toml"


def _library_bytes(hdf5: str = "2.2.0", threadsafety: str = "ON") -> bytes:
    """Bytes imitating the build-settings summary HDF5 embeds in its own binary.

    Args:
        hdf5: The version string to embed.
        threadsafety: The thread-safety setting to embed.

    Returns:
        A payload the checker's extractors can read, padded so it looks like a real object file
        rather than a bare string.
    """
    head = b"\x7fELF" + b"\x00" * 60
    body = f"    HDF5 Version: {hdf5}\n    Threadsafety: {threadsafety}\n".encode()
    return head + body + b"\x00" * 32


def _wheel(
    tmp_path: Path,
    platform_tag: str = "manylinux_2_28_x86_64",
    hdf5: str = "2.2.0",
    threadsafety: str = "ON",
    proj: str = "9.8.1",
    geos: str = "3.14.1",
    include_hdf5: bool = True,
) -> Path:
    """Build a synthetic wheel whose bundled libraries report the given versions.

    Args:
        tmp_path: Directory to write the wheel into.
        platform_tag: The wheel's platform tag, which is how the checker resolves exceptions.
        hdf5: HDF5 version to embed.
        threadsafety: HDF5 thread-safety setting to embed.
        proj: PROJ version to embed.
        geos: GEOS version to embed.
        include_hdf5: When False, omit the HDF5 library entirely.

    Returns:
        The path to the written wheel.
    """
    path = tmp_path / f"pyramids_core-9.9.9-cp312-cp312-{platform_tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        if include_hdf5:
            archive.writestr(
                "pyramids_core.libs/libhdf5-abcdef12.so.320.2.0",
                _library_bytes(hdf5=hdf5, threadsafety=threadsafety),
            )
        archive.writestr(
            "pyramids_core.libs/libproj-abcdef12.so.25", f"PROJ {proj}\x00".encode()
        )
        archive.writestr(
            "pyramids_core.libs/libgeos_c-abcdef12.so.1",
            f"{geos}-CAPI-1.20.5\x00".encode(),
        )
        archive.writestr("pyramids/__init__.py", b"")
    return path


def _run(*targets: Path) -> subprocess.CompletedProcess:
    """Run the checker over `targets` with the repo's real baseline.

    Args:
        targets: Wheel paths or directories to check.

    Returns:
        The completed process, with stdout and stderr captured as text.
    """
    return subprocess.run(
        [sys.executable, str(_CHECKER), *[str(t) for t in targets]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class TestTheCheckerShipsWithABaseline:
    """The checker and the baseline it reads are both present and parseable."""

    def test_the_checker_exists(self):
        """`ci/check-bundled-stack.py` is the script the repair steps invoke."""
        assert _CHECKER.is_file(), f"missing checker at {_CHECKER}"

    def test_the_baseline_exists_and_declares_the_byte_readable_libraries(self):
        """The baseline pins exactly the libraries whose versions a wheel can be read for.

        Test scenario:
            GDAL is deliberately absent: its version is not reliably greppable from the binary
            (a manylinux `libgdal` yields only a spurious `3.6.0`, and the Windows DLLs carry no
            version in their name), so GDAL parity is asserted from the installed wheel instead.
        """
        import tomllib

        baseline = tomllib.loads(_BASELINE.read_text(encoding="utf-8"))

        assert set(baseline["libraries"]) == {"hdf5", "proj", "geos"}, (
            f"unexpected baseline libraries: {sorted(baseline['libraries'])}"
        )
        assert baseline["libraries"]["hdf5"]["threadsafety"] == "ON", (
            "the baseline must require a thread-safe HDF5 (#1620)"
        )


class TestAConformingWheelPasses:
    """A wheel matching the baseline is accepted."""

    def test_a_wheel_matching_the_baseline_passes(self, tmp_path):
        """Every library reporting its baseline version exits 0."""
        result = _run(_wheel(tmp_path))

        assert result.returncode == 0, (
            f"expected pass, got {result.returncode}\n{result.stdout}\n{result.stderr}"
        )

    def test_a_directory_of_wheels_is_accepted(self, tmp_path):
        """The repair steps pass `{dest_dir}`, so a directory argument must work."""
        _wheel(tmp_path)

        result = _run(tmp_path)

        assert result.returncode == 0, (
            f"expected pass for a directory, got:\n{result.stdout}\n{result.stderr}"
        )


class TestDriftIsRefused:
    """Each property the baseline pins fails the build when it drifts."""

    def test_a_non_threadsafe_hdf5_is_refused(self, tmp_path):
        """The #1620 regression itself: thread-safety off must fail."""
        result = _run(_wheel(tmp_path, threadsafety="OFF"))

        assert result.returncode == 1, (
            f"a non-threadsafe HDF5 must fail, got {result.returncode}"
        )
        assert "threadsafety" in result.stdout.lower(), (
            f"the report should name the property: {result.stdout}"
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("hdf5", "2.1.0"), ("proj", "9.7.0"), ("geos", "3.13.0")],
        ids=["hdf5-behind", "proj-behind", "geos-behind"],
    )
    def test_a_library_version_off_the_baseline_is_refused(
        self, tmp_path, field, value
    ):
        """A route drifting to a different version of any pinned library fails.

        Args:
            field: Which library drifts.
            value: The off-baseline version it reports.
        """
        result = _run(_wheel(tmp_path, **{field: value}))

        assert result.returncode == 1, (
            f"{field}={value} must fail, got {result.returncode}"
        )
        assert field in result.stdout.lower(), (
            f"the report should name {field}: {result.stdout}"
        )

    def test_a_missing_library_is_refused(self, tmp_path):
        """A binary wheel without HDF5 cannot read NetCDF, so its absence is a regression too."""
        result = _run(_wheel(tmp_path, include_hdf5=False))

        assert result.returncode == 1, "a wheel with no bundled HDF5 must fail"


class TestDeclaredExceptionsAreHonoured:
    """A platform that cannot meet the baseline is pinned to what it can meet."""

    def test_win_arm64_passes_on_its_declared_hdf5_version(self, tmp_path):
        """`win_arm64` is allowed HDF5 2.1.1, because vcpkg's baseline has no 2.2.x port.

        Test scenario:
            A bare equality check would fail this platform on something unachievable, so the
            baseline carries the exception and the reason with it (#1627).
        """
        result = _run(_wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1"))

        assert result.returncode == 0, (
            f"win_arm64 on its declared version must pass:\n{result.stdout}"
        )

    def test_the_exception_does_not_leak_to_other_platforms(self, tmp_path):
        """The same version that `win_arm64` is allowed still fails on manylinux."""
        result = _run(
            _wheel(tmp_path, platform_tag="manylinux_2_28_x86_64", hdf5="2.1.1")
        )

        assert result.returncode == 1, (
            "the win_arm64 exception must not apply to manylinux"
        )

    def test_an_exception_does_not_waive_thread_safety(self, tmp_path):
        """A declared version exception is not a licence to ship a non-threadsafe build."""
        result = _run(
            _wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1", threadsafety="OFF")
        )

        assert result.returncode == 1, (
            "thread-safety must hold even where a version exception applies"
        )

    def test_an_unparseable_filename_says_exceptions_cannot_resolve(self, tmp_path):
        """A wheel whose name carries no platform tag warns rather than silently losing exceptions.

        Test scenario:
            Applying no exceptions quietly would compare a platform against the baseline instead
            of against its own declared values, and the mismatch would read as real drift -- the
            same silent-divergence failure this check exists to catch.
        """
        renamed = tmp_path / "renamed.whl"
        _wheel(tmp_path, platform_tag="win_arm64", hdf5="2.1.1").rename(renamed)

        result = _run(renamed)

        assert "no platform tag" in result.stderr, (
            f"an unparseable wheel name should warn on stderr, got: {result.stderr}"
        )
