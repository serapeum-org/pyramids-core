#!/bin/bash
#
# Install the wheel-build environment via micromamba for a cross-platform
# target. Used by ci/setup-gdal-from-pixi.sh when cibuildwheel builds for
# a target arch different from the host (currently: macos-14 / arm64 host
# producing an osx-64 wheel because GitHub's macos-13 Intel runner queue
# is unusable in practice).
#
# Why not pixi: pixi installs envs for the host arch only — there's no
# --platform flag on `pixi install`. micromamba ships a single static
# binary that supports `--platform osx-64` natively.
#
# Why micromamba re-solves: pixi.lock has osx-64 entries (we added the
# platform), but pixi can't materialize them on an osx-arm64 host, so
# micromamba solves from conda-forge against the range pins declared in
# [tool.pixi.feature.gdal.dependencies].
#
# A range does not pin a solve, though, and that shipped a real defect:
# re-resolving `gdal >=3.13.3,<3.13.4` here picked the newest matching
# build (np2py315..._12), which links proj 9.9.0 and geos 3.15.0, while
# every platform installed from pixi.lock got the _1 build with proj
# 9.8.1 and geos 3.14.1. One release therefore shipped two different
# native stacks under one version number, and only the osx-64 wheel was
# affected -- the divergence ci/check-bundled-stack.py caught (#1627).
#
# So the libraries whose versions the baseline asserts are pinned to
# exactly what pixi.lock resolves for this environment and platform,
# read by ci/gdal-pin.py --locked. The gdal family keeps its range: with
# proj and geos pinned the solver is steered to the matching gdal build
# anyway, verified by a cross-platform dry-run solve for osx-64.
#
# Inputs (must be exported by caller):
#   BUILD_PREFIX     destination for the micromamba binary
#   PIXI_ENV         path where the env will be created (kept as
#                    .pixi/envs/wheel-build/ so downstream extraction
#                    logic is identical to the pixi branch)
#   TARGET_PLATFORM  conda platform tag for the wheel (currently only
#                    "osx-64" is reached — the cross-compile target
#                    for macos-14 / arm64 host)
set -euo pipefail

: "${BUILD_PREFIX:?BUILD_PREFIX must be set}"
: "${PIXI_ENV:?PIXI_ENV must be set}"
: "${TARGET_PLATFORM:?TARGET_PLATFORM must be set}"

echo "=== setup-gdal-micromamba.sh ==="
echo "BUILD_PREFIX=${BUILD_PREFIX}"
echo "PIXI_ENV=${PIXI_ENV}"
echo "TARGET_PLATFORM=${TARGET_PLATFORM}"

# Pick the micromamba binary for the host arch (separate from the
# TARGET_PLATFORM we're installing packages for).
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)      HOST_PLATFORM="osx-arm64" ;;
    Darwin/x86_64)     HOST_PLATFORM="osx-64" ;;
    Linux/x86_64)      HOST_PLATFORM="linux-64" ;;
    Linux/aarch64)     HOST_PLATFORM="linux-aarch64" ;;
    *) echo "ERROR: unsupported host $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

# Pin micromamba version via the MICROMAMBA_VERSION env var (set in
# bundle-pypi-wheels.yml's env block; this script only runs on the macOS
# cross-compile path, where cibuildwheel runs before-all on the host
# and inherits it). The previous `…/latest` endpoint was a rolling
# target — a new micromamba release with a regressed `--platform osx-64`
# would have silently broken CI.
# TODO: pinning the version doesn't fully close the supply-chain gap;
# micro.mamba.pm + the S3 backing store could still serve a tampered
# tarball. A follow-up should add SHA256 verification per platform
# (4 SHAs: linux-64, linux-aarch64, osx-arm64, osx-64) committed to a
# `.micromamba-sha256.txt` manifest.
: "${MICROMAMBA_VERSION:?MICROMAMBA_VERSION not set}"

MAMBA_BIN="${BUILD_PREFIX}/bin/micromamba"
mkdir -p "${BUILD_PREFIX}/bin"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/${HOST_PLATFORM}/${MICROMAMBA_VERSION}"
echo "--- Installing micromamba ${MICROMAMBA_VERSION} for ${HOST_PLATFORM} ---"
echo "    from ${MICROMAMBA_URL}"
curl -fsSL "${MICROMAMBA_URL}" \
    | tar -xj -C "${BUILD_PREFIX}" bin/micromamba
chmod +x "${MAMBA_BIN}"
"${MAMBA_BIN}" --version

# MAMBA_ROOT_PREFIX is where micromamba stashes its package cache.
# Use a BUILD_PREFIX-local dir rather than $HOME/micromamba-root so a
# developer running this script locally doesn't end up with a stray
# ~/micromamba-root. The CI runner is ephemeral so location doesn't
# matter there.
export MAMBA_ROOT_PREFIX="${BUILD_PREFIX}/micromamba-root"
mkdir -p "${MAMBA_ROOT_PREFIX}"
# Remove ${PIXI_ENV} cleanly — micromamba 2.x's `create -p` creates
# the prefix dir itself and refuses if the path already exists as a
# non-conda directory ("Non-conda folder exists at prefix"). Do NOT
# pre-mkdir.
rm -rf "${PIXI_ENV}"

# All the wheel-build GDAL pins live once in pyproject.toml — the gdal-family libs
# (gdal + libgdal-netcdf/hdf4/grib/jp2openjpeg) in [tool.pixi.feature.gdal.dependencies],
# build-only swig in [tool.pixi.feature.wheel-build.dependencies]. ci/gdal-pin.py is the
# single reader of those tables; calling it here keeps this cross-compile branch from
# re-encoding (or drifting from) the pins. The subprocess emits one spec per requested
# name, newline-separated, mapped onto the bash vars via `read`. micromamba accepts a
# conda match-spec concatenated as ``<name><spec>`` (e.g. ``gdal>=3.13,<3.14``).
GDAL_PIN="$(cd "$(dirname "$0")" && pwd)/gdal-pin.py"
if [[ ! -f "${GDAL_PIN}" ]]; then
    echo "ERROR: gdal-pin.py not found at ${GDAL_PIN}" >&2
    exit 1
fi

{ read -r GDAL_SPEC; read -r LIBGDAL_NETCDF_SPEC; \
        read -r LIBGDAL_HDF4_SPEC; read -r LIBGDAL_GRIB_SPEC; \
    read -r LIBGDAL_JP2_SPEC; read -r SWIG_SPEC; } \
    < <(python3 "${GDAL_PIN}" gdal libgdal-netcdf libgdal-hdf4 libgdal-grib libgdal-jp2openjpeg swig)

# The baseline-asserted libraries are pinned to the locked solve rather than a range, so this
# branch materializes the same native stack as every lock-installed platform. gdal-pin.py raises
# if any name is absent from pixi.lock for this environment/platform, so a typo cannot quietly
# drop a pin and hand the choice back to the solver.
{ read -r PROJ_SPEC; read -r GEOS_SPEC; read -r HDF5_SPEC; } \
    < <(python3 "${GDAL_PIN}" --locked --platform "${TARGET_PLATFORM}" proj geos hdf5)

echo "--- Wheel-build pins (from pyproject.toml) ---"
echo "  gdal${GDAL_SPEC}"
echo "  libgdal-netcdf${LIBGDAL_NETCDF_SPEC}"
echo "  libgdal-hdf4${LIBGDAL_HDF4_SPEC}"
echo "  libgdal-grib${LIBGDAL_GRIB_SPEC}"
echo "  libgdal-jp2openjpeg${LIBGDAL_JP2_SPEC}"
echo "  swig${SWIG_SPEC}"
echo "--- Native-stack pins (from pixi.lock, ${TARGET_PLATFORM}) ---"
echo "  proj${PROJ_SPEC}"
echo "  geos${GEOS_SPEC}"
echo "  hdf5${HDF5_SPEC}"

echo "--- Creating ${TARGET_PLATFORM} env at ${PIXI_ENV} ---"
"${MAMBA_BIN}" create -p "${PIXI_ENV}" \
    --platform "${TARGET_PLATFORM}" \
    -c conda-forge \
    -y \
    "gdal${GDAL_SPEC}" \
    "libgdal-netcdf${LIBGDAL_NETCDF_SPEC}" \
    "libgdal-hdf4${LIBGDAL_HDF4_SPEC}" \
    "libgdal-grib${LIBGDAL_GRIB_SPEC}" \
    "libgdal-jp2openjpeg${LIBGDAL_JP2_SPEC}" \
    "swig${SWIG_SPEC}" \
    "proj${PROJ_SPEC}" \
    "geos${GEOS_SPEC}" \
    "hdf5${HDF5_SPEC}"

echo "=== setup-gdal-micromamba.sh complete ==="
