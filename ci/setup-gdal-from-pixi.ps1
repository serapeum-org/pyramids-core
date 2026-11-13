#Requires -Version 5.1
# Install GDAL + native dependencies via pixi (conda-forge), then extract
# DLLs, headers, and data files into $env:BUILD_PREFIX so downstream steps
# (install-and-vendor-osgeo.py, setuptools, delvewheel) can find them.
#
# Runs once per cibuildwheel Windows invocation (CIBW_BEFORE_ALL).
#
# bundle-pypi-wheels.yml invokes this via `powershell -File ...` which uses
# Windows PowerShell 5.1 on the windows-2022 runner — the `#Requires`
# above documents that minimum so any future move to `pwsh` (PS 7)
# still passes, but a regression to an older PowerShell would fail
# fast instead of silently miscompiling somewhere downstream.
#
# See docs/how-to/wheel-build-flow.md for the end-to-end pipeline.

$ErrorActionPreference = "Stop"

$BuildPrefix = if ($env:BUILD_PREFIX) { $env:BUILD_PREFIX } else { "C:\gdal-prefix" }
Write-Host "=== setup-gdal-from-pixi.ps1 ==="
Write-Host "BUILD_PREFIX=$BuildPrefix"

# 1. Install pixi.
#
# Pin the version via the PIXI_VERSION env var (set in bundle-pypi-wheels.yml's
# env block; on Windows cibuildwheel runs before-all on the host so it
# inherits it directly). Run locally with
# `$env:PIXI_VERSION = "X.Y.Z"; .\ci\setup-gdal-from-pixi.ps1` -- gh must be
# authenticated first, via GH_TOKEN or `gh auth login`.
#
# Download the pinned bare pixi binary directly from its GitHub release with an
# AUTHENTICATED `gh release download`, instead of piping prefix-dev's install.ps1:
# that install script downloads the release archive anonymously, and GitHub
# throttles anonymous downloads from the shared Actions-runner IP pool (HTTP 429),
# which was turning the wheel legs red. The `v<version>` tag pins which release we
# pull; integrity-verifying the fetched bytes is tracked in #1553.
if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    if (-not $env:PIXI_VERSION) {
        throw "PIXI_VERSION not set"
    }
    # Download the pixi binary with an AUTHENTICATED `gh` request. prefix-dev's
    # install.ps1 fetches the release archive anonymously (System.Net.WebClient),
    # which GitHub rate-limits (HTTP 429) on the shared Actions-runner IP pool —
    # that reddened the Windows/macOS wheel builds on 2026-11-13 (confirmed 429 on the
    # same-class vcpkg download). Authenticated downloads use the token's far higher
    # limit. The release ships a bare .exe, so no install script or unzip is needed;
    # 3x retry only covers a genuine transient blip now, not a rate-limit wall.
    $BinDir = Join-Path $BuildPrefix "bin"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    # windows-2022 host is x86_64 (win_arm64 is built via vcpkg, not this script).
    $Asset = "pixi-x86_64-pc-windows-msvc.exe"
    $installed = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "--- Downloading pixi $($env:PIXI_VERSION) ($Asset, attempt $attempt) ---"
        # 2>&1 | Write-Host merges gh's stderr into the output stream so a stderr line
        # cannot surface as a terminating NativeCommandError under ErrorActionPreference
        # 'Stop' and bypass the retry; success is read from $LASTEXITCODE + Test-Path only.
        & gh release download "v$($env:PIXI_VERSION)" --repo prefix-dev/pixi `
            --pattern $Asset --dir $BinDir --clobber 2>&1 | Write-Host
        if (($LASTEXITCODE -eq 0) -and (Test-Path (Join-Path $BinDir $Asset))) { $installed = $true; break }
        if ($attempt -lt 3) { Start-Sleep -Seconds ($attempt * 10) }
    }
    if (-not $installed) {
        throw ("pixi download failed after 3 attempts (see gh output above; " +
            "ensure gh is authenticated via GH_TOKEN or 'gh auth login', and that " +
            "PIXI_VERSION '$($env:PIXI_VERSION)' ships $Asset)")
    }
    Move-Item -Path (Join-Path $BinDir $Asset) -Destination (Join-Path $BinDir "pixi.exe") -Force
    $env:PATH = "$BinDir;$env:PATH"
}
pixi --version

# 2. Install the wheel-build environment using the committed pixi.lock.
Write-Host "--- Resolving wheel-build environment ---"
pixi install -e wheel-build --frozen
if ($LASTEXITCODE -ne 0) { throw "pixi install failed" }

$PixiEnv = Join-Path (Get-Location) ".pixi\envs\wheel-build"
if (-not (Test-Path $PixiEnv)) {
    throw "wheel-build env not found at $PixiEnv"
}
Write-Host "wheel-build env: $PixiEnv"

# Resolve the concrete GDAL version from the env we just materialized
# and persist it for install-and-vendor-osgeo.py. Single source of truth
# = pixi.lock — no more hardcoded duplicates in pyproject.toml or
# bundle-pypi-wheels.yml. conda-forge's Windows gdal package doesn't ship a
# usable gdal-config, so parse the version from conda-meta\gdal-X.Y.Z-*.json
# instead (the same file conda/pixi/micromamba all populate).
$GdalMeta = Get-ChildItem -Path (Join-Path $PixiEnv "conda-meta\gdal-*.json") -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $GdalMeta) {
    throw "gdal package not found in $PixiEnv\conda-meta"
}
if ($GdalMeta.BaseName -match '^gdal-(\d+\.\d+\.\d+)') {
    $GdalVersion = $Matches[1]
} else {
    throw "could not parse version from $($GdalMeta.Name)"
}
New-Item -ItemType Directory -Force -Path $BuildPrefix | Out-Null
Set-Content -Path (Join-Path $BuildPrefix "GDAL_VERSION") -Value $GdalVersion -NoNewline
Write-Host "resolved GDAL_VERSION=$GdalVersion"

# 3. Extract native artifacts into $BuildPrefix.
#
# Windows conda packages nest under Library/: Library/bin (DLLs),
# Library/include (headers), Library/lib (import .lib files),
# Library/share (data).
Write-Host "--- Extracting native artifacts into $BuildPrefix ---"

# Mirror the Library/ layout to keep paths predictable for downstream tools.
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\bin"     | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\include" | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\lib"     | Out-Null
New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\share"   | Out-Null

# DLLs
Copy-Item -Path "$PixiEnv\Library\bin\*" -Destination "$BuildPrefix\Library\bin" -Recurse -Force

# Headers
Copy-Item -Path "$PixiEnv\Library\include\*" -Destination "$BuildPrefix\Library\include" -Recurse -Force

# Import libs
if (Test-Path "$PixiEnv\Library\lib") {
    Copy-Item -Path "$PixiEnv\Library\lib\*" -Destination "$BuildPrefix\Library\lib" -Recurse -Force
}

# GDAL_DATA + PROJ_DATA
Copy-Item -Path "$PixiEnv\Library\share\gdal" -Destination "$BuildPrefix\Library\share" -Recurse -Force
Copy-Item -Path "$PixiEnv\Library\share\proj" -Destination "$BuildPrefix\Library\share" -Recurse -Force

# curl CA bundle (#412). conda-forge's libcurl bakes its default CA path to
# the wheel-build env's ssl\cacert.pem, absent in the consuming env — so
# vendored-GDAL /vsicurl HTTPS reads fail to load trust anchors. Extract it
# so install-and-vendor-osgeo.py ships it and the bootstrap re-points curl.
$CaBundle = Join-Path $PixiEnv "Library\ssl\cacert.pem"
if (Test-Path $CaBundle) {
    New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\ssl" | Out-Null
    Copy-Item -Path $CaBundle -Destination "$BuildPrefix\Library\ssl\cacert.pem" -Force
} else {
    Write-Warning "CA bundle not found at $CaBundle - vendored GDAL HTTPS reads will fail (#412)"
}

# GDAL plugins (NetCDF / HDF4 / HDF5 drivers)
if (Test-Path "$PixiEnv\Library\lib\gdalplugins") {
    New-Item -ItemType Directory -Force -Path "$BuildPrefix\Library\lib\gdalplugins" | Out-Null
    Copy-Item -Path "$PixiEnv\Library\lib\gdalplugins\*" -Destination "$BuildPrefix\Library\lib\gdalplugins" -Recurse -Force
}

# 4. Diagnostic output.
Write-Host "=== setup-gdal-from-pixi.ps1 complete ==="
& "$BuildPrefix\Library\bin\gdalinfo.exe" --version
Write-Host "DLLs bundled: $((Get-ChildItem "$BuildPrefix\Library\bin\*.dll").Count)"
