# Using pyramids in QGIS

[`pyramids-qgis`](https://github.com/serapeum-org/pyramids-qgis) is a **QGIS plugin** that exposes pyramids as a
QGIS **Processing provider**. Its algorithms appear in the Processing Toolbox under **Pyramids**, so they run in
the toolbox, the graphical modeler, batch mode, and the Python `processing.run(...)` API — just like any native
QGIS algorithm.

!!! warning "Experimental"
    `pyramids-qgis` is an early, experimental plugin maintained in its own repository. Check
    [its README](https://github.com/serapeum-org/pyramids-qgis) for the current status, version, and the
    authoritative algorithm list — this page summarises how it fits with pyramids.

## How it relates to pyramids

The plugin is a **thin adapter**, not a fork of any logic. Each algorithm translates its QGIS parameters into a
`pyramids <command> …` argument list and calls the pyramids [CLI](../reference/cli.md) in-process
(`pyramids.cli.main`). The CLI is pyramids' stable, documented contract, so the plugin carries no raster logic of
its own and cannot drift from the library.

Work is passed by **file path**, never by in-memory GDAL objects: QGIS hands pyramids an input path, pyramids
writes an output file, and QGIS loads it as a new layer. That deliberately avoids sharing live `gdal.Dataset`
handles across two GDAL builds in one process — which is also why the install rule below matters.

This makes `pyramids-qgis` an **integration**, not a [plugin of pyramids](../plugins.md): it depends on
`pyramids-core` and wraps it for a host application, rather than registering into one of pyramids' entry-point
hooks.

## Requirements

- **QGIS 3.22+** (works through 4.x).
- **`pyramids-core` importable from QGIS's own Python, bound to QGIS's GDAL.**

!!! danger "Do not install a vendored-GDAL wheel into QGIS"
    Install pyramids into the interpreter QGIS runs — **not** a platform wheel that vendors its own GDAL. Those
    inject a private `osgeo` and force `GDAL_DRIVER_PATH`, which conflicts with QGIS's own GDAL in the same
    process. The plugin detects a vendored build and warns. Prefer conda-forge inside a conda-based QGIS, or
    `pip` into QGIS's interpreter where that binds to the system/QGIS GDAL.

## Install

### 1. Install pyramids into QGIS's Python

```bash
# Windows (OSGeo4W shell)
python -m pip install pyramids-core

# conda-based QGIS
conda install -c conda-forge pyramids-core

# Linux / macOS (use the interpreter QGIS runs)
<qgis-python> -m pip install pyramids-core
```

Optional tiers, installed into the **same** QGIS Python, light up extra algorithms:

- `pyramids-mesh` — the non-UGRID mesh formats (2DM, Selafin, DFSU, XMDF, PLY, …) used by the mesh algorithms.
- `pyramids-eo` — the Earth-observation algorithms (contrast stretch, tasseled cap).

The plugin warns clearly when an optional tier an algorithm needs is absent.

### 2. Install the plugin

The published builds attach a QGIS custom-repository `plugins.xml` to each GitHub Release. Add that URL under
*Plugins → Manage and Install Plugins → Settings* to install and update from the QGIS plugin manager, then enable
**Pyramids**.

For a checkout, copy or symlink the `pyramids_qgis/` folder into your QGIS profile's plugin directory (the repo's
`scripts/deploy.py` does this), then enable **Pyramids** in *Plugins → Manage and Install Plugins*:

- Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins`
- Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins`
- macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins`

## Algorithms

The first release mirrors the `pyramids` CLI as file-in / file-out algorithms, grouped under **Pyramids** in the
Processing Toolbox. Core raster & vector algorithms wrap these CLI commands:

| Algorithm | pyramids command |
|-----------|------------------|
| Reproject raster (warp) | `warp` |
| Clip raster | `clip` |
| Assign CRS / NoData (edit info) | `edit-info` |
| Merge rasters (mosaic) | `merge` |
| Raster calculator | `calc` |
| Convert raster format | `convert` |
| Rasterize (vector → raster) | `rasterize` |
| Polygonize raster | `shapes` |
| Create Cloud Optimized GeoTIFF | `cog create` |

Mesh algorithms (convert mesh format, mesh → vector) wrap `UgridDataset`, and the Earth-observation algorithms
(contrast stretch, tasseled cap) wrap `pyramids-eo`. See the plugin's README for the up-to-date, complete list
and its roadmap.

## Run an algorithm from Python

Inside the QGIS Python console, the provider works like any other:

```python
import processing

processing.run(
    "pyramids:warp",
    {"INPUT": "dem.tif", "CRS": "EPSG:3857", "OUTPUT": "dem_3857.tif"},
)
```

(Algorithm ids and parameter names are shown in each algorithm's dialog and in the Processing Toolbox; consult the
plugin's docs for the exact signatures.)
