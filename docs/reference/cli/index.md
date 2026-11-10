# Command-line interface

`pyramids` ships a scriptable CLI (the `pyramids` console script, installed with `pip install pyramids-core`; also
runnable as `python -m pyramids`). It is built on the standard-library `argparse` — no extra dependency — and every
command maps 1:1 onto a library call.

## Convention

Every command follows `pyramids <domain> [<subsystem>] <verb>`. The ~10 domains below each group a data type's
operations; the large domains add a `<subsystem>` level (e.g. `vector geom buffer`, `nc select isel`). Rarely-one-shot
primitives live under a per-domain `advanced` subsystem (see each domain page). The old flat
spellings (`pyramids warp`, `pyramids to-cog`, …) were removed outright; see the [migration
guide](../../migration.md#cli).

`pyramids --help` lists the domains; `pyramids <domain> --help` and `pyramids <domain> <subsystem> --help` drill in. A
misspelled subcommand prints a *did you mean …?* suggestion.

## Global flags

Placed before the command, e.g. `pyramids --format json vector info roads.geojson`.

| flag | description |
|---|---|
| `--version`, `-V` | show program's version number and exit |
| `--quiet`, `-q` | suppress informational output, e.g. 'wrote ...' (errors still print); place it before the command, e.g. `pyramids --quiet raster warp ...` |
| `--format` | output format for data-emitting verbs (info/bounds/stats/list/...): 'json' is honored by every verb that supports --json; 'csv'/'table' by the tabular verbs. Place it before the command, e.g. `pyramids --format json vector info ...` (choices: `text`, `json`, `csv`, `table`) |
| `--config` | read per-command defaults from this pyramids.yaml (overrides discovery of a nearby pyramids.yaml / PYRAMIDS_CONFIG / ~/.config/pyramids/config.yaml) |
| `--no-config` | ignore any pyramids.yaml config file and use built-in defaults only |
| `--progress` | show a progress indicator on stderr for multi-item write verbs (a no-op when output is not an interactive terminal or when --quiet is set) |

## Domains

- [`raster`](raster.md) — single-raster (`Dataset`) operations
- [`vector`](vector.md) — `FeatureCollection` vector operations
- [`nc`](nc.md) — NetCDF / multidimensional operations
- [`collection`](collection.md) — temporal raster-stack (`DatasetCollection`) operations
- [`mesh`](mesh.md) — UGRID / `MeshDataset` operations
- [`process`](process.md) — geoprocessing pipelines & tools
- [`stac`](stac.md) — SpatioTemporal Asset Catalog access
- [`fetch`](fetch.md) — OGC web-service access (WMS / WCS / WFS)
- [`completion`](completion.md) — shell tab-completion
- [`plugins`](plugins.md) — third-party CLI plugin discovery

## Exit codes

`0` on success. An expected user error (missing file, bad CRS, unknown driver, …) prints a one-line `error: …` to
stderr and exits `1`. An unknown subcommand or bad flag exits `2` (argparse). Set `PYRAMIDS_DEBUG=1` to re-raise the
full traceback instead of the one-line message.

