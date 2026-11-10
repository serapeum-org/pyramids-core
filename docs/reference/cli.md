# Command line (`pyramids`)

`pyramids` ships a scriptable command-line interface — the `pyramids` console script (installed with
`pip install pyramids-core`), also runnable as `python -m pyramids`. It is built on the standard-library
`argparse`, and every command maps 1:1 onto a library call.

Commands follow one convention, `pyramids <domain> [<subsystem>] <verb>`: each data type has its own domain,
the large domains add a `<subsystem>` level (e.g. `vector geom buffer`, `nc select isel`), and rarely-one-shot
primitives sit under a per-domain `advanced` subsystem. The old flat spellings (`pyramids warp`,
`pyramids to-cog`, …) were removed — see the [migration guide](../migration.md#cli).

## Full command reference

The complete reference — every command, its subsystems and `advanced` tier, with per-verb usage and argument
tables — is split by domain:

- [Overview & global flags](cli/index.md) — the convention, the global `--version` / `--quiet` / `--format` /
  `--config` / `--no-config` / `--progress` flags, exit codes, and the did-you-mean behaviour.
- [`raster`](cli/raster.md) — single-raster (`Dataset`) operations.
- [`vector`](cli/vector.md) — `FeatureCollection` vector operations.
- [`nc`](cli/nc.md) — NetCDF / multidimensional operations.
- [`collection`](cli/collection.md) — temporal raster-stack (`DatasetCollection`) operations.
- [`mesh`](cli/mesh.md) — UGRID / `MeshDataset` operations.
- [`process`](cli/process.md) — geoprocessing pipelines and tools.
- [`stac`](cli/stac.md) — SpatioTemporal Asset Catalog access.
- [`fetch`](cli/fetch.md) — OGC web-service access (WMS / WCS / WFS).
- [`completion`](cli/completion.md) — shell tab-completion.
- [`plugins`](cli/plugins.md) — third-party CLI plugin discovery.

Run `pyramids --help` to list the domains, and `pyramids <domain> --help` (or
`pyramids <domain> <subsystem> --help`) to drill in from the shell.
