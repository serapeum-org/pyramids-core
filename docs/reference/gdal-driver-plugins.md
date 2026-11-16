# GDAL driver plugins

GDAL can load some of its format drivers from compiled plugin files rather than having them linked into
`libgdal` itself. Which plugins load is decided by one environment variable, `GDAL_DRIVER_PATH`, and pyramids
owns the value of that variable: the bundled plugins in a platform wheel are ABI-locked to that wheel's exact
`libgdal`, so letting a foreign path win would make GDAL try to load plugins built against a different library.

This page documents what pyramids puts on that path, in what order, and how a separate distribution can add a
plugin directory of its own.

## What ends up on the path

Three kinds of directory, highest precedence first:

**1. Bundled** — `pyramids/_data/gdalplugins/`, inside a platform wheel. Present on the conda-extract (macOS,
win_amd64) and vcpkg (win_arm64) wheels. The from-source Linux wheels compile the same drivers *into* `libgdal`,
so no directory exists there.

**2. Discovered** — a conda `Library/lib/gdalplugins`, or `/usr/lib/gdalplugins`. This is a conda or system GDAL
install, i.e. the usual development setup.

**3. Companion** — a directory contributed by a distribution that registers the `pyramids.gdal_plugin_path`
entry point. Present only when such a package is installed.

Companion directories are always appended **last**. A companion can therefore *add* a driver GDAL did not have,
but can never shadow the version-locked drivers the active `libgdal` was built with. Non-existent directories and
duplicates are dropped, so GDAL is never handed a path that is not there.

`pyramids.base.config.Config.initialize_gdal` composes the list and writes it once, immediately before
`gdal.AllRegister()`. The early value set during the wheel bootstrap only covers the window before
`from osgeo import …` runs; it is not the final word.

### Keeping your own value

Set `PYRAMIDS_KEEP_GDAL_ENV=1` and whatever `GDAL_DRIVER_PATH` the process inherited wins, with companion
directories still appended. This is the escape hatch for a deliberately hand-built GDAL; it is off by default
because an inherited path is usually a conda environment's, which is not safe for a wheel's bundled `libgdal`.

## Registering a plugin directory

Declare the entry point in your distribution's `pyproject.toml`. The **value is a module name**, optionally
followed by `:<relative-subdir>`:

```toml
[project.entry-points."pyramids.gdal_plugin_path"]
parquet = "pyramids_parquet"            # -> <pyramids_parquet>/gdalplugins/
other   = "some_pkg:native/plugins"     # -> <some_pkg>/native/plugins/
```

Omitting the subdir means `gdalplugins/`. The subdir must stay inside the package — an absolute path, or one
containing `..`, is refused.

Discovery is a **pure metadata scan**. pyramids resolves the module with `importlib.util.find_spec` and never
imports your package, because the scan runs during `import pyramids`, before GDAL registers drivers; importing a
companion there would drag its whole dependency tree into every `import pyramids`. The result is cached for the
process — call `pyramids.base._gdal_plugins.clear_companion_plugin_dir_cache()` if you install a package
mid-session.

### The ABI contract you are accepting

A GDAL plugin is **not** portable between GDAL builds:

- It must be compiled against the **exact** `libgdal` that will load it. GDAL's standalone driver builds enable
  `STRICT_VERSION_CHECK`, so a mismatched plugin is refused rather than crashing.
- pyramids' three wheel build models do not all ship the same GDAL (currently 3.13.3 for the conda-extract and
  from-source wheels, 3.12.4 for win_arm64), so a companion needs a build per platform.
- The wheel repair tools rename the vendored libraries with a per-build content hash — a published wheel contains
  `libgdal-<hash>.so.39`, not `libgdal.so.39`. A plugin built outside that wheel's own build job records the
  unmangled name and will fail to resolve it.

In short: a companion plugin wheel has to be produced by the same build that produced the core wheel it targets.
See `planning/bundle/ogr-parquet-driver-plan-2026-11-16.md` for the full analysis.

## Parquet and Arrow: what works where

The OGR Parquet and Arrow drivers ship as such a plugin (conda-forge's `libgdal-arrow-parquet`), and the pyramids
wheels bundle **neither** — the Arrow C++ runtime they link adds roughly 10–12 MB to every wheel for a capability
only one code path needs. What that means in practice:

| Operation | Needs the OGR driver? | Works on a bare `pip install`? |
|---|---|---|
| `FeatureCollection.to_parquet(...)` | No — geopandas → pyarrow | Yes, with the `[parquet]` extra |
| `FeatureCollection.read_parquet(...)` | No — `geopandas.read_parquet` | Yes, with the `[parquet]` extra |
| `FeatureCollection.iter_features("x.parquet", ...)` | **Yes** | **No** |

So reading and writing GeoParquet is fully supported everywhere; only the OGR *streaming* path over a `.parquet`
needs the driver. Attempting it without one raises a `VectorDriverError` naming both remedies — installing
`libgdal-arrow-parquet` into a conda environment, or using `read_parquet()` instead:

```python
>>> FeatureCollection.iter_features("reaches.parquet")   # doctest: +SKIP
Traceback (most recent call last):
    ...
pyramids.base._errors.VectorDriverError: Streaming 'reaches.parquet' requires reading it through
GDAL/OGR. The active GDAL has no OGR "Parquet" driver.
  - conda-forge: conda install -c conda-forge libgdal-arrow-parquet
  - or point GDAL_DRIVER_PATH at a directory holding a matching plugin, or register it
    through the 'pyramids.gdal_plugin_path' entry point
For an in-memory read that needs no GDAL driver, use FeatureCollection.read_parquet() instead.
```

Whether the drivers should be vendored, and in what shape, is tracked in
[#457](https://github.com/serapeum-org/pyramids-core/issues/457).

## Further reading

- [Plot backends](plot-backends.md) — the same entry-point discovery pattern, for renderers.
- [Writing a vector driver](feature/writing-a-vector-driver.md) — the Python-level driver hook, which is a
  different seam: that one adds a *pyramids* reader, this one adds a *GDAL* one.
- [Installation](../installation.md) — the wheel matrix and what each platform bundles.
