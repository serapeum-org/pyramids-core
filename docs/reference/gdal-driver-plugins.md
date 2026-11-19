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

`pyramids.base.config.Config.initialize_gdal` composes the list and writes it immediately before
`gdal.AllRegister()`. Earlier steps also set the variable — the wheel bootstrap, and the conda/system discovery
that runs one line before the composition — but this write happens after all of them, so it is the value GDAL
reads. When nothing composes, because not one candidate directory exists, the variable is cleared rather than
left holding a path the composition already rejected.

An inherited value is used when there is **no** bundled directory, where it is the best pointer to the GDAL that
is actually loaded, and it then leads the discovered directory. It is refused outright alongside a wheel's own
plugins, which are ABI-locked to that wheel's `libgdal`.

### Keeping your own value

Set `PYRAMIDS_KEEP_GDAL_ENV=1` and whatever `GDAL_DRIVER_PATH` the process inherited leads the path, with the
bundled, discovered and companion directories ranked behind it rather than discarded — the hatch re-ranks, it
does not remove, so a wheel keeps its own drivers. A value that points nowhere is also left alone under the
hatch. This is the escape hatch for a deliberately hand-built GDAL; it is off by default because an inherited
path is usually a conda environment's, which is not safe for a wheel's bundled `libgdal`.

### Leaving the host's GDAL alone

When pyramids is imported into a process that already owns GDAL — QGIS is the motivating case — set
`PYRAMIDS_MANAGE_GDAL_ENV=0` **before** importing pyramids. pyramids then does not touch `GDAL_DRIVER_PATH`,
`GDAL_DATA`, `PROJ_LIB` or `PATH` and does not activate its vendored GDAL tree: it binds to the host's GDAL as-is.
pyramids also auto-detects this case (a host that imported `osgeo` first *and* had already configured the GDAL
env), but the env var is the reliable switch for an embedder because it is read before the import-time setup runs.
This is distinct from `PYRAMIDS_KEEP_GDAL_ENV` above: that re-ranks the composed driver path, this makes pyramids
hands-off entirely. `PYRAMIDS_MANAGE_GDAL_ENV=1` forces the opposite — always manage, even inside a host — and
`Config(manage_gdal_env=False)` is the programmatic equivalent for a single `Config`, though it governs only the
four env vars (not `gdal.UseExceptions()` or the YAML config options, which always apply).

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
life of the process, and `GDAL_DRIVER_PATH` is composed once at import, so installing a companion into a running
interpreter has no effect — **restart it** to pick the plugin up.

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
This is why no companion plugin wheel ships today, and it is the main open question in
[#457](https://github.com/serapeum-org/pyramids-core/issues/457).

## Parquet and Arrow: what works where

The OGR Parquet and Arrow drivers ship as such a plugin (conda-forge's `libgdal-arrow-parquet`), and the pyramids
wheels bundle **neither** — the Arrow C++ runtime they link adds roughly 10–12 MB to every wheel for a capability
only one code path needs. What that means in practice:

| Operation | Needs the OGR driver? | Works on a bare `pip install`? |
|---|---|---|
| `FeatureCollection.to_parquet(...)` | No — geopandas → pyarrow | Yes, with the `[parquet]` extra |
| `FeatureCollection.read_parquet(...)` | No — `geopandas.read_parquet` | Yes, with the `[parquet]` extra |
| `FeatureCollection.read_file("x.parquet")` | **Yes** — routed through OGR | **No** |
| `FeatureCollection.iter_features("x.parquet", ...)` | **Yes** | **No** |

So the pyarrow-backed pair works everywhere, and **both** OGR entry points need the driver — `read_file` as much
as `iter_features`, since `read_file` hands the path to pyogrio rather than special-casing the suffix. Use
`read_parquet()` to read a GeoParquet without a GDAL driver.

Either OGR call without the driver raises a `VectorDriverError` naming both remedies — installing
`libgdal-arrow-parquet` into a conda environment, or using `read_parquet()` instead. The same error appears when
the plugin *is* installed but cannot load its own dependencies, which is what conda-Windows'
`Can't load requested DLL: duckdb.dll 126` means: GDAL registers a deferred proxy for it, so the driver looks
present until it is first used.

`iter_features` is a generator, so the call itself returns normally and the error surfaces on the first item.
Wrap the iteration, not the call:

```python
>>> list(FeatureCollection.iter_features("reaches.parquet"))   # doctest: +SKIP
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
