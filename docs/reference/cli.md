# Command line (`pyramids`)

Installing pyramids adds a `pyramids` command — a thin CLI over the same API, for quick jobs and scripts without
writing Python. Every command has `-h/--help`:

```bash
pyramids --help                 # list all commands
pyramids <command> --help       # options for one command
pyramids --version              # print the installed version
python -m pyramids info x.tif   # the module form, equivalent to the `pyramids` script
```

Place the global **`--quiet`** (`-q`) before the command (`pyramids --quiet warp …`) to suppress the `wrote …`
chatter; errors still print. Commands that write a file refuse to clobber an existing output unless you pass
**`--overwrite`**. A handful take
spatial options: `bounds`, `warp`, `georeference`, `orthorectify`, and `edit-info` accept a CRS as an EPSG code,
WKT, or PROJ string; `warp`, `overview`, `georeference`, and `orthorectify` accept a `--resampling` method
(`nearest`, `bilinear`, `cubic`, `average`, `mode`, …).

## Cross-cutting options

These global flags go **before** the command and apply across the CLI:

- **`--format {text,json,csv,table}`** — output format for data-emitting verbs (`info`, `bounds`, `stats`,
  `list-*`, …). `json` is honored by every verb that supports `--json` (the per-command `--json` flag still works
  as an alias); `csv` and `table` are rendered by the tabular verbs (e.g. `vector info/bounds`, `nc
  groups/list-vars`, `mesh mesh-names`). Example: `pyramids --format csv vector info roads.gpkg`.
- **`--config PATH`** / **`--no-config`** — read per-command defaults from a `pyramids.toml`. Discovery order:
  `--config PATH`, then `PYRAMIDS_CONFIG`, then the nearest `pyramids.toml` searched upward from the working
  directory, then `~/.config/pyramids/config.toml`. An explicit flag on the command line always overrides a
  config default; `--no-config` ignores any file. A top-level `[defaults]` table applies to every verb that has
  the option, and a `[group.command]` table applies to that one verb. Only **value-taking** options are
  configurable — boolean flags such as `--overwrite` are deliberately not settable from config (there is no
  `--no-overwrite` to override a configured `true` back to false, so the guard can never be silently disabled):

  ```toml
  [defaults]
  driver = "gpkg"

  [vector.warp]
  crs = "EPSG:4326"
  driver = "gpkg"
  ```

- **`--progress`** — show a progress indicator on stderr for multi-item write verbs (e.g. `collection resample`,
  `vector explode-gpx`). It is a no-op when stderr is not an interactive terminal (pipes, logs, CI) or when
  `--quiet` is set.

**Shell completion** — `pyramids completion {bash,zsh,fish}` prints the completion hook for your shell. Completion
is driven by [argcomplete](https://pypi.org/project/argcomplete/) (`pip install argcomplete`); pyramids keeps it
an optional convenience and works without it.

**Plugins** — a third-party package that registers a `pyramids.cli_commands` entry point contributes its own verbs
to `pyramids` automatically. `pyramids plugins list` reports the installed plugins and whether each one loaded.

## Inspect & edit metadata

```bash
pyramids info dem.tif                 # human-readable metadata (add --json for machine output)
pyramids info dem.tif --json
pyramids bounds dem.tif               # bounding box; --crs 4326 reprojects the corners; --json for JSON
pyramids edit-info dem.tif --crs 4326 --nodata -9999 --tag units=m   # edit CRS / nodata / tags in place
pyramids blocks dem.tif --json        # native block (tile) windows; --band N (0-based, default 0)
```

- **`info`** — print raster metadata (`--json` for a machine-readable dump).
- **`bounds`** — print the bounding box; `--crs` reprojects the corners; `--json` emits JSON.
- **`edit-info`** — set `--crs`, `--nodata`, and/or repeatable `--tag KEY=VALUE` on a raster **in place**.
- **`blocks`** — list a raster's native block (tile) windows, each with its pixel window and map bounds;
  `--band` selects the band whose blocking to report (0-based, default 0); `--json` emits JSON.

## Convert, warp, clip, merge, overviews

```bash
pyramids convert in.asc out.tif                       # re-encode; --driver GTiff to force a driver
pyramids warp in.tif out.tif --crs 4326 --resampling bilinear
pyramids clip in.tif out.tif --bbox 440000 475000 480000 515000    # or: --vector aoi.geojson
pyramids merge tileA.tif tileB.tif tileC.tif mosaic.tif            # two or more inputs, last arg = output
pyramids merge tileA.tif tileB.tif aoi.tif --bbox 4.1 51.9 4.6 52.2 --bbox-crs 4326   # read only the window
pyramids overview dem.tif --levels 2 4 8 --resampling average      # build image pyramids IN PLACE
```

- **`convert`** — re-save a raster in another format (driver inferred from the extension, or `--driver`).
- **`subdataset`** — extract one subdataset of a container (NetCDF/HDF store, multi-subdataset GeoTIFF, Sentinel
  product, …) to a standalone raster, selected by 0-based index (negatives allowed) or by its full GDAL name.
- **`warp`** — reproject to `--crs` with an optional `--resampling`.
- **`clip`** — crop by `--bbox MINX MINY MAXX MAXY` (in the raster CRS) **or** by a polygon `--vector`. This is a
  *spatial* crop (it wraps `Dataset.crop`); it is **not** the numpy value-clamp `Dataset.clip` (clamp values to a
  `[min, max]` range), which has no CLI verb — use `calc` with `np.clip(A, lo, hi)` for that.
- **`merge`** — mosaic two-or-more rasters; the **last** positional argument is the output. `--bbox MINX MINY
  MAXX MAXY` restricts the mosaic to that window and reads only it, which is what makes a small area of interest
  cheap to pull from remote sources instead of transferring every source in full. The window is in the mosaic's
  own CRS unless `--bbox-crs` names another.
- **`overview`** — build power-of-two overviews (`--levels 2 4 8 …`) into the file itself.

## Cloud-Optimized GeoTIFF

```bash
pyramids cog create in.tif out.cog.tif --profile zstd --blocksize 512   # write + auto-validate
pyramids cog validate out.cog.tif --strict                              # --strict: warnings are errors
pyramids cog info out.cog.tif                                           # structured COG layout / overviews
```

`cog create` profiles: `deflate`, `zstd`, `lzw`, `webp`, `jpeg`, `lerc`, `lerc_deflate`, `lerc_zstd`,
`packbits`, `raw`. Pass `--compress`/`--blocksize` to override, `--no-validate` to skip the post-write check.
See the [COG CLI reference](cog/cli.md) for the full option set.

## Band math

```bash
pyramids calc '(A - B) / (A + B)' nir.tif red.tif ndvi.tif --dtype float32
```

- **`calc`** — evaluate an expression over inputs `A, B, …` (the operands, in order) into a new raster; the
  **last** operand is the output path. `--dtype` sets the output NumPy dtype. Every input must already sit on
  the first one's grid — `calc` does not resample, and refuses a mismatch rather than writing the answer on
  a grid the inputs never agreed on. Warp them onto a common grid first (`pyramids warp`). An input carrying
  no CRS tag at all is compared on its pixel grid alone, so an untagged mask or QA layer that sits on the
  template's cells still works. The output's no-data value is `-9999` whatever the inputs declare — or, when
  the output dtype cannot hold `-9999`, that dtype's own fallback (`255` for `uint8`, `65535` for `uint16`), which
  `calc` reports on stderr. The in-process equivalent is
  [`Dataset.combine`](dataset/analysis.md#combining-two-rasters) and the `-` / `+` / `*` / `/` operators, which do
  apply no-data masking — `calc` evaluates over the raw arrays.

## Georeferencing

```bash
# Fit a transform through ground-control points (pixel/line -> map x/y)
pyramids georeference raw.tif out.tif \
    --gcp 0 0 100000 500000 --gcp 512 0 105000 500000 --gcp 0 512 100000 495000 \
    --gcp-crs 32636 --transform polynomial --order 1 --to-crs 4326

# Orthorectify from the raster's RPC sensor model (needs a DEM or a constant height)
pyramids orthorectify scene.tif ortho.tif --dem dem.tif --to-crs 4326
```

- **`georeference`** — warp from repeatable `--gcp PIXEL LINE X Y` points (with `--gcp-crs`); `--transform`
  `polynomial` (order 1–3 via `--order`) or `tps`; optional `--to-crs`.
- **`orthorectify`** — apply the raster's RPC model using `--dem` (or a constant `--rpc-height`).

## Raster ↔ vector

```bash
pyramids rasterize parcels.geojson parcels.tif --cell-size 10 --column value   # or --like template.tif
pyramids shapes classes.tif classes.geojson --geometry polygon                 # vectorize (one feature/cell)
pyramids sample dem.tif --points "440000,510000;450000,505000" --json          # read values at points
```

- **`rasterize`** — burn a vector into a new raster; set `--cell-size` **or** adopt a template grid with
  `--like`; `--column` selects the attribute to burn (default: all non-geometry columns).
- **`shapes`** — vectorize a raster to a vector file, one feature per cell (`--geometry polygon|point`,
  `--driver`); guarded for huge rasters — pass `--allow-large` to override the ~4M-cell safety limit.
- **`sample`** — read band values at `--points` (`'x,y'` pairs separated by `;`); `--json` for JSON output.

## Quick-look render

```bash
pyramids render dem.tif quicklook.png --band 0 --title "DEM" --dpi 150   # headless PNG via cleopatra
```

- **`render`** — save a PNG preview of a raster. Plotting goes through
  [cleopatra](https://github.com/serapeum-org/cleopatra) (the Agg backend is forced, so it is headless); reach for
  `Dataset.plot` in Python for anything beyond a quick look.

## Full raster (`Dataset`) surface

Beyond the core verbs above, the CLI wraps the full GIS/raster surface of `Dataset` — each command is a thin 1:1
wrapper over an existing method, grouped here by theme. Commands that write a file refuse to clobber an existing
output without `--overwrite`; inspection commands take `--json`.

**Read / construct** — build a raster from another representation, writing the result to the output path:
`from-bytes` (`--stdin` or a bytes file), `from-dataframe` (a `(band, y, x)` CSV), `from-points` (a point vector),
`from-point-arrays` (an `x,y,z` CSV), `from-band-files` (stack single-band rasters), `from-archive` (a member of a
zip/tar/gzip), `from-zarr` (needs the `[lazy]` extra), and the blank-canvas constructors `create`, `create-empty`,
`empty-like`, `dataset-like` (stamp a template's georeference onto a `.npy` array).

```bash
pyramids from-band-files red.tif green.tif blue.tif rgb.tif        # stack bands into one raster
pyramids create-empty blank.tif --rows 512 --columns 512 --epsg 4326 --cell-size 0.01
```

**Inspect / metadata** (`--json`) — `stats` (`--exact`), `histogram`, `get-meta-data` (`--domain`),
`get-attribute-table` / `set-attribute-table`, `mask-flags`, `footprint` (valid-data polygon → an optional vector
output), `cell-area`, `domain-area`, `count-domain-cells`, `same-grid` (compare two rasters' grids), and
`checksum` (per-band GDAL checksum for integrity / round-trip verification).

**Bands / georeferencing edit** (`SRC OUT`) — `add-band` (`--source`), `select-bands` (`--bands`, 1-based),
`get-band-by-color`, `create-mask-band`, `read-masks` / `get-mask` (write the mask as a raster), `set-color-ramp`,
`set-rpcs` (`--rpc-json`), `geolocate` (warp from the GDAL geolocation domain). The metadata-mutating commands edit
an in-memory copy and save to a new output, leaving the input untouched.

**Spatial ops** — `align` (resample onto a reference grid), `overlay` / `combine` (two aligned rasters;
`combine --op add|subtract|...`), `warped-view` (materialise a reprojected view), `wrap-longitude`, `translate`
(gdal_translate-style `--src-win` / `--proj-win` / `--scale` / ...), `transect --line` (profile along a line → CSV
or vector), `extract` (values, optional `--mask`). Two-raster ops raise cleanly on a CRS mismatch, except `align`,
which legitimately reprojects the source onto the reference grid.

**Analysis / terrain** — `proximity`, `cluster`, `zonal-stats` (`--stats`; CSV / JSON), `match-histogram`,
`percentile-stretch`, `pan-sharpen`, `fillna`, `where --cond`, `convert-units`, and `contour` (`--interval` or
`--fixed-levels` → a vector of contour lines). The fixed-kernel terrain operators `slope`, `aspect`, `hillshade`,
`resample`, `fill`, `sieve`, `focal-mean`, `focal-std` are reached through the existing `apply` command.

**Overviews / windowed reads** — `recreate-overviews`, `read-overview-array` (a decimated level → raster or
`--json`), `read-part --bbox` (a pixel/geo window → raster), `preview --max-size` (a decimated thumbnail), `point`
(sample at a map coordinate), `read-tile z x y` (an XYZ/Web-Mercator tile → raster).

**Export / convert** — `to-bytes`, `to-cog-bytes`, `to-dataframe` (CSV / Parquet — Parquet needs `[parquet]`),
`to-polygons` (region-dissolving vectorize), `to-raster` (band/tile/creation-option control over `convert`),
`to-image` (PNG/JPEG via a colormap — needs `[viz]`), `to-xyz`, `to-zarr` (needs `[lazy]`), `to-stac-item`,
`to-terrain-rgb` (Mapbox Terrain-RGB encode).

**Plot (headless, via cleopatra)** — `plot-histogram`, `plot-vector-field`, `animate`; the Agg backend is forced so
these run headless (`render` remains the quick-look PNG, and `contour` lives under analysis above).

**Network readers** — the `fetch` group gains `fetch wmts` and `fetch ogc-coverages` alongside `fetch wms` / `wcs`
/ `wfs` (see [Remote / OGC services](#remote-ogc-services)).

Deferred to the Python API (documented in each sub-module): the callable-taking methods (`apply`, `map_blocks`,
`focal_apply`), the low-level coordinate/block primitives (`array_to_map_coordinates`, `rowcol`, `xy`,
`block_windows`, `read_windows`, `iter_blocks`, `get_overview` / `get_overview_dataset`), the in-place
`write_array`, the deprecated `cluster2`, the internal `fill_gaps`, and the array predicates / lifecycle helpers
(`isnull`, `astype`, `round`, `copy`, ...), which compose in Python.

## Vector

```bash
pyramids vector info parcels.shp --json                      # feature count, geometry, CRS, columns, bounds
pyramids vector bounds parcels.shp --crs 3857                # bounding box (optionally reprojected)
pyramids vector convert parcels.shp parcels.gpkg             # driver inferred from the extension, or --driver
pyramids vector warp parcels.shp parcels_3857.gpkg --crs 3857   # reproject
pyramids vector clip parcels.shp aoi.gpkg --bbox 4.1 51.9 4.6 52.2   # or --mask mask.gpkg
```

- **Core:** `info` / `bounds` (with `--json`), `convert`, `warp` (reproject), and `clip` by `--bbox` (reads only
  the window) or a `--mask` polygon layer. `--layer` selects a layer in a multi-layer source.

The group wraps the full vector/GIS surface of `FeatureCollection` — each command is a thin 1:1 wrapper over an
existing method. Writes refuse to clobber without `--overwrite`.

**Inspect** — `feature-count`, `feature-info`, `list-layers`, `estimate-utm`, `is-valid-reason`,
`is-valid-coverage`, `invalid-coverage-edges`, `count-coordinates` / `count-geometries` / `count-interior-rings`
(`--json` on each).

**Unary geometry** (`SRC OUT`) — `buffer`, `simplify` / `simplify-coverage`, `make-valid`, `normalize`, `reverse`,
`force-2d` / `force-3d`, `segmentize`, `remove-repeated-points`, `offset-curve`, `representative-point`,
`concave-hull`, `minimum-bounding-circle` / `minimum-rotated-rectangle` / `maximum-inscribed-circle`,
`affine-transform`, `rotate`, `scale`, `translate`, `snap --other`, `build-area`, `extract-unique-points`,
`set-precision`.

```bash
pyramids vector buffer gauges.geojson buf.geojson --distance 500
pyramids vector simplify coast.geojson simple.geojson --tolerance 10
```

**Overlay / set ops** (need a second layer) — `intersection` / `difference` / `symmetric-difference` / `union`
(`--other`), `intersection-all`, `union-all`, `overlay --right --how`, `dissolve [--by]`, `sjoin` / `sjoin-nearest`
(`--right`), `clip-by-rect --rect`, `shared-paths` / `shortest-line` (`--other`), `explode`, and `filter
--predicate <contains|within|intersects|covers|covered-by|crosses|disjoint|overlaps|touches|dwithin> --mask`.

```bash
pyramids vector overlay parcels.geojson zones.geojson out.geojson --right zones.geojson --how intersection
pyramids vector filter parcels.geojson in_aoi.geojson --predicate intersects --mask aoi.geojson
```

**Tessellation / binning / sampling** — `voronoi-polygons`, `delaunay-triangles`,
`constrained-delaunay-triangles`, `fishnet --bounds --cell-size --epsg`, `to-h3 --resolution`,
`h3-bin --resolution`, `sample-points --size`, `polygonize`.

**Interpolate / tessellate / build** — the value-add constructors and in-memory readers now exposed as one-shot
verbs: `interpolate-to-raster --column` (IDW-grid a point column onto a raster → GeoTIFF), `voronoi` (pyramids'
own Thiessen tessellation of a point layer, with optional `--values` choropleth and `--clip` boundary — distinct
from the geopandas-delegated `voronoi-polygons`), `quadtree` (adaptive quad-tree point binning, `--nmax`/`--nmin`),
`with-coordinates` / `with-centroid` (append per-vertex / per-feature coordinate columns), and the readers
`from-xyz` (CSV `--x`/`--y`/`--z` columns → points), `from-features` (a GeoJSON features file), `from-records`
(a CSV with a WKT `--geometry` column), and `read-gpx` (every non-empty GPX sub-layer → one multi-layer file).

```bash
pyramids vector voronoi stations.geojson cells.geojson --values rainfall --clip aoi.geojson
pyramids vector interpolate-to-raster stations.geojson surface.tif --column rainfall --cell-size 100
```

**Measurements** (print; `--json`) — `geodesic-area` / `geodesic-length` / `geodesic-distance`, `distance`,
`hausdorff-distance`, `frechet-distance`, `minimum-clearance`, `hilbert-distance`, `get-coordinates` (to CSV).

**I/O & export** — `from-bbox`, `read-parquet`, `to-parquet`, `to-feather`, `to-json` (stdout or `--out`),
`to-geo-dict`, `to-wkt` / `to-wkb` (to CSV), `to-mvt`, `to-pmtiles`.

**Remote readers** — `from-ogc-features --endpoint --collection`, `from-featureserver --url`,
`from-vectortileserver --url` (plus `fetch wfs`).

The two-layer ops (overlay/set ops, `sjoin`, `snap`, the predicate `filter --mask`, and the planar distance
measurements) require both layers in the **same CRS** and exit 1 with a clear message on a mismatch — reproject
one first with `vector warp`. (The exception is `clip --mask`, which reprojects the mask to the source CRS for
convenience; `geodesic-distance` is CRS-agnostic.) A GEOS geometry error on the input (e.g. `shared-paths` on
non-lineal geometry) exits 1 with a one-line message, not a traceback. Not exposed as one-shot CLI verbs (use
Python): `from-dict` / `from-arrow` (need in-memory objects), `transform` (needs a callable), `to-arrow` /
`to-xarray` / `iter-features` (return Python objects), and `from-postgis` / `to-postgis` (need a live PostGIS
connection). Inherited pandas/GeoDataFrame tabular and arithmetic methods are out of scope.

## NetCDF

The `nc` group wraps the NetCDF-specific surface of `NetCDF`. Every command is a thin 1:1 wrapper over a `NetCDF`
method; reads go through GDAL's multidim driver (never xarray). Commands that write a file refuse to clobber an
existing output unless you pass `--overwrite`.

**Inspect**

```bash
pyramids nc info climate.nc --json              # variables, dimensions, CRS, bounds, time axis
pyramids nc summary climate.nc --vars tas pr    # per-variable statistics (count/min/max/mean/std)
pyramids nc metadata climate.nc --json          # the full metadata record
pyramids nc list-vars climate.nc                # variables GDAL can expose as rasters
pyramids nc dim-values climate.nc --dim lat     # a dimension's coordinate values
pyramids nc time-values climate.nc --json       # the formatted time-axis labels
```

- **`nc info --full`** adds the per-variable detail to the summary; **`nc time-values --raw`** emits the raw
  numeric time offsets instead of the formatted calendar labels.

**Select & subset** (each writes a new NetCDF)

```bash
pyramids nc sel climate.nc out.nc --time 2020-06-15 --var tas   # one time step (exact date label)
pyramids nc isel climate.nc out.nc --index time=0              # select by integer position
pyramids nc crop climate.nc out.nc --bbox 4.1 51.9 4.6 52.2    # spatial subset (file CRS)
pyramids nc thin climate.nc out.nc --step time=2               # keep every Nth step
pyramids nc head climate.nc out.nc --count time=3              # first N along a dimension (also: tail)
pyramids nc drop-sel climate.nc out.nc --label time=2020-06-15 # drop by label (also: drop-isel by index)
pyramids nc drop-duplicates climate.nc out.nc --dim time       # drop duplicate coordinates
pyramids nc dropna climate.nc out.nc --dim time --how all      # drop empty slices
pyramids nc pad climate.nc out.nc --width time=2 --mode constant --value 0
```

- **`nc sel`** matches by a `--time` date label (sub-daily axes may match several steps); `info`/`sel` assume the
  CF-standard time axis named `time`. `nc info` shows the step count and the first/last labels.
- **`nc subset`** reads one windowed slice in a single call — combine any of `--var`, `--time`, and `--bbox`
  (`--crs` sets the bbox CRS) instead of chaining `sel` / `crop` / `extract`.

**Variables, dimensions & coordinates** (each writes a new NetCDF)

```bash
pyramids nc extract climate.nc t2m.tif --var tas          # one variable -> a georeferenced raster
pyramids nc rename-var climate.nc out.nc --old tas --new t2m
pyramids nc remove-var climate.nc out.nc --var msk_rgn
pyramids nc reproject-var climate.nc out.nc --var tas --epsg 3857
pyramids nc resample-var climate.nc out.nc --var tas --cell-size 1.0
pyramids nc rename-dims climate.nc out.nc --map lat=y lon=x
pyramids nc drop-dims climate.nc out.nc --dims bnds
pyramids nc set-coords climate.nc out.nc --names lat lon    # also: reset-coords
pyramids nc squeeze climate.nc out.nc --dim time            # drop length-one dimensions
```

**Variable mutation** (each writes a new NetCDF) — `add-var --source` (copy variables in from another NetCDF),
`set-var --raster --var` (write a raster in as a named variable), `crop-var --var --mask` (crop one variable to a
vector mask), `expand-dims --var --dim` (lift a variable onto a new band dimension), `assign-coords --coord`
(restamp existing band dimensions).

**CF / time / calendar / calculus** (each writes a new NetCDF)

```bash
pyramids nc convert-calendar climate.nc out.nc --calendar noleap
pyramids nc interp-calendar climate.nc out.nc --target other.nc
pyramids nc groupby-bins climate.nc out.nc --dim time --bins 4 --how mean
pyramids nc rolling climate.nc out.nc --dim time --window 3 --how mean   # also: rolling-exp --alpha
pyramids nc shift climate.nc out.nc --dim time --periods 1
pyramids nc integrate climate.nc out.nc --dim time          # also: cumulative-integrate, differentiate
pyramids nc polyfit climate.nc out.nc --dim time --deg 2
pyramids nc interp climate.nc out.nc --coord plev=50000 --method linear
pyramids nc interpolate-na climate.nc out.nc --dim time
pyramids nc reduce climate.nc out.nc --dim time --how mean
```

**More reductions & interpolation** (each writes a new NetCDF) — `coarsen --dim --window --how` (block-reduce a
dimension by a window), `cumulative --dim --how` (running total/product), `weighted --weights --dims --how`
(weighted reduction), and `interp-like --target` (interpolate onto a target cube's band axes).

**Sub-groups** — `groups` lists a file's root-level sub-group names (`--json`); `get-group --group` writes one
sub-group out as a standalone NetCDF.

**Construct & combine** (each writes a new NetCDF) — `from-dataframe` (a MultiIndex CSV → a cube, `--x`/`--y`/
`--vars`), `from-dict` (a JSON structure dump → a cube), `concat --inputs --dim` (join cubes end to end along a
dimension), `merge --inputs` (put several cubes' variables together).

**Kerchunk / multi-file** — `to-kerchunk` writes a reference manifest for a single file, `combine-kerchunk
--inputs` a combined manifest over many files, and `open-mf --paths` reports a lazy variable stack across files
(`--json`).

**I/O export**

```bash
pyramids nc convert climate.nc out.nc           # re-write as a NetCDF
pyramids nc to-dataframe climate.nc out.csv --vars tas
pyramids nc to-dict climate.nc --json           # structure (add --data for the arrays)
pyramids nc to-zarr climate.nc out.zarr --var tas   # per-variable Zarr store (needs the [lazy] extra)
```

**Plot** (headless PNG/animation via cleopatra; the Agg backend is forced)

```bash
pyramids nc plot climate.nc map.png --var tas --kind contour
pyramids nc plot-histogram climate.nc hist.png --var tas
pyramids nc animate climate.nc anim.gif --var tas --dim time --fps 4
```

Not exposed as one-shot CLI verbs (use Python): `where`/`isin` (boolean-array conditions), `convert-units`
(per-variable unit metadata), `curvefit` (needs a Python callable), `open-subdataset` (no container-level target
on the shipped files), the xarray bridges `to_xarray`/`from_xarray`, the in-memory
constructors/combiners `from_array`/`combine`/`create_main_dimension`, and the array/arithmetic/reduction methods
inherited from `Dataset` (reachable via the top-level raster commands once a variable is `extract`ed). Generic
raster ops on a NetCDF plane (warp, slope, COG, …) are the top-level raster commands, not duplicated here.

## Raster stacks

```bash
pyramids collection info "tiles/*.tif" --json                # count, CRS, grid of a co-registered stack
pyramids collection stats "tiles/*.tif" mean.tif --reduce mean   # time reduction (mean|sum|min|max|std|var)
pyramids collection resample "tiles/*.tif" monthly/ --freq 1MS --op mean   # calendar-frequency binning
```

The `collection` group wraps the full `DatasetCollection` surface over a folder/glob of co-registered GeoTIFFs.
Every opening verb takes a shared `--source {files,zarr,archive}` option (default `files` — a glob or directory;
`zarr` a Zarr store; `archive` a zip/tar), plus `--member-glob` / `--date-format` to pick and stamp the members.
STAC and point-cloud sources are deferred to the Python API.

- **Inspect** — `info` (count, CRS, grid; `--json`).
- **Time reduction** — `stats` (one reduction → one raster; `mean|sum|min|max|std|var`) and `resample`, which
  bins the time axis to a pandas `--freq` (e.g. `1MS`, `7D`, `6h`) and reduces each window with a rich `--op`
  (`mean|sum|min|max|std|var|median|quantile|percentile|mode|trend|sens_slope`), writing one raster per window.
- **Batch geoprocessing** (applied to every timestep) — `reproject --crs`, `crop` (`--mask` or `--bbox`/`--epsg`),
  and `align --reference` (resample each step onto a reference raster's grid).
- **Export** — `to-file` (every step to a directory), `to-cog` (each step as its own COG), `to-netcdf` /
  `to-zarr` (the whole `(T, B, R, C)` cube to one store), `to-kerchunk` (a reference JSON over NetCDF/HDF5
  sources), and `merge` (mosaic every timestep into one raster, with `--bbox`/`--method`). `to-zarr` and
  `to-kerchunk` (and the `--source zarr` opener) need the `[lazy]` extra.

## UGRID mesh

The `mesh` group now wraps the full `MeshDataset` surface (the earlier "Phase 1" label is obsolete) — each command
is a thin 1:1 wrapper over a `MeshDataset` method. Writes refuse to clobber without `--overwrite`; inspection
verbs take `--json`. The original Phase-1 `extract` (rasterize onto a `--like` grid) now coexists with the new
`sample-point` (sample a variable at arbitrary x/y coordinates).

```bash
pyramids mesh info mesh.nc --json                            # node/face/edge counts, CRS, bounds, variables
pyramids mesh extract mesh.nc z.tif --like grid.tif --var mesh2d_node_z   # rasterize a variable onto a grid
pyramids mesh convert mesh.nc mesh.gpkg --var mesh2d_node_z --location node  # mesh -> vector
pyramids mesh sample-point mesh.nc --var mesh2d_node_z --x 4.1 4.6 --y 51.9 52.2 --json  # sample at x/y
```

**Read / construct** — `from-arrays` (node/face `.npy` arrays), `from-structured` (a quad mesh from a raster),
`from-geodataframe` (from a polygon layer), `from-bytes`, `from-zarr` (needs `[lazy]`), `read-all` (write every
topology to its own NetCDF), `open-mf` (concatenate several files), `mesh-names` (list topologies in a file).

**Inspect** — `info`, `summary`, `get` / `items` / `keys` / `values` (describe variables), `masked-values`,
`active-mask`, `histogram`, `count-domain-cells`, `domain-area`, `circumcenters`, `sparse-adjacency` (adjacency
matrix summary / `.npz`).

**Select / subset / sample** — `isel` / `sel` (by position / coordinate), `sel-time` / `sel-time-range`, `head` /
`tail` / `thin`, `squeeze`, `isel-layer` / `sel-layer` / `collapse-layer`, `drop-sel` / `drop-isel` /
`drop-duplicates` / `drop-variables` / `dropna`, `where` (mask by comparison), `crop` / `clip` (polygon mask or
bbox),
`subset-by-bounds`, and the samplers `sample`, `sample-line`, `sample-point`, `sample-random`.

**Convert / export** — `convert` / `to-file` / `to-bytes` (UGRID NetCDF or a driver), `to-crs` (reproject) /
`set-crs` (declare only), `to-geodataframe` / `to-dataframe` / `to-image` / `to-zarr`, `to-dataset` (rasterize to
GeoTIFF), `to-location` (remap onto another facet), `polygonize` / `contour` / `bounding-polygon` /
`exterior-edges` / `exterior-faces` / `footprint`,
`voronoi-polygons` (Voronoi cells of the nodes), `burn` (a vector onto faces), `with-variable` /
`rename-variable` / `attach-data`.

**Analysis / terrain** — `slope`, `aspect`, `hillshade`, `proximity`, `cluster`, `connected-components`,
`zonal-stats`, `sieve`, `binary-dilation` / `binary-erosion`, `focal-mean` / `focal-std`, `regrid`, `polyfit`
(per-element polynomial trend along time).

**Interp / fill / time / calculus** — `interp-like` / `interp-time` / `interp-calendar`, `interpolate-na` /
`fill-na-spatial` / `laplace-interpolate` / `ffill` / `bfill` / `fillna`, `clip-values`, `convert-units`,
`coarsen` / `resample` / `rolling` / `shift`, `integrate` / `cumulative-integrate` / `differentiate`,
`convert-calendar`.

**Plot (headless, via cleopatra)** — `plot`, `plot-histogram`, `plot-outline`, `plot-vector-field`, `animate`;
the Agg backend is forced so these run headless.

**Global attributes** — `set-attr` / `del-attr` (edit a global attribute into a new mesh file).

**Multi-mesh join** — `concat` (stack several same-topology meshes) and `merge` (combine their variables).

Not exposed (use Python, or pending a library fix): `curvefit` and `focal_apply` take a Python callable, so
they have no shell form. (`polyfit`, `voronoi-polygons` and `sel-time` are now verbs — listed under Analysis,
Convert/export and Select/subset above.)

## STAC

```bash
pyramids stac search https://earth-search.aws.element84.com/v1 \
    --collections sentinel-2-l2a --bbox 4.1 51.9 4.6 52.2 --datetime 2023-06-01/2023-06-30 --json
pyramids stac download <stac-url> --collections sentinel-2-l2a --bbox 4.1 51.9 4.6 52.2 \
    --assets red nir --out ./scenes
```

`stac` needs the `[stac]` extra (`pystac-client`, `stac-asset`); it reports a clear install hint if missing.

## Remote / OGC services

```bash
pyramids fetch wms https://example/wms scene.tif --layers topo --bbox 4.1 51.9 4.6 52.2 --crs 4326 --size 1024 1024
pyramids fetch wcs https://example/wcs dem.tif --coverage elevation --bbox 4.1 51.9 4.6 52.2
pyramids fetch wfs https://example/ows roads.geojson --typename topp:roads --bbox 4.1 51.9 4.6 52.2
```

- **`fetch`** pulls a bounding-box window from an OGC web service to a local file — `wms` / `wcs` (raster) and
  `wfs` (vector), via GDAL's native drivers.

## Run one tool, or a pipeline

```bash
pyramids tools                                               # list the registered processing tools
pyramids tool slope                                          # one tool's parameter schema
pyramids apply slope dem.tif slope.tif --set units=degrees   # run a single tool, no YAML needed
pyramids run pipeline.yaml --inputs a.tif b.tif --out ./out --manifest run.yaml   # batch + provenance
```

- **`apply`** runs one registered tool over a single input into an explicit output, taking typed `--set KEY=VALUE`
  parameters (discover them with `pyramids tool NAME`).
- **`run`** batches a pipeline YAML over one or more `--inputs` (paths or a glob) into `--out`; `--manifest` writes a
  reproducible recipe plus per-input timings, and `--parallel` runs the batch across a process pool.

!!! tip "The CLI mirrors the API"
    Each command maps to a `Dataset` / `FeatureCollection` method — e.g. `warp` → `to_crs`, `clip` → `crop`,
    `rasterize` → `Dataset.from_features`, `shapes` → `to_geodataframe`. Reach for Python when you need to
    compose steps or stay in-memory; reach for the CLI for one-off file jobs.
