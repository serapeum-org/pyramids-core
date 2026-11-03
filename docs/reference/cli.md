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

## Inspect & edit metadata

```bash
pyramids info dem.tif                 # human-readable metadata (add --json for machine output)
pyramids info dem.tif --json
pyramids bounds dem.tif               # bounding box; --crs 4326 reprojects the corners; --json for JSON
pyramids edit-info dem.tif --crs 4326 --nodata -9999 --tag units=m   # edit CRS / nodata / tags in place
```

- **`info`** — print raster metadata (`--json` for a machine-readable dump).
- **`bounds`** — print the bounding box; `--crs` reprojects the corners; `--json` emits JSON.
- **`edit-info`** — set `--crs`, `--nodata`, and/or repeatable `--tag KEY=VALUE` on a raster **in place**.

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
- **`warp`** — reproject to `--crs` with an optional `--resampling`.
- **`clip`** — crop by `--bbox MINX MINY MAXX MAXY` (in the raster CRS) **or** by a polygon `--vector`.
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

## Vector

```bash
pyramids vector info parcels.shp --json                      # feature count, geometry, CRS, columns, bounds
pyramids vector bounds parcels.shp --crs 3857                # bounding box (optionally reprojected)
pyramids vector convert parcels.shp parcels.gpkg             # driver inferred from the extension, or --driver
pyramids vector warp parcels.shp parcels_3857.gpkg --crs 3857   # reproject
pyramids vector clip parcels.shp aoi.gpkg --bbox 4.1 51.9 4.6 52.2   # or --mask mask.gpkg
```

- The vector group mirrors the raster single-shot commands on a `FeatureCollection`: `info` / `bounds` (with
  `--json`), `convert`, `warp` (reproject), and `clip` by `--bbox` (reads only the window) or a `--mask` polygon
  layer. `--layer` selects a layer in a multi-layer source.

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

Not exposed as one-shot CLI verbs (use Python): `where`/`isin` (boolean-array conditions), `subset` (slice/tuple
arguments), `expand-dims`/`open-subdataset` (no container-level target on the shipped files), `interp-like`/
`convert-units` (operate per gridded variable, which this group selects by other means), `curvefit`/`weighted`/
`cumulative` (need a Python callable or weights array), and the array/arithmetic methods inherited from `Dataset`
(reachable via the top-level raster commands once a variable is `extract`ed). Generic raster ops on a NetCDF plane
(warp, slope, COG, …) are the top-level raster commands, not duplicated here.

## Raster stacks

```bash
pyramids collection info "tiles/*.tif" --json                # count, CRS, grid of a co-registered stack
pyramids collection stats "tiles/*.tif" mean.tif --reduce mean   # time reduction (mean|sum|min|max|std|var)
```

## UGRID mesh

```bash
pyramids mesh info mesh.nc --json                            # node/face/edge counts, CRS, bounds, variables
pyramids mesh extract mesh.nc z.tif --like grid.tif --var mesh2d_node_z   # rasterize a variable onto a grid
pyramids mesh convert mesh.nc mesh.gpkg --var mesh2d_node_z --location node  # mesh -> vector
```

- **`mesh info`** — summarise a UGRID mesh (`--json` for a machine-readable dump).
- **`mesh extract`** — rasterize a mesh variable onto the `--like` template raster's grid/CRS.
- **`mesh convert`** — write the mesh as a vector file (`--location face|node|edge`; driver inferred from the
  extension or `--driver`).

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
