# collection — temporal raster-stack (`DatasetCollection`) operations

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids collection --help` for the live
listing and `pyramids collection <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?*
suggestion.

## Command tree

- `align` — align every timestep to a reference raster grid
- `crop` — crop every timestep to a mask or bbox
- `info` — summarize a raster stack
- `merge` — mosaic every timestep into one raster
- `reduce` — write a time reduction of a stack
- `reproject` — reproject every timestep to a target CRS
- `resample` — calendar-frequency time reduction (one raster per window)
- `export` …
  - `cog` — write each timestep as its own COG
  - `file` — write every timestep to a directory
  - `kerchunk` — write a kerchunk reference JSON (NetCDF/HDF5 sources)
  - `netcdf` — write the (T,B,R,C) cube to a NetCDF file
  - `zarr` — write the (T,B,R,C) cube to a Zarr store
- `import` …
  - `point` — build a point-centred STAC cube (needs the [stac] extra + a live endpoint)
  - `stac` — build a cube from a saved STAC ItemCollection / item-list JSON file

## Core verbs

Reached directly as `pyramids collection <verb>`.

#### `pyramids collection align`

align every timestep to a reference raster grid

```text
pyramids collection align [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] --reference REFERENCE [--resampling RESAMPLING] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination directory (one file per timestep) |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--reference` | **required** | reference raster whose grid to adopt |
| `--resampling` | optional | resampling method |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection crop`

crop every timestep to a mask or bbox

```text
pyramids collection crop [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] (--mask MASK | --bbox WEST SOUTH EAST NORTH) [--epsg EPSG] [--no-touch] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination directory (one file per timestep) |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--mask` | optional | raster whose footprint crops the stack |
| `--bbox` | optional | crop to this bounding box instead of a mask |
| `--epsg` | optional | CRS of --bbox (default: the stack's own CRS) |
| `--no-touch` | optional | keep only cells whose centre is inside, not every touched cell |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection info`

summarize a raster stack

```text
pyramids collection info [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--json] glob
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--json` | optional | emit JSON |

#### `pyramids collection merge`

mosaic every timestep into one raster

```text
pyramids collection merge [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--method {first,last,min,max,sum,count,mean}] [--nodata NODATA] [--bbox WEST SOUTH EAST NORTH] [--bbox-crs BBOX_CRS] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--method` | optional | overlap-resolution rule (default: last) (choices: `first`, `last`, `min`, `max`, `sum`, `count`, `mean`) |
| `--nodata` | optional | no-data value for the mosaic (default: inherit from the timesteps) |
| `--bbox` | optional | restrict the mosaic to this window |
| `--bbox-crs` | optional | CRS of --bbox (default: the mosaic's own CRS) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection reduce`

write a time reduction of a stack

```text
pyramids collection reduce [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] --reduce {mean,sum,min,max,std,var} [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--reduce` | **required** | time-reduction function (choices: `mean`, `sum`, `min`, `max`, `std`, `var`) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection reproject`

reproject every timestep to a target CRS

```text
pyramids collection reproject [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] --crs CRS [--resampling RESAMPLING] [--maintain-alignment] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination directory (one file per timestep) |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--crs` | **required** | target CRS (EPSG code, WKT, PROJ4) |
| `--resampling` | optional | resampling method |
| `--maintain-alignment` | optional | keep the source grid alignment through the reprojection |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection resample`

calendar-frequency time reduction (one raster per window)

```text
pyramids collection resample [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] --freq FREQ [--op {mean,sum,min,max,std,var,median,quantile,percentile,mode,trend,sens_slope}] [--q Q] [--no-skipna] [--times TIMES] [--pattern PATTERN] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination directory (one raster per window) |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--freq` | **required** | pandas offset alias, e.g. 1MS, 7D, 6h |
| `--op` | optional | reduction operation (choices: `mean`, `sum`, `min`, `max`, `std`, `var`, `median`, `quantile`, `percentile`, `mode`, `trend`, `sens_slope`) |
| `--q` | optional | level for --op percentile (0<q<=100) / quantile (0<q<=1) |
| `--no-skipna` | optional | propagate NaN instead of ignoring it within each window |
| `--times` | optional | comma-separated timestamps, one per timestep (default: the stack's own time axis) |
| `--pattern` | optional | per-window filename template with an {i} index placeholder |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `collection export`

Write the data out to a file or format.

#### `pyramids collection export cog`

write each timestep as its own COG

```text
pyramids collection export cog [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--pattern PATTERN] [--name NAME] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination directory (one COG per timestep) |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--pattern` | optional | per-slice filename template with {name} and {i} placeholders |
| `--name` | optional | replacement for the {name} placeholder |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection export file`

write every timestep to a directory

```text
pyramids collection export file [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--driver DRIVER] [--band BAND] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination directory (one file per timestep) |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--driver` | optional | output driver / catalog key (default: geotiff) |
| `--band` | optional | band to write for single-band drivers (ignored by GeoTIFF) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection export kerchunk`

write a kerchunk reference JSON (NetCDF/HDF5 sources)

```text
pyramids collection export kerchunk [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--concat-dim CONCAT_DIM] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination kerchunk reference JSON path |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--concat-dim` | optional | name of the concatenation dimension (default: time) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection export netcdf`

write the (T,B,R,C) cube to a NetCDF file

```text
pyramids collection export netcdf [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--time-dim TIME_DIM] [--no-var-per-band] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--time-dim` | optional | name of the time dimension |
| `--no-var-per-band` | optional | write one variable for all bands instead of one variable per band |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection export zarr`

write the (T,B,R,C) cube to a Zarr store

```text
pyramids collection export zarr [-h] [--source {files,zarr,archive}] [--member-glob MEMBER_GLOB] [--date-format DATE_FORMAT] [--mode MODE] [--overwrite] glob output
```

| argument | | description |
|---|---|---|
| `glob` | required | glob (quote it) or directory of co-registered rasters; with --source zarr it is a Zarr store path, and with --source archive a zip/tar path |
| `output` | required | destination Zarr store path |

| option | | description |
|---|---|---|
| `--source` | optional | how to open the stack: 'files' (default, glob/dir), 'zarr' store, or 'archive' (zip/tar) (choices: `files`, `zarr`, `archive`) |
| `--member-glob` | optional | with --source archive: glob selecting members inside the archive |
| `--date-format` | optional | with --source files: strptime format to parse each file's timestamp from its name (e.g. %%Y.%%m.%%d), giving the stack a time axis |
| `--mode` | optional | Zarr write mode (default: w) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `collection import`

Read / construct the data from another source.

#### `pyramids collection import point`

build a point-centred STAC cube (needs the [stac] extra + a live endpoint)

```text
pyramids collection import point [-h] --lat LAT --lon LON --collection COLLECTION --bands BANDS --start START --end END --edge-size EDGE_SIZE --resolution RESOLUTION [--units {px,m}] [--stac STAC] [--query QUERY] [--no-align] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination NetCDF cube path |

| option | | description |
|---|---|---|
| `--lat` | **required** | centre latitude (EPSG:4326) |
| `--lon` | **required** | centre longitude (EPSG:4326) |
| `--collection` | **required** | STAC collection id to search |
| `--bands` | **required** | asset key, or comma-separated keys for a band axis |
| `--start` | **required** | search start (YYYY-MM-DD / RFC 3339) |
| `--end` | **required** | search end (YYYY-MM-DD / RFC 3339) |
| `--edge-size` | **required** | cube side length, in pixels (--units px) or metres (--units m) |
| `--resolution` | **required** | pixel size in metres |
| `--units` | optional | unit of --edge-size (default: px) (choices: `px`, `m`) |
| `--stac` | optional | STAC API root URL. The default (Planetary Computer) needs a request signer, which is Python-only, so point this at an unauthenticated STAC root to use it from the CLI |
| `--query` | optional | STAC query-extension filter, as a JSON string |
| `--no-align` | optional | skip the multi-asset resolution alignment |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids collection import stac`

build a cube from a saved STAC ItemCollection / item-list JSON file

```text
pyramids collection import stac [-h] --asset ASSET [--bbox MINX MINY MAXX MAXY] [--max-items MAX_ITEMS] [--groupby GROUPBY] [--overwrite] items output
```

| argument | | description |
|---|---|---|
| `items` | required | path to a STAC ItemCollection / item-list JSON file |
| `output` | required | destination NetCDF cube path |

| option | | description |
|---|---|---|
| `--asset` | **required** | asset key, or comma-separated keys for a band axis |
| `--bbox` | optional | lon/lat (EPSG:4326) item filter: drop items whose footprint misses it |
| `--max-items` | optional | cap the number of items consumed (after --bbox filtering) |
| `--groupby` | optional | collapse items into timesteps: solar_day, id, time, or an item property key |
| `--overwrite` | optional | replace the output if it already exists |
