# raster — single-raster (`Dataset`) operations

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids raster --help` for the live
listing and `pyramids raster <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?*
suggestion.

## Command tree

- `add-band` — append a band read from another raster
- `align` — resample a raster onto a reference grid
- `animate` — save a multi-band animation (GIF/MP4)
- `bounds` — print the raster bounding box
- `calc` — evaluate a band expression into a new raster
- `cell-area` — per-cell ground area summary
- `cluster` — label connected cells within a value range
- `combine` — combine two aligned rasters cell by cell
- `contour` — trace iso-value contours into a vector file
- `convert` — re-save a raster in another format
- `convert-units` — transform band values to another unit
- `count-domain-cells` — number of a band's valid (non-no-data) cells
- `create` — allocate a new raster, optionally no-data filled
- `create-empty` — allocate a header-only raster (out-of-core sibling)
- `crop` — crop a raster by bbox or vector mask
- `dataset-like` — stamp a template's georeference onto a .npy array
- `domain-area` — ground area of a band's valid cells
- `edit-info` — edit a raster's CRS / nodata / tags in place
- `empty-like` — allocate a header-only raster aligned to a template
- `extract` — pull a raster's values (optionally at point locations)
- `fillna` — give every gap a constant value
- `footprint` — the covered extent of a band as polygons
- `georeference` — warp a raster from ground-control points
- `get-metadata` — a GDAL metadata domain's contents
- `histogram` — a band's value histogram
- `info` — print raster metadata
- `match-histogram` — match a band onto a reference's distribution
- `merge` — mosaic rasters into one file
- `orthorectify` — orthorectify a raster from its RPC sensor model
- `overlay` — group a raster's values by a class raster
- `overview` — build image pyramids in place
- `pan-sharpen` — fuse a multispectral raster with a finer pan band
- `percentile-stretch` — stretch a band between two percentiles into [0, 1]
- `plot-histogram` — save a band's value-distribution histogram (PNG)
- `plot-vector-field` — save a two-band (u, v) vector-field plot (PNG)
- `proximity` — distance-to-target raster
- `rasterize` — burn a vector into a new raster
- `recreate-overviews` — regenerate a raster's existing overviews in place
- `render` — save a quick-look PNG of a raster
- `sample` — read band values at points
- `select-bands` — subset bands into a new raster
- `set-color-ramp` — attach an interpolated colour table and save a copy
- `shapes` — vectorize a raster to a vector file (one feature per cell)
- `stats` — band statistics (min/max/mean/std)
- `subdataset` — extract one subdataset of a container to a raster
- `warp` — reproject a raster
- `where` — keep the cells a condition selects and mask the rest
- `zonal-stats` — statistics per polygon zone or raster category
- `advanced` …
  - `blocks` — list a raster's native block (tile) windows
  - `checksum` — per-band GDAL checksum for integrity verification
  - `create-mask-band` — add a mask band and save the result
  - `geolocate` — warp a raster from its geolocation arrays onto a grid
  - `get-attribute-table` — a band's raster attribute table
  - `get-band-by-color` — print the band index for a colour interpretation
  - `get-mask` — write one band's mask as a raster (0/255)
  - `mask-flags` — decode a band's GDAL mask flags
  - `point` — sample band value(s) at a map coordinate
  - `preview` — write a whole-image decimated thumbnail raster
  - `read-masks` — write per-band mask array(s) as a raster (0/255)
  - `read-overview-array` — read one overview level to a raster or --json stats
  - `read-part` — read a geographic window to a raster (overview-decimated)
  - `read-tile` — read a Web-Mercator XYZ tile to an EPSG:3857 raster
  - `same-grid` — whether two rasters share a grid (and CRS)
  - `set-attribute-table` — attach an attribute table to a raster band (edits the raster in place)
  - `set-rpcs` — attach RPC sensor-model metadata and save a copy
  - `transect` — sample a band along a line into a profile
  - `translate` — subset / rescale / retype a raster (gdal_translate-style)
  - `warped-view` — write an on-the-fly reprojected raster (optional grid/window)
  - `wrap-longitude` — shift a global grid from the 0..360 to the -180..180 frame
- `cog` …
  - `create` — write a raster as a COG
  - `info` — print structured COG metadata
  - `validate` — validate a COG
- `export` …
  - `bytes` — serialize a raster to raw bytes
  - `cog-bytes` — serialize a raster to Cloud Optimized GeoTIFF bytes
  - `dataframe` — write a per-cell table (CSV, or Parquet by extension)
  - `image` — write a colour-mapped RGB image of a band (PNG/JPEG)
  - `polygons` — dissolve connected equal-valued cells into polygons (vector file)
  - `raster` — write a raster (to_file alias) with band/tiling options
  - `stac-item` — write a STAC Item JSON document
  - `terrain-rgb` — encode a DEM to Mapbox/Terrarium terrain-RGB
  - `xyz` — write a lon/lat/value XYZ point table
  - `zarr` — write the raster to a Zarr store
- `import` …
  - `archive` — stack the raster members of a zip/tar/gzip archive
  - `band-files` — stack single-band rasters into a multiband raster
  - `bytes` — open a raster held as raw bytes
  - `dataframe` — rebuild a raster from a (band, y, x) CSV table
  - `point-arrays` — grid raw x/y/z columns of a CSV
  - `points` — interpolate a point vector layer to a grid
  - `zarr` — read a pyramids-written Zarr store

## Core verbs

Reached directly as `pyramids raster <verb>`.

#### `pyramids raster add-band`

append a band read from another raster

```text
pyramids raster add-band [-h] --source SOURCE [--source-band SOURCE_BAND] [--unit UNIT] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--source` | **required** | raster the new band's values are read from |
| `--source-band` | optional | 0-based band of --source to append (default 0) |
| `--unit` | optional | unit of the new band's values |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster align`

resample a raster onto a reference grid

```text
pyramids raster align [-h] [--resampling RESAMPLING] [--overwrite] input reference output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `reference` | required | reference raster whose grid/CRS to adopt |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--resampling` | optional | resampling method |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster animate`

save a multi-band animation (GIF/MP4)

```text
pyramids raster animate [-h] [--bands BANDS [BANDS ...]] [--interval INTERVAL] [--dpi DPI] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination animation path (e.g. out.gif) |

| option | | description |
|---|---|---|
| `--bands` | optional | zero-based band indices to animate, in order (default: all, >= 2) |
| `--interval` | optional | milliseconds between frames |
| `--dpi` | optional | output DPI (default: 100) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster bounds`

print the raster bounding box

```text
pyramids raster bounds [-h] [--crs CRS] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--crs` | optional | reproject the corners to this CRS (corner-based approximation) |
| `--json` | optional | emit JSON |

#### `pyramids raster calc`

evaluate a band expression into a new raster

```text
pyramids raster calc [-h] [--dtype DTYPE] [--overwrite] expr operands [operands ...]
```

| argument | | description |
|---|---|---|
| `expr` | required | expression over inputs A, B, ... e.g. '(A - B) / (A + B)' |
| `operands` | required | one or more input rasters followed by the output path |

| option | | description |
|---|---|---|
| `--dtype` | optional | output numpy dtype (e.g. float32) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster cell-area`

per-cell ground area summary

```text
pyramids raster cell-area [-h] [--unit {m2,km2,ha}] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--unit` | optional | area unit (default m2) (choices: `m2`, `km2`, `ha`) |
| `--json` | optional | emit JSON |

#### `pyramids raster cluster`

label connected cells within a value range

```text
pyramids raster cluster [-h] --lower LOWER --upper UPPER [--json] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination label raster path |

| option | | description |
|---|---|---|
| `--lower` | **required** | lower bound of the value range |
| `--upper` | **required** | upper bound of the value range |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster combine`

combine two aligned rasters cell by cell

```text
pyramids raster combine [-h] --op {add,divide,maximum,minimum,multiply,subtract} [--band BAND] [--overwrite] a b output
```

| argument | | description |
|---|---|---|
| `a` | required | first raster |
| `b` | required | second raster (must share a's grid) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--op` | **required** | element-wise binary operation applied to the matching cells (choices: `add`, `divide`, `maximum`, `minimum`, `multiply`, `subtract`) |
| `--band` | optional | combine only this band (default: every band) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster contour`

trace iso-value contours into a vector file

```text
pyramids raster contour [-h] (--interval INTERVAL | --fixed-levels FIXED_LEVELS [FIXED_LEVELS ...]) [--base BASE] [--band BAND] [--attribute ATTRIBUTE] [--polygonize] [--driver DRIVER] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--interval` | optional | spacing between contour levels (anchored at --base) |
| `--fixed-levels` | optional | explicit levels to contour |
| `--base` | optional | level the interval is anchored to (default: 0) |
| `--band` | optional | zero-based band index (default: 0) |
| `--attribute` | optional | elevation attribute name (default: elev) |
| `--polygonize` | optional | emit filled polygons between levels instead of lines |
| `--driver` | optional | OGR vector driver (default: inferred from the output extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster convert`

re-save a raster in another format

```text
pyramids raster convert [-h] [--driver DRIVER] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--driver` | optional | catalog (geotiff) or GDAL (GTiff) driver name (default: from extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster convert-units`

transform band values to another unit

```text
pyramids raster convert-units [-h] --target TARGET [--band BAND] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--target` | **required** | target unit label (e.g. celsius, hPa, knots) |
| `--band` | optional | band to convert (default: every band) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster count-domain-cells`

number of a band's valid (non-no-data) cells

```text
pyramids raster count-domain-cells [-h] [--band BAND] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--json` | optional | emit JSON |

#### `pyramids raster create`

allocate a new raster, optionally no-data filled

```text
pyramids raster create [-h] --rows ROWS --columns COLUMNS --dtype DTYPE --bands BANDS [--geo X DX RX Y RY DY] [--top-left X Y] [--cell-size CELL_SIZE] [--epsg EPSG] [--no-data-value NO_DATA_VALUE] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--rows` | **required** | number of rows |
| `--columns` | **required** | number of columns |
| `--dtype` | **required** | numpy dtype name (e.g. float32) |
| `--bands` | **required** | number of bands |
| `--geo` | optional | affine geotransform (x_min, pixel_width, row_rot, y_max, col_rot, pixel_height) |
| `--top-left` | optional | top-left corner (used with --cell-size to build a north-up grid) |
| `--cell-size` | optional | square pixel size |
| `--epsg` | optional | EPSG code of the output CRS |
| `--no-data-value` | optional | no-data value to stamp and fill |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster create-empty`

allocate a header-only raster (out-of-core sibling)

```text
pyramids raster create-empty [-h] --rows ROWS --cols COLS [--bands BANDS] [--dtype DTYPE] [--geo X DX RX Y RY DY] [--top-left X Y] [--cell-size CELL_SIZE] [--epsg EPSG] [--no-data-value NO_DATA_VALUE] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--rows` | **required** | number of rows |
| `--cols` | **required** | number of columns |
| `--bands` | optional | number of bands (default: 1) |
| `--dtype` | optional | numpy dtype name (default: float32) |
| `--geo` | optional | affine geotransform (x_min, pixel_width, row_rot, y_max, col_rot, pixel_height) |
| `--top-left` | optional | top-left corner (used with --cell-size to build a north-up grid) |
| `--cell-size` | optional | square pixel size |
| `--epsg` | optional | EPSG code of the output CRS |
| `--no-data-value` | optional | no-data sentinel (default: -9999) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster crop`

crop a raster by bbox or vector mask

```text
pyramids raster crop [-h] (--bbox MINX MINY MAXX MAXY | --vector VECTOR) [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--bbox` | optional | clip extent in the raster CRS |
| `--vector` | optional | vector file whose polygons clip the raster |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster dataset-like`

stamp a template's georeference onto a .npy array

```text
pyramids raster dataset-like [-h] [--overwrite] template array output
```

| argument | | description |
|---|---|---|
| `template` | required | template raster whose georeference the output copies |
| `array` | required | NumPy .npy array to store (must match the template's shape) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster domain-area`

ground area of a band's valid cells

```text
pyramids raster domain-area [-h] [--band BAND] [--unit {m2,km2,ha}] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--unit` | optional | area unit (default m2) (choices: `m2`, `km2`, `ha`) |
| `--json` | optional | emit JSON |

#### `pyramids raster edit-info`

edit a raster's CRS / nodata / tags in place

```text
pyramids raster edit-info [-h] [--crs CRS] [--nodata NODATA] [--tag KEY=VALUE] input
```

| argument | | description |
|---|---|---|
| `input` | required | raster path to edit in place |

| option | | description |
|---|---|---|
| `--crs` | optional | set the CRS (EPSG code, WKT, PROJ4) |
| `--nodata` | optional | set the no-data value; 'none'/'unset' removes it, 'nan' sets NaN |
| `--tag` | optional | set a metadata tag; repeat for each tag |

#### `pyramids raster empty-like`

allocate a header-only raster aligned to a template

```text
pyramids raster empty-like [-h] [--dtype DTYPE] [--bands BANDS] [--no-data-value NO_DATA_VALUE] [--overwrite] template output
```

| argument | | description |
|---|---|---|
| `template` | required | template raster whose footprint the output copies |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--dtype` | optional | numpy dtype name (default: the template's) |
| `--bands` | optional | number of bands (default: the template's) |
| `--no-data-value` | optional | no-data sentinel (default: inherit) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster extract`

pull a raster's values (optionally at point locations)

```text
pyramids raster extract [-h] [--band BAND] [--exclude-value EXCLUDE_VALUE] [--mask MASK] [--json] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination CSV path (one row per band) |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band index |
| `--exclude-value` | optional | value to drop from the extracted values (in physical units) |
| `--mask` | optional | point vector at which to read values instead of every cell |
| `--json` | optional | emit JSON instead of text |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster fillna`

give every gap a constant value

```text
pyramids raster fillna [-h] --value VALUE [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--value` | **required** | value written into every gap |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster footprint`

the covered extent of a band as polygons

```text
pyramids raster footprint [-h] [--band BAND] [--exclude-values EXCLUDE_VALUES [EXCLUDE_VALUES ...]] [--max-samples MAX_SAMPLES] [--driver DRIVER] [--overwrite] [--json] file [output]
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |
| `output` | optional | destination vector path (omit to print GeoJSON) |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--exclude-values` | optional | values treated as uncovered in addition to the band's no-data sentinel |
| `--max-samples` | optional | cap pixels read to build the mask (yields an approximate footprint) |
| `--driver` | optional | OGR vector driver for --output (default: from extension) |
| `--overwrite` | optional | replace the output if it already exists |
| `--json` | optional | emit GeoJSON |

#### `pyramids raster georeference`

warp a raster from ground-control points

```text
pyramids raster georeference [-h] --gcp PIXEL LINE X Y --gcp-crs GCP_CRS [--transform {polynomial,tps}] [--order ORDER] [--to-crs TO_CRS] [--resampling RESAMPLING] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--gcp` | **required** | a ground-control point 'PIXEL LINE X Y'; repeat for each point |
| `--gcp-crs` | **required** | CRS of the GCP map coordinates |
| `--transform` | optional | transform fitted through the GCPs (default: polynomial) (choices: `polynomial`, `tps`) |
| `--order` | optional | polynomial order 1-3 (default: 1) |
| `--to-crs` | optional | reproject the result to this CRS |
| `--resampling` | optional | resampling method |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster get-metadata`

a GDAL metadata domain's contents

```text
pyramids raster get-metadata [-h] [--domain DOMAIN] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--domain` | optional | GDAL metadata domain (default: the default domain) |
| `--json` | optional | emit JSON |

#### `pyramids raster histogram`

a band's value histogram

```text
pyramids raster histogram [-h] [--band BAND] [--bins BINS] [--min MIN_VALUE] [--max MAX_VALUE] [--include-out-of-range] [--approx] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--bins` | optional | number of buckets (default 6) |
| `--min` | optional | low edge, physical units |
| `--max` | optional | high edge, physical units |
| `--include-out-of-range` | optional | fold out-of-range values into the first and last buckets |
| `--approx` | optional | approximate from a subsample / overviews |
| `--json` | optional | emit JSON |

#### `pyramids raster info`

print raster metadata

```text
pyramids raster info [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids raster match-histogram`

match a band onto a reference's distribution

```text
pyramids raster match-histogram [-h] [--band BAND] [--reference-band REFERENCE_BAND] [--bins BINS] [--overwrite] input reference output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `reference` | required | reference raster whose distribution to match |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band index (default: 0) |
| `--reference-band` | optional | zero-based band of the reference (default: same as --band) |
| `--bins` | optional | quantile levels sampled (default: 256) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster merge`

mosaic rasters into one file

```text
pyramids raster merge [-h] [--bbox MINX MINY MAXX MAXY] [--bbox-crs BBOX_CRS] [--overwrite] inputs [inputs ...] output
```

| argument | | description |
|---|---|---|
| `inputs` | required | source raster paths (two or more) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--bbox` | optional | restrict the merge to this window, so only it is read |
| `--bbox-crs` | optional | CRS of --bbox (default: the mosaic's own CRS) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster orthorectify`

orthorectify a raster from its RPC sensor model

```text
pyramids raster orthorectify [-h] [--dem DEM] [--rpc-height RPC_HEIGHT] [--to-crs TO_CRS] [--resampling RESAMPLING] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--dem` | optional | elevation model raster path |
| `--rpc-height` | optional | constant elevation (map units) to use when no --dem is given |
| `--to-crs` | optional | reproject the result to this CRS |
| `--resampling` | optional | resampling method |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster overlay`

group a raster's values by a class raster

```text
pyramids raster overlay [-h] [--band BAND] [--exclude-value EXCLUDE_VALUE] [--json] input classes
```

| argument | | description |
|---|---|---|
| `input` | required | base raster whose values to group |
| `classes` | required | aligned class raster |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band index |
| `--exclude-value` | optional | value to drop from the grouped output (in physical units) |
| `--json` | optional | emit JSON instead of text |

#### `pyramids raster overview`

build image pyramids in place

```text
pyramids raster overview [-h] [--resampling RESAMPLING] [--levels LEVELS [LEVELS ...]] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to build overviews for |

| option | | description |
|---|---|---|
| `--resampling` | optional | overview resampling: nearest (alias 'nearest neighbor'), average, gauss, cubic, mode, ... |
| `--levels` | optional | power-of-two decimation levels (2, 4, 8, ... 2048); e.g. 2 4 8 |

#### `pyramids raster pan-sharpen`

fuse a multispectral raster with a finer pan band

```text
pyramids raster pan-sharpen [-h] [--algorithm ALGORITHM] [--weights WEIGHTS [WEIGHTS ...]] [--resampling RESAMPLING] [--bands BANDS [BANDS ...]] [--nodata NODATA] [--overwrite] multispectral pan output
```

| argument | | description |
|---|---|---|
| `multispectral` | required | coarse multispectral raster path |
| `pan` | required | fine single-band panchromatic raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--algorithm` | optional | pan-sharpening algorithm (default: brovey) |
| `--weights` | optional | per-band Brovey weights (one per selected band) |
| `--resampling` | optional | spectral upsampling: nearest, bilinear, cubic, cubicspline, lanczos, average |
| `--bands` | optional | zero-based multispectral bands to sharpen (default: all) |
| `--nodata` | optional | no-data value set on the fused output |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster percentile-stretch`

stretch a band between two percentiles into [0, 1]

```text
pyramids raster percentile-stretch [-h] [--lower LOWER] [--upper UPPER] [--band BAND] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--lower` | optional | lower percentile mapped to 0 (default: 2) |
| `--upper` | optional | upper percentile mapped to 1 (default: 98) |
| `--band` | optional | zero-based band index (default: 0) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster plot-histogram`

save a band's value-distribution histogram (PNG)

```text
pyramids raster plot-histogram [-h] [--band BAND] [--bins BINS] [--exclude-value EXCLUDE_VALUE] [--max-samples MAX_SAMPLES] [--dpi DPI] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination image path (e.g. hist.png) |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band (default 0) |
| `--bins` | optional | number of bins (default 15) |
| `--exclude-value` | optional | an extra value to drop before binning, besides no-data |
| `--max-samples` | optional | cap on pixels read (decimates a large band; default: read all) |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster plot-vector-field`

save a two-band (u, v) vector-field plot (PNG)

```text
pyramids raster plot-vector-field [-h] [--u-band U_BAND] [--v-band V_BAND] [--kind {quiver,barbs,streamplot}] [--dpi DPI] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination image path (e.g. field.png) |

| option | | description |
|---|---|---|
| `--u-band` | optional | x-component band index (default 0) |
| `--v-band` | optional | y-component band index (default 1) |
| `--kind` | optional | render kind (default: quiver) (choices: `quiver`, `barbs`, `streamplot`) |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster proximity`

distance-to-target raster

```text
pyramids raster proximity [-h] [--band BAND] [--target-values TARGET_VALUES [TARGET_VALUES ...]] [--distance-units {GEO,PIXEL}] [--max-distance MAX_DISTANCE] [--nodata NODATA] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band index (default: 0) |
| `--target-values` | optional | pixel values that count as targets (default: every non-zero pixel) |
| `--distance-units` | optional | measure distance in georeferenced units (GEO) or pixels (PIXEL) (choices: `GEO`, `PIXEL`) |
| `--max-distance` | optional | stop searching beyond this distance |
| `--nodata` | optional | no-data value, also filling pixels beyond --max-distance |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster rasterize`

burn a vector into a new raster

```text
pyramids raster rasterize [-h] [--cell-size CELL_SIZE] [--like LIKE] [--column COLUMN] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source vector path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--cell-size` | optional | output cell size (required unless --like; ignored when --like is given) |
| `--like` | optional | template raster whose grid/CRS the output adopts |
| `--column` | optional | attribute column to burn (default: all non-geometry columns) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster recreate-overviews`

regenerate a raster's existing overviews in place

```text
pyramids raster recreate-overviews [-h] [--resampling RESAMPLING] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path whose overviews to regenerate |

| option | | description |
|---|---|---|
| `--resampling` | optional | overview resampling (nearest, average, gauss, cubic, mode, ...) |

#### `pyramids raster render`

save a quick-look PNG of a raster

```text
pyramids raster render [-h] [--band BAND] [--title TITLE] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |
| `output` | required | destination image path (e.g. out.png) |

| option | | description |
|---|---|---|
| `--band` | optional | band index to render (default: auto) |
| `--title` | optional | plot title |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster sample`

read band values at points

```text
pyramids raster sample [-h] --points POINTS [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to sample |

| option | | description |
|---|---|---|
| `--points` | **required** | semicolon-separated 'x,y' pairs |
| `--json` | optional | emit JSON |

#### `pyramids raster select-bands`

subset bands into a new raster

```text
pyramids raster select-bands [-h] --bands BANDS [BANDS ...] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--bands` | **required** | 1-based band indices to keep, in order (duplicates allowed) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster set-color-ramp`

attach an interpolated colour table and save a copy

```text
pyramids raster set-color-ramp [-h] [--band BAND] --start-value START_VALUE --end-value END_VALUE [--start-color START_COLOR] [--end-color END_COLOR] [--colormap COLORMAP] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--band` | optional | 1-based band to colour (default 1) |
| `--start-value` | **required** | first value in the ramp (>= 0) |
| `--end-value` | **required** | last value in the ramp (must exceed --start-value) |
| `--start-color` | optional | hex colour at --start-value |
| `--end-color` | optional | hex colour at --end-value |
| `--colormap` | optional | named matplotlib colormap sampled across the range (instead of --start-color/--end-color) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster shapes`

vectorize a raster to a vector file (one feature per cell)

```text
pyramids raster shapes [-h] [--geometry {polygon,point}] [--driver DRIVER] [--allow-large] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--geometry` | optional | per-cell geometry to emit (default: polygon) (choices: `polygon`, `point`) |
| `--driver` | optional | OGR vector driver (default: geojson) |
| `--allow-large` | optional | proceed even when the raster has more than ~4M cells (one feature per cell can exhaust memory) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster stats`

band statistics (min/max/mean/std)

```text
pyramids raster stats [-h] [--band BAND] [--exact] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default: all bands) |
| `--exact` | optional | scan every pixel instead of letting GDAL approximate from overviews |
| `--json` | optional | emit JSON |

#### `pyramids raster subdataset`

extract one subdataset of a container to a raster

```text
pyramids raster subdataset [-h] [--overwrite] input key output
```

| argument | | description |
|---|---|---|
| `input` | required | container raster path (NetCDF/HDF, …) |
| `key` | required | 0-based subdataset index (negatives allowed) or its full name |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster warp`

reproject a raster

```text
pyramids raster warp [-h] --crs CRS [--resampling RESAMPLING] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--crs` | **required** | target CRS (EPSG code, WKT, PROJ4) |
| `--resampling` | optional | resampling method |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster where`

keep the cells a condition selects and mask the rest

```text
pyramids raster where [-h] --cond COND [--other OTHER] [--drop] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--cond` | **required** | condition over the input bound to A, e.g. 'A > 5' |
| `--other` | optional | value for unselected cells: a number, 'nan', or 'none' (default: the raster's no-data value) |
| `--drop` | optional | trim the result to the rows/columns/bands the condition selected |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster zonal-stats`

statistics per polygon zone or raster category

```text
pyramids raster zonal-stats [-h] [--classes CLASSES] [--stats STATS [STATS ...]] [--band BAND] [--output OUTPUT] [--json] [--overwrite] raster [zones]
```

| argument | | description |
|---|---|---|
| `raster` | required | source raster path |
| `zones` | optional | polygon vector file defining the zones |

| option | | description |
|---|---|---|
| `--classes` | optional | categorical raster whose distinct values define the zones (an alternative to the vector 'zones') |
| `--stats` | optional | stats to compute: mean sum min max std var count (default: mean) |
| `--band` | optional | zero-based band index (default: 0) |
| `--output` | optional | write the table to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `raster advanced`

Power-user **primitives** — low-level / rarely-one-shot operations (array, coord & dimension mechanics, internal
reads, geometry micro-ops). Kept here, rather than at the domain's top level, so the common task verbs stay uncluttered;
the capability is unchanged, only the path is longer.

#### `pyramids raster advanced blocks`

list a raster's native block (tile) windows

```text
pyramids raster advanced blocks [-h] [--band BAND] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band whose native blocking to report (default 0) |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced checksum`

per-band GDAL checksum for integrity verification

```text
pyramids raster advanced checksum [-h] [--band BAND] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default: all bands) |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced create-mask-band`

add a mask band and save the result

```text
pyramids raster advanced create-mask-band [-h] [--per-band] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--per-band` | optional | create a per-band mask (default: one mask shared by every band) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced geolocate`

warp a raster from its geolocation arrays onto a grid

```text
pyramids raster advanced geolocate [-h] [--to-crs TO_CRS] [--resampling RESAMPLING] [--cell-size CELL_SIZE] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--to-crs` | optional | reproject the result to this CRS |
| `--resampling` | optional | resampling method |
| `--cell-size` | optional | output pixel size in target-CRS units |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced get-attribute-table`

a band's raster attribute table

```text
pyramids raster advanced get-attribute-table [-h] [--band BAND] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced get-band-by-color`

print the band index for a colour interpretation

```text
pyramids raster advanced get-band-by-color [-h] --color COLOR [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--color` | **required** | colour interpretation (e.g. red, green, blue, alpha, gray_index) |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced get-mask`

write one band's mask as a raster (0/255)

```text
pyramids raster advanced get-mask [-h] [--band BAND] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination mask raster path |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced mask-flags`

decode a band's GDAL mask flags

```text
pyramids raster advanced mask-flags [-h] [--band BAND] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default 0) |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced point`

sample band value(s) at a map coordinate

```text
pyramids raster advanced point [-h] [--point-crs POINT_CRS] [--band BAND] [--json] input x y
```

| argument | | description |
|---|---|---|
| `input` | required | raster path to sample |
| `x` | required | X / longitude / easting in --point-crs |
| `y` | required | Y / latitude / northing in --point-crs |

| option | | description |
|---|---|---|
| `--point-crs` | optional | CRS of the coordinate (default: the raster's own CRS) |
| `--band` | optional | 0-based band index (default: all bands) |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced preview`

write a whole-image decimated thumbnail raster

```text
pyramids raster advanced preview [-h] [--max-size MAX_SIZE] [--resampling RESAMPLING] [--band BAND] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--max-size` | optional | maximum pixels on the longer edge (default: 1024) |
| `--resampling` | optional | resampling method |
| `--band` | optional | 0-based band index (default: all bands) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced read-masks`

write per-band mask array(s) as a raster (0/255)

```text
pyramids raster advanced read-masks [-h] [--band BAND] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination mask raster path |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index to read (default: every band, stacked) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced read-overview-array`

read one overview level to a raster or --json stats

```text
pyramids raster advanced read-overview-array [-h] [--band BAND] [--overview-index OVERVIEW_INDEX] [--json] [--overwrite] input [output]
```

| argument | | description |
|---|---|---|
| `input` | required | raster path to inspect |
| `output` | optional | destination raster for the decimated level (omit with --json) |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default: all bands) |
| `--overview-index` | optional | 0-based overview level to read (default: 0, the coarsest-but-one) |
| `--json` | optional | print shape/dtype/stats instead of writing |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced read-part`

read a geographic window to a raster (overview-decimated)

```text
pyramids raster advanced read-part [-h] --bbox MINX MINY MAXX MAXY [--dst-width DST_WIDTH] [--dst-height DST_HEIGHT] [--gsd GSD] [--bbox-crs BBOX_CRS] [--resampling RESAMPLING] [--band BAND] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--bbox` | **required** | window extent in --bbox-crs (default: the raster's own CRS) |
| `--dst-width` | optional | output width in pixels (decimates the read) |
| `--dst-height` | optional | output height in pixels (decimates the read) |
| `--gsd` | optional | target cell size in --bbox-crs units (instead of --dst-width/--dst-height) |
| `--bbox-crs` | optional | CRS of --bbox (default: the raster's CRS) |
| `--resampling` | optional | resampling method |
| `--band` | optional | 0-based band index (default: all bands) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced read-tile`

read a Web-Mercator XYZ tile to an EPSG:3857 raster

```text
pyramids raster advanced read-tile [-h] [--tilesize TILESIZE] [--resampling RESAMPLING] [--band BAND] [--overwrite] input output z x y
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |
| `z` | required | zoom level |
| `x` | required | tile column index |
| `y` | required | tile row index (origin north-west) |

| option | | description |
|---|---|---|
| `--tilesize` | optional | output tile size in pixels (default: 256) |
| `--resampling` | optional | resampling method |
| `--band` | optional | 0-based band index (default: all bands) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced same-grid`

whether two rasters share a grid (and CRS)

```text
pyramids raster advanced same-grid [-h] [--no-compare-crs] [--json] a b
```

| argument | | description |
|---|---|---|
| `a` | required | first raster path |
| `b` | required | second raster path |

| option | | description |
|---|---|---|
| `--no-compare-crs` | optional | compare only the grid (rows/cols/geotransform), not the CRS |
| `--json` | optional | emit JSON |

#### `pyramids raster advanced set-attribute-table`

attach an attribute table to a raster band (edits the raster in place)

```text
pyramids raster advanced set-attribute-table [-h] [--band BAND] input table
```

| argument | | description |
|---|---|---|
| `input` | required | raster path to edit in place |
| `table` | required | attribute-table source (.csv or .json) |

| option | | description |
|---|---|---|
| `--band` | optional | 0-based band index (default: band 0) |

#### `pyramids raster advanced set-rpcs`

attach RPC sensor-model metadata and save a copy

```text
pyramids raster advanced set-rpcs [-h] --rpc-json RPC_JSON [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--rpc-json` | **required** | path to a JSON object of RPC coefficients (offsets, scales, vectors) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced transect`

sample a band along a line into a profile

```text
pyramids raster advanced transect [-h] --line LINE (--n N | --step STEP) [--band BAND] [--driver DRIVER] [--json] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination profile path (.csv or a vector file) |

| option | | description |
|---|---|---|
| `--line` | **required** | semicolon-separated 'x,y' vertices in the raster CRS, e.g. '0,0;4,4' |
| `--n` | optional | number of positions along the line (endpoints included) |
| `--step` | optional | ground spacing between positions, in metres |
| `--band` | optional | zero-based band index |
| `--driver` | optional | OGR driver for a vector output (default: from extension) |
| `--json` | optional | emit JSON instead of text |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced translate`

subset / rescale / retype a raster (gdal_translate-style)

```text
pyramids raster advanced translate [-h] [--unscale] [--output-type OUTPUT_TYPE] [--nodata NODATA] [--src-win XOFF YOFF XSIZE YSIZE] [--proj-win ULX ULY LRX LRY] [--scale V [V ...]] [--bands N [N ...]] [--resampling RESAMPLING] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--unscale` | optional | bake scale/offset into the stored values and clear them |
| `--output-type` | optional | output numpy dtype (e.g. float32, int16, uint8) |
| `--nodata` | optional | set the output no-data value |
| `--src-win` | optional | pixel subwindow to extract |
| `--proj-win` | optional | projected-coordinate subwindow to extract |
| `--scale` | optional | rescale values: SRC_MIN SRC_MAX [DST_MIN DST_MAX] |
| `--bands` | optional | 1-based band numbers to keep / re-order |
| `--resampling` | optional | resampling for a resolution change |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced warped-view`

write an on-the-fly reprojected raster (optional grid/window)

```text
pyramids raster advanced warped-view [-h] --crs CRS [--resampling RESAMPLING] [--cell-size CELL_SIZE] [--bbox MINX MINY MAXX MAXY] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--crs` | **required** | target CRS (EPSG code, WKT, PROJ4) |
| `--resampling` | optional | resampling method |
| `--cell-size` | optional | output cell size in target-CRS units (default: preserve resolution) |
| `--bbox` | optional | output extent in the target CRS |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster advanced wrap-longitude`

shift a global grid from the 0..360 to the -180..180 frame

```text
pyramids raster advanced wrap-longitude [-h] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `raster cog`

Cloud-Optimized GeoTIFF create / validate / inspect.

#### `pyramids raster cog create`

write a raster as a COG

```text
pyramids raster cog create [-h] [--profile {deflate,jpeg,lerc,lerc_deflate,lerc_zstd,lzw,packbits,raw,webp,zstd}] [--compress COMPRESS] [--blocksize BLOCKSIZE] [--no-validate] [--overwrite] [--json] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination COG path |

| option | | description |
|---|---|---|
| `--profile` | optional | named compression profile (choices: `deflate`, `jpeg`, `lerc`, `lerc_deflate`, `lerc_zstd`, `lzw`, `packbits`, `raw`, `webp`, `zstd`) |
| `--compress` | optional | compression method (e.g. DEFLATE, ZSTD); overrides --profile and skips its jpeg/webp dtype/band pre-check |
| `--blocksize` | optional | internal tile size |
| `--no-validate` | optional | skip post-write validation |
| `--overwrite` | optional | replace the output if it already exists |
| `--json` | optional | emit a JSON result instead of text |

#### `pyramids raster cog info`

print structured COG metadata

```text
pyramids raster cog info [-h] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to inspect |

#### `pyramids raster cog validate`

validate a COG

```text
pyramids raster cog validate [-h] [--strict] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | raster path to validate |

| option | | description |
|---|---|---|
| `--strict` | optional | treat warnings as errors |
| `--json` | optional | emit a JSON validation result |

## Subsystem `raster export`

Write the data out to a file or format.

#### `pyramids raster export bytes`

serialize a raster to raw bytes

```text
pyramids raster export bytes [-h] [--driver DRIVER] [--overwrite] input [output]
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | optional | destination file (default: write to stdout) |

| option | | description |
|---|---|---|
| `--driver` | optional | GDAL raster driver (default: GTiff) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export cog-bytes`

serialize a raster to Cloud Optimized GeoTIFF bytes

```text
pyramids raster export cog-bytes [-h] [--compression COMPRESSION] [--blocksize BLOCKSIZE] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination .tif path |

| option | | description |
|---|---|---|
| `--compression` | optional | compression method (e.g. DEFLATE, ZSTD) |
| `--blocksize` | optional | internal tile size |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export dataframe`

write a per-cell table (CSV, or Parquet by extension)

```text
pyramids raster export dataframe [-h] [--bands BANDS [BANDS ...]] [--dropna] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination .csv or .parquet path |

| option | | description |
|---|---|---|
| `--bands` | optional | zero-based band indices to include (default: all) |
| `--dropna` | optional | drop no-data (NaN) cells |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export image`

write a colour-mapped RGB image of a band (PNG/JPEG)

```text
pyramids raster export image [-h] [--band BAND] [--cmap CMAP] [--exclude-value EXCLUDE_VALUE] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination image path (e.g. out.png) |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band index (default 0) |
| `--cmap` | optional | matplotlib colormap (default: viridis) |
| `--exclude-value` | optional | an extra value to mask out, besides no-data |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export polygons`

dissolve connected equal-valued cells into polygons (vector file)

```text
pyramids raster export polygons [-h] [--band BAND] [--driver DRIVER] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--band` | optional | zero-based band index (default 0) |
| `--driver` | optional | OGR vector driver (default: inferred from the extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export raster`

write a raster (to_file alias) with band/tiling options

```text
pyramids raster export raster [-h] [--band BAND] [--tile-length TILE_LENGTH] [--driver DRIVER] [--creation-option KEY=VALUE] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--band` | optional | band index forwarded to to_file; selects the band only for ASCII grid drivers (other drivers write every band). Must be in range (default 0) |
| `--tile-length` | optional | tile edge for a windowed write |
| `--driver` | optional | catalog (geotiff) or GDAL (GTiff) driver name (default: from extension) |
| `--creation-option` | optional | GDAL creation option; repeat for each |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export stac-item`

write a STAC Item JSON document

```text
pyramids raster export stac-item [-h] --item-id ITEM_ID [--asset-href ASSET_HREF] [--datetime DATETIME] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination .json path |

| option | | description |
|---|---|---|
| `--item-id` | **required** | STAC item identifier |
| `--asset-href` | optional | asset href for the raster (default: the input path) |
| `--datetime` | optional | item datetime (ISO 8601; default: the item's own default) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export terrain-rgb`

encode a DEM to Mapbox/Terrarium terrain-RGB

```text
pyramids raster export terrain-rgb [-h] [--encoding {mapbox,terrarium}] [--band BAND] [--tiles] [--min-zoom MIN_ZOOM] [--max-zoom MAX_ZOOM] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source elevation raster path |
| `output` | required | destination raster (or tile-root directory with --tiles) |

| option | | description |
|---|---|---|
| `--encoding` | optional | RGB encoding (default: mapbox) (choices: `mapbox`, `terrarium`) |
| `--band` | optional | zero-based elevation band (default 0) |
| `--tiles` | optional | write an XYZ PNG pyramid instead of a single raster |
| `--min-zoom` | optional | lowest XYZ zoom (with --tiles) |
| `--max-zoom` | optional | highest XYZ zoom (with --tiles; default: from resolution) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export xyz`

write a lon/lat/value XYZ point table

```text
pyramids raster export xyz [-h] [--bands BANDS [BANDS ...]] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination .xyz path |

| option | | description |
|---|---|---|
| `--bands` | optional | zero-based band indices to include (default: all) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster export zarr`

write the raster to a Zarr store

```text
pyramids raster export zarr [-h] [--mode MODE] [--overwrite] input output
```

| argument | | description |
|---|---|---|
| `input` | required | source raster path |
| `output` | required | destination Zarr store path |

| option | | description |
|---|---|---|
| `--mode` | optional | Zarr open mode (default: w) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `raster import`

Read / construct the data from another source.

#### `pyramids raster import archive`

stack the raster members of a zip/tar/gzip archive

```text
pyramids raster import archive [-h] [--kind KIND] [--member-glob MEMBER_GLOB] [--band-names BAND_NAMES [BAND_NAMES ...]] [--align] [--no-data-value NO_DATA_VALUE] [--overwrite] archive output
```

| argument | | description |
|---|---|---|
| `archive` | required | archive path or URL (.zip / .tar / .tar.gz / .gz) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--kind` | optional | archive kind: zip, tar, gzip, or auto (default: auto) |
| `--member-glob` | optional | fnmatch pattern for members (default: *) |
| `--band-names` | optional | explicit band names, one per member |
| `--align` | optional | resample mismatched members onto the first member's grid |
| `--no-data-value` | optional | no-data sentinel for the output bands |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster import band-files`

stack single-band rasters into a multiband raster

```text
pyramids raster import band-files [-h] [--band-names BAND_NAMES [BAND_NAMES ...]] [--align] [--resampling RESAMPLING] [--no-data-value NO_DATA_VALUE] [--overwrite] files [files ...] output
```

| argument | | description |
|---|---|---|
| `files` | required | single-band raster paths (one per band, in order) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--band-names` | optional | explicit band names, one per file |
| `--align` | optional | resample mismatched inputs onto the first file's grid |
| `--resampling` | optional | alignment resampling method (requires --align) |
| `--no-data-value` | optional | no-data sentinel for the output bands |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster import bytes`

open a raster held as raw bytes

```text
pyramids raster import bytes [-h] [--stdin] [--suffix SUFFIX] [--name NAME] [--overwrite] [input] output
```

| argument | | description |
|---|---|---|
| `input` | optional | file of raw raster bytes (omit with --stdin) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--stdin` | optional | read the bytes from standard input |
| `--suffix` | optional | extension hint for GDAL driver detection (default: .tif) |
| `--name` | optional | cosmetic dataset label |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster import dataframe`

rebuild a raster from a (band, y, x) CSV table

```text
pyramids raster import dataframe [-h] --x X --y Y [--band BAND] [--value VALUE] [--crs CRS] [--no-data-value NO_DATA_VALUE] [--overwrite] csv output
```

| argument | | description |
|---|---|---|
| `csv` | required | source CSV path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--x` | **required** | column holding the x (column) coordinate |
| `--y` | **required** | column holding the y (row) coordinate |
| `--band` | optional | column holding the 0-based band axis (single band if omitted) |
| `--value` | optional | the value column (required if the CSV has extra columns) |
| `--crs` | optional | CRS for the result (EPSG code or CRS string) |
| `--no-data-value` | optional | no-data sentinel for gaps |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster import point-arrays`

grid raw x/y/z columns of a CSV

```text
pyramids raster import point-arrays [-h] --x X --y Y --z Z [--algorithm ALGORITHM] [--cell-size CELL_SIZE] [--width WIDTH] [--height HEIGHT] [--bbox MINX MINY MAXX MAXY] [--crs CRS] [--epsg EPSG] [--overwrite] csv output
```

| argument | | description |
|---|---|---|
| `csv` | required | source CSV path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--x` | **required** | column holding the x coordinates |
| `--y` | **required** | column holding the y coordinates |
| `--z` | **required** | column holding the values to interpolate |
| `--algorithm` | optional | gdal.Grid algorithm string (default: inverse-distance weighting) |
| `--cell-size` | optional | output pixel size |
| `--width` | optional | output width in pixels |
| `--height` | optional | output height in pixels |
| `--bbox` | optional | output extent MINX MINY MAXX MAXY |
| `--crs` | optional | CRS to stamp on the output (label only) |
| `--epsg` | optional | output EPSG code (takes precedence over --crs) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster import points`

interpolate a point vector layer to a grid

```text
pyramids raster import points [-h] --value-column VALUE_COLUMN [--algorithm ALGORITHM] [--cell-size CELL_SIZE] [--width WIDTH] [--height HEIGHT] [--bbox MINX MINY MAXX MAXY] [--epsg EPSG] [--overwrite] vector output
```

| argument | | description |
|---|---|---|
| `vector` | required | source point vector path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--value-column` | **required** | numeric attribute to interpolate |
| `--algorithm` | optional | gdal.Grid algorithm string (default: inverse-distance weighting) |
| `--cell-size` | optional | output pixel size |
| `--width` | optional | output width in pixels |
| `--height` | optional | output height in pixels |
| `--bbox` | optional | output extent MINX MINY MAXX MAXY |
| `--epsg` | optional | output EPSG code (default: the points' CRS) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids raster import zarr`

read a pyramids-written Zarr store

```text
pyramids raster import zarr [-h] [--level LEVEL] [--data-name DATA_NAME] [--storage-options STORAGE_OPTIONS] [--overwrite] store output
```

| argument | | description |
|---|---|---|
| `store` | required | input store (path / fsspec URL) |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--level` | optional | pyramid downsample factor to read (1 = full resolution) |
| `--data-name` | optional | explicit data-array name to read |
| `--storage-options` | optional | fsspec storage options as a JSON object |
| `--overwrite` | optional | replace the output if it already exists |
