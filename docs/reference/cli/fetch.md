# fetch — OGC web-service access (WMS / WCS / WFS)

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids fetch --help` for the live
listing and `pyramids fetch <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?*
suggestion.

## Command tree

- `ogc-coverages` — read an OGC API Coverages subset to a raster
- `wcs` — pull a WCS coverage window to a raster
- `wfs` — pull WFS features to a vector file
- `wms` — render a WMS map window to a raster
- `wmts` — crop a WMTS tile-pyramid window to a raster

## Core verbs

Reached directly as `pyramids fetch <verb>`.

#### `pyramids fetch ogc-coverages`

read an OGC API Coverages subset to a raster

```text
pyramids fetch ogc-coverages [-h] --coverage COVERAGE --bbox MINX MINY MAXX MAXY [--resolution RESOLUTION] [--coverage-crs COVERAGE_CRS] [--output-crs OUTPUT_CRS] [--resample RESAMPLE] [--overwrite] endpoint output
```

| argument | | description |
|---|---|---|
| `endpoint` | required | OGC API landing-page / base URL |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--coverage` | **required** | coverage identifier |
| `--bbox` | **required** | request window (lon/lat unless --crs says otherwise) |
| `--resolution` | optional | approximate pixel size in the coverage's native CRS units |
| `--coverage-crs` | optional | coverage CRS when the service omits one |
| `--output-crs` | optional | reproject the result to this CRS |
| `--resample` | optional | resampling method (default: nearest) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids fetch wcs`

pull a WCS coverage window to a raster

```text
pyramids fetch wcs [-h] --coverage COVERAGE --bbox MINX MINY MAXX MAXY [--crs CRS] [--overwrite] url output
```

| argument | | description |
|---|---|---|
| `url` | required | WCS endpoint URL |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--coverage` | **required** | coverage identifier |
| `--bbox` | **required** | request window (lon/lat unless --crs says otherwise) |
| `--crs` | optional | bbox CRS (default: the service default) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids fetch wfs`

pull WFS features to a vector file

```text
pyramids fetch wfs [-h] --typename TYPENAME [--bbox MINX MINY MAXX MAXY] [--driver DRIVER] [--overwrite] url output
```

| argument | | description |
|---|---|---|
| `url` | required | WFS endpoint URL |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--typename` | **required** | feature type name |
| `--bbox` | optional | request window (lon/lat unless --crs says otherwise) |
| `--driver` | optional | OGR driver (default: geojson) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids fetch wms`

render a WMS map window to a raster

```text
pyramids fetch wms [-h] --layers LAYERS --bbox MINX MINY MAXX MAXY [--crs CRS] (--size W H | --resolution RESOLUTION) [--overwrite] url output
```

| argument | | description |
|---|---|---|
| `url` | required | WMS endpoint URL |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--layers` | **required** | WMS layer name |
| `--bbox` | **required** | request window (lon/lat unless --crs says otherwise) |
| `--crs` | optional | request CRS (default: the service default) |
| `--size` | optional | output pixel size |
| `--resolution` | optional | output cell size in CRS units |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids fetch wmts`

crop a WMTS tile-pyramid window to a raster

```text
pyramids fetch wmts [-h] --layer LAYER --bbox MINX MINY MAXX MAXY [--crs CRS] [--tile-matrix-set TILE_MATRIX_SET] [--resolution RESOLUTION] [--layer-crs LAYER_CRS] [--output-crs OUTPUT_CRS] [--resample RESAMPLE] [--overwrite] endpoint output
```

| argument | | description |
|---|---|---|
| `endpoint` | required | WMTS GetCapabilities URL |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--layer` | **required** | WMTS layer identifier |
| `--bbox` | **required** | request window (lon/lat unless --crs says otherwise) |
| `--crs` | optional | CRS of --bbox (default: EPSG:4326) |
| `--tile-matrix-set` | optional | tile-matrix-set id to pin |
| `--resolution` | optional | output pixel size in the layer's native CRS units |
| `--layer-crs` | optional | layer CRS when the service omits one |
| `--output-crs` | optional | reproject the result to this CRS |
| `--resample` | optional | resampling method (default: nearest) |
| `--overwrite` | optional | replace the output if it exists |

