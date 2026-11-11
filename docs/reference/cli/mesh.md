# mesh — UGRID / `MeshDataset` operations

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids mesh --help` for the live
listing and `pyramids mesh <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?* suggestion.

## Command tree

- `convert` — convert a mesh to a vector file
- `extract` — rasterize a mesh variable onto a grid
- `info` — print mesh metadata
- `advanced` …
  - `active-mask` — a variable's active-cell mask
  - `bfill` — back-fill gaps along a dimension
  - `coarsen` — block-reduce the time axis by a window
  - `collapse-layer` — collapse a 3-D volume to a 2-D face field
  - `connected-components` — label the mesh's connected components
  - `cumulative-integrate` — running integral along the time axis
  - `differentiate` — derivative of a variable along the time axis
  - `drop-duplicates` — keep the first of each repeated time stamp
  - `drop-isel` — drop time steps by position
  - `drop-sel` — drop time steps by coordinate value
  - `drop-variables` — drop whole data variables
  - `dropna` — drop time steps that carry missing values
  - `ffill` — forward-fill gaps along a dimension
  - `get` — describe a single data variable
  - `integrate` — integrate a variable over the time axis
  - `items` — describe every (name, variable) pair
  - `keys` — list the data-variable names
  - `masked-values` — a variable's no-data-masked values
  - `rename-variable` — rename a data variable
  - `rolling` — moving-window reduction along a dimension
  - `set-crs` — declare the mesh's CRS without reprojecting
  - `shift` — shift values along the time axis
  - `sparse-adjacency` — the mesh adjacency matrix (summary / .npz)
  - `squeeze` — drop a length-1 time dimension
  - `thin` — keep every N-th time step
  - `values` — describe every data variable
  - `with-variable` — add a data variable from a NumPy .npy array
- `analysis` …
  - `aspect` — per-face aspect (clockwise from north)
  - `binary-dilation` — grow a boolean face mask over its neighbours
  - `binary-erosion` — shrink a boolean face mask at its border
  - `cluster` — label connected faces whose values fall in a range
  - `contour` — trace iso-value contour lines to a vector
  - `count-domain-cells` — number of active mesh faces
  - `domain-area` — total area of the mesh domain
  - `focal-mean` — k-ring neighbourhood mean
  - `focal-std` — k-ring neighbourhood standard deviation
  - `footprint` — dissolved outline of the valid-value faces
  - `hillshade` — per-face shaded relief
  - `histogram` — the variable's value distribution as data
  - `polyfit` — per-element polynomial trend along time (writes coefficients)
  - `proximity` — topological hop distance to target faces
  - `sample` — element-resolved values at point locations
  - `sample-line` — one sample per face crossing along a line
  - `sample-point` — variable values at arbitrary x/y coordinates
  - `sample-random` — draw spatially random samples from the domain
  - `sieve` — drop value-connected groups below a size
  - `slope` — per-face slope magnitude
  - `zonal-stats` — per-zone statistics of a mesh variable
- `attrs` …
  - `del-attr` — delete a global attribute into a new mesh file
  - `set-attr` — set a global attribute into a new mesh file
- `export` …
  - `attach-data` — attach a results-only file's variables to the mesh
  - `burn` — burn a vector layer onto the mesh faces
  - `polygonize` — dissolve equal-valued faces into polygons (vector file)
  - `regrid` — regrid a variable onto another mesh
  - `to-bytes` — serialise the mesh to a UGRID NetCDF byte buffer
  - `to-crs` — reproject the mesh to a target CRS
  - `to-dataframe` — write a tidy value table (CSV, or Parquet by extension)
  - `to-dataset` — rasterize a mesh variable to a regular-grid GeoTIFF
  - `to-file` — re-write a mesh (UGRID NetCDF or a driver)
  - `to-geodataframe` — write the mesh elements as a vector file
  - `to-image` — write a colour-mapped image of a mesh variable (PNG/JPEG)
  - `to-location` — remap a variable onto another mesh facet (node/edge/face)
  - `to-zarr` — write the mesh to a Zarr store
  - `voronoi-polygons` — Voronoi (Thiessen) cells of the mesh nodes, as a vector file
- `inspect` …
  - `bounding-polygon` — write the mesh outline polygon
  - `circumcenters` — write triangle circumcenters as points
  - `exterior-edges` — write the boundary edges as a vector
  - `exterior-faces` — write the boundary faces as a vector
  - `mesh-names` — list the topology names in a file
  - `summary` — per-variable summary statistics
- `interp` …
  - `convert-calendar` — recast the time axis to a calendar
  - `convert-units` — convert a variable to target units
  - `fill-na-spatial` — fill gaps from nearby cells in space
  - `fillna` — replace missing values with a constant
  - `interp-calendar` — interpolate onto a target's time axis
  - `interp-like` — interpolate onto another mesh's grid
  - `interp-time` — resample variables onto new time coordinates
  - `interpolate-na` — fill gaps by interpolating along a dimension
  - `laplace-interpolate` — fill gaps by solving Laplace's equation
  - `resample` — bin the time axis to a frequency and reduce
- `join` …
  - `concat` — concatenate several same-topology meshes into one
  - `merge` — merge the variables of several same-topology meshes into one
- `plot` …
  - `animate` — save an animation of a temporal mesh variable (GIF/MP4)
  - `plot` — save a filled plot of a mesh variable (PNG)
  - `plot-histogram` — save a variable's value-distribution histogram (PNG)
  - `plot-outline` — save the mesh wireframe/outline (PNG)
  - `plot-vector-field` — save a two-variable (u, v) vector-field plot (PNG)
- `read` …
  - `from-arrays` — build a mesh from node/face .npy arrays
  - `from-bytes` — open a mesh held as raw bytes
  - `from-geodataframe` — build a mesh from a polygon vector layer
  - `from-structured` — build a quad mesh from a raster (one quad per cell)
  - `from-zarr` — read a pyramids-written Zarr store (needs the [lazy] extra)
  - `open-mf` — concatenate several mesh files into one dataset
  - `read-all` — write every topology in a file to its own NetCDF
- `select` …
  - `clip` — clip the mesh to a polygon mask
  - `clip-values` — clamp a variable's values to [lower, upper]
  - `crop` — keep faces inside a polygon mask or a bbox
  - `head` — keep the first N time steps
  - `isel` — select by time and/or element position
  - `isel-layer` — select a layer by position
  - `isel-time` — select a single time step by position (collapses the axis)
  - `isel-time-range` — select a half-open range of time steps
  - `sel` — select time steps by coordinate value
  - `sel-layer` — select a layer by coordinate value
  - `subset-by-bounds` — keep the faces inside a bounding box
  - `tail` — keep the last N time steps
  - `where` — mask elements a comparison does not select

## Core verbs

Reached directly as `pyramids mesh <verb>`.

#### `pyramids mesh convert`

convert a mesh to a vector file

```text
pyramids mesh convert [-h] [--var VAR] [--location LOCATION] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to carry onto each cell (optional) |
| `--location` | optional | mesh location to convert: face / node / edge (default: face) |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids mesh extract`

rasterize a mesh variable onto a grid

```text
pyramids mesh extract [-h] --like LIKE [--var VAR] [--method METHOD] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--like` | **required** | template raster whose grid/CRS the output adopts |
| `--var` | optional | mesh variable to rasterize; required when the mesh has >1 data variable |
| `--method` | optional | rasterization method (default: nearest) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids mesh info`

print mesh metadata

```text
pyramids mesh info [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

## Subsystem `mesh advanced`

Power-user **primitives** — low-level / rarely-one-shot operations (array, coord & dimension mechanics, internal
reads, geometry micro-ops). Kept here, rather than at the domain's top level, so the common task verbs stay
uncluttered; the capability is unchanged, only the path is one level deeper.

#### `pyramids mesh advanced active-mask`

a variable's active-cell mask

```text
pyramids mesh advanced active-mask [-h] [--output OUTPUT] [--overwrite] [--json] file variable
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `variable` | required | mesh variable name |

| option | | description |
|---|---|---|
| `--output` | optional | write the array to this .npy file |
| `--overwrite` | optional | replace --output if it exists |
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced bfill`

back-fill gaps along a dimension

```text
pyramids mesh advanced bfill [-h] [--overwrite] [--limit LIMIT] [--dim DIM] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--limit` | optional | maximum number of consecutive gaps to fill (default: no limit) |
| `--dim` | optional | the dimension to operate along (default: the time dimension) |

#### `pyramids mesh advanced coarsen`

block-reduce the time axis by a window

```text
pyramids mesh advanced coarsen [-h] [--overwrite] --window WINDOW [--how HOW] [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--window` | **required** | number of steps per block |
| `--how` | optional | aggregation per block (mean, sum, min, max, ...) |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |

#### `pyramids mesh advanced collapse-layer`

collapse a 3-D volume to a 2-D face field

```text
pyramids mesh advanced collapse-layer [-h] [--overwrite] --layer LAYER [--location LOCATION] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--layer` | **required** | 0-based layer position to take per column |
| `--location` | optional | target mesh location (only 'face' is supported) |

#### `pyramids mesh advanced connected-components`

label the mesh's connected components

```text
pyramids mesh advanced connected-components [-h] [--output OUTPUT] [--overwrite] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--output` | optional | write the array to this .npy file |
| `--overwrite` | optional | replace --output if it exists |
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced cumulative-integrate`

running integral along the time axis

```text
pyramids mesh advanced cumulative-integrate [-h] [--overwrite] [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |

#### `pyramids mesh advanced differentiate`

derivative of a variable along the time axis

```text
pyramids mesh advanced differentiate [-h] [--overwrite] [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |

#### `pyramids mesh advanced drop-duplicates`

keep the first of each repeated time stamp

```text
pyramids mesh advanced drop-duplicates [-h] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh advanced drop-isel`

drop time steps by position

```text
pyramids mesh advanced drop-isel [-h] [--overwrite] --indices INDICES [INDICES ...] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--indices` | **required** | 0-based step position(s) to drop |

#### `pyramids mesh advanced drop-sel`

drop time steps by coordinate value

```text
pyramids mesh advanced drop-sel [-h] [--overwrite] --values VALUES [VALUES ...] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--values` | **required** | coordinate value(s) to drop |

#### `pyramids mesh advanced drop-variables`

drop whole data variables

```text
pyramids mesh advanced drop-variables [-h] [--overwrite] --names NAMES [NAMES ...] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--names` | **required** | variable name(s) to drop |

#### `pyramids mesh advanced dropna`

drop time steps that carry missing values

```text
pyramids mesh advanced dropna [-h] [--overwrite] [--how {any,all}] [--thresh THRESH] [--dim DIM] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--how` | optional | drop a step with 'any' missing value (default) or only when 'all' are (choices: `any`, `all`) |
| `--thresh` | optional | keep a step with at least this many valid values |
| `--dim` | optional | the dimension to operate along (default: the time dimension) |

#### `pyramids mesh advanced ffill`

forward-fill gaps along a dimension

```text
pyramids mesh advanced ffill [-h] [--overwrite] [--limit LIMIT] [--dim DIM] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--limit` | optional | maximum number of consecutive gaps to fill (default: no limit) |
| `--dim` | optional | the dimension to operate along (default: the time dimension) |

#### `pyramids mesh advanced get`

describe a single data variable

```text
pyramids mesh advanced get [-h] [--json] file variable
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `variable` | required | mesh variable name |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced integrate`

integrate a variable over the time axis

```text
pyramids mesh advanced integrate [-h] [--overwrite] [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |

#### `pyramids mesh advanced items`

describe every (name, variable) pair

```text
pyramids mesh advanced items [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced keys`

list the data-variable names

```text
pyramids mesh advanced keys [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced masked-values`

a variable's no-data-masked values

```text
pyramids mesh advanced masked-values [-h] [--output OUTPUT] [--overwrite] [--json] file variable
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `variable` | required | mesh variable name |

| option | | description |
|---|---|---|
| `--output` | optional | write the array to this .npy file |
| `--overwrite` | optional | replace --output if it exists |
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced rename-variable`

rename a data variable

```text
pyramids mesh advanced rename-variable [-h] --old OLD --new NEW [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--old` | **required** | current variable name |
| `--new` | **required** | new variable name |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh advanced rolling`

moving-window reduction along a dimension

```text
pyramids mesh advanced rolling [-h] [--overwrite] --window WINDOW [--how HOW] [--center] [--min-periods MIN_PERIODS] [--q Q] [--dim DIM] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--window` | **required** | window length in steps |
| `--how` | optional | aggregation per window (mean, sum, min, max, ...) |
| `--center` | optional | center the window on each step |
| `--min-periods` | optional | minimum valid steps in a window to produce a value |
| `--q` | optional | quantile in [0, 1] when --how quantile |
| `--dim` | optional | the dimension to operate along (default: the time dimension) |

#### `pyramids mesh advanced set-crs`

declare the mesh's CRS without reprojecting

```text
pyramids mesh advanced set-crs [-h] [--epsg EPSG] [--crs CRS] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--epsg` | optional | EPSG code to assign |
| `--crs` | optional | WKT CRS string to assign |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh advanced shift`

shift values along the time axis

```text
pyramids mesh advanced shift [-h] [--overwrite] [--periods PERIODS] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--periods` | optional | number of steps to shift (default: 1) |

#### `pyramids mesh advanced sparse-adjacency`

the mesh adjacency matrix (summary / .npz)

```text
pyramids mesh advanced sparse-adjacency [-h] [--kind KIND] [--output OUTPUT] [--overwrite] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--kind` | optional | adjacency kind: face / node / edge (default: face) |
| `--output` | optional | write the matrix to this .npz file |
| `--overwrite` | optional | replace --output if it exists |
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced squeeze`

drop a length-1 time dimension

```text
pyramids mesh advanced squeeze [-h] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh advanced thin`

keep every N-th time step

```text
pyramids mesh advanced thin [-h] [--overwrite] --step STEP file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--step` | **required** | stride (>= 1) |

#### `pyramids mesh advanced values`

describe every data variable

```text
pyramids mesh advanced values [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids mesh advanced with-variable`

add a data variable from a NumPy .npy array

```text
pyramids mesh advanced with-variable [-h] --name NAME --data DATA [--location LOCATION] [--nodata NODATA] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--name` | **required** | name for the new variable |
| `--data` | **required** | .npy file holding the values array |
| `--location` | optional | mesh location: face / node / edge (default: face) |
| `--nodata` | optional | no-data value for the new variable |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh analysis`

#### `pyramids mesh analysis aspect`

per-face aspect (clockwise from north)

```text
pyramids mesh analysis aspect [-h] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis binary-dilation`

grow a boolean face mask over its neighbours

```text
pyramids mesh analysis binary-dilation [-h] [--iterations ITERATIONS] [--border-value] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--iterations` | optional | how many times to grow (default: 1) |
| `--border-value` | optional | treat the region outside the mesh as True |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis binary-erosion`

shrink a boolean face mask at its border

```text
pyramids mesh analysis binary-erosion [-h] [--iterations ITERATIONS] [--border-value] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--iterations` | optional | how many times to shrink (default: 1) |
| `--border-value` | optional | treat the region outside the mesh as True |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis cluster`

label connected faces whose values fall in a range

```text
pyramids mesh analysis cluster [-h] --lower LOWER --upper UPPER [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--lower` | **required** | lower bound of the value range |
| `--upper` | **required** | upper bound of the value range |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis contour`

trace iso-value contour lines to a vector

```text
pyramids mesh analysis contour [-h] [--levels LEVELS [LEVELS ...]] [--var VAR] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--levels` | optional | a single level count, or several explicit levels (default: 10) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--driver` | optional | OGR vector driver (default: inferred from the output extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis count-domain-cells`

number of active mesh faces

```text
pyramids mesh analysis count-domain-cells [-h] [--var VAR] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--json` | optional | emit JSON |

#### `pyramids mesh analysis domain-area`

total area of the mesh domain

```text
pyramids mesh analysis domain-area [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids mesh analysis focal-mean`

k-ring neighbourhood mean

```text
pyramids mesh analysis focal-mean [-h] [--rings RINGS] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--rings` | optional | neighbourhood radius in rings (default: 1) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis focal-std`

k-ring neighbourhood standard deviation

```text
pyramids mesh analysis focal-std [-h] [--rings RINGS] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--rings` | optional | neighbourhood radius in rings (default: 1) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis footprint`

dissolved outline of the valid-value faces

```text
pyramids mesh analysis footprint [-h] [--var VAR] [--output OUTPUT] [--driver DRIVER] [--json] [--overwrite] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--output` | optional | write the outline polygon to this vector path |
| `--driver` | optional | OGR vector driver (default: inferred from the output extension) |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis hillshade`

per-face shaded relief

```text
pyramids mesh analysis hillshade [-h] [--azimuth AZIMUTH] [--altitude ALTITUDE] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--azimuth` | optional | light azimuth (default: 315) |
| `--altitude` | optional | light altitude (default: 45) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis histogram`

the variable's value distribution as data

```text
pyramids mesh analysis histogram [-h] [--bins BINS] [--range LOW HIGH] [--var VAR] [--output OUTPUT] [--json] [--overwrite] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--bins` | optional | number of equal-width bins (default: 256) |
| `--range` | optional | value range to bin within (default: the data range) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--output` | optional | write bin_left/bin_right/count to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis polyfit`

per-element polynomial trend along time (writes coefficients)

```text
pyramids mesh analysis polyfit [-h] --degree DEGREE [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--degree`, `-d` | **required** | polynomial degree (>= 0) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis proximity`

topological hop distance to target faces

```text
pyramids mesh analysis proximity [-h] --target-faces TARGET_FACES [TARGET_FACES ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--target-faces` | **required** | zero-based face indices to measure distance from |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis sample`

element-resolved values at point locations

```text
pyramids mesh analysis sample [-h] --var VAR --x X [X ...] --y Y [Y ...] [--method {contains,nearest}] [--time-index TIME_INDEX] [--output OUTPUT] [--json] [--overwrite] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | mesh variable to act on; defaults to the only data variable |
| `--x` | **required** | query x-coordinate(s) |
| `--y` | **required** | query y-coordinate(s) |
| `--method` | optional | how a point resolves to an element (default: contains) (choices: `contains`, `nearest`) |
| `--time-index` | optional | time step to sample (default: 0) |
| `--output` | optional | write x/y/value to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis sample-line`

one sample per face crossing along a line

```text
pyramids mesh analysis sample-line [-h] --coords COORDS [COORDS ...] [--var VAR] [--output OUTPUT] [--json] [--overwrite] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--coords` | **required** | flat x/y pairs defining the line, e.g. 0 0 5 5 |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--output` | optional | write distance/value/face_index to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis sample-point`

variable values at arbitrary x/y coordinates

```text
pyramids mesh analysis sample-point [-h] --x X [X ...] --y Y [Y ...] [--var VAR] [--output OUTPUT] [--json] [--overwrite] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--x` | **required** | query x-coordinate(s) |
| `--y` | **required** | query y-coordinate(s) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--output` | optional | write x/y/value to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis sample-random`

draw spatially random samples from the domain

```text
pyramids mesh analysis sample-random [-h] -n N [--var VAR] [--seed SEED] [--output OUTPUT] [--json] [--overwrite] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `-n`, `--n` | **required** | number of samples |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--seed` | optional | RNG seed for reproducibility |
| `--output` | optional | write x/y/value to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis sieve`

drop value-connected groups below a size

```text
pyramids mesh analysis sieve [-h] --threshold THRESHOLD [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--threshold` | **required** | minimum group size to keep |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis slope`

per-face slope magnitude

```text
pyramids mesh analysis slope [-h] [--units {degrees,radians}] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh (NetCDF) path |

| option | | description |
|---|---|---|
| `--units` | optional | slope units (default: degrees) (choices: `degrees`, `radians`) |
| `--var` | optional | mesh variable to act on; defaults to the only data variable |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh analysis zonal-stats`

per-zone statistics of a mesh variable

```text
pyramids mesh analysis zonal-stats [-h] --var VAR [--stats STATS [STATS ...]] [--unweighted] [--time-index TIME_INDEX] [--output OUTPUT] [--json] [--overwrite] file zones
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `zones` | required | polygon vector file defining the zones |

| option | | description |
|---|---|---|
| `--var` | **required** | mesh variable to act on; defaults to the only data variable |
| `--stats` | optional | stats to compute: mean sum min max std … (default: mean) |
| `--unweighted` | optional | weight each face equally instead of by its area |
| `--time-index` | optional | time step to summarise (default: 0) |
| `--output` | optional | write the table to this CSV path |
| `--json` | optional | emit JSON |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh attrs`

#### `pyramids mesh attrs del-attr`

delete a global attribute into a new mesh file

```text
pyramids mesh attrs del-attr [-h] --name NAME [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--name` | **required** | attribute name |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh attrs set-attr`

set a global attribute into a new mesh file

```text
pyramids mesh attrs set-attr [-h] --name NAME --value VALUE [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--name` | **required** | attribute name |
| `--value` | **required** | attribute value (stored as a string) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh export`

Write the data out to a file or format.

#### `pyramids mesh export attach-data`

attach a results-only file's variables to the mesh

```text
pyramids mesh export attach-data [-h] --data DATA [--driver DRIVER] [--no-lazy] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--data` | **required** | results file to read values from |
| `--driver` | optional | registered driver to use (default: detected from the suffix) |
| `--no-lazy` | optional | read variable values eagerly (default: defer until first access) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export burn`

burn a vector layer onto the mesh faces

```text
pyramids mesh export burn [-h] --geometry GEOMETRY --value VALUE [--name NAME] [--fill FILL] [--no-all-touched] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--geometry` | **required** | vector file whose geometry is burned |
| `--value` | **required** | value burned into covered faces |
| `--name` | optional | name for the new variable (default: burned) |
| `--fill` | optional | value for missed faces (default: NaN) |
| `--no-all-touched` | optional | only burn faces whose centroid falls inside (default: every touched face) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export polygonize`

dissolve equal-valued faces into polygons (vector file)

```text
pyramids mesh export polygonize [-h] [--var VAR] [--decimals DECIMALS] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to operate on (default: the only data variable) (the face field to dissolve by) |
| `--decimals` | optional | round to this many decimals before grouping |
| `--driver` | optional | OGR driver (default: inferred from the extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export regrid`

regrid a variable onto another mesh

```text
pyramids mesh export regrid [-h] --target TARGET [--method METHOD] [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--target` | **required** | destination mesh whose topology is the target |
| `--method` | optional | overlap / conservative / centroid / barycentric (default: overlap) |
| `--var` | optional | mesh variable to operate on (default: the only data variable) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-bytes`

serialise the mesh to a UGRID NetCDF byte buffer

```text
pyramids mesh export to-bytes [-h] [--overwrite] file [output]
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | optional | destination file (default: write to stdout) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-crs`

reproject the mesh to a target CRS

```text
pyramids mesh export to-crs [-h] --epsg EPSG [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--epsg` | **required** | target EPSG code (e.g. 4326) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-dataframe`

write a tidy value table (CSV, or Parquet by extension)

```text
pyramids mesh export to-dataframe [-h] [--variables VARIABLES [VARIABLES ...]] [--dropna] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination .csv or .parquet path |

| option | | description |
|---|---|---|
| `--variables` | optional | variable names to include (default: all) |
| `--dropna` | optional | drop rows whose value is no-data (NaN) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-dataset`

rasterize a mesh variable to a regular-grid GeoTIFF

```text
pyramids mesh export to-dataset [-h] --var VAR --cell-size CELL_SIZE [--method METHOD] [--bounds XMIN YMIN XMAX YMAX] [--epsg EPSG] [--nodata NODATA] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--var` | **required** | mesh variable to rasterize |
| `--cell-size` | **required** | grid cell size in coordinate units |
| `--method` | optional | interpolation method (default: nearest) |
| `--bounds` | optional | target bounds (default: the mesh bounds) |
| `--epsg` | optional | target EPSG code (default: mesh CRS) |
| `--nodata` | optional | output no-data value (default: -9999) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-file`

re-write a mesh (UGRID NetCDF or a driver)

```text
pyramids mesh export to-file [-h] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--driver` | optional | registered mesh driver (default: built-in UGRID NetCDF writer) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-geodataframe`

write the mesh elements as a vector file

```text
pyramids mesh export to-geodataframe [-h] [--var VAR] [--location LOCATION] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to operate on (default: the only data variable) (carried as a column) |
| `--location` | optional | mesh location: face / node / edge (default: face) |
| `--driver` | optional | OGR driver (default: inferred from the extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-image`

write a colour-mapped image of a mesh variable (PNG/JPEG)

```text
pyramids mesh export to-image [-h] [--var VAR] [--cell-size CELL_SIZE] [--cmap CMAP] [--method METHOD] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination image path (e.g. out.png) |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to operate on (default: the only data variable) |
| `--cell-size` | optional | rasterisation cell size (default: ~512 px across the longer side) |
| `--cmap` | optional | matplotlib colormap (default: viridis) |
| `--method` | optional | rasterisation method (default: nearest) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-location`

remap a variable onto another mesh facet (node/edge/face)

```text
pyramids mesh export to-location [-h] --location LOCATION [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--location` | **required** | target facet: node / edge / face |
| `--var` | optional | mesh variable to operate on (default: the only data variable) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export to-zarr`

write the mesh to a Zarr store

```text
pyramids mesh export to-zarr [-h] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination Zarr store path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh export voronoi-polygons`

Voronoi (Thiessen) cells of the mesh nodes, as a vector file

```text
pyramids mesh export voronoi-polygons [-h] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from the extension) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh inspect`

#### `pyramids mesh inspect bounding-polygon`

write the mesh outline polygon

```text
pyramids mesh inspect bounding-polygon [-h] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids mesh inspect circumcenters`

write triangle circumcenters as points

```text
pyramids mesh inspect circumcenters [-h] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids mesh inspect exterior-edges`

write the boundary edges as a vector

```text
pyramids mesh inspect exterior-edges [-h] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids mesh inspect exterior-faces`

write the boundary faces as a vector

```text
pyramids mesh inspect exterior-faces [-h] [--driver DRIVER] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids mesh inspect mesh-names`

list the topology names in a file

```text
pyramids mesh inspect mesh-names [-h] [--driver DRIVER] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--driver` | optional | mesh driver name (default: detect) |
| `--json` | optional | emit JSON |

#### `pyramids mesh inspect summary`

per-variable summary statistics

```text
pyramids mesh inspect summary [-h] [--variables VARIABLES [VARIABLES ...]] [--metrics METRICS [METRICS ...]] [--no-skipna] [--ddof DDOF] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--variables` | optional | variables to summarise (default: all) |
| `--metrics` | optional | statistics to report (default: count min max mean std) |
| `--no-skipna` | optional | count no-data cells instead of skipping them |
| `--ddof` | optional | delta degrees of freedom for std (default: 0) |
| `--json` | optional | emit JSON |

## Subsystem `mesh interp`

#### `pyramids mesh interp convert-calendar`

recast the time axis to a calendar

```text
pyramids mesh interp convert-calendar [-h] [--overwrite] --calendar CALENDAR [--align-on ALIGN_ON] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--calendar` | **required** | target calendar, e.g. noleap, 360_day |
| `--align-on` | optional | date alignment rule |

#### `pyramids mesh interp convert-units`

convert a variable to target units

```text
pyramids mesh interp convert-units [-h] [--overwrite] --to TARGET [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--to` | **required** | target units, e.g. km, m/s, degC |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |

#### `pyramids mesh interp fill-na-spatial`

fill gaps from nearby cells in space

```text
pyramids mesh interp fill-na-spatial [-h] [--overwrite] [--var VAR] [--max-distance MAX_DISTANCE] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |
| `--max-distance` | optional | only fill from donors within this distance (default: no limit) |

#### `pyramids mesh interp fillna`

replace missing values with a constant

```text
pyramids mesh interp fillna [-h] [--overwrite] --value VALUE [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--value` | **required** | the constant to fill with |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |

#### `pyramids mesh interp interp-calendar`

interpolate onto a target's time axis

```text
pyramids mesh interp interp-calendar [-h] [--overwrite] --target TARGET file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--target` | **required** | mesh whose time axis to interpolate onto |

#### `pyramids mesh interp interp-like`

interpolate onto another mesh's grid

```text
pyramids mesh interp interp-like [-h] [--overwrite] --other OTHER file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | mesh whose grid/time axis the output adopts |

#### `pyramids mesh interp interp-time`

resample variables onto new time coordinates

```text
pyramids mesh interp interp-time [-h] [--overwrite] --time TIME [TIME ...] [--method {linear,nearest}] [--bounds {nan,clamp}] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--time` | **required** | target time coordinate(s) on the existing axis's scale |
| `--method` | optional | interpolation method (default: linear) (choices: `linear`, `nearest`) |
| `--bounds` | optional | value for targets outside the axis (default: nan) (choices: `nan`, `clamp`) |

#### `pyramids mesh interp interpolate-na`

fill gaps by interpolating along a dimension

```text
pyramids mesh interp interpolate-na [-h] [--overwrite] [--method METHOD] [--limit LIMIT] [--dim DIM] [--no-use-coordinate] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--method` | optional | interpolation method (default: linear) |
| `--limit` | optional | maximum number of consecutive gaps to fill (default: no limit) |
| `--dim` | optional | the dimension to operate along (default: the time dimension) |
| `--no-use-coordinate` | optional | interpolate against index positions, not the coordinate values |

#### `pyramids mesh interp laplace-interpolate`

fill gaps by solving Laplace's equation

```text
pyramids mesh interp laplace-interpolate [-h] [--overwrite] [--var VAR] [--no-weights] [--direct-solve] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--var` | optional | data variable to act on; required only when the mesh has >1 data variable |
| `--no-weights` | optional | use an unweighted (uniform) graph Laplacian |
| `--direct-solve` | optional | use a direct sparse solve instead of the iterative one |

#### `pyramids mesh interp resample`

bin the time axis to a frequency and reduce

```text
pyramids mesh interp resample [-h] [--overwrite] --freq FREQ [--how HOW] [--q Q] [--no-skipna] file output
```

| argument | | description |
|---|---|---|
| `file` | required | UGRID NetCDF path |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--freq` | **required** | pandas offset alias, e.g. 1D, 2W, MS |
| `--how` | optional | aggregation per bin (mean, sum, min, max, ...) |
| `--q` | optional | quantile in [0, 1] when --how quantile |
| `--no-skipna` | optional | let missing values propagate instead of skipping them |

## Subsystem `mesh join`

#### `pyramids mesh join concat`

concatenate several same-topology meshes into one

```text
pyramids mesh join concat [-h] --inputs MESH [MESH ...] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--inputs` | **required** | input UGRID NetCDF paths to join (at least two) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh join merge`

merge the variables of several same-topology meshes into one

```text
pyramids mesh join merge [-h] --inputs MESH [MESH ...] [--compat COMPAT] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--inputs` | **required** | input UGRID NetCDF paths to join (at least two) |
| `--compat` | optional | conflict policy for shared variables (default: no_conflicts; use 'override' to let a later input win) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh plot`

#### `pyramids mesh plot animate`

save an animation of a temporal mesh variable (GIF/MP4)

```text
pyramids mesh plot animate [-h] [--var VAR] [--interval INTERVAL] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination animation path (e.g. out.gif) |

| option | | description |
|---|---|---|
| `--var` | optional | temporal variable to animate (default: the only temporal one) |
| `--interval` | optional | milliseconds between frames |
| `--dpi` | optional | output DPI (default: 100) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh plot plot`

save a filled plot of a mesh variable (PNG)

```text
pyramids mesh plot plot [-h] --var VAR [--cmap CMAP] [--title TITLE] [--basemap BASEMAP] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination image path (e.g. mesh.png) |

| option | | description |
|---|---|---|
| `--var` | **required** | mesh variable to plot |
| `--cmap` | optional | matplotlib colormap (default: viridis) |
| `--title` | optional | figure title (default: none) |
| `--basemap` | optional | web-tile basemap provider name (default: none; needs a CRS) |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh plot plot-histogram`

save a variable's value-distribution histogram (PNG)

```text
pyramids mesh plot plot-histogram [-h] [--var VAR] [--bins BINS] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination image path (e.g. hist.png) |

| option | | description |
|---|---|---|
| `--var` | optional | mesh variable to histogram (default: the only data variable) |
| `--bins` | optional | number of bins (default 15) |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh plot plot-outline`

save the mesh wireframe/outline (PNG)

```text
pyramids mesh plot plot-outline [-h] [--basemap BASEMAP] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination image path (e.g. outline.png) |

| option | | description |
|---|---|---|
| `--basemap` | optional | web-tile basemap provider name (default: none; needs a CRS) |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh plot plot-vector-field`

save a two-variable (u, v) vector-field plot (PNG)

```text
pyramids mesh plot plot-vector-field [-h] --u U --v V [--kind {quiver,barbs,streamplot}] [--thin THIN] [--time-index TIME_INDEX] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source UGRID NetCDF path |
| `output` | required | destination image path (e.g. field.png) |

| option | | description |
|---|---|---|
| `--u` | **required** | x-component variable name |
| `--v` | **required** | y-component variable name |
| `--kind` | optional | render kind (default: quiver) (choices: `quiver`, `barbs`, `streamplot`) |
| `--thin` | optional | keep every n-th arrow to declutter (default: keep all) |
| `--time-index` | optional | time step to draw for a temporal component (default: 0) |
| `--dpi` | optional | output DPI (default: 150) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh read`

#### `pyramids mesh read from-arrays`

build a mesh from node/face .npy arrays

```text
pyramids mesh read from-arrays [-h] [--data NAME=PATH] [--data-location NAME=LOC] [--epsg EPSG] [--mesh-name MESH_NAME] [--overwrite] node_x node_y faces output
```

| argument | | description |
|---|---|---|
| `node_x` | required | .npy of node x coordinates |
| `node_y` | required | .npy of node y coordinates |
| `faces` | required | .npy of face-node connectivity (n_face, max_nodes) |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--data` | optional | a per-variable .npy of values (repeatable) |
| `--data-location` | optional | mesh location for a --data variable: face / node / edge (repeatable) |
| `--epsg` | optional | EPSG code of the mesh CRS (default: 4326) |
| `--mesh-name` | optional | topology name to stamp on the mesh (default: mesh2d) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh read from-bytes`

open a mesh held as raw bytes

```text
pyramids mesh read from-bytes [-h] [--stdin] [--suffix SUFFIX] [--overwrite] [input] output
```

| argument | | description |
|---|---|---|
| `input` | optional | file of raw mesh bytes (omit with --stdin) |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--stdin` | optional | read the bytes from standard input |
| `--suffix` | optional | extension hint for driver detection (default: .nc) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh read from-geodataframe`

build a mesh from a polygon vector layer

```text
pyramids mesh read from-geodataframe [-h] [--mesh-name MESH_NAME] [--overwrite] vector output
```

| argument | | description |
|---|---|---|
| `vector` | required | source polygon vector path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--mesh-name` | optional | topology name to stamp on the mesh (default: mesh2d) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh read from-structured`

build a quad mesh from a raster (one quad per cell)

```text
pyramids mesh read from-structured [-h] [--band-names BAND_NAMES [BAND_NAMES ...]] [--mesh-name MESH_NAME] [--overwrite] raster output
```

| argument | | description |
|---|---|---|
| `raster` | required | source raster path |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--band-names` | optional | names for the per-face variables, one per band |
| `--mesh-name` | optional | topology name to stamp on the mesh (default: mesh2d) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh read from-zarr`

read a pyramids-written Zarr store (needs the [lazy] extra)

```text
pyramids mesh read from-zarr [-h] [--overwrite] store output
```

| argument | | description |
|---|---|---|
| `store` | required | input Zarr store (path / fsspec URL) |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh read open-mf`

concatenate several mesh files into one dataset

```text
pyramids mesh read open-mf [-h] [--driver DRIVER] [--mesh-name MESH_NAME] [--no-sort] [--overwrite] files [files ...] output
```

| argument | | description |
|---|---|---|
| `files` | required | source mesh file paths (in order) |
| `output` | required | destination UGRID NetCDF path |

| option | | description |
|---|---|---|
| `--driver` | optional | mesh driver name, bypassing detection (default: detect) |
| `--mesh-name` | optional | topology to read from each file (default: the first) |
| `--no-sort` | optional | keep the given file order instead of sorting by time |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids mesh read read-all`

write every topology in a file to its own NetCDF

```text
pyramids mesh read read-all [-h] [--driver DRIVER] [--eager] [--overwrite] file output_dir
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh file path |
| `output_dir` | required | directory to write one <mesh_name>.nc per topology into |

| option | | description |
|---|---|---|
| `--driver` | optional | mesh driver name, bypassing detection (default: detect) |
| `--eager` | optional | read variable values eagerly instead of lazily |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `mesh select`

#### `pyramids mesh select clip`

clip the mesh to a polygon mask

```text
pyramids mesh select clip [-h] [--overwrite] --mask MASK [--no-touch] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--mask` | **required** | polygon vector file to clip to |
| `--no-touch` | optional | keep only faces fully inside the mask (default: also those touching it) |

#### `pyramids mesh select clip-values`

clamp a variable's values to [lower, upper]

```text
pyramids mesh select clip-values [-h] [--overwrite] --lower LOWER --upper UPPER [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--lower` | **required** | minimum allowed value |
| `--upper` | **required** | maximum allowed value |
| `--var` | optional | variable to act on (default: the mesh's only data variable) |

#### `pyramids mesh select crop`

keep faces inside a polygon mask or a bbox

```text
pyramids mesh select crop [-h] [--overwrite] (--mask MASK | --bbox MINX MINY MAXX MAXY) [--epsg EPSG] [--no-touch] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--mask` | optional | polygon vector file whose faces to keep |
| `--bbox` | optional | bounding box to crop to |
| `--epsg` | optional | EPSG of --bbox (default: the mesh CRS) |
| `--no-touch` | optional | keep only faces fully inside the mask (default: also those touching it) |

#### `pyramids mesh select head`

keep the first N time steps

```text
pyramids mesh select head [-h] [--overwrite] [--n N] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--n` | optional | how many leading steps to keep (default: 5) |

#### `pyramids mesh select isel`

select by time and/or element position

```text
pyramids mesh select isel [-h] [--overwrite] [--time TIME [TIME ...]] [--element ELEMENT [ELEMENT ...]] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--time` | optional | 0-based time step(s); one collapses the axis |
| `--element` | optional | 0-based face index/indices to keep (renumbers the mesh) |

#### `pyramids mesh select isel-layer`

select a layer by position

```text
pyramids mesh select isel-layer [-h] [--overwrite] --index INDEX file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--index` | **required** | 0-based layer position |

#### `pyramids mesh select isel-time`

select a single time step by position (collapses the axis)

```text
pyramids mesh select isel-time [-h] [--overwrite] --index INDEX file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--index` | **required** | 0-based time step position (positional, not a coordinate value; use 'sel --time' to select by coordinate) |

#### `pyramids mesh select isel-time-range`

select a half-open range of time steps

```text
pyramids mesh select isel-time-range [-h] [--overwrite] --start START --stop STOP file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--start` | **required** | start step index (inclusive) |
| `--stop` | **required** | stop step index (exclusive) |

#### `pyramids mesh select sel`

select time steps by coordinate value

```text
pyramids mesh select sel [-h] [--overwrite] --time TIME [TIME ...] [--method {nearest}] [--tolerance TOLERANCE] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--time` | **required** | coordinate value(s); one value collapses the time axis, several keep it |
| `--method` | optional | 'nearest' snaps each value to the closest coordinate (default: exact) (choices: `nearest`) |
| `--tolerance` | optional | furthest a --method nearest snap may reach |

#### `pyramids mesh select sel-layer`

select a layer by coordinate value

```text
pyramids mesh select sel-layer [-h] [--overwrite] --value VALUE file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--value` | **required** | layer coordinate value |

#### `pyramids mesh select subset-by-bounds`

keep the faces inside a bounding box

```text
pyramids mesh select subset-by-bounds [-h] [--overwrite] --bbox MINX MINY MAXX MAXY file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--bbox` | **required** | bounding box in the mesh's own CRS |

#### `pyramids mesh select tail`

keep the last N time steps

```text
pyramids mesh select tail [-h] [--overwrite] [--n N] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--n` | optional | how many trailing steps to keep (default: 5) |

#### `pyramids mesh select where`

mask elements a comparison does not select

```text
pyramids mesh select where [-h] [--overwrite] --cond COND [--other OTHER] [--var VAR] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source mesh path (UGRID NetCDF) |
| `output` | required | destination mesh path (.nc keeps it a UGRID NetCDF) |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |
| `--cond` | **required** | comparison '[VAR] OP NUMBER', e.g. 'depth > 3' or '> 3' |
| `--other` | optional | value for unselected elements (default: NaN) |
| `--var` | optional | variable to act on (default: the mesh's only data variable) |

