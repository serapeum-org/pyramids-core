# nc — NetCDF / multidimensional operations

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids nc --help` for the live listing
and `pyramids nc <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?* suggestion.

## Command tree

- `crop` — spatially subset to a bbox
- `extract` — extract one variable to a raster
- `get-group` — write one sub-group out as a NetCDF
- `groups` — list root-level sub-group names
- `info` — summarize a NetCDF file
- `list-vars` — list variables/subdatasets
- `sel` — select along time into a new NetCDF
- `advanced` …
  - `assign-coords` — restamp existing band dimensions
  - `coarsen` — block-reduce a dimension by a window
  - `cumulative` — running total/product along a dimension
  - `cumulative-integrate` — running integral along a dimension
  - `differentiate` — derivative of values along a dimension
  - `drop-dims` — drop band dimensions
  - `drop-duplicates` — drop repeated-stamp bands
  - `drop-isel` — drop bands by position
  - `drop-sel` — drop bands by coordinate label
  - `expand-dims` — lift a variable onto a new band dim
  - `groupby-bins` — bin a dimension and reduce each bin
  - `integrate` — integrate over a dimension
  - `interp-like` — interpolate onto a target cube's band axes
  - `pad` — grow dimensions with a constant fill
  - `rename-dims` — rename band dimensions
  - `reset-coords` — demote aux coordinates to variables
  - `rolling` — moving-window reduction along a dimension
  - `rolling-exp` — exponentially weighted moving reduction
  - `set-coords` — promote variables to aux coordinates
  - `shift` — shift values along a dimension
  - `squeeze` — drop length-one band dimensions
  - `weighted` — weighted reduction over dimension(s)
- `calc` …
  - `convert-calendar` — recast the time axis to a calendar
  - `interp` — interpolate onto new coordinate values
  - `interp-calendar` — interpolate onto a target's time axis
  - `interpolate-na` — fill gaps along a dimension
  - `polyfit` — least-squares polynomial fit along a dimension
  - `reduce` — collapse a dimension with an aggregation
- `export` …
  - `combine-kerchunk` — write a combined kerchunk manifest over many files
  - `convert` — re-write a file as a NetCDF
  - `dataframe` — export variables to a CSV table
  - `dict` — print the file's structure as a dict
  - `kerchunk` — write a kerchunk reference manifest
  - `zarr` — write one variable to a Zarr store
- `import` …
  - `concat` — join cubes end to end along a dimension
  - `dataframe` — build a cube from a MultiIndex CSV
  - `dict` — build a cube from a JSON structure dump
  - `merge` — put the variables of several cubes together
  - `open-mf` — report a lazy variable stack over files
- `inspect` …
  - `del-attr` — delete a global attribute
  - `dim-values` — a dimension's coordinate values
  - `metadata` — the full metadata record
  - `set-attr` — set a global attribute
  - `summary` — per-variable statistics
  - `time-values` — the formatted time-axis labels
- `plot` …
  - `animate` — save an animation over a dimension
  - `plot-histogram` — save a variable's value histogram (PNG)
  - `render` — save a map of one variable (PNG)
- `select` …
  - `dropna` — drop steps that hold missing cells
  - `head` — keep the first N bands
  - `isel` — select bands by position
  - `subset` — read a windowed (variable, time, bbox) slice
  - `tail` — keep the last N bands
  - `thin` — keep every N-th band
- `vars` …
  - `add-var` — copy variables in from another NetCDF
  - `crop-var` — crop one variable to a vector mask
  - `remove-var` — remove a variable
  - `rename-var` — rename a variable
  - `reproject-var` — reproject one variable
  - `resample-var` — resample one variable
  - `set-var` — write a raster in as a variable

## Core verbs

Reached directly as `pyramids nc <verb>`.

#### `pyramids nc crop`

spatially subset to a bbox

```text
pyramids nc crop [-h] --bbox MINX MINY MAXX MAXY [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--bbox` | **required** | crop window in the file's CRS |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc extract`

extract one variable to a raster

```text
pyramids nc extract [-h] --var VAR [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination raster path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable name (supports 'group/var') |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc get-group`

write one sub-group out as a NetCDF

```text
pyramids nc get-group [-h] --group GROUP [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--group` | **required** | sub-group name (nested 'a/b' accepted) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc groups`

list root-level sub-group names

```text
pyramids nc groups [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids nc info`

summarize a NetCDF file

```text
pyramids nc info [-h] [--json | --full] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit the compact summary as JSON |
| `--full` | optional | print the full ncdump-style report (dims/vars/attrs) as text |

#### `pyramids nc list-vars`

list variables/subdatasets

```text
pyramids nc list-vars [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids nc sel`

select along time into a new NetCDF

```text
pyramids nc sel [-h] --time TIME [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--time` | **required** | a date label on the time axis, e.g. 2020-06-15 ('nc info' shows the step count and endpoints) |
| `--var` | optional | narrow to one variable before selecting |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `nc advanced`

Power-user **primitives** — low-level / rarely-one-shot operations (array, coord & dimension mechanics, internal
reads, geometry micro-ops). Kept here, rather than at the domain's top level, so the common task verbs stay
uncluttered; the capability is unchanged, only the path is one level deeper.

#### `pyramids nc advanced assign-coords`

restamp existing band dimensions

```text
pyramids nc advanced assign-coords [-h] --coord DIM=v1,v2,... [DIM=v1,v2,... ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--coord` | **required** | one or more DIM=v1,v2,... band-dimension restamps |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced coarsen`

block-reduce a dimension by a window

```text
pyramids nc advanced coarsen [-h] --dim DIM --window WINDOW [--how HOW] [--boundary {exact,trim,pad}] [--q Q] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--window` | **required** | number of steps per block |
| `--how` | optional | aggregation to apply (mean, sum, min, max, std, median, ...) |
| `--boundary` | optional | how to treat a trailing partial block (default: exact) (choices: `exact`, `trim`, `pad`) |
| `--q` | optional | quantile in [0, 1] when --how quantile |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced cumulative`

running total/product along a dimension

```text
pyramids nc advanced cumulative [-h] --dim DIM [--how {sum,prod}] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--how` | optional | 'sum' for a running total (default) or 'prod' for a running product (choices: `sum`, `prod`) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced cumulative-integrate`

running integral along a dimension

```text
pyramids nc advanced cumulative-integrate [-h] --dim DIM [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced differentiate`

derivative of values along a dimension

```text
pyramids nc advanced differentiate [-h] --dim DIM [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced drop-dims`

drop band dimensions

```text
pyramids nc advanced drop-dims [-h] --dims DIMS [DIMS ...] [--errors {raise,ignore}] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--dims` | **required** | band dimensions to drop |
| `--errors` | optional | how to treat an unknown dimension (default: raise) (choices: `raise`, `ignore`) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced drop-duplicates`

drop repeated-stamp bands

```text
pyramids nc advanced drop-duplicates [-h] --dim DIM [--keep {first,last}] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | band dimension to de-duplicate |
| `--keep` | optional | which plane of a repeated stamp to keep (choices: `first`, `last`) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced drop-isel`

drop bands by position

```text
pyramids nc advanced drop-isel [-h] --index DIM=N [DIM=N ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--index` | **required** | DIM=N pairs, N a 0-based band index to drop |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced drop-sel`

drop bands by coordinate label

```text
pyramids nc advanced drop-sel [-h] --label DIM=VALUE [DIM=VALUE ...] [--errors {raise,ignore}] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--label` | **required** | DIM=VALUE pairs (a CF date label resolves on a time axis) |
| `--errors` | optional | 'raise' (default) refuses an absent label; 'ignore' skips it (choices: `raise`, `ignore`) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced expand-dims`

lift a variable onto a new band dim

```text
pyramids nc advanced expand-dims [-h] --var VAR --dim DIM [--value VALUE] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable name |
| `--dim` | **required** | name of the new length-one band dim |
| `--value` | optional | the new dimension's single coordinate value (default: none) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced groupby-bins`

bin a dimension and reduce each bin

```text
pyramids nc advanced groupby-bins [-h] --dim DIM --bins BINS [--how HOW] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--bins` | **required** | number of equal-width bins |
| `--how` | optional | aggregation to apply (mean, sum, min, max, std, median, ...) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced integrate`

integrate over a dimension

```text
pyramids nc advanced integrate [-h] --dim DIM [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced interp-like`

interpolate onto a target cube's band axes

```text
pyramids nc advanced interp-like [-h] --target TARGET [--method METHOD] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--target` | **required** | NetCDF whose band coordinates to interpolate onto |
| `--method` | optional | interpolation method (default: linear) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced pad`

grow dimensions with a constant fill

```text
pyramids nc advanced pad [-h] --width DIM=N [DIM=N ...] [--mode MODE] [--value VALUE] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--width` | **required** | DIM=N pairs, N steps added on each side (band or x/y/lon/lat) |
| `--mode` | optional | pad mode (only 'constant' is supported) |
| `--value` | optional | constant fill (default: the variable's no-data value) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced rename-dims`

rename band dimensions

```text
pyramids nc advanced rename-dims [-h] --map OLD=NEW [OLD=NEW ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--map` | **required** | one or more OLD=NEW dimension renames |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced reset-coords`

demote aux coordinates to variables

```text
pyramids nc advanced reset-coords [-h] [--names [NAMES ...]] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--names` | optional | coordinates to demote (default: all) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced rolling`

moving-window reduction along a dimension

```text
pyramids nc advanced rolling [-h] --dim DIM --window WINDOW [--how HOW] [--center] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--window` | **required** | window length in steps |
| `--how` | optional | aggregation to apply (mean, sum, min, max, std, median, ...) |
| `--center` | optional | center the window on each step |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced rolling-exp`

exponentially weighted moving reduction

```text
pyramids nc advanced rolling-exp [-h] --dim DIM --alpha ALPHA [--how HOW] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--alpha` | **required** | smoothing factor in (0, 1] |
| `--how` | optional | aggregation to apply (mean, sum, min, max, std, median, ...) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced set-coords`

promote variables to aux coordinates

```text
pyramids nc advanced set-coords [-h] --names NAMES [NAMES ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--names` | **required** | variables to promote |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced shift`

shift values along a dimension

```text
pyramids nc advanced shift [-h] --dim DIM [--periods PERIODS] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--periods` | optional | number of steps to shift |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced squeeze`

drop length-one band dimensions

```text
pyramids nc advanced squeeze [-h] [--dim DIM] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--dim` | optional | the one length-one dimension to drop (default: all) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc advanced weighted`

weighted reduction over dimension(s)

```text
pyramids nc advanced weighted [-h] --weights WEIGHTS [--dims DIMS [DIMS ...]] [--how HOW] [--q Q] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--weights` | **required** | the keyword 'area' (cos-latitude cell weights) or a weights-raster path |
| `--dims` | optional | dimension(s) to reduce over (default: both spatial axes) |
| `--how` | optional | aggregation to apply (mean, sum, min, max, std, median, ...) |
| `--q` | optional | quantile in [0, 1] when --how quantile |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `nc calc`

#### `pyramids nc calc convert-calendar`

recast the time axis to a calendar

```text
pyramids nc calc convert-calendar [-h] --calendar CALENDAR [--dim DIM] [--align-on ALIGN_ON] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--calendar` | **required** | target calendar, e.g. noleap, 360_day |
| `--dim` | optional | the dimension to operate along |
| `--align-on` | optional | date alignment rule |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc calc interp`

interpolate onto new coordinate values

```text
pyramids nc calc interp [-h] --coord DIM=VALUE [DIM=VALUE ...] [--method METHOD] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--coord` | **required** | target coordinate(s), e.g. time=5 (non-spatial dimensions only) |
| `--method` | optional | interpolation method (default: linear) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc calc interp-calendar`

interpolate onto a target's time axis

```text
pyramids nc calc interp-calendar [-h] --target TARGET [--dim DIM] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--target` | **required** | NetCDF whose time axis to interpolate onto |
| `--dim` | optional | the dimension to operate along |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc calc interpolate-na`

fill gaps along a dimension

```text
pyramids nc calc interpolate-na [-h] --dim DIM [--method METHOD] [--limit LIMIT] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--method` | optional | interpolation method (default: linear) |
| `--limit` | optional | max consecutive gaps to fill |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc calc polyfit`

least-squares polynomial fit along a dimension

```text
pyramids nc calc polyfit [-h] --dim DIM --deg DEG [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--deg` | **required** | polynomial degree (0 or more) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc calc reduce`

collapse a dimension with an aggregation

```text
pyramids nc calc reduce [-h] --dim DIM [--how HOW] [--groupby GROUPBY] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | the dimension to operate along |
| `--how` | optional | aggregation to apply (mean, sum, min, max, std, median, ...) |
| `--groupby` | optional | optional coordinate to group by first |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `nc export`

Write the data out to a file or format.

#### `pyramids nc export combine-kerchunk`

write a combined kerchunk manifest over many files

```text
pyramids nc export combine-kerchunk [-h] --inputs INPUTS [INPUTS ...] [--concat-dims CONCAT_DIMS [CONCAT_DIMS ...]] [--identical-dims IDENTICAL_DIMS [IDENTICAL_DIMS ...]] [--inline-threshold INLINE_THRESHOLD] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination manifest JSON path |

| option | | description |
|---|---|---|
| `--inputs` | **required** | two or more NetCDF paths |
| `--concat-dims` | optional | dimension(s) to concatenate along (default: time) |
| `--identical-dims` | optional | dimensions expected identical across files (default: lat lon) |
| `--inline-threshold` | optional | embed chunks smaller than this many bytes (default: 500) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc export convert`

re-write a file as a NetCDF

```text
pyramids nc export convert [-h] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc export dataframe`

export variables to a CSV table

```text
pyramids nc export dataframe [-h] [--vars VARS [VARS ...]] [--dropna] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination CSV path |

| option | | description |
|---|---|---|
| `--vars` | optional | variables to export (default: all) |
| `--dropna` | optional | drop rows with any missing value |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc export dict`

print the file's structure as a dict

```text
pyramids nc export dict [-h] [--json] [--data] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |
| `--data` | optional | include variable arrays (large) |

#### `pyramids nc export kerchunk`

write a kerchunk reference manifest

```text
pyramids nc export kerchunk [-h] [--inline-threshold INLINE_THRESHOLD] [--vlen-encode VLEN_ENCODE] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination manifest JSON path |

| option | | description |
|---|---|---|
| `--inline-threshold` | optional | embed chunks smaller than this many bytes (default: 500) |
| `--vlen-encode` | optional | variable-length string encoding mode (default: embed) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc export zarr`

write one variable to a Zarr store

```text
pyramids nc export zarr [-h] --var VAR [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination .zarr store path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable to store |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `nc import`

Read / construct the data from another source.

#### `pyramids nc import concat`

join cubes end to end along a dimension

```text
pyramids nc import concat [-h] --inputs INPUTS [INPUTS ...] --dim DIM [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--inputs` | **required** | two or more NetCDF paths |
| `--dim` | **required** | dimension to join along |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc import dataframe`

build a cube from a MultiIndex CSV

```text
pyramids nc import dataframe [-h] --index-col INDEX_COL [INDEX_COL ...] [--crs CRS] [--x X] [--y Y] [--vars VARS [VARS ...]] [--no-data-value NO_DATA_VALUE] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source CSV path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--index-col` | **required** | index columns, innermost last (y/x grid axes; outer are band dims) |
| `--crs` | optional | CRS for the result (EPSG or string) |
| `--x` | optional | index level holding the x coordinates |
| `--y` | optional | index level holding the y coordinates |
| `--vars` | optional | columns to use (default: all) |
| `--no-data-value` | optional | sentinel for gaps (default: NaN) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc import dict`

build a cube from a JSON structure dump

```text
pyramids nc import dict [-h] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | source JSON path (as 'nc export dict --json --data') |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc import merge`

put the variables of several cubes together

```text
pyramids nc import merge [-h] --inputs INPUTS [INPUTS ...] [--compat {no_conflicts,override}] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--inputs` | **required** | two or more NetCDF paths |
| `--compat` | optional | how to treat a shared variable (default: no_conflicts) (choices: `no_conflicts`, `override`) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc import open-mf`

report a lazy variable stack over files

```text
pyramids nc import open-mf [-h] --paths PATHS [PATHS ...] --var VAR [--json]
```

| option | | description |
|---|---|---|
| `--paths` | **required** | NetCDF paths to stack |
| `--var` | **required** | variable to stack across the files |
| `--json` | optional | emit JSON |

## Subsystem `nc inspect`

#### `pyramids nc inspect del-attr`

delete a global attribute

```text
pyramids nc inspect del-attr [-h] --name NAME [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--name` | **required** | attribute name |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc inspect dim-values`

a dimension's coordinate values

```text
pyramids nc inspect dim-values [-h] --dim DIM [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--dim` | **required** | dimension name |
| `--json` | optional | emit JSON |

#### `pyramids nc inspect metadata`

the full metadata record

```text
pyramids nc inspect metadata [-h] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

#### `pyramids nc inspect set-attr`

set a global attribute

```text
pyramids nc inspect set-attr [-h] --name NAME --value VALUE [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--name` | **required** | attribute name |
| `--value` | **required** | attribute value |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc inspect summary`

per-variable statistics

```text
pyramids nc inspect summary [-h] [--vars VARS [VARS ...]] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--vars` | optional | variables to summarize (default: all) |
| `--json` | optional | emit JSON |

#### `pyramids nc inspect time-values`

the formatted time-axis labels

```text
pyramids nc inspect time-values [-h] [--var VAR] [--format FORMAT] [--raw] [--json] file
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |

| option | | description |
|---|---|---|
| `--var` | optional | time variable name (empty result if it is not a time axis) |
| `--format` | optional | strftime format for the labels |
| `--raw` | optional | emit the stored numeric offsets instead of decoded labels |
| `--json` | optional | emit JSON |

## Subsystem `nc plot`

#### `pyramids nc plot animate`

save an animation over a dimension

```text
pyramids nc plot animate [-h] --var VAR [--dim DIM] [--fps FPS] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination animation path (e.g. .gif) |

| option | | description |
|---|---|---|
| `--var` | **required** | variable to plot |
| `--dim` | optional | dimension to animate over |
| `--fps` | optional | frames per second |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc plot plot-histogram`

save a variable's value histogram (PNG)

```text
pyramids nc plot plot-histogram [-h] --var VAR [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination image path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable to plot |
| `--dpi` | optional | output DPI |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc plot render`

save a map of one variable (PNG)

```text
pyramids nc plot render [-h] --var VAR [--kind {auto,imshow,pcolormesh,contour}] [--title TITLE] [--dpi DPI] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination image path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable to plot |
| `--kind` | optional | render kind (default: auto) (choices: `auto`, `imshow`, `pcolormesh`, `contour`) |
| `--title` | optional | plot title |
| `--dpi` | optional | output DPI |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `nc select`

#### `pyramids nc select dropna`

drop steps that hold missing cells

```text
pyramids nc select dropna [-h] --dim DIM [--how {any,all}] [--thresh THRESH] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--dim` | **required** | non-spatial dimension to drop from |
| `--how` | optional | 'any' drops a step with any gap; 'all' only a fully empty step (choices: `any`, `all`) |
| `--thresh` | optional | keep a step with at least this many valid cells (overrides --how) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc select head`

keep the first N bands

```text
pyramids nc select head [-h] --count DIM=N [DIM=N ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--count` | **required** | DIM=N pairs, N how many leading bands to keep |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc select isel`

select bands by position

```text
pyramids nc select isel [-h] --index DIM=N [DIM=N ...] [--drop] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--index` | **required** | DIM=N pairs, N a 0-based band index |
| `--drop` | optional | drop axes a scalar index collapsed |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc select subset`

read a windowed (variable, time, bbox) slice

```text
pyramids nc select subset [-h] --var VAR [--time TIME] [--bbox MINX MINY MAXX MAXY] [--crs CRS] [--dim DIM=N [DIM=N ...]] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--var` | **required** | data variable to window |
| `--time` | optional | 0-based time-step index (default: all) |
| `--bbox` | optional | crop window in --crs (default: the full grid) |
| `--crs` | optional | CRS of --bbox: an EPSG code or a CRS string (e.g. EPSG:4326); default EPSG:4326 |
| `--dim` | optional | DIM=N index per extra non-spatial dimension (e.g. plev=0) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc select tail`

keep the last N bands

```text
pyramids nc select tail [-h] --count DIM=N [DIM=N ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--count` | **required** | DIM=N pairs, N how many trailing bands to keep |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc select thin`

keep every N-th band

```text
pyramids nc select thin [-h] --step DIM=N [DIM=N ...] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination path (.nc keeps it a NetCDF) |

| option | | description |
|---|---|---|
| `--step` | **required** | DIM=N pairs, N the stride (at least 1) |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `nc vars`

#### `pyramids nc vars add-var`

copy variables in from another NetCDF

```text
pyramids nc vars add-var [-h] --source SOURCE [--var VAR] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--source` | **required** | NetCDF to copy variables from |
| `--var` | optional | one variable to copy (default: all of the source) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc vars crop-var`

crop one variable to a vector mask

```text
pyramids nc vars crop-var [-h] --var VAR --mask MASK [--no-touch] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable name |
| `--mask` | **required** | polygon vector mask to crop to |
| `--no-touch` | optional | exclude cells that only touch the mask boundary |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc vars remove-var`

remove a variable

```text
pyramids nc vars remove-var [-h] --var VAR [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable name |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc vars rename-var`

rename a variable

```text
pyramids nc vars rename-var [-h] --old OLD --new NEW [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--old` | **required** | current variable name |
| `--new` | **required** | new variable name |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc vars reproject-var`

reproject one variable

```text
pyramids nc vars reproject-var [-h] --var VAR --epsg EPSG [--method METHOD] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable name |
| `--epsg` | **required** | target EPSG code |
| `--method` | optional | resampling method (default: nearest neighbor) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc vars resample-var`

resample one variable

```text
pyramids nc vars resample-var [-h] --var VAR --cell-size CELL_SIZE [--method METHOD] [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | variable name |
| `--cell-size` | **required** | new cell size |
| `--method` | optional | resampling method (default: nearest neighbor) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids nc vars set-var`

write a raster in as a variable

```text
pyramids nc vars set-var [-h] --var VAR --raster RASTER [--overwrite] file output
```

| argument | | description |
|---|---|---|
| `file` | required | NetCDF path |
| `output` | required | destination NetCDF path |

| option | | description |
|---|---|---|
| `--var` | **required** | name for the new variable |
| `--raster` | **required** | raster to store as the variable |
| `--overwrite` | optional | replace the output if it already exists |
