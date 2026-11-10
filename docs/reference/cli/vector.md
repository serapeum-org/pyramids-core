# vector — `FeatureCollection` operations

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids vector --help` for the live
listing and `pyramids vector <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?*
suggestion.

## Command tree

- `bounds` — print the vector bounding box
- `clip` — spatially subset a vector
- `convert` — re-save a vector in another format
- `info` — print vector metadata
- `warp` — reproject a vector
- `advanced` …
  - `affine-transform` — apply a 2-D affine matrix
  - `build-area` — build polygonal areas from linework
  - `extract-unique-points` — the distinct vertices of each geometry
  - `force-2d` — drop the Z coordinate
  - `force-3d` — add a constant Z coordinate
  - `frechet-distance` — Frechet distance to --other
  - `hausdorff-distance` — Hausdorff distance to --other
  - `hilbert-distance` — Hilbert-curve index per feature
  - `maximum-inscribed-circle` — the largest circle fitting inside
  - `minimum-clearance` — minimum clearance per feature
  - `minimum-rotated-rectangle` — the smallest enclosing rectangle
  - `normalize` — normalize geometries to canonical form
  - `offset-curve` — offset a line by a distance
  - `remove-repeated-points` — drop duplicate vertices
  - `representative-point` — a point guaranteed inside each geometry
  - `reverse` — reverse coordinate (winding) order
  - `segmentize` — densify edges to a max segment length
  - `set-precision` — snap coordinates to a grid size
  - `shared-paths` — shared paths with --other
  - `shortest-line` — shortest line with --other
  - `simplify-coverage` — topology-preserving coverage simplification
  - `snap` — snap geometries to another layer's vertices
  - `with-centroid` — attach per-feature centroid columns
  - `with-coordinates` — attach per-vertex x/y coordinate columns
- `geom` …
  - `buffer` — buffer each geometry by a distance
  - `concave-hull` — the concave hull of each geometry
  - `make-valid` — repair invalid geometries
  - `minimum-bounding-circle` — the smallest enclosing circle
  - `rotate` — rotate geometries about an origin
  - `scale` — scale geometries about their center
  - `simplify` — Douglas-Peucker simplification
  - `translate` — shift geometries by an offset
- `inspect` …
  - `count-coordinates` — per-feature coordinate count
  - `count-geometries` — per-feature geometry count
  - `count-interior-rings` — per-feature interior-ring count
  - `estimate-utm` — estimate the best UTM CRS
  - `feature-count` — count features
  - `feature-info` — metadata read without geometry
  - `invalid-coverage-edges` — write the invalid coverage edges
  - `is-valid-coverage` — whether polygons form a valid coverage
  - `is-valid-reason` — per-feature geometry-validity reason
  - `list-layers` — list the vector-layer names
- `io` …
  - `from-bbox` — write a one-feature bbox polygon
  - `from-features` — build an FC from a GeoJSON features file
  - `from-featureserver` — read an Esri FeatureServer layer
  - `from-ogc-features` — read an OGC API Features service
  - `from-records` — build an FC from a CSV with a WKT geometry column
  - `from-vectortileserver` — read a vector-tile service
  - `from-xyz` — build a point layer from CSV x/y/z columns
  - `read-gpx` — write every non-empty GPX sub-layer to a multi-layer output
  - `read-parquet` — convert a GeoParquet to a vector file
  - `to-feather` — write a GeoFeather file
  - `to-geo-dict` — print the GeoJSON mapping
  - `to-json` — emit the GeoJSON string (stdout or --out)
  - `to-mvt` — write a Mapbox Vector Tiles archive
  - `to-parquet` — write a GeoParquet file
  - `to-pmtiles` — write a PMTiles archive
  - `to-wkb` — write a WKB table to CSV
  - `to-wkt` — write a WKT table to CSV
- `overlay` …
  - `clip-by-rect` — clip to an axis-aligned window
  - `difference` — difference with --other
  - `dissolve` — merge features, optionally grouped
  - `explode` — split multi-part geometries into parts
  - `filter` — keep features matching a mask predicate
  - `intersection` — intersection with --other
  - `intersection-all` — intersection of all features
  - `overlay` — spatial overlay of two layers
  - `sjoin` — predicate-based spatial join
  - `sjoin-nearest` — nearest-neighbour spatial join
  - `symmetric-difference` — symmetric difference with --other
  - `union` — union with --other
  - `union-all` — the single union of all features
- `tess` …
  - `constrained-delaunay-triangles` — constrained Delaunay triangulation of each polygon
  - `delaunay-triangles` — Delaunay triangulation
  - `distance` — planar distance to --other
  - `fishnet` — a grid of square cell polygons
  - `geodesic-area` — ellipsoidal area per feature
  - `geodesic-distance` — ground distance to --other
  - `geodesic-length` — ellipsoidal length per feature
  - `get-coordinates` — the vertex coordinate table
  - `h3-bin` — aggregate points into H3 hexagon cells
  - `interpolate-to-raster` — grid a point column onto a raster via IDW
  - `polygonize` — build polygons from noded line work
  - `quadtree` — adaptive quad-tree binning of points
  - `sample-points` — random points within each geometry
  - `to-h3` — attach each point's H3 cell index
  - `voronoi` — Thiessen tessellation with optional values/clip
  - `voronoi-polygons` — Voronoi cells of the inputs

## Core verbs

Reached directly as `pyramids vector <verb>`.

#### `pyramids vector bounds`

print the vector bounding box

```text
pyramids vector bounds [-h] [--layer LAYER] [--crs CRS] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index |
| `--crs` | optional | reproject bounds to this CRS first |
| `--json` | optional | emit JSON |

#### `pyramids vector clip`

spatially subset a vector

```text
pyramids vector clip [-h] (--bbox MINX MINY MAXX MAXY | --mask MASK) [--driver DRIVER] [--layer LAYER] [--overwrite] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--bbox` | optional | keep features intersecting this window (source CRS) |
| `--mask` | optional | vector file whose polygons clip the source |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--layer` | optional | layer name/index |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector convert`

re-save a vector in another format

```text
pyramids vector convert [-h] [--driver DRIVER] [--layer LAYER] [--overwrite] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--layer` | optional | layer name/index |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector info`

print vector metadata

```text
pyramids vector info [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector warp`

reproject a vector

```text
pyramids vector warp [-h] --crs CRS [--driver DRIVER] [--layer LAYER] [--overwrite] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--crs` | **required** | target CRS (EPSG, WKT, PROJ4) |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--layer` | optional | layer name/index |
| `--overwrite` | optional | replace the output if it exists |

## Subsystem `vector advanced`

Power-user **primitives** — low-level / rarely-one-shot operations (array, coord & dimension mechanics, internal
reads, geometry micro-ops). Kept here, rather than at the domain's top level, so the common task verbs stay uncluttered;
the capability is unchanged, only the path is longer.

#### `pyramids vector advanced affine-transform`

apply a 2-D affine matrix

```text
pyramids vector advanced affine-transform [-h] [--layer LAYER] [--overwrite] --matrix A B D E XOFF YOFF src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--matrix` | **required** | affine coefficients [a, b, d, e, xoff, yoff] |

#### `pyramids vector advanced build-area`

build polygonal areas from linework

```text
pyramids vector advanced build-area [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced extract-unique-points`

the distinct vertices of each geometry

```text
pyramids vector advanced extract-unique-points [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced force-2d`

drop the Z coordinate

```text
pyramids vector advanced force-2d [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced force-3d`

add a constant Z coordinate

```text
pyramids vector advanced force-3d [-h] [--layer LAYER] [--overwrite] [--z Z] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--z` | optional | Z value to assign |

#### `pyramids vector advanced frechet-distance`

Frechet distance to --other

```text
pyramids vector advanced frechet-distance [-h] --other OTHER [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--other` | **required** | second vector file, compared element-wise against the source |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector advanced hausdorff-distance`

Hausdorff distance to --other

```text
pyramids vector advanced hausdorff-distance [-h] --other OTHER [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--other` | **required** | second vector file, compared element-wise against the source |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector advanced hilbert-distance`

Hilbert-curve index per feature

```text
pyramids vector advanced hilbert-distance [-h] [--level LEVEL] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--level` | optional | Hilbert-curve level |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector advanced maximum-inscribed-circle`

the largest circle fitting inside

```text
pyramids vector advanced maximum-inscribed-circle [-h] [--layer LAYER] [--overwrite] [--tolerance TOLERANCE] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--tolerance` | optional | search tolerance |

#### `pyramids vector advanced minimum-clearance`

minimum clearance per feature

```text
pyramids vector advanced minimum-clearance [-h] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector advanced minimum-rotated-rectangle`

the smallest enclosing rectangle

```text
pyramids vector advanced minimum-rotated-rectangle [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced normalize`

normalize geometries to canonical form

```text
pyramids vector advanced normalize [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced offset-curve`

offset a line by a distance

```text
pyramids vector advanced offset-curve [-h] [--layer LAYER] [--overwrite] --distance DISTANCE src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--distance` | **required** | offset distance |

#### `pyramids vector advanced remove-repeated-points`

drop duplicate vertices

```text
pyramids vector advanced remove-repeated-points [-h] [--layer LAYER] [--overwrite] [--tolerance TOLERANCE] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--tolerance` | optional | merge vertices within this distance |

#### `pyramids vector advanced representative-point`

a point guaranteed inside each geometry

```text
pyramids vector advanced representative-point [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced reverse`

reverse coordinate (winding) order

```text
pyramids vector advanced reverse [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector advanced segmentize`

densify edges to a max segment length

```text
pyramids vector advanced segmentize [-h] [--layer LAYER] [--overwrite] --max-length MAX_LENGTH src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--max-length` | **required** | maximum segment length |

#### `pyramids vector advanced set-precision`

snap coordinates to a grid size

```text
pyramids vector advanced set-precision [-h] [--layer LAYER] [--overwrite] --grid-size GRID_SIZE src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--grid-size` | **required** | coordinate grid size |

#### `pyramids vector advanced shared-paths`

shared paths with --other

```text
pyramids vector advanced shared-paths [-h] [--layer LAYER] [--overwrite] --other OTHER src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | second vector path (the right-hand operand) |

#### `pyramids vector advanced shortest-line`

shortest line with --other

```text
pyramids vector advanced shortest-line [-h] [--layer LAYER] [--overwrite] --other OTHER src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | second vector path (the right-hand operand) |

#### `pyramids vector advanced simplify-coverage`

topology-preserving coverage simplification

```text
pyramids vector advanced simplify-coverage [-h] [--layer LAYER] [--overwrite] --tolerance TOLERANCE src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--tolerance` | **required** | simplify tolerance |

#### `pyramids vector advanced snap`

snap geometries to another layer's vertices

```text
pyramids vector advanced snap [-h] [--layer LAYER] [--overwrite] --other OTHER --tolerance TOLERANCE src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | vector file to snap towards |
| `--tolerance` | **required** | maximum snapping distance |

#### `pyramids vector advanced with-centroid`

attach per-feature centroid columns

```text
pyramids vector advanced with-centroid [-h] [--layer LAYER] [--overwrite] [--driver DRIVER] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--driver` | optional | OGR driver (default: inferred from extension) |

#### `pyramids vector advanced with-coordinates`

attach per-vertex x/y coordinate columns

```text
pyramids vector advanced with-coordinates [-h] [--layer LAYER] [--overwrite] [--driver DRIVER] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--driver` | optional | OGR driver (default: inferred from extension) |

## Subsystem `vector geom`

#### `pyramids vector geom buffer`

buffer each geometry by a distance

```text
pyramids vector geom buffer [-h] [--layer LAYER] [--overwrite] --distance DISTANCE [--resolution RESOLUTION] [--cap-style {round,flat,square}] [--join-style {round,mitre,bevel}] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--distance` | **required** | buffer distance |
| `--resolution` | optional | segments per quadrant |
| `--cap-style` | optional | end-cap style (choices: `round`, `flat`, `square`) |
| `--join-style` | optional | segment join style (choices: `round`, `mitre`, `bevel`) |

#### `pyramids vector geom concave-hull`

the concave hull of each geometry

```text
pyramids vector geom concave-hull [-h] [--layer LAYER] [--overwrite] [--ratio RATIO] [--allow-holes] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--ratio` | optional | hull tightness in [0, 1] |
| `--allow-holes` | optional | permit holes in the hull |

#### `pyramids vector geom make-valid`

repair invalid geometries

```text
pyramids vector geom make-valid [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector geom minimum-bounding-circle`

the smallest enclosing circle

```text
pyramids vector geom minimum-bounding-circle [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector geom rotate`

rotate geometries about an origin

```text
pyramids vector geom rotate [-h] [--layer LAYER] [--overwrite] --angle ANGLE [--origin ORIGIN] [--radians] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--angle` | **required** | rotation angle |
| `--origin` | optional | rotation origin |
| `--radians` | optional | interpret --angle as radians |

#### `pyramids vector geom scale`

scale geometries about their center

```text
pyramids vector geom scale [-h] [--layer LAYER] [--overwrite] [--xfact XFACT] [--yfact YFACT] [--zfact ZFACT] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--xfact` | optional | X scale factor |
| `--yfact` | optional | Y scale factor |
| `--zfact` | optional | Z scale factor |

#### `pyramids vector geom simplify`

Douglas-Peucker simplification

```text
pyramids vector geom simplify [-h] [--layer LAYER] [--overwrite] --tolerance TOLERANCE [--no-preserve-topology] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--tolerance` | **required** | simplify tolerance |
| `--no-preserve-topology` | optional | allow topology to change (faster) |

#### `pyramids vector geom translate`

shift geometries by an offset

```text
pyramids vector geom translate [-h] [--layer LAYER] [--overwrite] [--xoff XOFF] [--yoff YOFF] [--zoff ZOFF] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it already exists |
| `--xoff` | optional | X offset |
| `--yoff` | optional | Y offset |
| `--zoff` | optional | Z offset |

## Subsystem `vector inspect`

#### `pyramids vector inspect count-coordinates`

per-feature coordinate count

```text
pyramids vector inspect count-coordinates [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect count-geometries`

per-feature geometry count

```text
pyramids vector inspect count-geometries [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect count-interior-rings`

per-feature interior-ring count

```text
pyramids vector inspect count-interior-rings [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect estimate-utm`

estimate the best UTM CRS

```text
pyramids vector inspect estimate-utm [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect feature-count`

count features

```text
pyramids vector inspect feature-count [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect feature-info`

metadata read without geometry

```text
pyramids vector inspect feature-info [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect invalid-coverage-edges`

write the invalid coverage edges

```text
pyramids vector inspect invalid-coverage-edges [-h] [--layer LAYER] [--gap-width GAP_WIDTH] [--driver DRIVER] [--overwrite] src out
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `out` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--gap-width` | optional | coverage gap-width tolerance in CRS units |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector inspect is-valid-coverage`

whether polygons form a valid coverage

```text
pyramids vector inspect is-valid-coverage [-h] [--layer LAYER] [--gap-width GAP_WIDTH] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--gap-width` | optional | coverage gap-width tolerance in CRS units |

#### `pyramids vector inspect is-valid-reason`

per-feature geometry-validity reason

```text
pyramids vector inspect is-valid-reason [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--json` | optional | emit JSON |

#### `pyramids vector inspect list-layers`

list the vector-layer names

```text
pyramids vector inspect list-layers [-h] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

## Subsystem `vector io`

#### `pyramids vector io from-bbox`

write a one-feature bbox polygon

```text
pyramids vector io from-bbox [-h] --bbox MINX MINY MAXX MAXY --epsg EPSG [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--bbox` | **required** | bounding box as MINX MINY MAXX MAXY |
| `--epsg` | **required** | EPSG code of the bbox |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io from-features`

build an FC from a GeoJSON features file

```text
pyramids vector io from-features [-h] [--crs CRS] [--driver DRIVER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source JSON path (FeatureCollection or feature list) |
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--crs` | optional | CRS to attach to the result (EPSG int, 'EPSG:4326', WKT, PROJ4) |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io from-featureserver`

read an Esri FeatureServer layer

```text
pyramids vector io from-featureserver [-h] --url URL [--where WHERE] [--out-fields OUT_FIELDS] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--url` | **required** | FeatureServer layer URL |
| `--where` | optional | attribute filter |
| `--out-fields` | optional | comma-separated fields |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io from-ogc-features`

read an OGC API Features service

```text
pyramids vector io from-ogc-features [-h] --endpoint ENDPOINT --collection COLLECTION [--bbox MINX MINY MAXX MAXY] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--endpoint` | **required** | service base URL |
| `--collection` | **required** | collection id |
| `--bbox` | optional | bounding box as MINX MINY MAXX MAXY |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io from-records`

build an FC from a CSV with a WKT geometry column

```text
pyramids vector io from-records [-h] [--geometry GEOMETRY] [--crs CRS] [--driver DRIVER] [--overwrite] csv output
```

| argument | | description |
|---|---|---|
| `csv` | required | source CSV path |
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--geometry` | optional | name of the WKT geometry column |
| `--crs` | optional | CRS to attach to the result (EPSG int, 'EPSG:4326', WKT, PROJ4) |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io from-vectortileserver`

read a vector-tile service

```text
pyramids vector io from-vectortileserver [-h] --url URL [--layer LAYER] [--bbox MINX MINY MAXX MAXY] [--overwrite] output
```

| argument | | description |
|---|---|---|
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--url` | **required** | vector-tile service URL |
| `--layer` | optional | layer name |
| `--bbox` | optional | bounding box as MINX MINY MAXX MAXY |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io from-xyz`

build a point layer from CSV x/y/z columns

```text
pyramids vector io from-xyz [-h] --x X --y Y [--z Z] [--crs CRS] [--z-column Z_COLUMN] [--driver DRIVER] [--overwrite] csv output
```

| argument | | description |
|---|---|---|
| `csv` | required | source CSV path |
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--x` | **required** | name of the X-coordinate column |
| `--y` | **required** | name of the Y-coordinate column |
| `--z` | optional | name of the optional value column |
| `--crs` | optional | CRS to attach to the result (EPSG int, 'EPSG:4326', WKT, PROJ4) |
| `--z-column` | optional | name to store --z under |
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io read-gpx`

write every non-empty GPX sub-layer to a multi-layer output

```text
pyramids vector io read-gpx [-h] [--driver DRIVER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source .gpx path |
| `output` | required | destination multi-layer path (e.g. .gpkg) |

| option | | description |
|---|---|---|
| `--driver` | optional | OGR driver (default: inferred from extension) |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io read-parquet`

convert a GeoParquet to a vector file

```text
pyramids vector io read-parquet [-h] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source GeoParquet path |
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-feather`

write a GeoFeather file

```text
pyramids vector io to-feather [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination .feather path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-geo-dict`

print the GeoJSON mapping

```text
pyramids vector io to-geo-dict [-h] [--layer LAYER] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--json` | optional | emit JSON |

#### `pyramids vector io to-json`

emit the GeoJSON string (stdout or --out)

```text
pyramids vector io to-json [-h] [--layer LAYER] [--out OUT] [--overwrite] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--out` | optional | write to this path instead of stdout |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-mvt`

write a Mapbox Vector Tiles archive

```text
pyramids vector io to-mvt [-h] [--layer LAYER] [--min-zoom MIN_ZOOM] [--max-zoom MAX_ZOOM] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination .mvt path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--min-zoom` | optional | minimum zoom level |
| `--max-zoom` | optional | maximum zoom level |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-parquet`

write a GeoParquet file

```text
pyramids vector io to-parquet [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination .parquet path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-pmtiles`

write a PMTiles archive

```text
pyramids vector io to-pmtiles [-h] [--layer LAYER] [--min-zoom MIN_ZOOM] [--max-zoom MAX_ZOOM] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination .pmtiles path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--min-zoom` | optional | minimum zoom level |
| `--max-zoom` | optional | maximum zoom level |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-wkb`

write a WKB table to CSV

```text
pyramids vector io to-wkb [-h] [--layer LAYER] [--hex] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination .csv path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--hex` | optional | encode geometries as hex |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector io to-wkt`

write a WKT table to CSV

```text
pyramids vector io to-wkt [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination .csv path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name or index in a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |

## Subsystem `vector overlay`

#### `pyramids vector overlay clip-by-rect`

clip to an axis-aligned window

```text
pyramids vector overlay clip-by-rect [-h] [--layer LAYER] [--overwrite] --rect XMIN YMIN XMAX YMAX src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--rect` | **required** | clip window in the source CRS |

#### `pyramids vector overlay difference`

difference with --other

```text
pyramids vector overlay difference [-h] [--layer LAYER] [--overwrite] --other OTHER src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | second vector path (the right-hand operand) |

#### `pyramids vector overlay dissolve`

merge features, optionally grouped

```text
pyramids vector overlay dissolve [-h] [--layer LAYER] [--overwrite] [--by BY] [--aggfunc AGGFUNC] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--by` | optional | column to group by before dissolving |
| `--aggfunc` | optional | aggregation for the non-geometry columns |

#### `pyramids vector overlay explode`

split multi-part geometries into parts

```text
pyramids vector overlay explode [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector overlay filter`

keep features matching a mask predicate

```text
pyramids vector overlay filter [-h] [--layer LAYER] [--overwrite] --predicate {contains,within,intersects,covers,covered_by,crosses,disjoint,overlaps,touches,dwithin} --mask MASK [--distance DISTANCE] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--predicate` | **required** | spatial predicate (choices: `contains`, `within`, `intersects`, `covers`, `covered_by`, `crosses`, `disjoint`, `overlaps`, `touches`, `dwithin`) |
| `--mask` | **required** | vector whose geometry is the mask |
| `--distance` | optional | distance for the 'dwithin' predicate |

#### `pyramids vector overlay intersection`

intersection with --other

```text
pyramids vector overlay intersection [-h] [--layer LAYER] [--overwrite] --other OTHER src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | second vector path (the right-hand operand) |

#### `pyramids vector overlay intersection-all`

intersection of all features

```text
pyramids vector overlay intersection-all [-h] [--layer LAYER] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |

#### `pyramids vector overlay overlay`

spatial overlay of two layers

```text
pyramids vector overlay overlay [-h] [--layer LAYER] [--overwrite] --right RIGHT [--how {intersection,union,identity,symmetric_difference,difference}] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--right` | **required** | right-hand vector path to combine with the source |
| `--how` | optional | overlay mode (choices: `intersection`, `union`, `identity`, `symmetric_difference`, `difference`) |

#### `pyramids vector overlay sjoin`

predicate-based spatial join

```text
pyramids vector overlay sjoin [-h] [--layer LAYER] [--overwrite] --right RIGHT [--how {inner,left,right}] [--predicate {intersects,contains,within,covers,covered_by,crosses,overlaps,touches}] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--right` | **required** | right-hand vector path to combine with the source |
| `--how` | optional | join type (choices: `inner`, `left`, `right`) |
| `--predicate` | optional | spatial predicate for the join (choices: `intersects`, `contains`, `within`, `covers`, `covered_by`, `crosses`, `overlaps`, `touches`) |

#### `pyramids vector overlay sjoin-nearest`

nearest-neighbour spatial join

```text
pyramids vector overlay sjoin-nearest [-h] [--layer LAYER] [--overwrite] --right RIGHT [--how {inner,left,right}] [--max-distance MAX_DISTANCE] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--right` | **required** | right-hand vector path to combine with the source |
| `--how` | optional | join type (choices: `inner`, `left`, `right`) |
| `--max-distance` | optional | only join within this distance |

#### `pyramids vector overlay symmetric-difference`

symmetric difference with --other

```text
pyramids vector overlay symmetric-difference [-h] [--layer LAYER] [--overwrite] --other OTHER src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | second vector path (the right-hand operand) |

#### `pyramids vector overlay union`

union with --other

```text
pyramids vector overlay union [-h] [--layer LAYER] [--overwrite] --other OTHER src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--other` | **required** | second vector path (the right-hand operand) |

#### `pyramids vector overlay union-all`

the single union of all features

```text
pyramids vector overlay union-all [-h] [--layer LAYER] [--overwrite] [--method {unary,coverage}] [--grid-size GRID_SIZE] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for a multi-layer source |
| `--overwrite` | optional | replace the output if it already exists |
| `--method` | optional | union algorithm (choices: `unary`, `coverage`) |
| `--grid-size` | optional | snapping grid size for the union |

## Subsystem `vector tess`

#### `pyramids vector tess constrained-delaunay-triangles`

constrained Delaunay triangulation of each polygon

```text
pyramids vector tess constrained-delaunay-triangles [-h] [--layer LAYER] [--overwrite] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector tess delaunay-triangles`

Delaunay triangulation

```text
pyramids vector tess delaunay-triangles [-h] [--layer LAYER] [--overwrite] [--tolerance TOLERANCE] [--only-edges] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--tolerance` | optional | snap input coordinates to this grid before tessellating |
| `--only-edges` | optional | return the tessellation edges as lines instead of polygons |

#### `pyramids vector tess distance`

planar distance to --other

```text
pyramids vector tess distance [-h] --other OTHER [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--other` | **required** | second vector file, compared element-wise against the source |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector tess fishnet`

a grid of square cell polygons

```text
pyramids vector tess fishnet [-h] --bounds MINX MINY MAXX MAXY --cell-size CELL_SIZE --epsg EPSG [--overwrite] dst
```

| argument | | description |
|---|---|---|
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--bounds` | **required** | extent the grid covers, in the units of --epsg |
| `--cell-size` | **required** | side length of each square cell |
| `--epsg` | **required** | EPSG code for the grid |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector tess geodesic-area`

ellipsoidal area per feature

```text
pyramids vector tess geodesic-area [-h] [--layer LAYER] [--unit UNIT] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--unit` | optional | m2 (default), km2 or ha |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector tess geodesic-distance`

ground distance to --other

```text
pyramids vector tess geodesic-distance [-h] --other OTHER [--unit UNIT] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--other` | **required** | second vector file, compared element-wise against the source |
| `--unit` | optional | m (default), km, mi or nmi |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector tess geodesic-length`

ellipsoidal length per feature

```text
pyramids vector tess geodesic-length [-h] [--unit UNIT] [--json] src
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |

| option | | description |
|---|---|---|
| `--unit` | optional | m (default), km, mi or nmi |
| `--json` | optional | emit the raw per-feature values as JSON |

#### `pyramids vector tess get-coordinates`

the vertex coordinate table

```text
pyramids vector tess get-coordinates [-h] [--json] [--overwrite] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination CSV path |

| option | | description |
|---|---|---|
| `--json` | optional | emit the raw per-feature values as JSON |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector tess h3-bin`

aggregate points into H3 hexagon cells

```text
pyramids vector tess h3-bin [-h] [--layer LAYER] [--overwrite] --resolution RESOLUTION [--agg AGG] [--column COLUMN] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--resolution` | **required** | H3 resolution, 0 (coarsest) to 15 (finest) |
| `--agg` | optional | per-cell reducer applied to --column |
| `--column` | optional | numeric column to aggregate (default: count) |

#### `pyramids vector tess interpolate-to-raster`

grid a point column onto a raster via IDW

```text
pyramids vector tess interpolate-to-raster [-h] [--layer LAYER] --column COLUMN [--method METHOD] [--cell-size CELL_SIZE] [--bounds MINX MINY MAXX MAXY] [--power POWER] [--n-neighbors N_NEIGHBORS] [--nodata NODATA] [--overwrite] src output
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `output` | required | destination raster path (GeoTIFF) |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--column` | **required** | numeric column to interpolate |
| `--method` | optional | interpolation method (only idw) |
| `--cell-size` | optional | output pixel size (CRS units) |
| `--bounds` | optional | output extent (default: the points' total bounds) |
| `--power` | optional | IDW distance exponent |
| `--n-neighbors` | optional | limit each estimate to the nearest N points |
| `--nodata` | optional | value for un-interpolable cells |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids vector tess polygonize`

build polygons from noded line work

```text
pyramids vector tess polygonize [-h] [--layer LAYER] [--overwrite] [--full] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--full` | optional | also report cut/dangle/invalid counts |

#### `pyramids vector tess quadtree`

adaptive quad-tree binning of points

```text
pyramids vector tess quadtree [-h] [--layer LAYER] [--overwrite] [--column COLUMN] [--agg AGG] [--nmax NMAX] [--nmin NMIN] [--clip CLIP] [--driver DRIVER] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--column` | optional | numeric column to aggregate (default: count) |
| `--agg` | optional | per-cell reducer applied to --column |
| `--nmax` | optional | max points in a cell before it splits |
| `--nmin` | optional | drop cells with fewer than this many points |
| `--clip` | optional | boundary vector file each cell is intersected with |
| `--driver` | optional | OGR driver (default: inferred from extension) |

#### `pyramids vector tess sample-points`

random points within each geometry

```text
pyramids vector tess sample-points [-h] [--layer LAYER] [--overwrite] --size SIZE [--method METHOD] [--seed SEED] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--size` | **required** | points per geometry |
| `--method` | optional | sampling method |
| `--seed` | optional | random seed for reproducibility |

#### `pyramids vector tess to-h3`

attach each point's H3 cell index

```text
pyramids vector tess to-h3 [-h] [--layer LAYER] [--overwrite] --resolution RESOLUTION src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--resolution` | **required** | H3 resolution, 0 (coarsest) to 15 (finest) |

#### `pyramids vector tess voronoi`

Thiessen tessellation with optional values/clip

```text
pyramids vector tess voronoi [-h] [--layer LAYER] [--overwrite] [--values VALUES] [--clip CLIP] [--driver DRIVER] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--values` | optional | column copied onto each cell (choropleth) |
| `--clip` | optional | boundary vector file each cell is intersected with |
| `--driver` | optional | OGR driver (default: inferred from extension) |

#### `pyramids vector tess voronoi-polygons`

Voronoi cells of the inputs

```text
pyramids vector tess voronoi-polygons [-h] [--layer LAYER] [--overwrite] [--tolerance TOLERANCE] [--only-edges] src dst
```

| argument | | description |
|---|---|---|
| `src` | required | source vector path |
| `dst` | required | destination vector path |

| option | | description |
|---|---|---|
| `--layer` | optional | layer name/index for multi-layer sources |
| `--overwrite` | optional | replace the output if it exists |
| `--tolerance` | optional | snap input coordinates to this grid before tessellating |
| `--only-edges` | optional | return the tessellation edges as lines instead of polygons |
