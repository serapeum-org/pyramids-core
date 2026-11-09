# Geometry Helpers

Pure functions for building and inspecting shapely geometries. These
sit on `pyramids.feature.geometry` and are re-exposed as
`FeatureCollection` static methods for convenience (e.g.
`FeatureCollection.create_polygon` is a thin delegate to
`pyramids.feature.geometry.create_polygon`).

## Factories

Create shapely geometries from coordinate sequences. ARC-15 split these
by return type so each function has a single, unambiguous output:

| Function | Returns |
|----------|---------|
| `create_polygon(coords)` | `shapely.Polygon` |
| `polygon_wkt(coords)` | WKT `str` |
| `create_points(coords)` | `list[shapely.Point]` |
| `point_collection(coords, crs)` | `geopandas.GeoDataFrame` |

::: pyramids.feature.geometry.create_polygon
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.polygon_wkt
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.create_points
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.point_collection
    options:
        show_root_heading: true
        heading_level: 3

## Coordinate Extraction

Per-geometry coordinate access goes through `GeometryCoords`, which exposes
`.x` / `.y` / `.xy` and dispatches by geometry type internally (checking
`coord_type` lazily, before touching a degenerate geometry's attributes). The
low-level per-type extractors are private implementation details.

## Multi-geometry Handling

Per-part coordinates of any geometry are available through `GeometryCoords`,
which dispatches by type (including `GeometryCollection`). To split multi-part
frames into one-geometry-per-row, use `FeatureCollection.explode`.
