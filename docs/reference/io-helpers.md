# In-memory rasters & array layout helpers

Top-level I/O and array utilities exported from `pyramids`.

## MemoryFile

An in-memory, `/vsimem`-backed raster file handle. Stage a raster by handing it encoded bytes
(`MemoryFile(ds.to_bytes())` or `write()`), open it as a `Dataset`, or read its bytes back; the buffer is
unlinked on exit.

::: pyramids._io.MemoryFile
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

## Array layout helpers

Convert between the band-first `(bands, rows, cols)` layout GDAL and pyramids use and the band-last
`(rows, cols, bands)` layout image tooling expects. Both return a view (no copy).

::: pyramids.base._utils.reshape_as_image
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base._utils.reshape_as_raster
    options:
        show_root_heading: true
        heading_level: 3
