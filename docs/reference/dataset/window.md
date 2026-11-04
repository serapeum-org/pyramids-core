# Window

A first-class pixel-space window for raster reads and writes (x-first `col_off, row_off, cols, rows`), accepted
everywhere pyramids takes a window. Build one from map-space bounds with
[`from_bounds`](#pyramids.dataset.window.Window.from_bounds), or from a geometry's bounding box with
[`covering`](#pyramids.dataset.window.Window.covering).

::: pyramids.dataset.window.Window
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
