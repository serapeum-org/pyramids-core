# Configuration

Process-wide and scoped GDAL configuration helpers, exported at the `pyramids` top level.

- [`configure`](#pyramids.base._configure.configure) applies GDAL options process-wide and permanently (and,
  with a dask client, to every worker).
- [`gdal_config`](#pyramids.base._configure.gdal_config) applies options only for the duration of a `with`
  block and restores the previous values on exit — a scoped GDAL environment. It is applied process-wide (so
  options like `GDAL_CACHEMAX` work) and is therefore **not safe to use from multiple threads concurrently** —
  use it from a single thread or guard it with your own lock.

::: pyramids.base._configure.configure
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.base._configure.gdal_config
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.base._configure.configure_lazy_vector
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
