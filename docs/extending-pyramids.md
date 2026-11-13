# Extending pyramids

pyramids is extensible through **typed plugin seams**. A separate distribution can add new raster sources, new
vector or mesh formats, or new renderers without any change to pyramids itself. Discovery goes through Python
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/): pyramids scans a named
group with `importlib.metadata.entry_points(group=...)` the first time it needs it, and loads only what it finds.

The coupling runs **one way** — your package imports pyramids' driver types and builds on them; pyramids knows
only the group name and never imports a plugin. A bare pyramids install therefore behaves exactly as before, and
an optional backend is imported only when a source actually needs it.

Already-published plugins are listed in the [Plugins & extensions catalog](plugins.md).

## The plugin seams

Each seam is an entry-point group with a small, versioned contract and (for the driver seams) a **conformance
kit** you can assert against in your own test suite. Pick the one that matches what you are adding:

| You want to add… | Entry-point group | Contract type | Conformance kit | Guide |
|------------------|-------------------|---------------|-----------------|-------|
| A raster source GDAL can't open (an asset id, a scene directory, a scheme) | `pyramids.raster_drivers` | `RasterDriver` protocol | `pyramids.dataset.driver_conformance.assert_raster_driver_conforms` | [Writing a raster driver](reference/dataset/writing-a-raster-driver.md) |
| A vector format | `pyramids.vector_drivers` | `VectorDriver` protocol | `pyramids.feature.driver_conformance.assert_vector_driver_conforms` | [Writing a vector driver](reference/feature/writing-a-vector-driver.md) |
| An unstructured-mesh format | `pyramids.mesh_drivers` | `MeshDriver` protocol | — | [Writing a mesh driver](reference/netcdf/ugrid/writing-a-mesh-driver.md) |
| A non-matplotlib renderer for `plot(backend=...)` | `pyramids.plot_backends` | plot-backend protocol | `pyramids.base.plot_backend_conformance.assert_plot_backend_conforms` | [Writing a plot backend](reference/writing-a-plot-backend.md) |

The renderers a plugin registers show up in [Plot backends](reference/plot-backends.md) once it is installed.

A fifth seam, `pyramids.compute_backend`, is different in kind: it is a **single-backend** hook (one engine serves
the whole lazy/dask layer, not a family of named drivers), contracted by the `ComputeBackend` protocol in
`pyramids.base`. It is filled by `pyramids-lazy` (see the [Plugins & extensions catalog](plugins.md)); replacing
it is an advanced, rarely needed task, so it has no author guide of its own — read the protocol and the plugin's
source if you need to.

## How registration works

A plugin declares its entry points in its own `pyproject.toml` under the matching group. For example, a package
that adds an Earth Engine raster driver:

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.raster_drivers"]
earthengine = "your_package.drivers:EarthEngineDriver"
```

The left-hand name (`earthengine`) is the driver's public name — the value a caller passes to `driver=`, and the
name that must be unique within the group. The right-hand side points at the class (or callable, for a plot
backend) that implements the contract. After the package is installed in the same environment, pyramids finds it
automatically; there is nothing to import and no registry call to make.

Each driver guide gives the full contract — the required class attributes, the `can_open` / capability rules, the
API-version constant to match, and how ties with GDAL's built-in drivers are resolved.

## Runtime extension: the processing registry

The entry-point seams above are resolved at import time from installed packages. pyramids also has an
**in-process** extension point for geoprocessing tools: `pyramids.processing.register(...)` adds a tool to the
processing catalog at runtime, so a `Pipeline` can resolve and run it by name. This is the right seam for a tool
you compose in your own code or notebook rather than ship as an installable driver — see
[Processing pipelines](reference/processing.md) and
[ADR-0007: processing registry approach](adr/0007-processing-registry-approach.md).

## Publishing your plugin

1. Generate the package from the shared cookiecutter so it gets the same CI, docs, and release pipeline as the
   rest of the stack, and depend on `pyramids-core`.
2. Implement the contract for your seam and assert against its conformance kit in your tests.
3. Declare your entry points in `pyproject.toml` as shown above.
4. Ship your own documentation site (the repos in the catalog all publish to
   `serapeum-org.github.io/<your-repo>`); keep your full API reference there.
5. Open a pull request on [`serapeum-org/pyramids-core`](https://github.com/serapeum-org/pyramids-core) adding
   your package to the [Plugins & extensions catalog](plugins.md) so users can find it.
