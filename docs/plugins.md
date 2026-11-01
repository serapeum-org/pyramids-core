# Plugins & extensions

pyramids has a small **plugin system**: a separate package can teach the core new file formats, new data
sources, or new renderers by registering into one of pyramids' entry-point groups. Installing such a package
into the same environment is all it takes — pyramids discovers the plugin through Python
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/) at import time, with **no
import coupling**: pyramids never imports a plugin, and a bare install behaves exactly as before.

This page is the **catalog** of known plugins. Each one is maintained in its own repository and ships its own
documentation site — follow the **Docs** link for its full API and examples. To learn how to *write* one, see
[Extending pyramids](extending-pyramids.md).

| Plugin | Hook group | What it adds | Docs |
|--------|-----------|--------------|------|
| [`pyramids-eo`](https://github.com/serapeum-org/pyramids-eo) | `pyramids.raster_drivers` | Remote-sensing / Earth-observation tier — raster drivers for Google Earth Engine, Sentinel-2, SEVIRI and EUMETSAT FCI sources | [serapeum-org.github.io/pyramids-eo](https://serapeum-org.github.io/pyramids-eo) |
| [`pyramids-mesh`](https://github.com/serapeum-org/pyramids-mesh) | `pyramids.mesh_drivers` | Mesh file-format drivers — 2DM, Selafin, DFSU, XMDF, PLY, SWW, HEC-RAS, ADCIRC, SCHISM, GMSH and more | [serapeum-org.github.io/pyramids-mesh](https://serapeum-org.github.io/pyramids-mesh) |
| [`digitalearth`](https://github.com/serapeum-org/Digital-Earth) | `pyramids.plot_backends` | Geospatial visualization — `interactive`, `3d` and `web` plot backends for `Dataset.plot(backend=...)` | [serapeum-org.github.io/Digital-Earth](https://serapeum-org.github.io/Digital-Earth) |
| [`pyramids-citymodels`](https://github.com/serapeum-org/pyramids-citymodels) | `pyramids.vector_drivers` | 3D city-model vector drivers — CityJSON, CityGML and 3D Tiles | [serapeum-org.github.io/pyramids-citymodels](https://serapeum-org.github.io/pyramids-citymodels) |

## pyramids-eo — Earth-observation raster sources

`pyramids-eo` adds an EO tier on top of pyramids and registers raster drivers for sources GDAL cannot open on its
own: an Earth Engine asset id (`ee://…`), a Sentinel-2 `.SAFE` product, a SEVIRI granule, a EUMETSAT FCI `.nat`
file. Once it is installed, `Dataset.read_file` routes those sources through the plugin automatically.

```bash
pip install pyramids-eo
```

```python
from pyramids.dataset import Dataset

# resolved by the pyramids-eo "earthengine" driver, discovered via entry points
scene = Dataset.read_file("ee://COPERNICUS/S2_SR_HARMONIZED/20200701T...")
```

## pyramids-mesh — hydraulic & simulation mesh formats

`pyramids-mesh` registers UGRID mesh drivers for ~19 hydraulic-modelling and interchange formats (SMS 2DM, Telemac
Selafin, DHI DFSU, XMDF, Stanford PLY, ANUGA SWW, HEC-RAS, ADCIRC, SCHISM, GMSH, …). `UgridDataset.read_file`
picks the right driver from the file.

```bash
pip install pyramids-mesh
```

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.2dm")   # resolved by the pyramids-mesh "2dm" driver
```

## digitalearth — interactive, 3-D and web visualization

`digitalearth` is a geospatial visualization package that also exposes its renderers to pyramids as **plot
backends**. With it installed, `Dataset.plot(backend="interactive" | "3d" | "web")` draws through digitalearth
instead of the default matplotlib backend.

```bash
pip install digitalearth
```

```python
ds.plot(backend="interactive")   # digitalearth renderer, selected by entry-point name
```

## pyramids-citymodels — 3D city models

`pyramids-citymodels` registers vector drivers for 3D city-model formats (CityJSON, CityGML, OGC 3D Tiles), read
through `FeatureCollection.read_file`.

```bash
pip install pyramids-citymodels
```

```python
from pyramids.feature import FeatureCollection

city = FeatureCollection.read_file("city.json")   # resolved by the "cityjson" driver
```

## Adding your plugin to this catalog

Built a package that registers into one of pyramids' hooks? Follow [Extending pyramids](extending-pyramids.md)
to write and publish it, then open a pull request on
[`serapeum-org/pyramids-core`](https://github.com/serapeum-org/pyramids-core) adding a row to the table above and
a short section here. Keep the full API docs in your own repository's site; this page only needs the one-line
summary and the link.
