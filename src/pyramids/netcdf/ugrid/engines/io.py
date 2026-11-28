"""I/O engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.io``).

Owns the serialisation / hand-off-to-disk family -- writing the mesh out (``to_file`` /
``to_bytes`` / ``to_zarr``) and the bridges that turn it into a different data model
(``to_dataset`` to a regular raster, ``rasterize_like`` onto an existing raster's grid, ``burn``
of vector geometry onto the faces). It mirrors the raster ``Dataset`` / ``NetCDF`` write surface;
the flat ``mesh.<name>`` methods remain as deprecating facades that delegate here.

This is the last and most import-entangled of the mesh engines. ``to_file`` / ``to_bytes`` /
``to_dataset`` lean on the module-level NetCDF writer plumbing in ``ugrid.dataset``
(``write_ugrid_topology`` / ``write_ugrid_data_variable`` / ``_require_one_time_length`` /
``_write_time_coordinate``, the ``tempfile`` round trip, :class:`pyramids.dataset.Dataset` and
``mesh_to_grid``), which this engine cannot import without a cycle (``dataset`` imports this engine
for its ``io`` accessor), so each keeps its body on :class:`MeshDataset` as a ``_<name>_impl``
method and this engine delegates to it. ``to_zarr`` / ``rasterize_like`` / ``burn`` carry their
body here, pulling what they need from :mod:`~pyramids.netcdf.ugrid.interop_xarray`,
:mod:`~pyramids.netcdf.ugrid.regrid` and :mod:`~pyramids.netcdf.ugrid.vector_ops`, none of which
imports ``ugrid.dataset``; ``burn`` forwards to the already-migrated ``mesh.variables.with_variable``
engine so it does not re-emit the variables family's deprecation warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.netcdf.ugrid import interop_xarray as _interop
from pyramids.netcdf.ugrid import regrid as _regrid
from pyramids.netcdf.ugrid import vector_ops as _vector_ops
from pyramids.netcdf.ugrid.engines._base import _MeshEngine

if TYPE_CHECKING:
    from pathlib import Path

    from pyramids.dataset import Dataset


class _MeshIO(_MeshEngine):
    """Serialisation / hand-off ops for a mesh (``mesh.io``)."""

    __slots__ = ()

    def to_file(
        self, path: str | Path, driver: str | None = None, **options: Any
    ) -> None:
        """Write the dataset to a file.

        Writes UGRID-compliant NetCDF by default: a topology variable, node coordinates,
        connectivity arrays, face/edge centres, data variables, and global attributes following
        the UGRID convention.

        With ``driver``, writes through a registered third-party driver instead — 2DM, Selafin,
        PLY and others are writable. The output format is never inferred from the suffix, so
        ``to_file("mesh.slf")`` still writes NetCDF; name the driver to change that.

        Args:
            path: Output file path.
            driver: Name of a registered **mesh** driver to write with, as declared by the
                plugin providing it (see :func:`~pyramids.netcdf.ugrid.mesh_drivers`) — not a
                GDAL driver short name, which is what ``driver`` means elsewhere in pyramids.
                ``None`` (the default) writes UGRID NetCDF through the built-in writer, and
                the output format is never inferred from ``path``'s suffix.
            **options: Format-specific settings for the driver, passed through uninterpreted
                (for example Selafin endianness, or PLY ascii versus binary). The built-in
                writer takes none, so passing any without a ``driver`` raises ``TypeError``
                rather than dropping them silently.

        Raises:
            DriverNotExistError: If ``driver`` names a driver that is not installed.
            UnsupportedMeshFormatError: If the named driver cannot write a mesh.
            TypeError: If ``options`` are given without a ``driver``.
        """
        return self._ds._to_file_impl(path, driver, **options)

    def to_bytes(self) -> bytes:
        """Serialise the dataset to an in-memory UGRID NetCDF buffer.

        Writes through a temporary file because GDAL's NetCDF driver
        does not support ``/vsimem/`` for multidimensional creation.

        Returns:
            bytes: The NetCDF file content, suitable for writing to
                cloud storage or embedding in a response payload.

        See Also:
            to_file: Write directly to a file on disk.
            read_file: Class method that reads from a file path or
                byte buffer.
        """
        return self._ds._to_bytes_impl()

    def to_dataset(
        self,
        variable_name: str,
        cell_size: float,
        method: str = "nearest",
        bounds: tuple[float, float, float, float] | None = None,
        epsg: int | None = None,
        nodata: float = -9999.0,
    ) -> Dataset:
        """Convert a mesh variable to a regular-grid Dataset.

        Interpolates mesh data onto a regular grid and returns a
        standard pyramids Dataset. This is the bridge between
        unstructured (UGRID) and structured (raster) worlds.

        Args:
            variable_name: Name of the data variable to rasterize.
            cell_size: Target grid cell size in coordinate units.
            method: Interpolation method ("nearest" or "linear").
            bounds: Target (xmin, ymin, xmax, ymax). Defaults to mesh bounds.
            epsg: Target EPSG code. Defaults to mesh CRS.
            nodata: No-data value for the output raster.

        Returns:
            pyramids Dataset with the interpolated data.

        Raises:
            KeyError: No variable is named ``variable_name``.
            ValueError: The variable exists but holds no data (nothing was loaded and it
                carries no lazy loader), so there is nothing to interpolate.
        """
        return self._ds._to_dataset_impl(
            variable_name,
            cell_size,
            method=method,
            bounds=bounds,
            epsg=epsg,
            nodata=nodata,
        )

    def to_zarr(self, store: Any, **kwargs: Any) -> Any:
        """Write the mesh and its variables to a zarr store, through xarray.

        Args:
            store: The zarr store (a path or a mapping).
            **kwargs: Passed to ``xarray.Dataset.to_zarr``.

        Returns:
            The value ``xarray.Dataset.to_zarr`` returns.

        Raises:
            OptionalPackageDoesNotExist: xarray or zarr is not installed.
        """
        return _interop.to_zarr(
            self._ds._mesh,
            self._ds._data_variables,
            store,
            mesh_name=self._ds.mesh_name,
            crs_wkt=self._ds.crs_wkt,
            **kwargs,
        )

    def rasterize_like(
        self,
        reference: Any,
        variable_name: str | None = None,
        *,
        method: str = "nearest",
        nodata: float = -9999.0,
        max_distance: float | None = None,
    ) -> Any:
        """Interpolate a variable onto an existing raster's grid.

        Where :meth:`to_dataset` builds its own grid from a cell size, this snaps to the
        geotransform, shape and CRS of a raster you already have, so the result is
        co-registered with it.

        Args:
            reference: The raster whose grid to match (a pyramids ``Dataset`` / ``NetCDF``).
            variable_name: The variable to interpolate, or ``None`` for the only one.
            method: ``"nearest"``, ``"linear"`` or ``"idw"``. Defaults to ``"nearest"``.
            nodata: The value for cells the mesh does not cover.
            max_distance: How far a ``"nearest"`` / ``"idw"`` sample may reach. ``None`` uses
                twice the longer of the reference's two cell edges. Defaults to ``None``.

        Returns:
            Dataset: A raster on ``reference``'s grid.

        Raises:
            ValueError: An unknown ``method``, or the mesh declares an EPSG while
                ``reference`` declares no CRS to reproject into.
        """
        var = self._ds._facade_variable("rasterize_like", variable_name)
        return _regrid.rasterize_like(
            self._ds._mesh,
            np.asarray(var.data),
            reference,
            location=var.location,
            method=method,
            nodata=nodata,
            source_epsg=self._ds.epsg,
            max_distance=max_distance,
        )

    def burn(
        self,
        geometry: Any,
        value: float,
        *,
        name: str = "burned",
        fill: float = float("nan"),
        all_touched: bool = True,
    ) -> Any:
        """Burn vector geometry onto the faces as a new variable.

        The mesh analogue of rasterising a vector layer: every face the geometry covers takes
        ``value``, the rest take ``fill``.

        Args:
            geometry: A shapely geometry, GeoDataFrame, FeatureCollection, or iterable of them.
            value: The value burned into the covered faces.
            name: The new variable's name. Defaults to ``"burned"``.
            fill: The value for the faces the geometry misses. Defaults to NaN.
            all_touched: Burn every face the geometry touches, rather than only those whose
                centroid falls inside. Defaults to True.

        Returns:
            MeshDataset: A new dataset carrying the burned variable.
        """
        values = _vector_ops.burn(
            self._ds._mesh,
            geometry,
            value,
            fill=fill,
            all_touched=all_touched,
        )
        # Declare the fill as no-data only when it is NaN. A NaN fill means "untouched", so
        # naming it is what makes an untouched face distinguishable once written. An explicit
        # numeric fill is a value the caller chose, not a sentinel: `burn(geom, 1.0, fill=0.0)`
        # is the 0/1 mask idiom, and declaring 0.0 as missing made `dropna` and every `skipna`
        # reduction discard the unburned faces.
        burned = np.asarray(values)
        declared = float(fill) if np.isnan(fill) else None
        return self._ds.variables.with_variable(
            name, burned, location="face", nodata=declared
        )
