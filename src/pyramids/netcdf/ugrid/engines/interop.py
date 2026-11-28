"""Interop engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.interop``).

Owns the hand-off to the wider PyData stack -- ``to_xarray`` (UGRID-CF encoded), the tidy
``to_dataframe``, and its lazy ``to_dask_dataframe`` sibling. Mirrors the raster
``NetCDF.interop`` engine; the UGRID-CF marshalling lives in
:mod:`pyramids.netcdf.ugrid.interop_xarray` and the tidy-frame layout on ``MeshDataset`` itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from pyramids.base._utils import import_dask, require_optional
from pyramids.netcdf.ugrid import interop_xarray as _interop
from pyramids.netcdf.ugrid.engines._base import _MeshEngine


class _MeshInterop(_MeshEngine):
    """PyData hand-off for a mesh: ``to_xarray`` / ``to_dataframe`` / ``to_dask_dataframe``."""

    __slots__ = ()

    def to_xarray(self) -> Any:
        """Hand the mesh and its variables to xarray, in the UGRID-CF encoding.

        xarray is a **peer dependency**, imported only here (as the raster
        :meth:`pyramids.netcdf.NetCDF.to_xarray` does), so nothing here imports it at module
        scope. Note the parent ``pyramids.netcdf`` package does currently pull xarray and dask
        at import time for unrelated reasons, so that cost is not avoided in practice yet.

        Returns:
            xarray.Dataset: The mesh's UGRID-CF variables plus the data variables.

        Raises:
            OptionalPackageDoesNotExist: xarray is not installed.
        """
        return _interop.to_xarray(
            self._ds._mesh,
            self._ds._data_variables,
            # The dataset's own topology name, not the default: exporting a mesh called
            # anything else wrote its topology as "mesh2d" while the data variables kept
            # their own element dimension, leaving two unrelated dimensions in one file.
            mesh_name=self._ds.mesh_name,
            crs_wkt=self._ds.crs_wkt,
            volume_topology=self._ds._volume_topology,
        )

    def to_dataframe(
        self, variables: Sequence[str] | None = None, *, dropna: bool = False
    ) -> pd.DataFrame:
        """A tidy frame of the mesh values, one row per element (and step).

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.to_dataframe`, in the **tidy**
        shape rather than the raster ``(y, x)`` MultiIndex one: an unstructured element has no
        row/column address, so each row carries the element index, its centroid coordinates,
        the time step where the variable has one, and the value.

        Args:
            variables: Which data variables to include. ``None`` (the default) takes them all.
            dropna: Drop rows whose value is no-data. Defaults to False.

        Returns:
            pandas.DataFrame: Columns ``variable``, ``element``, ``x``, ``y``, ``layer``,
            ``time`` and ``value``. ``element`` indexes the facet the variable sits on and
            ``x``/``y`` are that facet's coordinates, so a node variable is addressed by node.
            ``layer`` is the layer coordinate, or None for an unlayered variable, and ``time``
            the step coordinate, or None for a static one.

        Raises:
            KeyError: A requested variable is not a data variable of this dataset.
        """
        names = list(self._ds._data_variables) if variables is None else list(variables)
        frames: list[pd.DataFrame] = []
        for name in names:
            if name not in self._ds._data_variables:
                raise KeyError(name)
            var = self._ds._data_variables[name]
            if var.data is None:
                continue
            frames.extend(self._ds._variable_frames(name, var))
        if not frames:
            # Match the populated frame's dtypes, so a caller that concatenates or
            # dtype-checks sees one schema either way.
            return pd.DataFrame(
                {
                    "variable": pd.Series(dtype="str"),
                    "element": pd.Series(dtype="int64"),
                    "x": pd.Series(dtype="float64"),
                    "y": pd.Series(dtype="float64"),
                    "layer": pd.Series(dtype="object"),
                    "time": pd.Series(dtype="object"),
                    "value": pd.Series(dtype="float64"),
                }
            )
        frame = pd.concat(frames, ignore_index=True)
        if dropna:
            frame = frame[~frame["value"].isna()].reset_index(drop=True)
        return frame

    def to_dask_dataframe(
        self,
        variables: Sequence[str] | None = None,
        *,
        dropna: bool = False,
        npartitions: int = 1,
    ) -> Any:
        """Hand the mesh to dask in tidy form — the lazy sibling of :meth:`to_dataframe`.

        The mesh's own frame is already tidy (one row per element and step), so this is the
        same shape wrapped in a dask dataframe rather than a different layout.

        **What is and is not deferred:** a mesh's values live in memory, not in a per-element
        chunked store, so there is nothing to stream and the frame is built then partitioned.
        The result is a real dask graph — it composes with the rest of a dask pipeline and
        computes lazily from here on — but it does not reduce the peak memory of building it.
        That is the same thing :meth:`NetCDF.to_dask_dataframe` does for a cube with no
        on-disk store.

        Args:
            variables: Which data variables to include. ``None`` (default) takes them all.
            dropna: Drop rows whose value is no-data. Defaults to False.
            npartitions: How many partitions to split the frame into. Defaults to 1.

        Returns:
            dask.dataframe.DataFrame: The tidy frame as a dask dataframe.

        Raises:
            ImportError: dask (the ``lazy`` extra) or ``dask.dataframe`` (which needs pyarrow,
                the ``parquet`` extra) is not installed.
            ValueError: ``npartitions`` is below 1, or a named variable does not exist.

        See Also:
            to_dataframe: The eager pandas frame this wraps.
        """
        if npartitions < 1:
            raise ValueError(
                f"to_dask_dataframe() got npartitions={npartitions}; it must be at least 1."
            )
        import_dask("to_dask_dataframe() needs dask; install the `lazy` extra.")
        # dask.dataframe (dask-expr) imports pyarrow, which ships in the `parquet` extra rather
        # than `lazy`; guard it so a lazy-only install gets a hint, not a ModuleNotFoundError.
        require_optional(
            "dask.dataframe",
            "to_dask_dataframe() needs dask.dataframe, which requires pyarrow — install the "
            "`parquet` extra (e.g. `pip install pyramids-core[parquet]`).",
        )
        import dask.dataframe as dd

        frame = self.to_dataframe(variables, dropna=dropna)
        return dd.from_pandas(frame, npartitions=npartitions)
