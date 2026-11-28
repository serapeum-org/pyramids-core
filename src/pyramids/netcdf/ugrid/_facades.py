"""The capability facades `MeshDataset` exposes, as a mixin.

The algorithms live in focused modules — :mod:`~pyramids.netcdf.ugrid.graph` (topology and
graph operations), :mod:`~pyramids.netcdf.ugrid.fill` (spatial gap filling),
:mod:`~pyramids.netcdf.ugrid.vector_ops` (mesh↔vector), :mod:`~pyramids.netcdf.ugrid.regrid`
(sampling and regridding) and :mod:`~pyramids.netcdf.ugrid.interop_xarray` (xarray / zarr /
dask hand-off). Each takes a :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d` plus plain arrays, so
none of them imports ``ugrid.dataset`` and the import graph stays acyclic.

This mixin is the user-facing half: it resolves a variable by name, hands the kernel the mesh
and the values, and wraps the answer back into a dataset / variable / GeoDataFrame. Keeping the
facades here rather than in ``dataset.py`` keeps that module about the data model.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from pyramids.base._array_ops import nodata_mask
from pyramids.netcdf.ugrid import analysis as _analysis
from pyramids.netcdf.ugrid import regrid as _regrid
from pyramids.netcdf.ugrid.messages import unknown_variable_message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.netcdf.ugrid.engines.analysis import _MeshAnalysis
    from pyramids.netcdf.ugrid.engines.glyph import _MeshGlyph
    from pyramids.netcdf.ugrid.engines.interop import _MeshInterop
    from pyramids.netcdf.ugrid.engines.io import _MeshIO
    from pyramids.netcdf.ugrid.engines.selection import _MeshSelection
    from pyramids.netcdf.ugrid.engines.spatial import _MeshSpatial
    from pyramids.netcdf.ugrid.engines.terrain import _MeshTerrain
    from pyramids.netcdf.ugrid.engines.topology import _MeshTopology
    from pyramids.netcdf.ugrid.engines.variables import _MeshVariables
    from pyramids.netcdf.ugrid.engines.vector import _MeshVector
    from pyramids.netcdf.ugrid.models import MeshVariable

_KEEP = object()
"""``_rewrap``'s default for ``nodata``: carry the source's sentinel if the dtype holds it."""


def _representable(nodata: Any, dtype: np.dtype) -> bool:
    """Whether ``dtype`` can hold ``nodata`` as a distinguishable value.

    A facade that changes a variable's dtype must not carry the old no-data value over: a
    boolean mask declaring ``-9999.0`` promises a sentinel no element can equal, and NaN means
    nothing to an integer array.

    Three kinds of answer, in order:

    * **A boolean result refuses every declaration.** ``0.0`` and ``1.0`` survive any round
      trip through ``bool``, so a round-trip test kept them -- and a mask declaring ``1.0`` as
      missing marks every ``True`` face missing, which is the outcome this helper exists to
      prevent. A two-valued domain has no spare sentinel.
    * **A non-number is not representable**, and answering that is the point: the question is
      asked from inside a facade, so raising there would surface as a bare numpy ``TypeError``.
    * **Anything else round-trips through the dtype and back to its own type.** Comparing via
      ``float64`` instead could not show that ``int64``'s largest value is representable in
      ``int64``, because a double's 53-bit mantissa cannot hold it.

    Args:
        nodata: The declared no-data value, or ``None``.
        dtype: The dtype of the new data.

    Returns:
        bool: True when the declaration still means something, including for ``None``, which
        declares nothing and so is always safe to keep.
    """
    if nodata is None:
        return True
    if np.issubdtype(dtype, np.bool_):
        return False
    try:
        original = np.asarray(nodata)
        if not np.issubdtype(original.dtype, np.number):
            return False
        if bool(np.isnan(original)):
            # NaN is a real fill value for a float array and meaningless in an integer one.
            return bool(np.issubdtype(dtype, np.floating))
        if np.issubdtype(dtype, np.integer):
            # An integer dtype holds exactly the integers in its range, and the range has to
            # be tested directly: a round trip cannot see an out-of-range value, because the
            # two's-complement wrap is symmetric -- 2**63 cast to int64 and back to uint64
            # gives 2**63 again. Comparing as Python ints keeps it exact at the boundary,
            # where float64 cannot tell int64's largest value from one past it.
            info = np.iinfo(dtype)
            if np.issubdtype(original.dtype, np.integer):
                return bool(info.min <= int(original) <= info.max)
            as_float = float(original)
            if not as_float.is_integer():
                return False
            return bool(info.min <= int(as_float) <= info.max)
        with np.errstate(invalid="ignore", over="ignore"):
            # The cast is the question, so its failure modes are answers rather than
            # problems: a value past the dtype's range warns before producing the wrong
            # number this helper exists to reject. Leaving the warnings in place would make a
            # facade call noisy, and fatal under `-W error`.
            roundtrip = original.astype(dtype).astype(original.dtype)
        return bool(roundtrip == original)
    except (TypeError, ValueError, OverflowError):
        return False


def _axis_distances(stamps: list) -> np.typing.NDArray | None:
    """Time coordinates as distances along the axis, or ``None`` when they carry none.

    Numeric coordinates are already distances. Date-like ones are not, but they do carry a real
    spacing, so they are converted to nanoseconds since the epoch -- without that step a
    `float64` cast raises on a `Timestamp` and the whole coordinate reading silently degraded to
    step positions, which is the one case `use_coordinate=True` exists for.

    Args:
        stamps: The time coordinate values, in step order.

    Returns:
        np.ndarray: One float distance per step, or ``None`` for coordinates with no measurable
        spacing (plain labels, say), where an evenly spaced axis is the sensible reading.
    """
    # datetime64 is tested FIRST, because numpy casts it to float64 happily (as nanoseconds)
    # and that would take the numeric branch -- leaving a `datetime64[ns]` axis on a different
    # scale from the same instants held as Timestamps, which the pandas conversion below
    # normalises. A 1000x disagreement between two spellings of one coordinate silently turns
    # a valid target into all-NaN.
    as_array = np.asarray(stamps)
    if not np.issubdtype(as_array.dtype, np.datetime64):
        try:
            return np.asarray(stamps, dtype="float64")
        except (TypeError, ValueError):
            pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.DatetimeIndex(pd.to_datetime(list(stamps)))
    except (TypeError, ValueError):
        return None
    # Pinned to nanoseconds rather than read with `.view("int64")`, which exposes whatever
    # resolution pandas chose from the input -- microseconds for a Timestamp list, nanoseconds
    # for a `datetime64[ns]` array. One instant then had two numeric values 1000x apart
    # depending on how the caller happened to hold it, so a target converted by one route
    # landed outside an axis converted by the other and came back all-NaN.
    return np.asarray(
        converted.astype("datetime64[ns]").astype("int64"), dtype="float64"
    )


def _numeric_stamps(stamps: Any) -> np.ndarray:
    """Time coordinates as distances along the axis, or a written error when they carry none.

    Date-like coordinates go through the same conversion
    :func:`pyramids.netcdf.ugrid.dataset._axis_distances` gives `interpolate_na` --
    nanoseconds since the epoch -- so a mesh read from a dated file can be interpolated. It
    previously refused them, which made `interp_time` unusable on exactly the data `resample`
    handles, since `time_values` returns whatever the file put in the attribute.

    Routing every spelling through one converter also settles the scale. A `datetime64[ns]`
    array used to arrive as nanoseconds and a `pd.DatetimeIndex` as microseconds -- a factor of
    1000 -- so a ``new_times`` value valid against one landed outside the other's axis and gave
    all-NaN.

    Args:
        stamps: The time coordinate values.

    Returns:
        np.ndarray: One float distance per step. Date-like values are nanoseconds since the
        epoch, which is a scale ``new_times`` shares because it is converted the same way.

    Raises:
        ValueError: The coordinates are neither numeric nor date-like, so they carry no
            distance to interpolate along. A bare ``TypeError`` about `Timestamp` from inside
            numpy is not an answer.
    """
    # Handed over as they come. Converting to an object array first turned a `datetime64[ns]`
    # array into plain ints, which then satisfied the numeric branch and came out as
    # nanoseconds while every other spelling came out as microseconds -- the 1000x scale
    # disagreement this routing exists to remove.
    distances = _axis_distances(stamps)
    if distances is None:
        raise ValueError(
            "interp_time needs time coordinates it can measure along -- numbers, or dates it "
            "can read. This dataset's are neither, so there is no distance to interpolate "
            "between them."
        )
    return distances


class _MeshFacades:
    """Capability facades mixed into :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset`."""

    if TYPE_CHECKING:  # engine accessors provided by the MeshDataset subclass

        @property
        def analysis(self) -> _MeshAnalysis: ...

        @property
        def terrain(self) -> _MeshTerrain: ...

        @property
        def interop(self) -> _MeshInterop: ...

        @property
        def vector(self) -> _MeshVector: ...

        @property
        def glyph(self) -> _MeshGlyph: ...

        @property
        def variables(self) -> _MeshVariables: ...

        @property
        def topology(self) -> _MeshTopology: ...

        @property
        def spatial(self) -> _MeshSpatial: ...

        @property
        def selection(self) -> _MeshSelection: ...

        @property
        def io(self) -> _MeshIO: ...

    def _facade_variable(self, operation: str, variable_name: str | None) -> Any:
        """The variable ``operation`` acts on, defaulting to the only data variable.

        Most of these capabilities act on one field. Naming it is optional when the dataset
        carries exactly one, which is the common case for a derived mesh; with several, the
        name is required rather than guessed.

        Args:
            operation: The calling method's name, for the message.
            variable_name: The requested name, or ``None`` to take the only one.

        Returns:
            MeshVariable: The resolved variable.

        Raises:
            KeyError: The named variable is not in this dataset.
            ValueError: ``None`` was given but the dataset does not hold exactly one variable,
                or the resolved variable has no loaded data.
        """
        names = list(self._data_variables)  # type: ignore[attr-defined]
        if variable_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"{operation}: this dataset holds {len(names)} data variable(s) "
                    f"{names}; name the one to use."
                )
            variable_name = names[0]
        if variable_name not in names:
            raise KeyError(unknown_variable_message(operation, variable_name, names))
        var = self._data_variables[variable_name]  # type: ignore[attr-defined]
        if var.data is None:
            raise ValueError(
                f"{operation}: variable {variable_name!r} has no loaded data."
            )
        return var

    def _rewrap(
        self, var: MeshVariable, values: np.ndarray, *, nodata: Any = _KEEP
    ) -> Any:
        """A new dataset on the same mesh, with ``var`` replaced by ``values``.

        Args:
            var: The variable being replaced; everything but its data is carried over.
            values: The new data.
            nodata: The sentinel the result declares. :data:`_KEEP` (the default) carries the
                source's own sentinel onto the result when it is representable in the result's
                dtype -- the morphology / fill facades that return the same quantity. ``None``
                declares no sentinel, for a facade whose answer is a **different quantity**
                (a boolean mask, a cluster label). Any other value is the one the shared
                cell-wise kernel already worked out and snapped to the result's type, so it is
                taken as given rather than guessed from dtype representability.

        Returns:
            MeshDataset: A new dataset carrying ``values`` under ``var``'s name.
        """
        new_vars = dict(self._data_variables)  # type: ignore[attr-defined]
        array = np.asarray(values)
        replacement = var.with_data(array)
        if nodata is _KEEP:
            # A float sentinel like -9999.0 carried onto a boolean mask declares a
            # `_FillValue` no element of the result can equal, which `to_file` would then
            # stamp on the variable; drop it when the result's dtype cannot hold it.
            if not _representable(replacement.nodata, array.dtype):
                replacement.nodata = None
        else:
            # The kernel answered the sentinel (or `None` to declare none); it already fits
            # the result's dtype, so it is set directly rather than re-guessed.
            replacement.nodata = nodata
        new_vars[var.name] = replacement
        return self._rebuild(new_vars)  # type: ignore[attr-defined]

    def connected_components(self) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTopology.connected_components`."""
        warnings.warn(
            "`mesh.connected_components` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.connected_components`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.connected_components()

    def _focal_inputs(
        self, operation: str, variable_name: str | None
    ) -> tuple[Any, np.ndarray, Any]:
        """The variable, its values and its activity mask for a focal aggregation.

        Args:
            operation: The calling method's name, for messages.
            variable_name: Variable to read, or ``None`` for the only one.

        Returns:
            tuple: ``(variable, values, active)``, where ``active`` is ``None`` when the
            variable carries no activity mask.
        """
        var = self._facade_variable(operation, variable_name)
        values = np.asarray(var.data)
        active = None if var.active is None else np.asarray(var.active, dtype=bool)
        return var, values, active

    def focal_mean(
        self,
        rings: int = 1,
        variable_name: str | None = None,
        reach: Any = None,
    ) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshAnalysis.focal_mean`."""
        warnings.warn(
            "`mesh.focal_mean` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.focal_mean`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.focal_mean(
            rings=rings, variable_name=variable_name, reach=reach
        )

    def focal_std(
        self,
        rings: int = 1,
        variable_name: str | None = None,
        reach: Any = None,
    ) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshAnalysis.focal_std`."""
        warnings.warn(
            "`mesh.focal_std` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.focal_std`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.focal_std(
            rings=rings, variable_name=variable_name, reach=reach
        )

    def focal_apply(
        self,
        func: Any,
        rings: int = 1,
        variable_name: str | None = None,
        reach: Any = None,
    ) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshAnalysis.focal_apply`."""
        warnings.warn(
            "`mesh.focal_apply` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.focal_apply`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.focal_apply(
            func, rings=rings, variable_name=variable_name, reach=reach
        )

    def _elevation_nodes(self, operation: str, variable_name: str | None) -> np.ndarray:
        """One elevation per node for the terrain derivatives (issue #1374).

        The plane fit needs node elevations. A node-located variable supplies them directly; a
        face-located one -- which is where a bed level usually lives (``mesh2d_flowelem_bl``) --
        is averaged onto the nodes first, through the same helper the contouring path uses. An
        edge-located variable has no meaningful route to the nodes and is refused.

        Args:
            operation: The calling method's name, for the message.
            variable_name: The requested variable, or ``None`` for the only one.

        Note:
            A declared ``nodata`` is set to ``NaN`` here, so every derivative built on these
            elevations reports ``NaN`` for the faces that touch the void rather than treating
            the sentinel as a measurement.

        Returns:
            np.ndarray: One elevation per node, with voids as ``NaN``.

        Raises:
            ValueError: The variable is edge-located, or is temporal / layered rather than one
                value per element.
        """
        var = self._facade_variable(operation, variable_name)
        values = np.asarray(var.data, dtype="float64").copy()
        mesh = self._mesh  # type: ignore[attr-defined]
        if values.ndim != 1:
            raise ValueError(
                f"{operation}: expected one value per element -- a 1-D array, got shape "
                f"{values.shape}. Select a single step or layer first, e.g. with isel()."
            )
        # A declared sentinel is a void, not an elevation. Setting it to NaN hands it to the
        # plane fit's existing NaN propagation, so the faces touching it report NaN instead of
        # a fabricated cliff -- which is the contract the raster kernels document ("a centred
        # difference straddling a void has no defined derivative").
        values[nodata_mask(values, var.nodata)] = np.nan
        if var.location == "node":
            return values
        if var.location == "face":
            return _analysis._face_values_to_nodes(mesh, values)
        raise ValueError(
            f"{operation}: needs a node- or face-located elevation, not "
            f"{var.location!r}. An edge-located variable has no defined route to the nodes "
            f"the plane fit is built from."
        )

    def slope(
        self, units: str = "degrees", variable_name: str | None = None
    ) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTerrain.slope`."""
        warnings.warn(
            "`mesh.slope` is deprecated and will be removed in 0.90.0; use "
            "`mesh.terrain.slope`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.terrain.slope(units=units, variable_name=variable_name)

    def aspect(self, variable_name: str | None = None) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTerrain.aspect`."""
        warnings.warn(
            "`mesh.aspect` is deprecated and will be removed in 0.90.0; use "
            "`mesh.terrain.aspect`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.terrain.aspect(variable_name=variable_name)

    def hillshade(
        self,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        variable_name: str | None = None,
    ) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTerrain.hillshade`."""
        warnings.warn(
            "`mesh.hillshade` is deprecated and will be removed in 0.90.0; use "
            "`mesh.terrain.hillshade`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.terrain.hillshade(
            azimuth=azimuth, altitude=altitude, variable_name=variable_name
        )

    def cluster(
        self,
        lower: float,
        upper: float,
        variable_name: str | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshVector.cluster`."""
        warnings.warn(
            "`mesh.cluster` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.cluster`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.cluster(lower, upper, variable_name=variable_name)

    def footprint(self, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.footprint`."""
        warnings.warn(
            "`mesh.footprint` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.footprint`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.footprint(variable_name)

    def exterior_edges(self) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTopology.exterior_edges`."""
        warnings.warn(
            "`mesh.exterior_edges` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.exterior_edges`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.exterior_edges()

    def exterior_faces(self) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTopology.exterior_faces`."""
        warnings.warn(
            "`mesh.exterior_faces` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.exterior_faces`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.exterior_faces()

    def bounding_polygon(self) -> Any:
        """Facade -- delegates to :meth:`_MeshTopology.bounding_polygon`."""
        warnings.warn(
            "`mesh.bounding_polygon` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.bounding_polygon`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.bounding_polygon()

    def circumcenters(self) -> np.ndarray:
        """Facade -- delegates to :meth:`_MeshTopology.circumcenters`."""
        warnings.warn(
            "`mesh.circumcenters` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.circumcenters`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.circumcenters()

    def voronoi(self) -> Any:
        """Facade -- delegates to :meth:`_MeshVector.voronoi`."""
        warnings.warn(
            "`mesh.voronoi` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.voronoi`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.voronoi()

    def sparse_adjacency(self, kind: str = "face") -> Any:
        """Facade -- delegates to :meth:`_MeshTopology.sparse_adjacency`."""
        warnings.warn(
            "`mesh.sparse_adjacency` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.sparse_adjacency`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.sparse_adjacency(kind)

    def binary_dilation(
        self,
        variable_name: str | None = None,
        *,
        iterations: int = 1,
        border_value: bool = False,
        fixed: np.ndarray | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshTopology.binary_dilation`."""
        warnings.warn(
            "`mesh.binary_dilation` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.binary_dilation`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.binary_dilation(
            variable_name,
            iterations=iterations,
            border_value=border_value,
            fixed=fixed,
        )

    def binary_erosion(
        self,
        variable_name: str | None = None,
        *,
        iterations: int = 1,
        border_value: bool = False,
        fixed: np.ndarray | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshTopology.binary_erosion`."""
        warnings.warn(
            "`mesh.binary_erosion` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.binary_erosion`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.binary_erosion(
            variable_name,
            iterations=iterations,
            border_value=border_value,
            fixed=fixed,
        )

    def to_location(self, location: str, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.to_location`."""
        warnings.warn(
            "`mesh.to_location` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.to_location`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.to_location(location, variable_name)

    def fill_na_spatial(
        self, variable_name: str | None = None, *, max_distance: float | None = None
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.fill_na_spatial`."""
        warnings.warn(
            "`mesh.fill_na_spatial` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.fill_na_spatial`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.fill_na_spatial(variable_name, max_distance=max_distance)

    def laplace_interpolate(
        self,
        variable_name: str | None = None,
        *,
        use_weights: bool = True,
        direct_solve: bool = False,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.laplace_interpolate`."""
        warnings.warn(
            "`mesh.laplace_interpolate` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.laplace_interpolate`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.laplace_interpolate(
            variable_name, use_weights=use_weights, direct_solve=direct_solve
        )

    def polygonize(
        self, variable_name: str | None = None, *, decimals: int | None = None
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshVector.polygonize`."""
        warnings.warn(
            "`mesh.polygonize` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.polygonize`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.polygonize(variable_name, decimals=decimals)

    def sample_line(self, line: Any, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.sample_line`."""
        warnings.warn(
            "`mesh.sample_line` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.sample_line`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.sample_line(line, variable_name)

    def burn(
        self,
        geometry: Any,
        value: float,
        *,
        name: str = "burned",
        fill: float = float("nan"),
        all_touched: bool = True,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshIO.burn`."""
        warnings.warn(
            "`mesh.burn` is deprecated and will be removed in 0.90.0; use "
            "`mesh.io.burn`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.io.burn(  # type: ignore[attr-defined]
            geometry, value, name=name, fill=fill, all_touched=all_touched
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
        """Facade -- delegates to :meth:`_MeshIO.rasterize_like`."""
        warnings.warn(
            "`mesh.rasterize_like` is deprecated and will be removed in 0.90.0; use "
            "`mesh.io.rasterize_like`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.io.rasterize_like(  # type: ignore[attr-defined]
            reference,
            variable_name,
            method=method,
            nodata=nodata,
            max_distance=max_distance,
        )

    def interp_time(
        self, new_times: Any, *, method: str = "linear", bounds: str = "nan"
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshSelection.interp_time`."""
        warnings.warn(
            "`mesh.interp_time` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.interp_time`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.interp_time(new_times, method=method, bounds=bounds)

    def _interp_time_impl(
        self, new_times: Any, *, method: str = "linear", bounds: str = "nan"
    ) -> Any:
        """Implementation of :meth:`_MeshSelection.interp_time`; see that method for the contract."""
        stamps = self._time_coords("interp_time")  # type: ignore[attr-defined]
        targets = list(np.atleast_1d(np.asarray(new_times)))
        # `new_times` goes through the same converter as the existing axis, so a date target
        # is measured on the same scale as the dated coordinates it sits between. Casting it
        # with a bare `astype("float64")` was what made a `Timestamp` target unusable even
        # once the axis itself could be read.
        numeric_targets = _numeric_stamps(targets)
        new_vars: dict[str, Any] = {}
        for name, var in self._data_variables.items():  # type: ignore[attr-defined]
            if not var.has_time:
                new_vars[name] = var
                continue
            moved = _regrid.interp_time(
                _numeric_stamps(stamps),
                np.asarray(var.data),
                numeric_targets,
                method=method,
                axis=cast("int", var.time_axis_index),
                bounds=bounds,
            )
            new_var = var.with_data(np.asarray(moved))
            new_var.attributes = {**var.attributes, "time_values": targets}
            new_vars[name] = new_var
        return self._rebuild(new_vars)  # type: ignore[attr-defined]

    def regrid(
        self, target: Any, *, method: str = "overlap", variable_name: str | None = None
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshSpatial.regrid`."""
        warnings.warn(
            "`mesh.regrid` is deprecated and will be removed in 0.90.0; use "
            "`mesh.spatial.regrid`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.spatial.regrid(target, method=method, variable_name=variable_name)

    def to_xarray(self) -> Any:
        """Facade -- delegates to :meth:`_MeshInterop.to_xarray`."""
        warnings.warn(
            "`mesh.to_xarray` is deprecated and will be removed in 0.90.0; use "
            "`mesh.interop.to_xarray`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.interop.to_xarray()

    def to_zarr(self, store: Any, **kwargs: Any) -> Any:
        """Facade -- delegates to :meth:`_MeshIO.to_zarr`."""
        warnings.warn(
            "`mesh.to_zarr` is deprecated and will be removed in 0.90.0; use "
            "`mesh.io.to_zarr`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.io.to_zarr(store, **kwargs)  # type: ignore[attr-defined]

    def fillna(self, value: Any, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.fillna`."""
        warnings.warn(
            "`mesh.fillna` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.fillna`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.fillna(value, variable_name)

    def isnull(self, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.isnull`."""
        warnings.warn(
            "`mesh.isnull` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.isnull`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.isnull(variable_name)

    def notnull(self, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.notnull`."""
        warnings.warn(
            "`mesh.notnull` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.notnull`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.notnull(variable_name)

    def clip_values(
        self, lower: Any, upper: Any, variable_name: str | None = None
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.clip_values`."""
        warnings.warn(
            "`mesh.clip_values` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.clip_values`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.clip_values(lower, upper, variable_name)

    def round(self, decimals: int = 0, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.round`."""
        warnings.warn(
            "`mesh.round` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.round`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.round(decimals, variable_name)

    def astype(self, dtype: Any, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.astype`."""
        warnings.warn(
            "`mesh.astype` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.astype`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.astype(dtype, variable_name)

    def where(
        self,
        cond: Any,
        other: Any = np.nan,
        variable_name: str | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.where`."""
        warnings.warn(
            "`mesh.where` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.where`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.where(cond, other, variable_name)

    def apply(self, func: Any, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.apply`."""
        warnings.warn(
            "`mesh.apply` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.apply`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.apply(func, variable_name)

    def get_histogram(
        self,
        bins: int = 256,
        value_range: tuple[float, float] | None = None,
        variable_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Facade -- delegates to :meth:`_MeshAnalysis.get_histogram`."""
        warnings.warn(
            "`mesh.get_histogram` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.get_histogram`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.get_histogram(bins, value_range, variable_name)

    def isin(self, test_values: Any, variable_name: str | None = None) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.isin`."""
        warnings.warn(
            "`mesh.isin` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.isin`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.isin(test_values, variable_name)

    def domain_area(self) -> float:
        """Total area of the mesh domain.

        Returns:
            float: Sum of all face areas in the mesh's coordinate
                units squared.

        See Also:
            count_domain_cells: Count of active faces rather than
                their total area.
        """
        warnings.warn(
            "`mesh.domain_area` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.domain_area`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.domain_area()

    def count_domain_cells(self, variable_name: str | None = None) -> int:
        """Facade -- delegates to :meth:`_MeshAnalysis.count_domain_cells`."""
        warnings.warn(
            "`mesh.count_domain_cells` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.count_domain_cells`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.count_domain_cells(variable_name)

    def extract(
        self,
        x: Any,
        y: Any,
        variable_name: str | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.extract`."""
        warnings.warn(
            "`mesh.extract` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.extract`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.extract(x, y, variable_name)

    def sample_random(
        self,
        n: int,
        variable_name: str | None = None,
        seed: int | None = None,
    ) -> tuple:
        """Facade -- delegates to :meth:`_MeshAnalysis.sample_random`."""
        warnings.warn(
            "`mesh.sample_random` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.sample_random`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.sample_random(n, variable_name, seed)

    def contour(
        self,
        levels: Any = None,
        variable_name: str | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshVector.contour`."""
        warnings.warn(
            "`mesh.contour` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.contour`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.contour(levels, variable_name=variable_name)

    def plot_histogram(
        self,
        bins: int = 15,
        variable_name: str | None = None,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshGlyph.plot_histogram`."""
        warnings.warn(
            "`mesh.plot_histogram` is deprecated and will be removed in 0.90.0; use "
            "`mesh.glyph.plot_histogram`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glyph.plot_histogram(
            bins=bins, variable_name=variable_name, ax=ax, **kwargs
        )

    def proximity(self, target_faces: Any) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.proximity`."""
        warnings.warn(
            "`mesh.proximity` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.proximity`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.proximity(target_faces)

    def sieve(
        self,
        threshold: int,
        variable_name: str | None = None,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshAnalysis.sieve`."""
        warnings.warn(
            "`mesh.sieve` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.sieve`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.sieve(threshold, variable_name)
