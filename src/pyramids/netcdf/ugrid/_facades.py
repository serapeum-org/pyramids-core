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
from pyramids.netcdf.ugrid import graph as _graph
from pyramids.netcdf.ugrid import interop_xarray as _interop
from pyramids.netcdf.ugrid import regrid as _regrid
from pyramids.netcdf.ugrid import vector_ops as _vector_ops
from pyramids.netcdf.ugrid.messages import unknown_variable_message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.netcdf.ugrid.engines.analysis import _MeshAnalysis
    from pyramids.netcdf.ugrid.engines.glyph import _MeshGlyph
    from pyramids.netcdf.ugrid.engines.interop import _MeshInterop
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
        """The mean of each face's k-ring neighbourhood, centre included (issue #1376).

        The mesh counterpart of the raster ``focal_mean``. The raster version aggregates over a
        ``(2*radius+1)`` **pixel window**, which has no meaning here — faces have neither a
        row/column address nor a fixed neighbour count. This aggregates over the faces within
        ``rings`` hops of the face adjacency instead, which is why the parameter is ``rings``
        and not ``radius``.

        Boundary faces aggregate over whatever neighbours they have: no padding, no fabricated
        values. ``nodata`` faces are excluded, and so are inactive ones when the variable
        carries an active mask — a dry neighbour should not drag a wet face's mean down.

        Args:
            rings: Adjacency hops the neighbourhood spans. Defaults to 1.
            variable_name: Variable to act on (default: the only one).
            reach: A neighbourhood matrix to reuse across calls, built with
                ``pyramids.netcdf.ugrid.focal.neighbourhood(ds.mesh, rings)``. Without it each
                call builds its own, so a sweep over many variables or steps rebuilds the same
                sparse power every time. Leave ``rings`` at its default when passing it.

        Returns:
            numpy.ndarray: Per-face mean of length ``n_face``; ``NaN`` where the neighbourhood
            holds no usable value.

        Raises:
            ValueError: ``rings`` is less than 1, or the variable is not one value per face.

        See Also:
            focal_std: Spread over the same neighbourhood.
            focal_apply: An arbitrary aggregation over the same neighbourhood.
        """
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
        """The standard deviation of each face's k-ring neighbourhood (issue #1376).

        The spread counterpart of :meth:`focal_mean`, with the same neighbourhood, exclusion
        and boundary conventions.

        Args:
            rings: Adjacency hops the neighbourhood spans. Defaults to 1.
            variable_name: Variable to act on (default: the only one).
            reach: A neighbourhood matrix to reuse across calls, built with
                ``pyramids.netcdf.ugrid.focal.neighbourhood(ds.mesh, rings)``. Without it each
                call builds its own, so a sweep over many variables or steps rebuilds the same
                sparse power every time. Leave ``rings`` at its default when passing it.

        Returns:
            numpy.ndarray: Per-face population standard deviation; ``NaN`` where the
            neighbourhood holds no usable value.

        Raises:
            ValueError: ``rings`` is less than 1, or the variable is not one value per face.

        See Also:
            focal_mean: The mean over the same neighbourhood.
        """
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
        """Aggregate each face's k-ring neighbourhood with ``func`` (issue #1376).

        ``func`` receives the 1-D array of usable values in one neighbourhood and returns a
        scalar, so ``np.ptp``, ``np.median`` or a lambda all work. Unlike :meth:`focal_mean`
        and :meth:`focal_std` this cannot be expressed as a sparse product, so it visits each
        face in turn — noticeably slower on a large mesh, which is the price of an arbitrary
        aggregation.

        Args:
            func: Callable taking a 1-D array and returning a scalar.
            rings: Adjacency hops the neighbourhood spans. Defaults to 1.
            variable_name: Variable to act on (default: the only one).
            reach: A neighbourhood matrix to reuse across calls, built with
                ``pyramids.netcdf.ugrid.focal.neighbourhood(ds.mesh, rings)``. Without it each
                call builds its own, so a sweep over many variables or steps rebuilds the same
                sparse power every time. Leave ``rings`` at its default when passing it.

        Returns:
            numpy.ndarray: Per-face aggregate; ``NaN`` where the neighbourhood holds no usable
            value, so ``func`` is never handed an empty array.

        Raises:
            ValueError: ``rings`` is less than 1, or the variable is not one value per face.

        See Also:
            focal_mean: The vectorised mean, when that is all you need.
        """
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
        """Label connected groups of faces whose values fall in ``[lower, upper]`` (#1379).

        The value-aware sibling of :meth:`connected_components`, and the mesh counterpart of
        the raster ``cluster``. :meth:`connected_components` labels by **topology alone**, so
        it cannot answer "which separate pools of flooded cells are there"; this restricts the
        face adjacency to the faces inside the range before labelling.

        Faces outside the range — and no-data faces, which cannot be inside one — are labelled
        ``0``, so a label is truthy exactly when the face belongs to a cluster. Clusters are
        numbered ``1..k``.

        Args:
            lower: Inclusive lower bound.
            upper: Inclusive upper bound.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset whose variable carries the per-face cluster labels.

        Raises:
            ValueError: The variable is not one value per face (select a step first).

        See Also:
            connected_components: Topology-only labelling, ignoring values.
            footprint: The dissolved outline of the faces carrying valid values.
        """
        warnings.warn(
            "`mesh.cluster` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.cluster`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.cluster(lower, upper, variable_name=variable_name)

    def footprint(self, variable_name: str | None = None) -> Any:
        """The dissolved outline of the faces carrying valid values (#1379).

        The mesh counterpart of the raster ``footprint`` — "the real coverage of the values".
        Distinct from :meth:`bounding_polygon`, which dissolves **every** face and so describes
        the mesh *domain* rather than where data exists, and from :meth:`polygonize`, which
        returns one polygon per distinct value.

        "Valid" means not ``nodata`` and not NaN. It deliberately does **not** consult the
        per-timestep active mask: a dry element has a real value and is still covered by the
        data, so excluding it would conflate "no data here" with "dry here", which is the
        distinction issue #1331 exists to preserve. For the wet extent, mask first
        (:meth:`masked_values` or :meth:`where`) and take the footprint of that.

        Args:
            variable_name: Variable to act on (default: the only one).

        Returns:
            shapely.Polygon | shapely.MultiPolygon: The dissolved coverage, empty when no face
            carries a valid value. Carries no CRS — attach one if you need it.

        Raises:
            ValueError: The variable is not one value per face (select a step first).

        See Also:
            bounding_polygon: The whole mesh outline, data or not.
            polygonize: One polygon per distinct value.
        """
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
        """Remap a variable onto another facet of the mesh.

        Averages the values of the contributing elements, so a face field asked for on the
        nodes becomes each node's mean of the faces around it.

        Args:
            location: The facet to remap onto — ``"node"``, ``"edge"`` or ``"face"``.
            variable_name: The variable to remap, or ``None`` for the only one.

        Returns:
            MeshDataset: A new dataset whose variable sits on ``location``.

        Raises:
            ValueError: An unknown ``location``, or a remap the kernel does not offer.
        """
        var = self._facade_variable("to_location", variable_name)
        remap = {"node": _graph.to_node, "edge": _graph.to_edge, "face": _graph.to_face}
        if location not in remap:
            raise ValueError(
                f"to_location: location must be one of {sorted(remap)}, got {location!r}."
            )
        if var.location == location:
            # Already there. The averaging kernels only accept the two *other* facets as a
            # source, so without this an identity remap failed with "source must be 'node' or
            # 'edge'" -- a confusing answer to a request that is trivially already satisfied.
            return self._rebuild(dict(self._data_variables))  # type: ignore[attr-defined]
        values = remap[location](
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            source=var.location,
        )
        moved = var.with_data(np.asarray(values))
        moved.location = location
        # An element with no contributor (an orphan node, say) averages to NaN, so the remapped
        # variable has to declare NaN as its no-data value -- the same reasoning the regrid facade
        # already applies. Otherwise `to_file` writes those NaNs under the old `_FillValue`.
        # Through the same check `_rewrap` applies, rather than declared unconditionally.
        # `graph._average_over` answers float64 today, so NaN is always representable -- but a
        # future integer-preserving remap would reintroduce exactly the defect `_rewrap` guards.
        moved_values = np.asarray(moved.data)
        if _representable(float("nan"), moved_values.dtype):
            moved.nodata = float("nan")
        if moved.dimensions:
            # The element axis is the last dimension, by the mesh-variable invariant. Renaming it
            # to the new facet matters because a file-read variable carries real dimension names
            # and the xarray bridge prefers them: leaving the old facet's name on an axis that now
            # has the new facet's length makes the export fail on conflicting dimension sizes.
            mesh_name = self.mesh_name  # type: ignore[attr-defined]
            moved.dimensions = moved.dimensions[:-1] + (
                f"{mesh_name}_n{location.capitalize()}s",
            )
        new_vars = dict(self._data_variables)  # type: ignore[attr-defined]
        new_vars[var.name] = moved
        return self._rebuild(new_vars)  # type: ignore[attr-defined]

    def fill_na_spatial(
        self, variable_name: str | None = None, *, max_distance: float | None = None
    ) -> Any:
        """Fill each gap from the nearest valid element in space.

        Where :meth:`interpolate_na` fills along time, this fills across the mesh: a gap takes
        the value of the closest element that has one.

        Args:
            variable_name: The variable to fill, or ``None`` for the only one.
            max_distance: Leave a gap unfilled when no valid element is this close. ``None``
                (the default) is unbounded.

        Returns:
            MeshDataset: A new dataset with the gaps filled.
        """
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
        """Fill the interior gaps by solving Laplace's equation over the mesh.

        The valid values are the boundary condition and each gap becomes the smooth solution
        between them — a better hole-filler than nearest when the field varies smoothly.

        Args:
            variable_name: The variable to fill, or ``None`` for the only one.
            use_weights: Weight the graph Laplacian by inverse distance. Defaults to True.
            direct_solve: Solve directly rather than iteratively. Defaults to False.

        Returns:
            MeshDataset: A new dataset with the interior gaps filled.
        """
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
        """Dissolve adjacent faces of equal value into polygons.

        Args:
            variable_name: The face variable to dissolve by, or ``None`` for the only one.
            decimals: Round to this many decimals before grouping, for approximate equality.

        Returns:
            geopandas.GeoDataFrame: One row per merged region, with its value and geometry.
        """
        warnings.warn(
            "`mesh.polygonize` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.polygonize`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.polygonize(variable_name, decimals=decimals)

    def sample_line(self, line: Any, variable_name: str | None = None) -> Any:
        """Sample a variable along a line, ordered by distance along it.

        One sample per *crossing*, at the midpoint of each part of the line-face intersection.
        A line that leaves a face and re-enters it is sampled once per entry, so
        ``face_indices`` can repeat -- see :func:`pyramids.netcdf.ugrid.vector_ops.sample_line`.

        Args:
            line: A shapely ``LineString``, or a sequence of coordinates.
            variable_name: The variable to sample, or ``None`` for the only one.

        Returns:
            tuple: ``(distances, values, face_indices)``, one entry per crossing in
            along-line order. ``face_indices`` is not unique when the line re-enters a face.

        Raises:
            ValueError: ``variable_name`` is ``None`` while the dataset holds no variable or
                more than one, or the variable's last axis does not match ``n_face``.
        """
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
            self._mesh,  # type: ignore[attr-defined]
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
        return self.variables.with_variable(  # type: ignore[attr-defined]
            name, burned, location="face", nodata=declared
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
        var = self._facade_variable("rasterize_like", variable_name)
        return _regrid.rasterize_like(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            reference,
            location=var.location,
            method=method,
            nodata=nodata,
            source_epsg=self.epsg,  # type: ignore[attr-defined]
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
        """Regrid a variable onto another mesh.

        Args:
            target: The destination ``MeshDataset`` (its mesh is the target).
            method: ``"overlap"`` (area-weighted mean), ``"conservative"`` (integral
                preserving), ``"centroid"`` or ``"barycentric"``. Defaults to ``"overlap"``.
            variable_name: The variable to regrid, or ``None`` for the only one.

        Returns:
            MeshDataset: ``target``'s mesh carrying the regridded variable.

        Raises:
            ValueError: An unknown ``method``; the variable does not sit on faces (only
                ``"face"`` source data is weighted); or ``variable_name`` is ``None`` while the
                dataset holds no variable or more than one.
        """
        var = self._facade_variable("regrid", variable_name)
        regridder = _regrid.Regridder(
            self._mesh,  # type: ignore[attr-defined]
            target.mesh,
            method=method,
            source_location=var.location,
        )
        values = regridder.regrid(np.asarray(var.data))
        # A target element the source does not cover comes back NaN, so the regridded variable
        # has to declare NaN as its no-data value -- otherwise `to_file` writes those NaNs with
        # no `_FillValue` and a reader cannot tell them from data.
        return target.variables.with_variable(
            var.name, np.asarray(values), location="face", nodata=float("nan")
        )

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
            self._mesh,  # type: ignore[attr-defined]
            self._data_variables,  # type: ignore[attr-defined]
            store,
            mesh_name=self.mesh_name,  # type: ignore[attr-defined]
            crs_wkt=self.crs_wkt,  # type: ignore[attr-defined]
            **kwargs,
        )

    def fillna(self, value: Any, variable_name: str | None = None) -> Any:
        """Replace missing values with *value*.

        Args:
            value: The fill value.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with missing values replaced.

        See Also:
            isnull: Boolean mask of missing positions.
            fill_na_spatial: Fill gaps from the nearest valid spatial
                element.

        """
        warnings.warn(
            "`mesh.fillna` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.fillna`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.fillna(value, variable_name)

    def isnull(self, variable_name: str | None = None) -> Any:
        """Boolean mask of missing positions.

        Args:
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with boolean data.

        See Also:
            notnull: Inverse mask (non-missing positions).
            fillna: Replace missing values with a constant.

        """
        warnings.warn(
            "`mesh.isnull` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.isnull`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.isnull(variable_name)

    def notnull(self, variable_name: str | None = None) -> Any:
        """Boolean mask of non-missing positions.

        Args:
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with boolean data.

        See Also:
            isnull: Inverse mask (missing positions).

        """
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
        """Clamp variable values to ``[lower, upper]``.

        Named ``clip_values`` to avoid collision with the spatial
        :meth:`~pyramids.netcdf.ugrid.dataset.MeshDataset.clip` method.

        Args:
            lower: Minimum allowed value.
            upper: Maximum allowed value.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with clamped values.

        See Also:
            round: Round values to a number of decimal places.

        """
        warnings.warn(
            "`mesh.clip_values` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.clip_values`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.clip_values(lower, upper, variable_name)

    def round(self, decimals: int = 0, variable_name: str | None = None) -> Any:
        """Round variable values to *decimals* decimal places.

        Args:
            decimals: Number of decimal places.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with rounded values.

        See Also:
            clip_values: Clamp values to a range.

        """
        warnings.warn(
            "`mesh.round` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.round`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.round(decimals, variable_name)

    def astype(self, dtype: Any, variable_name: str | None = None) -> Any:
        """Cast variable values to a different dtype.

        Args:
            dtype: Target numpy dtype.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with the new dtype.

        See Also:
            round: Round floating-point values before an integer cast.

        """
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
        """Keep the elements ``cond`` selects and mask the rest (issue #1375).

        The mesh counterpart of the raster ``where``. A ``nodata`` / NaN element is never
        selected — there is no value there to keep — so it goes to ``other`` whatever ``cond``
        says, leaving the result with one masking convention rather than two.

        The per-timestep active mask (:attr:`~pyramids.netcdf.ugrid.models.MeshVariable.active`)
        is **not** applied here. Keeping "dry" distinct from "missing" is the whole point of
        that field (issue #1331), so activity-aware masking is something a caller composes
        explicitly: ``ds.where(ds.active_mask(name) & (values > 0))``.

        Args:
            cond: A boolean array broadcastable to the variable's values.
            other: What an unselected element holds. Defaults to NaN.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with the unselected elements masked.

        See Also:
            clip_values: Clamp values to a range instead of masking outside it.
            masked_values: Apply the active mask to the values, on demand.
        """
        warnings.warn(
            "`mesh.where` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.where`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.where(cond, other, variable_name)

    def apply(self, func: Any, variable_name: str | None = None) -> Any:
        """Apply ``func`` to every element of a variable (issue #1375).

        The mesh counterpart of the raster ``apply``. ``func`` receives the whole array, so a
        numpy ufunc runs vectorised rather than once per element. The shape must survive, since
        the result still has to line up with the mesh elements it describes.

        Args:
            func: A callable taking and returning an array of the same shape.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset carrying the transformed values.

        Raises:
            ValueError: ``func`` changed the array shape.

        See Also:
            astype: Cast values to another dtype.
            round: Round values to a number of decimal places.
        """
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
        """The variable's value distribution as data (issue #1375).

        The mesh counterpart of the raster ``get_histogram``. :meth:`plot_histogram` could
        already *draw* the distribution but gave no way to obtain it; both mask no-data through
        the same helper, so these counts describe exactly the samples those bars show.

        Args:
            bins: Number of equal-width bins. Defaults to 256.
            value_range: ``(low, high)`` to bin within, or ``None`` to span the data.
            variable_name: Variable to act on (default: the only one).

        Returns:
            tuple: ``(counts, edges)`` -- per-bin counts and the ``bins + 1`` edges, in the
            variable's own units.

        Raises:
            ValueError: Nothing survives the no-data masking, so there is no distribution.

        See Also:
            plot_histogram: The same distribution, drawn.
            summary: Per-variable min / max / mean / std over the whole mesh.
        """
        warnings.warn(
            "`mesh.get_histogram` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.get_histogram`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.get_histogram(bins, value_range, variable_name)

    def isin(self, test_values: Any, variable_name: str | None = None) -> Any:
        """Boolean mask where variable values are in *test_values*.

        Args:
            test_values: Values to test membership against.
            variable_name: Variable to act on (default: the only one).

        Returns:
            MeshDataset: A new dataset with boolean data.
        """
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
        """Number of active (non-masked) mesh faces.

        When the variable carries no active mask, all faces are counted.

        Args:
            variable_name: Variable whose active mask to consult.
                ``None`` counts all faces.

        Returns:
            int: Active face count.

        See Also:
            domain_area: Total area rather than face count.

        """
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
        """Sample variable values at arbitrary (x, y) coordinates.

        Uses an STRtree spatial index to locate the containing face
        for each query point.

        Args:
            x: Query x-coordinates (scalar or array).
            y: Query y-coordinates (scalar or array).
            variable_name: Variable to extract (default: the only one).

        Returns:
            numpy array of extracted values.  Points outside the mesh
            get ``NaN``.

        See Also:
            sample_random: Draw spatially random samples instead of
                querying specific coordinates.
            sample: Interpolated point sampling (existing method on
                MeshDataset).
        """
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
        """Draw *n* spatially random samples from the mesh domain.

        Args:
            n: Number of samples.
            variable_name: Variable to sample (default: the only one).
            seed: RNG seed for reproducibility.

        Returns:
            ``(x, y, values)`` — arrays of length *n*.

        See Also:
            extract: Sample at specific coordinates rather than random
                ones.

        """
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
        """Generate contour lines from face/node values.

        Args:
            levels: Contour levels (int for count, array for explicit).
            variable_name: Variable to contour (default: the only one).

        Returns:
            GeoDataFrame with contour ``LineString`` geometries and a
            ``level`` column.

        Raises:
            ValueError: If the variable has more than one time step;
                select a single step first.

        See Also:
            plot_histogram: Distribution of the same variable as a
                histogram.

        """
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
        """Graph distance from every face to the nearest target face.

        Uses BFS over the face adjacency graph.

        Args:
            target_faces: 1-D array of face indices that are targets
                (distance zero).

        Returns:
            numpy array of per-face integer distances.  ``-1`` for
            faces in disconnected components with no target.

        See Also:
            sieve: Remove small connected regions by area threshold.
        """
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
        """Remove small connected regions below *threshold* faces.

        Faces in a connected component smaller than *threshold* are
        blanked to NaN.  Operates on the face adjacency graph.

        Args:
            threshold: Minimum component size (face count) to keep.
            variable_name: Variable to sieve (default: the only one).

        Returns:
            MeshDataset: A new dataset with small-region faces
                replaced by NaN.

        Raises:
            ValueError: If the variable has more than one time step;
                select a single step first.

        See Also:
            proximity: Compute graph distances from target faces.

        """
        warnings.warn(
            "`mesh.sieve` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.sieve`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.sieve(threshold, variable_name)
