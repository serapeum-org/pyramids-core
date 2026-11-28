"""MeshDataset — top-level container for UGRID NetCDF mesh data.

Combines mesh topology, data variables, metadata, and provides
the user-facing API for reading, inspecting, and operating on
unstructured mesh data.
"""

from __future__ import annotations

import datetime
import functools
import inspect
import sys
import tempfile
import warnings
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload
from uuid import uuid4

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

if TYPE_CHECKING:
    from cleopatra.glyphs.gridded.array_glyph import PointOverlay
    from cleopatra.styling.colorbar import ColorBar
    from cleopatra.styling.params import Contour, DataStyle
    from cleopatra.styling.scaling import ColorScaling

    from pyramids.netcdf.ugrid.mesh1d import Mesh1d
from osgeo import gdal
from pyproj import CRS
from shapely.geometry import LineString

from pyramids.base import _timeseries
from pyramids.base._array_ops import array_equals, array_identical
from pyramids.base._compute_backend import accepts_keyword, require_compute_backend
from pyramids.base._errors import (
    AlignmentError,
    DriverNotExistError,
    OptionalPackageDoesNotExist,
)
from pyramids.base._plot_backends import (
    forward_kwargs,
    render_via_backend,
)
from pyramids.base._reductions import (
    COUNTING_REDUCERS,
    FLAG_NO_DATA,
    REDUCERS,
    _check_min_periods,
    _check_window,
    gaps_as_nan,
    reduce_axis,
    reduce_groups,
    validate_min_count,
    window_members,
    window_positions,
)
from pyramids.base._summary import DEFAULT_METRICS, variable_summary
from pyramids.base.crs import (
    crs_spec,
    sr_from_epsg,
)
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.ops.units import convert_array
from pyramids.feature import FeatureCollection
from pyramids.netcdf._mdim import no_data_value_of, open_mdarray
from pyramids.netcdf.cf import write_attributes_to_md_array, write_global_attributes
from pyramids.netcdf.ugrid import construct as _construct
from pyramids.netcdf.ugrid import interop_xarray as _interop
from pyramids.netcdf.ugrid import plot as _ugrid_plot
from pyramids.netcdf.ugrid._drivers import (
    NETCDF_SUFFIXES,
    RESERVED_NAMES,
    MeshDriver,
    NoMeshTopologyError,
    UnsupportedMeshFormatError,
    describe_drivers,
    drivers_for_suffix,
    mesh_drivers,
    unsupported_format_message,
)
from pyramids.netcdf.ugrid._facades import _MeshFacades
from pyramids.netcdf.ugrid.connectivity import Connectivity
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
from pyramids.netcdf.ugrid.fill import _location_coords
from pyramids.netcdf.ugrid.interpolation import mesh_to_grid
from pyramids.netcdf.ugrid.io import (
    parse_ugrid_topology,
    parse_volume_topology,
    write_ugrid_data_variable,
    write_ugrid_topology,
    write_volume_topology,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.messages import unknown_variable_message
from pyramids.netcdf.ugrid.models import (
    _KEEP_ACTIVE,
    ACTIVE_MASK_ATTR,
    ACTIVE_MASK_ROLE,
    ACTIVE_MASK_SUFFIX,
    DEFAULT_MESH_NAME,
    UGRID_CONVENTIONS,
    MeshMetadata,
    MeshTopologyInfo,
    MeshVariable,
    VolumeTopology,
    _consistent_active,
    read_mesh_variable,
)
from pyramids.netcdf.ugrid.spatial import (
    MeshSpatialIndex,
    subset_mesh_by_faces,
)
from pyramids.netcdf.utils import (
    _decimal_years,
    _dtype_to_str,
    _plan_calendar_conversion,
    _plan_calendar_interpolation,
    _read_attributes,
    decode_cf_time,
    encode_cf_time,
    is_cf_time_units,
    read_cf_attributes,
)


class _joins_meshes:  # noqa: N801
    """Descriptor making a join callable on the class and on a dataset.

    ``MeshDataset.concat([a, b])`` is the canonical spelling, but ``a.concat([b])`` reads
    naturally as "join this dataset with that one". A plain method only supports the second and
    a ``classmethod`` only the first — and as a classmethod the instance it was reached through
    would be invisible, silently dropping ``a``. This threads the receiver in instead: an
    instance call passes the receiver as ``self``, a class call promotes the first element of
    the list to ``self`` and joins the rest onto it, so both spellings answer the same dataset.
    """

    def __init__(self, function: Any) -> None:
        """Wrap the join.

        Args:
            function: The join, taking ``(self, objs, ...)``.
        """
        self._function = function
        self.__doc__ = function.__doc__
        self._class_calls: dict[type, Any] = {}

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        """Bind the join to the class or to a dataset.

        Args:
            instance: The dataset it was reached through, or ``None`` for a class call.
            owner: The owning class.

        Returns:
            The join, with the receiver threaded in.
        """
        function = self._function
        if instance is not None:
            # A real bound method, so the join keeps the signature tooling reads (`compat=`
            # stays visible) and the equality an ordinary method has -- `a.concat ==
            # a.concat` holds, which a freshly built closure could not give.
            return MethodType(function, instance)

        target = owner if owner is not None else MeshDataset
        cached = self._class_calls.get(target)
        if cached is not None:
            # Built once per owner. Rebuilding it per access made
            # `MeshDataset.concat is MeshDataset.concat` false and `pickle.dumps` fail with
            # "it's not the same object as MeshDataset.merge", and the signature repair added
            # an `inspect.signature` call to every read on top of that.
            return cached

        @functools.wraps(function)
        def call(objs: Any, *args: Any, **kwargs: Any) -> Any:
            # No docstring: `functools.wraps` replaces it with the wrapped join's immediately
            # below, so one written here is dead text that also needs a `noqa` to silence the
            # Args-mismatch lint. `objs` is one dataset or an iterable of them, the first
            # becoming the receiver; everything else is forwarded.
            #
            # A bare dataset is iterable over its variable names, so `list(objs)` would turn
            # `MeshDataset.concat(a)` into a list of strings. Promote a single dataset to a
            # one-element list, exactly as the instance path already accepts `a.concat(b)`.
            parts = [objs] if isinstance(objs, target) else list(objs)
            if not parts:
                raise ValueError(
                    f"{function.__name__}: needs at least one dataset to join."
                )
            return function(parts[0], parts[1:], *args, **kwargs)

        # `functools.wraps` sets `__wrapped__`, which `inspect.signature` follows -- so the
        # class call advertised the undecorated `(self, others, *, compat=...)`. That is the
        # instance spelling with a stray `self`, and wrong for `MeshDataset.merge([a, b])`,
        # which takes the whole join as one positional argument. Re-advertise it: drop `self`
        # and name the first parameter after the one this wrapper really takes.
        original = inspect.signature(function)
        tail = list(original.parameters.values())[1:]
        if tail:
            call.__signature__ = original.replace(  # type: ignore[attr-defined]
                parameters=[tail[0].replace(name="others"), *tail[1:]]
            )
        self._class_calls[target] = call
        return call


# The types whose emptiness the falsy-`chunks` guard can judge. Anything else -- an ndarray, a
# pandas object, a custom chunk spec -- is forwarded for the backend to accept or reject, because
# `not value` on such an object raises (numpy: "truth value of an array ... is ambiguous") and
# would replace a clear refusal with an unrelated error naming neither `chunks` nor this method.
_JUDGEABLE_CHUNKS = (bool, int, float, str, bytes, dict, list, tuple, set, frozenset)


class MeshDataset(_MeshFacades):
    """An unstructured mesh with data variables -- the mesh equivalent of the raster
    :class:`~pyramids.dataset.Dataset` and the vector :class:`~pyramids.feature.FeatureCollection`.

    Combines a mesh topology (:class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`), data variables,
    global attributes and a CRS into one object with GIS-aware operations. The UGRID NetCDF
    convention is one format it reads and writes (:meth:`read_file` / :meth:`to_file`), not
    its identity. Does NOT inherit from Dataset or RasterBase -- the raster paradigm does not
    apply to unstructured meshes.

    Attributes:
        _mesh: Mesh2d topology instance.
        _data_variables: Mapping of variable name to MeshVariable.
        _global_attributes: File-level NetCDF attributes.
        _topology_info: Parsed UGRID topology metadata.
        _crs_wkt: CRS in WKT format.
        _file_name: Source file path, if read from disk.
    """

    def __init__(
        self,
        mesh: Mesh2d | Mesh1d,
        data_variables: dict[str, MeshVariable],
        global_attributes: dict[str, Any],
        topology_info: MeshTopologyInfo | None = None,
        crs_wkt: str | None = None,
        file_name: str | None = None,
        volume_topology: VolumeTopology | None = None,
    ):
        """Assemble a dataset from an already-built mesh and its variables.

        The low-level constructor. Prefer :meth:`read_file` for a file, :meth:`from_arrays`
        for raw arrays, or one of the ``from_*`` bridges -- each of those builds the mesh and
        the metadata record and then calls this. It is also what every derivation
        (:meth:`isel`, :meth:`reduce`, :meth:`copy`, ...) calls through ``type(self)(...)``,
        which is how a subclass survives them.

        Args:
            mesh: The topology the variables sit on.
            data_variables: Mapping of variable name to :class:`MeshVariable`. Held by
                reference, not copied, so a caller must not mutate it afterwards.
            global_attributes: The file-level attributes, e.g. ``Conventions``.
            topology_info: The record of the on-disk variable names and the declared
                topology dimension. ``None`` for a mesh built from arrays, where there are
                no file variables to name. Defaults to ``None``.
            crs_wkt: WKT of the mesh CRS. ``None`` leaves the mesh unreferenced, and
                :attr:`epsg` then answers ``None``. Defaults to ``None``.
            file_name: The path this dataset was read from, carried through derivations so a
                :meth:`to_file` round trip and ``repr`` can name it. ``None`` for an
                in-memory mesh. Defaults to ``None``.
            volume_topology: The ragged (per-column) vertical layering for a genuinely 3-D
                mesh — the map from faces to the flat ``volume`` axis that a
                ``location="volume"`` variable is indexed through (issue #1330). ``None`` (the
                default) for a 2-D mesh or a uniform-layer 3-D one, where the dense
                ``(layer, element)`` path applies and :attr:`n_volume` answers ``None``.
        """
        self._mesh = mesh
        self._data_variables = data_variables
        self._global_attributes = global_attributes
        self._topology_info = topology_info
        self._crs_wkt = crs_wkt
        self._file_name = file_name
        self._volume_topology = volume_topology
        self._cached_crs: Any = None
        # D4: the dataset is the single owner of its coordinate system. `crs` / `epsg` /
        # `crs_wkt` and every geometry export (`contour`, `polygonize`, `to_geodataframe`)
        # resolve from `self._crs_wkt`, so they cannot disagree. `__init__` does NOT write
        # onto the passed `mesh`: an earlier approach filled `mesh.crs` here, but that
        # mutated a caller-held object (review M2) and needed a fragile, leaky mesh-vs-wkt
        # conflict check (review M1). The mesh's own `crs` is set only where it genuinely
        # belongs to the mesh -- the file read path and `set_crs`.

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        mesh_name: str | None = None,
        driver: str | None = None,
        lazy: bool = True,
    ) -> MeshDataset:
        """Open a mesh file.

        UGRID NetCDF is read here. Any other format — 2DM, Selafin, DFSU, XMDF, … — is read by a
        third-party driver registered into the ``pyramids.mesh_drivers`` entry-point group. A
        non-NetCDF suffix is dispatched by detection; a NetCDF suffix is read by the built-in
        reader first, and only a file that opens but carries no UGRID topology falls back to a
        driver claiming that suffix. Either way a successful UGRID read triggers no driver
        discovery, so a ``.nc`` read still pays nothing.

        Resolution order:

        1. ``driver`` given — that driver, or an error. Never falls back.
        2. A ``/vsi`` path or a NetCDF suffix — the built-in UGRID reader. If a NetCDF-suffixed
           file opens but declares no UGRID topology, a registered driver claiming that suffix
           is tried before giving up, so 3Di, TUFLOW FV and other NetCDF-based formats are
           reachable; discovery runs only after the built-in read has already failed.
        3. Otherwise a registered driver claiming the suffix whose ``can_open`` accepts the file.
        4. Nothing claimed it — the built-in reader anyway, so a NetCDF file under an unusual
           suffix still opens exactly as it always did.
        5. That failed too — :class:`UnsupportedMeshFormatError`, naming what is installed.

        Args:
            path: Path to the file. A GDAL virtual-filesystem path
                (``/vsimem/``, ``/vsizip/``, ``/vsicurl/``, …) is passed straight to GDAL
                rather than checked on disk.
            mesh_name: Which topology to read, for a file declaring more than one (see
                :meth:`mesh_names`). ``None`` (the default) takes the first.
            driver: Name of a registered **mesh** driver to use, bypassing detection — the
                name a plugin declares, not a GDAL driver short name. ``None`` (the default)
                resolves as described above.
            lazy: Defer reading variable values until first access. Forwarded to the driver,
                which honours it when it declares ``lazy_data``. The built-in UGRID reader is
                always lazy, so it rejects ``lazy=False`` rather than ignoring it.

        Returns:
            MeshDataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            DriverNotExistError: If ``driver`` names a driver that is not installed.
            UnsupportedMeshFormatError: If no installed driver reads the file, and the
                built-in UGRID reader cannot either.
            IsADirectoryError: The path is a directory that no driver claims.
            TypeError: ``lazy=False`` was asked of the built-in UGRID reader, which is always
                lazy and so cannot honour it.
            RuntimeError: GDAL cannot open a file that reached the built-in reader by its
                own suffix. A file whose suffix no driver claimed is reported as
                :class:`UnsupportedMeshFormatError` instead, so this is now the native
                route only.
            ValueError: If no UGRID topology is found in the file, or ``mesh_name`` names
                a topology the file does not declare.
        """
        selected, native = _resolve_read_driver(path, driver)
        if selected is not None:
            result = _read_through_driver(selected, path, mesh_name, cls, lazy)
        else:
            try:
                result = cls._read_ugrid_netcdf(
                    path, mesh_name, wrap_unreadable=not native, lazy=lazy
                )
            except NoMeshTopologyError:
                # A NetCDF file that opened but carries no UGRID topology -- a 3Di or
                # TUFLOW FV result. Only now, after the built-in read has failed, consult a
                # driver claiming the suffix; if none does, the original error stands. `lazy`
                # is forwarded to the driver, which may honour an eager read even though the
                # built-in reader cannot. A genuine UGRID file with `lazy=False` raised
                # TypeError inside `_read_ugrid_netcdf` instead, before reading any geometry.
                fallback = _netcdf_fallback_driver(path, driver)
                if fallback is None:
                    raise
                result = _read_through_driver(fallback, path, mesh_name, cls, lazy)
        return result

    @classmethod
    def _read_ugrid_netcdf(
        cls,
        path: str | Path,
        mesh_name: str | None,
        wrap_unreadable: bool = False,
        lazy: bool = True,
    ) -> MeshDataset:
        """Read a UGRID NetCDF file through GDAL's multidimensional API.

        The built-in reader, unchanged from before driver dispatch existed. It is reached both
        for a NetCDF suffix and as the fallback when no registered driver claims a file.

        Args:
            path: Path to the file, or a GDAL virtual-filesystem name.
            mesh_name: Which topology to read. ``None`` takes the first.
            wrap_unreadable: True when this call is step 4 of :meth:`read_file`'s resolution —
                a file whose suffix no driver claimed. A failure then reports the installed
                drivers rather than only GDAL's complaint, which would misleadingly suggest the
                file is corrupt rather than unsupported.
            lazy: Defer variable values to first access. The built-in reader is always lazy, so
                ``lazy=False`` is rejected with :class:`TypeError` once a topology is confirmed —
                before any geometry is read. Defaults to True.

        Returns:
            MeshDataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedMeshFormatError: If GDAL cannot open a file that reached here as a
                fallback from driver detection.
            NoMeshTopologyError: If the file opens but declares no UGRID topology. Subclasses
                :class:`ValueError`, and is what lets a caller fall back to a driver.
            TypeError: If ``lazy=False``; the built-in reader has no eager mode.
            ValueError: If ``mesh_name`` names a topology the file does not declare.
        """
        # A GDAL virtual-filesystem path (/vsimem/, /vsizip/, /vsicurl/, …) is not an OS path:
        # `Path` would mangle the separators and `exists()` would always be False, so hand
        # those straight to GDAL and let it report. `from_bytes` reads through /vsimem/.
        target = str(path)
        if target.startswith("/vsi"):
            opened_name = target
        else:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {Path(path)}"
                )  # pragma: no cover - `_require_existing_file` runs first on every public route
            opened_name = str(path)

        # GDAL exceptions are enabled package-wide (`Config` calls `gdal.UseExceptions()`),
        # so a file no driver recognises normally raises RuntimeError from inside the binding.
        # That is process-global state, though, and any dependency can call
        # `gdal.DontUseExceptions()` -- after which OpenEx answers None and this guard is the
        # difference between naming the file and an `AttributeError: 'NoneType' object has no
        # attribute 'GetRootGroup'`.
        ds = _open_multidim(opened_name, str(path), wrap_unreadable)
        rg = ds.GetRootGroup()
        if rg is None:  # pragma: no cover - no multidim driver answers None here
            # Kept as a clear message rather than letting `parse_ugrid_topology(None)`
            # raise an AttributeError, though every multidim driver that opens a file
            # does expose a root group.
            raise ValueError(f"Cannot get root group from: {path}")

        topologies = parse_ugrid_topology(rg)
        if not topologies:
            raise NoMeshTopologyError(f"No UGRID mesh topology found in: {path}")

        if not lazy:
            # The built-in reader always defers variable values to first `.data` access and has
            # no eager mode, so `lazy=False` cannot be honoured. Raise here -- after topology is
            # confirmed, before `_from_gdal_group` reads any geometry -- so the rejection is cheap
            # and both `read_file` and `read_all` get it. A driver-backed file never reaches this:
            # it raised `NoMeshTopologyError` above, and `lazy` is forwarded to the driver instead.
            raise TypeError(
                "lazy=False is not supported for UGRID NetCDF: the built-in reader always defers "
                "variable values to first access. Call `.load_array()` on a variable to read it "
                "now."
            )

        # A file may declare several meshes. The default stays the first one — changing that
        # would silently alter every existing caller — but `mesh_name` picks any of them, and
        # `read_all` reads them all. Without either, a multi-mesh file quietly dropped the rest.
        if mesh_name is None:
            topo_info = topologies[0]
        else:
            available = [topo.mesh_name for topo in topologies]
            chosen = [topo for topo in topologies if topo.mesh_name == mesh_name]
            if not chosen:
                raise ValueError(
                    f"No UGRID topology named {mesh_name!r} in {path}; it declares "
                    f"{available}."
                )
            topo_info = chosen[0]
        mesh = Mesh2d._from_gdal_group(rg, topo_info)

        # Resolve to an absolute path before threading it into the lazy variable loaders:
        # data reads are deferred to first `.data` access (PERF-3), which re-opens the file.
        # A relative path would break that deferred open if the process changed directory in
        # the meantime; the old eager read was immune because it read while still in `read_file`.
        # A virtual-filesystem name is already absolute and has no OS path to resolve.
        source_name = (
            opened_name if opened_name.startswith("/vsi") else str(Path(path).resolve())
        )
        data_variables = _read_data_variables(rg, topo_info, source_name)
        # Re-attach each variable's active (wet/dry) mask from its `<var>_active` companion,
        # and rebuild the ragged volume topology from its companion arrays, if present — both
        # are additive, so a file carrying neither reads exactly as before (issues #1331/#1330).
        _attach_active_masks(rg, data_variables)
        volume_topology = parse_volume_topology(rg, topo_info.mesh_name)
        # Decode the CF time coordinate onto the temporal variables, so `time_values` answers
        # real datetimes instead of positional indices wherever the file supplies the stamps
        # (issue #1381). Inert for a file with no CF time units.
        _decode_time_coordinate(rg, data_variables)

        global_attrs = _read_attributes(rg)

        ds = None

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=global_attrs,
            topology_info=topo_info,
            crs_wkt=topo_info.crs_wkt,
            file_name=str(path),
            volume_topology=volume_topology,
        )
        return result

    @property
    def mesh(self) -> Mesh2d | Mesh1d:
        """The mesh topology -- a :class:`Mesh2d` (2D) or :class:`Mesh1d` (1D)."""
        return self._mesh

    def _require_mesh2d(self, operation: str) -> Mesh2d:
        """Return the mesh narrowed to :class:`Mesh2d`, or raise for a 1D mesh.

        A handful of operations are defined over faces and have no 1D meaning -- rasterising
        to a grid, the UGRID file writer, the face spatial index, plane-fill coordinates and
        animation. Each routes through here so a 1D mesh is refused with a clear message
        rather than an ``AttributeError`` deep inside a face-only helper (issue #1652).

        Args:
            operation: The user-facing operation name, for the message.

        Returns:
            Mesh2d: The mesh, when it is 2D.

        Raises:
            TypeError: The dataset carries a 1D (:class:`Mesh1d`) mesh.
        """
        mesh = self._mesh
        if not isinstance(mesh, Mesh2d):
            raise TypeError(
                f"{operation} is not available on a 1D mesh "
                f"({type(mesh).__name__}); it is defined over faces."
            )
        return mesh

    @property
    def mesh_name(self) -> str:
        """Name of the mesh topology variable."""
        result = (
            self._topology_info.mesh_name if self._topology_info else DEFAULT_MESH_NAME
        )
        return result

    @property
    def analysis(self) -> _MeshAnalysis:
        """Analysis engine: cell-wise transforms, focal, sampling, reductions (``mesh.analysis``)."""
        return _MeshAnalysis(self)

    @property
    def terrain(self) -> _MeshTerrain:
        """Terrain engine: DEM derivatives -- slope, aspect, hillshade (``mesh.terrain``)."""
        return _MeshTerrain(self)

    @property
    def interop(self) -> _MeshInterop:
        """Interop engine: ``to_xarray`` / ``to_dataframe`` / ``to_dask_dataframe`` (``mesh.interop``)."""
        return _MeshInterop(self)

    @property
    def vector(self) -> _MeshVector:
        """Vector engine: geodataframe/feature-collection, contour, cluster, voronoi (``mesh.vector``)."""
        return _MeshVector(self)

    @property
    def glyph(self) -> _MeshGlyph:
        """Glyph engine: plot, histogram, vector field, outline, animate, to_image (``mesh.glyph``)."""
        return _MeshGlyph(self)

    @property
    def variables(self) -> _MeshVariables:
        """Variables engine: with_variable, attach_data, rename_variable, drop_variables (``mesh.variables``)."""
        return _MeshVariables(self)

    @property
    def topology(self) -> _MeshTopology:
        """Topology engine: circumcenters, exterior edges/faces, bounding_polygon, components, \
morphology, adjacency, active_mask (``mesh.topology``)."""
        return _MeshTopology(self)

    @property
    def spatial(self) -> _MeshSpatial:
        """Spatial engine: crop, subset_by_bounds, to_crs, set_crs, resample (``mesh.spatial``)."""
        return _MeshSpatial(self)

    @property
    def selection(self) -> _MeshSelection:
        """Selection/reduction engine: sel, isel, reduce, rolling, layers, fits (``mesh.selection``)."""
        return _MeshSelection(self)

    @property
    def io(self) -> _MeshIO:
        """I/O engine: to_file, to_bytes, to_zarr, to_dataset, rasterize_like, burn (``mesh.io``)."""
        return _MeshIO(self)

    @property
    def variable_names(self) -> list[str]:
        """Names of all data variables, in insertion order.

        The list counterpart of :attr:`data_vars` (the ``{name: variable}`` mapping), named
        to match the sibling :attr:`pyramids.netcdf.NetCDF.variable_names`.

        Returns:
            list[str]: The data-variable names; empty when the mesh carries no data variables.
        """
        result = list(self._data_variables.keys())
        return result

    @property
    def data_variable_names(self) -> list[str]:
        """Deprecated alias for :attr:`variable_names`.

        Deprecated: Renamed to ``variable_names`` for parity with the sibling
        :attr:`pyramids.netcdf.NetCDF.variable_names`.

        Returns:
            list[str]: The names of all data variables.
        """
        warnings.warn(
            "MeshDataset.data_variable_names is deprecated; use variable_names.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variable_names

    @property
    def crs(self) -> CRS | None:
        """CRS as a pyproj.CRS object, or None. Cached after first access."""
        if self._cached_crs is None and self._crs_wkt is not None:
            try:
                self._cached_crs = CRS.from_wkt(self._crs_wkt)
            except Exception:  # nosec B110 - best-effort CRS parse; falls back to None
                pass
        return cast("CRS | None", self._cached_crs)

    @property
    def epsg(self) -> int | None:
        """EPSG code of the CRS, or None."""
        crs = self.crs
        result = crs.to_epsg() if crs is not None else None
        return result

    @property
    def crs_wkt(self) -> str | None:
        """The mesh CRS as WKT, or `None` when there is none that parses.

        Gated on a successful parse rather than returning `_crs_wkt` verbatim.
        `_crs_wkt` is copied straight out of the file and :attr:`crs` tolerates
        an unparseable one on purpose, swallowing the error and reporting no
        CRS. Handing the raw string to a caller would move that parse out from
        under the handler, so a malformed WKT would start raising downstream
        instead of being reported as "no CRS" here.
        """
        return self._crs_wkt if self.crs is not None else None

    def set_crs(self, crs: str | None = None, epsg: int | None = None) -> None:
        """Facade -- delegates to :meth:`_MeshSpatial.set_crs`."""
        warnings.warn(
            "`mesh.set_crs` is deprecated and will be removed in 0.90.0; use "
            "`mesh.spatial.set_crs`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.spatial.set_crs(crs=crs, epsg=epsg)

    def _mesh_equals(self, other: MeshDataset) -> bool:
        """Compare mesh topology and coordinate arrays."""
        result = np.array_equal(self._mesh.node_x, other._mesh.node_x)
        if result:
            result = np.array_equal(self._mesh.node_y, other._mesh.node_y)
        if result:
            result = np.array_equal(
                self._mesh.face_node_connectivity.data,
                other._mesh.face_node_connectivity.data,
            )
        return result

    @staticmethod
    def _vars_value_equal(
        vars_a: dict[str, MeshVariable],
        vars_b: dict[str, MeshVariable],
    ) -> bool:
        """Compare variable values ignoring nodata/NaN positions."""
        for name in vars_a:
            va, vb = vars_a[name], vars_b[name]
            if va.data is None and vb.data is None:
                continue
            if va.data is None or vb.data is None:
                return False
            if not array_equals(
                np.asarray(va.data),
                np.asarray(vb.data),
                va.nodata,
                vb.nodata,
            ):
                return False
        return True

    def equals(self, other: object) -> bool:
        """Facade -- delegates to :meth:`_MeshAnalysis.equals`."""
        warnings.warn(
            "`mesh.equals` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.equals`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.equals(other)

    def _equals_impl(self, other: object) -> bool:
        """Implementation of :meth:`_MeshAnalysis.equals`; see that method for the contract."""
        if not isinstance(other, MeshDataset):
            return False
        result = self._mesh_equals(other)
        if result:
            result = self.epsg == other.epsg
        if result:
            result = set(self._data_variables) == set(other._data_variables)
        if result:
            result = self._vars_value_equal(self._data_variables, other._data_variables)
        return result

    @staticmethod
    def _nodata_matches(a, b) -> bool:
        """Return True when two nodata sentinels are semantically equal."""
        if a is None or b is None:
            return a is b
        if isinstance(a, float) and isinstance(b, float):
            return a == b or (np.isnan(a) and np.isnan(b))
        result: bool = a == b
        return result

    @staticmethod
    def _vars_identical(
        vars_a: dict[str, MeshVariable],
        vars_b: dict[str, MeshVariable],
    ) -> bool:
        """Strict per-variable comparison (dtype, nodata, values)."""
        for name in vars_a:
            va, vb = vars_a[name], vars_b[name]
            if not MeshDataset._nodata_matches(va.nodata, vb.nodata):
                return False
            if va.data is None and vb.data is None:
                continue
            if va.data is None or vb.data is None:
                return False
            if not array_identical(np.asarray(va.data), np.asarray(vb.data)):
                return False
        return True

    def _crs_identical(self, other: MeshDataset) -> bool:
        """Semantic CRS comparison, falling back to WKT string."""
        if self.crs is not None and other.crs is not None:
            return self.crs == other.crs
        return self._crs_wkt == other._crs_wkt

    def identical(self, other: object) -> bool:
        """Facade -- delegates to :meth:`_MeshAnalysis.identical`."""
        warnings.warn(
            "`mesh.identical` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.identical`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.identical(other)

    def _identical_impl(self, other: object) -> bool:
        """Implementation of :meth:`_MeshAnalysis.identical`; see that method for the contract."""
        if not isinstance(other, MeshDataset):
            return False
        result = self._mesh_equals(other)
        if result:
            result = self._crs_identical(other)
        if result:
            result = self._global_attributes == other._global_attributes
        if result:
            result = set(self._data_variables) == set(other._data_variables)
        if result:
            result = self._vars_identical(self._data_variables, other._data_variables)
        return result

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Mesh bounding box as (xmin, ymin, xmax, ymax)."""
        return self._mesh.bounds

    @property
    def global_attributes(self) -> dict[str, Any]:
        """File-level NetCDF attributes."""
        return self._global_attributes

    @property
    def n_node(self) -> int:
        """Number of mesh nodes."""
        return self._mesh.n_node

    @property
    def n_face(self) -> int:
        """Number of mesh faces."""
        return self._mesh.n_face

    @property
    def n_edge(self) -> int:
        """Number of mesh edges."""
        return self._mesh.n_edge

    def __getitem__(self, key: str) -> MeshVariable:
        """Get a data variable by name using bracket notation -- the xarray idiom ``mesh[name]``.

        Args:
            key: The variable name.

        Returns:
            MeshVariable: The variable.

        Raises:
            KeyError: No variable is named ``key``; the message lists the available names.
        """
        if key not in self._data_variables:
            raise KeyError(
                f"Variable '{key}' not found. Available: {self.variable_names}"
            )
        return self._data_variables[key]

    def get_data(self, variable_name: str) -> MeshVariable:
        """Deprecated alias for ``mesh[variable_name]``.

        Deprecated: Use subscription ``mesh[name]`` (or :meth:`get` for a default) instead; it is the same lookup and
            matches the xarray idiom.

        Args:
            variable_name: Name of the data variable.

        Returns:
            MeshVariable: The variable.

        Raises:
            KeyError: No variable is named ``variable_name``.
        """
        warnings.warn(
            "MeshDataset.get_data is deprecated; use mesh[name] (or mesh.get(name, default)).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self[variable_name]

    def __contains__(self, key: str) -> bool:
        """Whether ``key`` names a data variable of this dataset.

        Args:
            key: The variable name to test.

        Returns:
            bool: True when the dataset carries a variable of that name.
        """
        return key in self._data_variables

    def __len__(self) -> int:
        """The number of data variables.

        Returns:
            int: How many data variables the dataset carries. Note this counts *variables*,
            not elements -- :attr:`n_face` / :attr:`n_node` / :attr:`n_edge` are the mesh
            sizes.
        """
        return len(self._data_variables)

    def __iter__(self) -> Any:
        """Iterate over the data variable names.

        Returns:
            Iterator[str]: The variable names, in insertion order -- so ``list(dataset)``
            and ``dataset.keys()`` agree, and ``for name in dataset`` reads like a mapping.
        """
        return iter(self._data_variables)

    def keys(self) -> list[str]:
        """The data variable names (read-only mapping surface).

        Returns:
            list[str]: The names, in insertion order. A list rather than a view, so it does
            not track later edits -- every variable edit returns a new dataset anyway.
        """
        return list(self._data_variables.keys())

    def values(self) -> list[MeshVariable]:
        """The data variables (read-only mapping surface).

        Returns:
            list[MeshVariable]: The variables, in the same order as :meth:`keys`. The
            variables themselves are not copied, so reading ``.data`` on one still fires its
            lazy loader.
        """
        return list(self._data_variables.values())

    def items(self) -> list[tuple[str, MeshVariable]]:
        """The ``(name, variable)`` pairs (read-only mapping surface).

        Returns:
            list[tuple[str, MeshVariable]]: One pair per variable, in insertion order.
        """
        return list(self._data_variables.items())

    def get(self, key: str, default: Any = None) -> MeshVariable | Any:
        """The variable ``key``, or ``default`` when it is absent.

        Args:
            key: The variable name to look up.
            default: What to answer when no variable is named ``key``. Defaults to ``None``.

        Returns:
            MeshVariable | Any: The variable, or ``default``.
        """
        return self._data_variables.get(key, default)

    @property
    def data_vars(self) -> dict[str, MeshVariable]:
        """A copy of the ``{name: variable}`` mapping.

        A fresh dict, so mutating it never changes the dataset — variable edits go through
        :meth:`with_variable` / :meth:`drop_variables` / :meth:`rename_variable`, which each
        return a new dataset.
        """
        return dict(self._data_variables)

    def with_variable(
        self,
        name: str,
        data: np.ndarray,
        *,
        location: str = "face",
        nodata: float | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshVariables.with_variable`."""
        warnings.warn(
            "`mesh.with_variable` is deprecated and will be removed in 0.90.0; use "
            "`mesh.variables.with_variable`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variables.with_variable(
            name, data, location=location, nodata=nodata
        )

    def drop_variables(self, names: str | list[str]) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshVariables.drop_variables`."""
        warnings.warn(
            "`mesh.drop_variables` is deprecated and will be removed in 0.90.0; use "
            "`mesh.variables.drop_variables`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variables.drop_variables(names)

    def rename_variable(self, old: str, new: str) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshVariables.rename_variable`."""
        warnings.warn(
            "`mesh.rename_variable` is deprecated and will be removed in 0.90.0; use "
            "`mesh.variables.rename_variable`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variables.rename_variable(old, new)

    @property
    def metadata(self) -> MeshMetadata:
        """Full metadata summary for this dataset."""
        topo_tuple = (self._topology_info,) if self._topology_info else ()
        data_vars = {name: var.location for name, var in self._data_variables.items()}
        conventions = self._global_attributes.get("Conventions")
        result = MeshMetadata(
            mesh_topologies=topo_tuple,
            data_variables=data_vars,
            global_attributes=self._global_attributes,
            conventions=conventions,
            n_nodes=self.n_node,
            n_faces=self.n_face,
            n_edges=self.n_edge,
        )
        return result

    def to_dataset(
        self,
        variable_name: str,
        cell_size: float,
        method: str = "nearest",
        bounds: tuple[float, float, float, float] | None = None,
        epsg: int | None = None,
        nodata: float = -9999.0,
    ) -> Dataset:
        """Facade -- delegates to :meth:`_MeshIO.to_dataset`."""
        warnings.warn(
            "`mesh.to_dataset` is deprecated and will be removed in 0.90.0; use "
            "`mesh.io.to_dataset`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.io.to_dataset(
            variable_name,
            cell_size,
            method=method,
            bounds=bounds,
            epsg=epsg,
            nodata=nodata,
        )

    def _to_dataset_impl(
        self,
        variable_name: str,
        cell_size: float,
        method: str = "nearest",
        bounds: tuple[float, float, float, float] | None = None,
        epsg: int | None = None,
        nodata: float = -9999.0,
    ) -> Dataset:
        """Implementation of :meth:`_MeshIO.to_dataset`; see that method for the contract."""
        var = self[variable_name]
        data = var.data
        if data is None:
            raise ValueError(f"Variable '{variable_name}' has no data loaded.")
        if var.has_time:
            # First step along the variable's own time axis (not blindly axis 0, which for
            # a trailing-time variable would take the first face) — see `_element_values`.
            data = np.take(np.asarray(data), 0, axis=cast("int", var.time_axis_index))

        grid_array, geotransform = mesh_to_grid(
            mesh=self._require_mesh2d("rasterising to a grid (to_dataset)"),
            data=data,
            location=var.location,
            cell_size=cell_size,
            method=method,
            bounds=bounds,
            nodata=nodata,
        )

        # `crs_spec` rather than `self.epsg or 4326`: a mesh carrying a projected
        # WKT with no EPSG code has `epsg is None`, and the old default stamped
        # EPSG:4326 onto metre coordinates. That is the case this resolves.
        #
        # The `or 4326` behind it keeps the *other* case as it was. A mesh with
        # no CRS at all -- no code and no WKT -- has always rasterised to WGS 84
        # here, and `to_dataset(...).to_file("out.tif")` writing a GeoTIFF with
        # no CRS instead would be a silent behaviour change for every caller
        # holding such a mesh. Narrowing that default is a decision for its own
        # change, not a side effect of teaching this line about WKT.
        target_epsg = (
            epsg if epsg is not None else crs_spec(self.epsg, self.crs_wkt) or 4326
        )
        result = Dataset.from_array(
            grid_array,
            no_data_value=nodata,
            geo_ref=GeoReference(
                geo=cast(
                    "tuple[float, float, float, float, float, float]", geotransform
                ),
                epsg=target_epsg,
            ),
        )
        return result

    def crop(
        self,
        mask: Any = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: int | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSpatial.crop`."""
        warnings.warn(
            "`mesh.crop` is deprecated and will be removed in 0.90.0; use "
            "`mesh.spatial.crop`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.spatial.crop(mask, touch, bbox=bbox, epsg=epsg)

    def _wrap_subset(
        self, mesh: Mesh2d, data_variables: dict[str, MeshVariable]
    ) -> MeshDataset:
        """Wrap a ``(mesh, data_variables)`` pair from the spatial subsetters into a dataset.

        The spatial subsetting helpers (:func:`clip_mesh` / :func:`subset_by_bounds`) return
        the rebuilt mesh and data variables rather than a dataset (STR-3 — keeps
        ``ugrid.spatial`` independent of this module). This carries the source dataset's
        global attributes / topology info / CRS onto the subset.

        Args:
            mesh: The subset mesh.
            data_variables: The sliced data variables.

        Returns:
            MeshDataset: The wrapped subset.
        """
        return type(self)(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    def clip(self, mask: Any, touch: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.clip`."""
        warnings.warn(
            "`mesh.clip` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.clip`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.clip(mask, touch=touch)

    def subset_by_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSpatial.subset_by_bounds`."""
        warnings.warn(
            "`mesh.subset_by_bounds` is deprecated and will be removed in 0.90.0; use "
            "`mesh.spatial.subset_by_bounds`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.spatial.subset_by_bounds(xmin, ymin, xmax, ymax)

    def to_crs(self, to_epsg: int | str | Any) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSpatial.to_crs`."""
        warnings.warn(
            "`mesh.to_crs` is deprecated and will be removed in 0.90.0; use "
            "`mesh.spatial.to_crs`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.spatial.to_crs(to_epsg)

    @property
    def time_values(self) -> list | None:
        """Parsed time coordinate values from the first temporal variable.

        Returns None if no variables have a time dimension.
        """
        result = None
        for var in self._data_variables.values():
            if var.has_time:
                time_attr = var.attributes.get("time_values")
                if time_attr is not None:
                    result = list(time_attr)
                else:
                    result = list(range(var.n_time_steps))
                break
        return result

    def sel_time(self, index: int) -> MeshDataset:
        """Deprecated: use ``isel(time=index)``.

        `sel_time` took an **integer** position, which is `isel` semantics under a `sel` name,
        and the same class already exposes :meth:`isel`, which collapses a single step through
        the identical :func:`_collapse_variable_to_step` (so the result is byte-for-byte the
        same, including the #982 lazy slab read). This forwards to it.

        Args:
            index: Time step index (0-based; a negative index counts from the end).

        Returns:
            MeshDataset: The single step, with the time dimension dropped.
        """
        warnings.warn(
            "MeshDataset.sel_time is deprecated; use isel(time=index) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.isel(time=index)

    def sel_time_range(self, start: int, stop: int) -> MeshDataset:
        """Deprecated: use ``isel(time=slice(start, stop))``.

        `sel_time_range` took **integer** positions (`stop` exclusive), which is `isel` with a
        slice. :meth:`isel` keeps the time dimension for a slice exactly as this did, so the
        result is identical.

        Args:
            start: Start index (inclusive).
            stop: Stop index (exclusive).

        Returns:
            MeshDataset: The selected time range, time dimension kept.
        """
        warnings.warn(
            "MeshDataset.sel_time_range is deprecated; use "
            "isel(time=slice(start, stop)) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.isel(time=slice(start, stop))

    def attach_data(
        self, path: str | Path, driver: str | None = None, lazy: bool = True
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshVariables.attach_data`."""
        warnings.warn(
            "`mesh.attach_data` is deprecated and will be removed in 0.90.0; use "
            "`mesh.variables.attach_data`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variables.attach_data(path, driver, lazy)

    def _attach_data_impl(
        self, path: str | Path, driver: str | None = None, lazy: bool = True
    ) -> MeshDataset:
        """Implementation of :meth:`_MeshVariables.attach_data`.

        Kept on ``MeshDataset`` rather than moved onto the variables engine: the body reads the
        results file through the module-level ``_require_existing_file`` /
        ``_resolve_attach_driver`` / ``_read_data_through_driver`` helpers in this module, which
        the engine cannot import without a cycle (``dataset`` imports the engine for its
        ``variables`` accessor).

        Some formats carry values but no topology — an SMS ``.dat``, an XMDF, an XDMF — and must
        be paired with a mesh read from elsewhere. Those drivers declare ``needs_mesh``, and
        :meth:`read_file` refuses them, pointing here.

            >>> mesh = MeshDataset.read_file("domain.2dm")       # doctest: +SKIP
            >>> result = mesh.variables.attach_data("velocity.dat")  # doctest: +SKIP

        The driver returns **only the variables it read**; this method merges them onto the
        dataset. That division is deliberate — it makes a name the dataset already defines an
        unambiguous collision, and means a driver cannot drop the dataset's own variables by
        rebuilding the mapping.

        The driver checks the file's declared element count against this mesh. That check is the
        only safeguard there is: a bare results file carries nothing identifying the mesh it
        belongs to, so attaching it to the wrong one would otherwise produce a plausible and
        entirely wrong dataset.

        Args:
            path: The results file to read.
            driver: Name of a registered driver to use, bypassing detection. ``None`` (the
                default) picks the ``needs_mesh`` driver claiming the file's suffix.
            lazy: Defer reading variable values until first access. Defaults to True.

        Returns:
            A new dataset carrying this mesh plus the attached variables. This dataset is
            unchanged.

        Raises:
            FileNotFoundError: If the results file does not exist.
            DriverNotExistError: If ``driver`` names a driver that is not installed.
            UnsupportedMeshFormatError: If no installed driver can attach this file, the
                named driver does not declare ``needs_mesh``, or the driver returned
                something other than a mapping of name to :class:`MeshVariable`.
            ValueError: If the driver returns a variable whose name this dataset already
                uses, or reports an element-count mismatch against this mesh.
        """
        _require_existing_file(path)
        selected = _resolve_attach_driver(path, driver)
        produced = _read_data_through_driver(selected, path, self, lazy)
        collisions = sorted(set(produced) & set(self._data_variables))
        if collisions:
            raise ValueError(
                f"The {selected.name!r} driver read variable(s) {collisions} from "
                f"{str(path)!r}, which this dataset already defines. Rename one side before "
                f"attaching, or drop the existing variable first."
            )
        merged = dict(self._data_variables)
        merged.update(produced)
        return type(self)(
            mesh=self._mesh,
            data_variables=merged,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            volume_topology=self._volume_topology,
            file_name=self._file_name,
        )

    def to_file(
        self, path: str | Path, driver: str | None = None, **options: Any
    ) -> None:
        """Facade -- delegates to :meth:`_MeshIO.to_file`."""
        warnings.warn(
            "`mesh.to_file` is deprecated and will be removed in 0.90.0; use "
            "`mesh.io.to_file`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.io.to_file(path, driver, **options)

    def _to_file_impl(
        self, path: str | Path, driver: str | None = None, **options: Any
    ) -> None:
        """Implementation of :meth:`_MeshIO.to_file`; see that method for the contract."""
        if driver is not None:
            self._write_through_driver(path, driver, options)
        elif options:
            raise TypeError(
                f"to_file() got unexpected keyword argument(s) "
                f"{sorted(options)}; format options apply to a `driver=` write only."
            )
        else:
            self._write_ugrid_netcdf(Path(path))

    def to_bytes(self) -> bytes:
        """Facade -- delegates to :meth:`_MeshIO.to_bytes`."""
        warnings.warn(
            "`mesh.to_bytes` is deprecated and will be removed in 0.90.0; use "
            "`mesh.io.to_bytes`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.io.to_bytes()

    def _to_bytes_impl(self) -> bytes:
        """Implementation of :meth:`_MeshIO.to_bytes`; see that method for the contract."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._write_ugrid_netcdf(tmp_path)
            data = tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except PermissionError:
                pass
        return data

    def _write_ugrid_netcdf(self, path: Path) -> None:
        """Write UGRID-compliant NetCDF through GDAL's multidimensional API.

        The built-in writer, unchanged from before driver dispatch existed.

        Args:
            path: Output file path.
        """
        drv = gdal.GetDriverByName("netCDF")
        ds = drv.CreateMultiDimensional(str(path))
        rg = ds.GetRootGroup()

        mesh_name = self.mesh_name
        dims = write_ugrid_topology(
            rg, self._require_mesh2d("writing a UGRID file"), mesh_name, self._crs_wkt
        )
        if self._volume_topology is not None:
            # Before the data variables, so the `nVolumes` dimension exists when a
            # `location="volume"` variable is written against it (issue #1330).
            write_volume_topology(rg, mesh_name, self._volume_topology, dims)

        _require_one_time_length(self._data_variables)
        for var in self._data_variables.values():
            if var.has_time and DEFAULT_TIME_DIM not in dims:
                # The writer's time dimension is always called "time": the topology writer and
                # `write_ugrid_data_variable` both address it by that name, so a variable
                # whose own axis is named otherwise is written against this one. See
                # `_write_time_coordinate` for what that means for the coordinate.
                time_dim = rg.CreateDimension(
                    DEFAULT_TIME_DIM, None, None, var.n_time_steps
                )
                dims[DEFAULT_TIME_DIM] = time_dim
            # `load_array` reads each variable without memoising it on the shared dataset, so the
            # write streams one variable at a time instead of holding the whole cube resident (#982).
            write_ugrid_data_variable(
                rg, var.with_data(var.load_array()), mesh_name, dims
            )

        # Write the CF time coordinate beside its dimension, so a round trip keeps the stamps
        # the reader decoded instead of dropping them to positional indices (issue #1381).
        _write_time_coordinate(rg, dims, self._data_variables)

        global_attrs = dict(self._global_attributes)
        if "Conventions" not in global_attrs:
            global_attrs["Conventions"] = UGRID_CONVENTIONS
        write_global_attributes(rg, global_attrs)

        ds = None

    def _write_through_driver(
        self, path: str | Path, driver: str, options: dict[str, Any]
    ) -> None:
        """Write through a registered third-party driver.

        Args:
            path: Output file path.
            driver: Name of the driver to write with.
            options: Format-specific settings, passed to the driver uninterpreted.

        Raises:
            DriverNotExistError: If the driver is not installed.
            UnsupportedMeshFormatError: If the driver cannot write a mesh, or raised.
        """
        if driver.strip().lower() in RESERVED_NAMES:
            self._write_ugrid_netcdf(Path(path))
            return
        installed = mesh_drivers()
        if driver not in installed:
            raise DriverNotExistError(_unknown_driver_message(driver))
        selected = installed[driver]
        if "write_mesh" not in selected.capabilities:
            can_do = ", ".join(sorted(selected.capabilities))
            raise UnsupportedMeshFormatError(
                f"The {driver!r} driver cannot write a mesh; it declares: {can_do}."
            )
        try:
            selected.write(self, str(path), **options)
        except (UnsupportedMeshFormatError, DriverNotExistError, OSError):
            raise
        except Exception as error:
            raise UnsupportedMeshFormatError(
                f"The {driver!r} driver failed to write {str(path)!r}: {error!r}"
            ) from error

        # Warned after the write, not before: a failed write previously still reported that
        # variables "will not be in" a file that was never created.
        if "write_data" not in selected.capabilities and self._data_variables:
            warnings.warn(
                f"The {driver!r} driver wrote a mesh frame but no data; "
                f"{len(self._data_variables)} variable(s) are not in {str(path)!r}.",
                RuntimeWarning,
                stacklevel=4,
            )

    def to_geodataframe(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> gpd.GeoDataFrame:
        """Facade -- delegates to :meth:`_MeshVector.to_geodataframe`."""
        warnings.warn(
            "`mesh.to_geodataframe` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.to_geodataframe`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.to_geodataframe(variable_name, location)

    def _build_geometries(self, location: str) -> list:
        """Build the geometry list for a mesh location.

        Args:
            location: Mesh location ("face", "node", or "edge").

        Returns:
            List of shapely geometries — Polygons for "face", Points for
            "node", LineStrings for "edge".

        Raises:
            ValueError: If `location` is unknown, or edge connectivity is
                unavailable for an edge conversion.
        """
        if location == "face":
            geometries = MeshSpatialIndex(
                self._require_mesh2d("the face spatial index")
            ).face_polygons
        elif location == "node":
            # Vectorized point construction — meshes routinely have 1e5-1e7 nodes, so a per-node
            # Python `Point(...)` loop is a hot spot (ARC-59). `list(...)` keeps the return type
            # consistent with the face/edge branches (review N2).
            geometries = list(shapely.points(self._mesh.node_x, self._mesh.node_y))
        elif location == "edge":
            if self._mesh.edge_node_connectivity is None:
                raise ValueError("Edge connectivity not available.")
            geometries = self._edge_linestrings(self._mesh.edge_node_connectivity)
        else:
            raise ValueError(f"Unknown location: {location}")
        return geometries

    def _edge_linestrings(self, enc: Connectivity) -> Any:
        """Build one LineString per edge, vectorized for standard 2-node edges (ARC-59)."""
        node_idx = np.asarray(enc.data)
        # A `None` fill means no sentinels are present, so the fast path is valid without an
        # elementwise `node_idx != None` compare (which NumPy deprecates) (review N3).
        no_fill = enc.fill_value is None or bool(np.all(node_idx != enc.fill_value))
        if node_idx.ndim == 2 and node_idx.shape[1] == 2 and no_fill:
            xs = self._mesh.node_x[node_idx]
            ys = self._mesh.node_y[node_idx]
            return list(shapely.linestrings(np.stack([xs, ys], axis=-1)))
        # Rare ragged / filled edge connectivity: fall back to a per-edge build.
        # See `Mesh1d._edge_linestrings`: a row with fewer than two real nodes is no
        # edge, and GEOS refuses a one-point LineString, so such a row escaped as a raw
        # GEOSException. An empty geometry keeps one row per declared edge.
        return [
            LineString(
                [
                    (self._mesh.node_x[n], self._mesh.node_y[n])
                    for n in enc.get_element(i)
                ]
            )
            if len(enc.get_element(i)) >= 2
            else LineString()
            for i in range(enc.n_elements)
        ]

    def to_feature_collection(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> FeatureCollection:
        """Facade -- delegates to :meth:`_MeshVector.to_feature_collection`."""
        warnings.warn(
            "`mesh.to_feature_collection` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.to_feature_collection`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.to_feature_collection(variable_name, location)

    def voronoi_polygons(
        self, *, clip: FeatureCollection | None = None
    ) -> FeatureCollection:
        """Facade -- delegates to :meth:`_MeshVector.voronoi_polygons`."""
        warnings.warn(
            "`mesh.voronoi_polygons` is deprecated and will be removed in 0.90.0; use "
            "`mesh.vector.voronoi_polygons`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vector.voronoi_polygons(clip=clip)

    @classmethod
    def from_arrays(
        cls,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: np.ndarray,
        data: dict[str, np.ndarray] | None = None,
        data_locations: dict[str, str] | None = None,
        epsg: int = 4326,
        mesh_name: str = DEFAULT_MESH_NAME,
    ) -> MeshDataset:
        """Create a MeshDataset programmatically from arrays.

        Plural because an unstructured mesh is not one array: the topology
        needs node coordinates *and* a face-node connectivity table before any
        data can be attached. It therefore takes a flat `epsg` rather than the
        :class:`~pyramids.base.georeference.GeoReference` the gridded
        constructors take — a mesh carries its own coordinates, so there is no
        affine transform to describe.

        Args:
            node_x: Node x-coordinates.
            node_y: Node y-coordinates.
            face_node_connectivity: (n_faces, max_nodes) array of node
                indices. Use -1 as fill value for mixed meshes.
            data: Optional dict mapping variable name to data array.
            data_locations: Optional dict mapping variable name to
                location ("face", "node", "edge"). Defaults to "face".
            epsg: EPSG code for the CRS.
            mesh_name: Name for the topology variable.

        Returns:
            MeshDataset instance.

        Examples:
            - Build the smallest possible mesh — two triangles — and inspect
              its topology:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ... )
                >>> (mesh.n_node, mesh.n_face)
                (4, 2)
                >>> mesh.bounds
                (0.0, 0.0, 1.0, 1.0)

                ```
            - Attach a per-face variable and read it back:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([1.5, 2.5])},
                ... )
                >>> mesh.variable_names
                ['depth']
                >>> mesh["depth"].location
                'face'
                >>> mesh["depth"].data.tolist()
                [1.5, 2.5]

                ```
        """
        fnc = Connectivity(
            data=np.asarray(face_node_connectivity, dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.asarray(node_x, dtype=np.float64),
            node_y=np.asarray(node_y, dtype=np.float64),
            face_node_connectivity=fnc,
        )

        data_variables: dict[str, MeshVariable] = {}
        topo_data_vars: dict[str, str] = {}
        if data is not None:
            if data_locations is None:
                data_locations = {}
            for name, arr in data.items():
                loc = data_locations.get(name, "face")
                topo_data_vars[name] = loc
                data_variables[name] = MeshVariable(
                    name=name,
                    location=loc,
                    mesh_name=mesh_name,
                    shape=arr.shape,
                    _data=arr,
                )

        srs = sr_from_epsg(epsg)
        crs_wkt = srs.ExportToWkt()

        topo_info = MeshTopologyInfo(
            mesh_name=mesh_name,
            topology_dimension=2,
            node_x_var=f"{mesh_name}_node_x",
            node_y_var=f"{mesh_name}_node_y",
            face_node_var=f"{mesh_name}_face_nodes",
            data_variables=topo_data_vars,
            crs_wkt=crs_wkt,
        )

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={"Conventions": UGRID_CONVENTIONS},
            topology_info=topo_info,
            crs_wkt=crs_wkt,
        )
        return result

    @classmethod
    def from_mesh1d(
        cls,
        node_x: np.ndarray,
        node_y: np.ndarray,
        edge_node_connectivity: np.ndarray,
        data: dict[str, np.ndarray] | None = None,
        data_locations: dict[str, str] | None = None,
        directed: bool = False,
        epsg: int = 4326,
        mesh_name: str = DEFAULT_MESH_NAME,
    ) -> MeshDataset:
        """Build a dataset on a 1D topology (:class:`Mesh1d`) from raw arrays (issue #1652).

        The 1D counterpart of :meth:`from_arrays`: that one takes ``face_node_connectivity``
        and builds a :class:`Mesh2d`; this takes ``edge_node_connectivity`` and builds a
        :class:`Mesh1d` (nodes + edges, no faces), so a 1D mesh gets the same data container
        the 2D path has. Data variables default to the ``"node"`` location rather than
        ``"face"`` -- a 1D mesh has no faces.

        Args:
            node_x: Node x-coordinates, shape ``(n_node,)``.
            node_y: Node y-coordinates, shape ``(n_node,)``.
            edge_node_connectivity: ``(n_edge, 2)`` node indices, one pair per edge.
            data: Optional ``{name: array}`` of data variables. A 1-D array is static (one
                value per element); a 2-D array is temporal (leading axis = time).
            data_locations: Optional ``{name: location}`` with ``"node"`` or ``"edge"``.
                Defaults to ``"node"``.
            directed: Whether the edges carry a flow direction. Defaults to False.
            epsg: EPSG code for the mesh CRS. Defaults to 4326.
            mesh_name: Name of the topology variable. Defaults to the package default.

        Returns:
            MeshDataset: The dataset carrying the 1D mesh and its variables.

        Examples:
            - Attach a per-node variable to a 1D chain and read it back:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> ds = MeshDataset.from_mesh1d(
                ...     node_x=np.array([0.0, 1.0, 2.0]),
                ...     node_y=np.array([0.0, 0.0, 0.0]),
                ...     edge_node_connectivity=np.array([[0, 1], [1, 2]]),
                ...     data={"s1": np.array([1.0, 2.0, 3.0])},
                ...     data_locations={"s1": "node"},
                ... )
                >>> (ds.n_node, ds.n_edge)
                (3, 2)
                >>> ds["s1"].location
                'node'

                ```
        """
        # Local import: `mesh1d` imports from this module (the built-in reader builds a
        # MeshDataset), so a module-level import here would be a cycle.
        from pyramids.netcdf.ugrid.mesh1d import Mesh1d

        mesh = Mesh1d.from_arrays(
            node_x=np.asarray(node_x, dtype=np.float64),
            node_y=np.asarray(node_y, dtype=np.float64),
            edge_node_connectivity=np.asarray(edge_node_connectivity, dtype=np.intp),
            directed=directed,
            epsg=epsg,
        )

        data_variables: dict[str, MeshVariable] = {}
        topo_data_vars: dict[str, str] = {}
        if data is not None:
            if data_locations is None:
                data_locations = {}
            for name, arr in data.items():
                loc = data_locations.get(name, "node")
                topo_data_vars[name] = loc
                data_variables[name] = MeshVariable(
                    name=name,
                    location=loc,
                    mesh_name=mesh_name,
                    shape=arr.shape,
                    _data=arr,
                )

        srs = sr_from_epsg(epsg)
        crs_wkt = srs.ExportToWkt()

        topo_info = MeshTopologyInfo(
            mesh_name=mesh_name,
            topology_dimension=1,
            node_x_var=f"{mesh_name}_node_x",
            node_y_var=f"{mesh_name}_node_y",
            edge_node_var=f"{mesh_name}_edge_nodes",
            data_variables=topo_data_vars,
            crs_wkt=crs_wkt,
        )

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={"Conventions": UGRID_CONVENTIONS},
            topology_info=topo_info,
            crs_wkt=crs_wkt,
        )
        return result

    @classmethod
    def from_structured(
        cls,
        dataset: Any,
        band_names: list[str] | None = None,
        mesh_name: str = DEFAULT_MESH_NAME,
    ) -> MeshDataset:
        """Build a mesh from a raster, one quad face per cell.

        The bridge from the structured world: each cell becomes a quad over the corner
        lattice (shared corners deduplicated) and each band becomes a per-face variable, so a
        raster can be fed to the mesh operations.

        Args:
            dataset: The source raster (a pyramids ``Dataset`` / ``NetCDF`` variable).
            band_names: Names for the per-face variables, one per band. Defaults to
                ``band_1``, ``band_2``, ….
            mesh_name: The topology name to stamp. Defaults to the package default.

        Returns:
            MeshDataset: The quad mesh carrying the band values.
        """
        node_x, node_y, faces, data = _construct.from_structured(dataset, band_names)
        return cls.from_arrays(
            node_x,
            node_y,
            faces,
            data=data,
            epsg=dataset.epsg,
            mesh_name=mesh_name,
        )

    @classmethod
    def from_geodataframe(
        cls, gdf: Any, mesh_name: str = DEFAULT_MESH_NAME
    ) -> MeshDataset:
        """Build a mesh from polygon geometries, one face per polygon.

        Vertices are deduplicated into the node set and each ring is oriented
        counter-clockwise; the frame's attribute columns become per-face variables.

        Args:
            gdf: A GeoDataFrame of polygons. Its ``crs`` is carried onto the mesh verbatim,
                so a projected CRS stays projected even when it has no EPSG code.
            mesh_name: The topology name to stamp. Defaults to the package default.

        Returns:
            MeshDataset: The mesh built from the polygons.

        Raises:
            ValueError: The frame declares no CRS, so the mesh would carry no
                georeferencing.
        """
        node_x, node_y, faces, data = _construct.from_geodataframe(gdf)
        # Carry the frame's own CRS. Reading an EPSG code and defaulting to 4326 stamped degrees
        # on a projected frame whose CRS has no EPSG code, and on a frame with no CRS at all --
        # silent georeferencing corruption, since everything downstream trusts `epsg`.
        crs = getattr(gdf, "crs", None)
        if crs is None:
            raise ValueError(
                "from_geodataframe: the GeoDataFrame declares no CRS, so the mesh would have no "
                "georeferencing. Set `gdf.crs` (e.g. gdf.set_crs(4326)) before converting."
            )
        epsg = crs.to_epsg()
        built = cls.from_arrays(
            node_x,
            node_y,
            faces,
            data=data,
            epsg=epsg if epsg is not None else 4326,
            mesh_name=mesh_name,
        )
        if epsg is not None:
            return built
        # A CRS with no EPSG code (a projected one given as WKT, say) would otherwise be
        # reduced to the 4326 placeholder `from_arrays` needs, stamping degrees on metres.
        # Keep the frame's WKT verbatim instead.
        wkt = crs.to_wkt()
        return cls(
            mesh=built.mesh,
            data_variables=built.data_vars,
            global_attributes={"Conventions": UGRID_CONVENTIONS},
            topology_info=_topology_info_named(mesh_name, crs_wkt=wkt),
            crs_wkt=wkt,
            file_name=None,
        )

    @classmethod
    def from_xarray(cls, ds: Any) -> MeshDataset:
        """Read a mesh back from an ``xarray.Dataset`` in the UGRID-CF encoding.

        The inverse of :meth:`to_xarray`. xarray is a peer dependency, imported only inside
        the bridge, so nothing in this module imports it at module scope.

        Args:
            ds: The ``xarray.Dataset`` to read.

        Returns:
            MeshDataset: The reconstructed mesh and variables.

        Raises:
            OptionalPackageDoesNotExist: xarray is not installed.
        """
        mesh, data_variables = _interop.from_xarray(ds)
        mesh_name = _interop.read_mesh_name(ds)
        return cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={},
            # Keep the topology's own name: without it a mesh called anything other than
            # "mesh2d" came back renamed, and a re-export then wrote the wrong dimensions.
            topology_info=_topology_info_named(mesh_name),
            crs_wkt=_interop.read_crs_wkt(ds),
            file_name=None,
            # Rebuild the ragged volume topology from its companion arrays when present
            # (issue #1330); `None` for a 2-D / uniform mesh, leaving the dense path in force.
            volume_topology=_interop.volume_topology_from_xarray(ds),
        )

    @classmethod
    def from_zarr(cls, store: Any, **kwargs: Any) -> MeshDataset:
        """Read a mesh from a zarr store written by :meth:`to_zarr`.

        Args:
            store: The zarr store (a path or a mapping).
            **kwargs: Passed to ``xarray.open_zarr``.

        Returns:
            MeshDataset: The reconstructed mesh and variables.

        Raises:
            OptionalPackageDoesNotExist: xarray or zarr is not installed.
        """
        # One open, read twice: the mesh and the CRS both come off the same dataset, so the
        # store is not opened again just to recover the grid mapping.
        opened = _interop.open_zarr_dataset(store, **kwargs)
        mesh, data_variables = _interop.from_xarray(opened)
        return cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={},
            topology_info=_topology_info_named(_interop.read_mesh_name(opened)),
            crs_wkt=_interop.read_crs_wkt(opened),
            file_name=None,
        )

    @staticmethod
    def mesh_names(path: str | Path, driver: str | None = None) -> list[str]:
        """The names of every topology a file declares.

        A mesh file may carry several topologies; :meth:`read_file` takes the first unless told
        otherwise, so this is how a caller discovers what else is in there.

        Dispatch mirrors :meth:`read_file` exactly, including the short-circuit that keeps a
        NetCDF suffix from triggering driver discovery. Every driver is asked, not only those
        declaring ``multi_mesh``: a single-mesh format answers with a one-item list, which is
        the right answer rather than an error.

        Args:
            path: Path to the file.
            driver: Name of a registered driver to ask, bypassing detection. ``None`` (the
                default) resolves as described above.

        Returns:
            list[str]: One name per declared topology, in file order.

        Raises:
            FileNotFoundError: The file does not exist.
            DriverNotExistError: ``driver`` names a driver that is not installed.
            UnsupportedMeshFormatError: No installed driver claims the suffix and the
                built-in reader cannot open it either, or a driver failed or returned
                something other than a list of topology names.
            IsADirectoryError: The path is a directory that no driver claims.
            RuntimeError: GDAL cannot open a file that reached the built-in reader by its own
                suffix. GDAL exceptions are enabled package-wide, so the open failure
                surfaces as GDAL's own error.
            ValueError: The file declares no UGRID topology.
        """
        selected, native = _resolve_read_driver(path, driver)
        if selected is not None:
            result = _mesh_names_through_driver(selected, path)
        else:
            try:
                result = MeshDataset._ugrid_netcdf_mesh_names(
                    path, wrap_unreadable=not native
                )
            except NoMeshTopologyError:
                fallback = _netcdf_fallback_driver(path, driver)
                if fallback is None:
                    raise
                result = _mesh_names_through_driver(fallback, path)
        return result

    @staticmethod
    def _ugrid_netcdf_mesh_names(
        path: str | Path, wrap_unreadable: bool = False
    ) -> list[str]:
        """List the topology names in a UGRID NetCDF file.

        Args:
            path: Path to the ``.nc`` file.
            wrap_unreadable: True when this call is the fallback for a suffix no driver
                claimed, so a failure reports the installed drivers rather than GDAL's
                "not recognized" — the same treatment :meth:`read_file` gives it.

        Returns:
            list[str]: One name per declared topology, in file order.

        Raises:
            FileNotFoundError: The file does not exist.
            UnsupportedMeshFormatError: GDAL cannot open a file that reached here as a
                fallback from driver detection.
            ValueError: The file declares no UGRID topology.
        """
        target = str(path)
        if not target.startswith("/vsi"):
            resolved = Path(path)
            if not resolved.exists():
                raise FileNotFoundError(
                    f"File not found: {Path(resolved)}"
                )  # pragma: no cover - `_require_existing_file` runs first on every public route
            target = str(resolved)
        # See `read_file`: with GDAL exceptions on an unopenable file raises RuntimeError,
        # but that setting is process-global and reversible, so the guard stays.
        try:
            dataset = gdal.OpenEx(
                target, gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR
            )
        except RuntimeError as error:
            if not wrap_unreadable:
                raise
            if not target.startswith("/vsi") and Path(target).is_dir():
                raise IsADirectoryError(
                    f"Not a mesh file: {target} is a directory."
                ) from error
            suffix = "" if target.startswith("/vsi") else Path(target).suffix
            raise UnsupportedMeshFormatError(
                unsupported_format_message(str(path), suffix)
            ) from error
        if (
            dataset is None
        ):  # pragma: no cover - needs gdal.DontUseExceptions() process-wide
            raise ValueError(f"GDAL cannot open file: {path}")
        rg = dataset.GetRootGroup()
        if rg is None:  # pragma: no cover - no multidim driver answers None here
            # Kept as a clear message rather than letting `parse_ugrid_topology(None)`
            # raise an AttributeError, though every multidim driver that opens a file
            # does expose a root group.
            raise ValueError(f"Cannot get root group from: {path}")
        topologies = parse_ugrid_topology(rg)
        if not topologies:
            raise NoMeshTopologyError(f"No UGRID mesh topology found in: {path}")
        return [topo.mesh_name for topo in topologies]

    @classmethod
    def _open_mfdataset_lazy(
        cls,
        resolved: Sequence[str | Path],
        *,
        chunks: dict | str | int | tuple[int, ...] | None,
        driver: str | None,
        mesh_name: str | None,
        sort: bool,
    ) -> Any:
        """Hand already-resolved paths to the backend's lazy mesh reader.

        Deliberately probed with :func:`getattr` rather than declared on the
        :class:`~pyramids.base._compute_backend.ComputeBackend` protocol. That protocol is
        ``runtime_checkable`` and :func:`pyramids.base._compute_backend._load` drops any backend
        failing its ``isinstance`` check, so adding a member there would silently disable *every*
        already-released plugin -- not just the mesh path (issue #1458).

        Args:
            resolved: The concrete paths, already globbed and validated by the caller so a bad
                pattern fails identically whether the read is eager or lazy.
            chunks: The backend's chunking argument; never ``None`` here.
            driver: Mesh driver name, or ``None`` to detect.
            mesh_name: Topology to take from each file, or ``None`` for the first.
            sort: Forwarded for the backend to apply. Core cannot pre-sort without opening
                every file, which would defeat the lazy read, but the backend opens them all
                regardless -- so it is the layer that can order them for free.

        Returns:
            The backend's lazily-backed mesh object.

        Raises:
            OptionalPackageDoesNotExist: No backend is installed, the installed one does not
                provide ``open_mesh_mfdataset``, or it provides one that cannot accept ``sort``.
                The signature is checked as well as the name, so a backend predating the
                ``sort`` argument is refused with this branded error rather than a bare
                ``TypeError`` from the argument binder. The message names the backend that was
                resolved, because a third party may register one, and offers the first-party
                upgrade as a parenthesised hint rather than as the diagnosis.
        """
        backend = require_compute_backend()
        forwarder = getattr(backend, "open_mesh_mfdataset", None)
        if forwarder is None:
            raise OptionalPackageDoesNotExist(
                f"The installed lazy backend ({type(backend).__module__}) does not provide "
                "`open_mesh_mfdataset`, so chunks= cannot be served (for the first-party one: "
                "`pip install -U pyramids-lazy`). Omit chunks= to read eagerly."
            )
        if not accepts_keyword(forwarder, "sort"):
            raise OptionalPackageDoesNotExist(
                f"The installed lazy backend ({type(backend).__module__}) provides "
                "`open_mesh_mfdataset` but not its `sort=` argument, so core cannot tell it how "
                "to order the files (for the first-party one: `pip install -U pyramids-lazy`). "
                "Omit chunks= to read eagerly."
            )
        return forwarder(
            resolved, chunks=chunks, driver=driver, mesh_name=mesh_name, sort=sort
        )

    @classmethod
    def _resolved_mfdataset_inputs(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        driver: str | None,
        chunks: dict | str | int | tuple[int, ...] | None,
    ) -> list[str]:
        """Expand `paths` and refuse every input core can judge, before the eager/lazy split.

        Everything core owns about the inputs lives here so the eager and lazy paths reject bad
        input identically. Without it the lazy branch reported a typo'd filename or an unknown
        driver as "upgrade pyramids-lazy" -- the confusion `_require_existing_file` exists to
        stop (review M3/M4) -- and a results-only driver was refused eagerly but forwarded
        lazily (review L1).

        Args:
            paths: A glob pattern, one path, or a sequence of paths.
            driver: Mesh driver name, or `None` to detect per file.
            chunks: The caller's chunking argument, judged only for the values whose emptiness
                is meaningful.

        Returns:
            The resolved paths as `str`, in the order to open them. One spelling, because the
                glob branch yields `str` while the sequence and single-path branches would
                otherwise pass the caller's own objects through to the backend (review L2).

        Raises:
            ValueError: `paths` is empty, a glob pattern matches nothing, or `chunks` is a falsy
                value other than `None` and so selects neither path.
            FileNotFoundError: A resolved path does not exist.
            DriverNotExistError: `driver` names a driver that is not installed.
            UnsupportedMeshFormatError: `driver` names a results-only (`needs_mesh`) driver,
                which carries values but no topology and cannot open a file on its own.
        """
        if isinstance(paths, (str, Path)):
            pattern = str(paths)
            is_pattern = any(character in pattern for character in "*?[")
            matched = sorted(glob(pattern)) if is_pattern else []
            if matched:
                resolved: list[str] = list(matched)
            elif is_pattern:
                raise ValueError(
                    f"open_mfdataset() got the pattern {pattern!r}, which matches no file."
                )
            else:
                resolved = [str(paths)]
        else:
            resolved = [str(path) for path in paths]
        if not resolved:
            raise ValueError(
                "open_mfdataset() got an empty list of paths; there is nothing to open."
            )
        for path in resolved:
            _require_existing_file(path)
        cls._refuse_unusable_driver(driver, resolved[0])
        if isinstance(chunks, _JUDGEABLE_CHUNKS) and not chunks:
            # `chunks is not None` is the eager/lazy switch, so every falsy non-None value used
            # to dispatch lazily -- and `chunks=False` reads as "do not chunk" while doing the
            # opposite. Reject them instead of guessing which was meant (review N3).
            raise ValueError(
                f"open_mfdataset() got chunks={chunks!r}, which selects neither path: pass "
                "chunks=None to read eagerly, or a chunk mapping, 'auto', an int block "
                "size, -1, or a per-axis tuple to read lazily."
            )
        return resolved

    @classmethod
    def _refuse_unusable_driver(cls, driver: str | None, target: str) -> None:
        """Refuse a named driver that cannot open a file on its own.

        The eager path refuses a results-only (`needs_mesh`) driver inside
        `_resolve_read_driver`; doing it here keeps both paths symmetric. It is a capability
        lookup on the already-resolved driver, so it costs no file read.

        Args:
            driver: Mesh driver name, or `None` to detect per file.
            target: The first resolved path, shown in the message's suggested `attach_data`
                call. The driver name would render a nonsense suggestion there.

        Raises:
            DriverNotExistError: The name is neither reserved nor installed.
            UnsupportedMeshFormatError: The driver carries values but no topology.
        """
        if driver is None:
            return
        selected, _ = _named_read_driver(driver)
        if selected is not None and "needs_mesh" in selected.capabilities:
            raise _needs_mesh_error(selected, target)

    @overload
    @classmethod
    def open_mfdataset(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        chunks: None = None,
        driver: str | None = None,
        mesh_name: str | None = None,
        sort: bool = True,
    ) -> MeshDataset: ...

    @overload
    @classmethod
    def open_mfdataset(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        chunks: dict | str | int | tuple[int, ...],
        driver: str | None = None,
        mesh_name: str | None = None,
        sort: bool = True,
    ) -> Any: ...

    @classmethod
    def open_mfdataset(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        chunks: dict | str | int | tuple[int, ...] | None = None,
        driver: str | None = None,
        mesh_name: str | None = None,
        sort: bool = True,
    ) -> Any:
        """Open many mesh files sharing one topology and stack them along time (#1378).

        The many-**files** entry point. :meth:`read_all` answers every topology in **one**
        file; this answers one topology across many files, which is the shape a long run is
        written in — one file per day, per month, or per restart.

        The topology is read from the first file and every other file must match it; the data
        variables are concatenated along time by :meth:`concat`, so the rules about what may
        be joined are that method's rules, stated in one place.

        The eager and lazy results are declared as two overloads rather than one ``Any`` so the
        eager default -- the only shape whose type core owns -- stays checked for callers. A
        single ``Any`` would also drop the ``attr-defined`` errors consumers get today.

        Args:
            paths: A glob pattern (``"out/*_map.nc"``), a single path, or a sequence of paths.
                A pattern that matches nothing is an error rather than an empty dataset.
            chunks: ``None`` (the default) reads every file eagerly and concatenates, which is
                what this method has always done and needs no backend. Anything else hands the
                resolved paths to the ``pyramids-lazy`` backend and returns a lazily-backed mesh
                instead, so a long run is not held in memory. Accepts whatever that backend's
                chunking argument accepts: ``"auto"``, an ``int`` block size, ``-1`` for one
                chunk, a per-axis tuple, or a mapping. A mapping's keys are the backend's to
                define, and core forwards the value without interpreting it. Expect the
                vocabulary ``pyramids-lazy`` uses on its other chunked reads -- axis index, or
                the case-insensitive aliases ``rows`` / ``cols`` / ``columns`` / ``bands``, never
                a dimension name, so ``{0: 1}`` rather than ``{"time": 1}`` (see
                :meth:`pyramids.netcdf.NetCDF.read_array`) -- but the installed backend's own
                documentation is what binds here, since this is its function to define. A
                falsy value other than ``None`` -- ``{}``, ``""``, ``0``, ``False`` -- selects
                neither path and raises :class:`ValueError` rather than dispatching lazily.
            driver: Name of a registered mesh driver, bypassing detection. ``None`` detects.
            mesh_name: Which topology to take from each file. ``None`` takes the first.
            sort: Order the files by their first time stamp before stacking, so a run whose
                filenames do not sort chronologically still comes back in time order —
                ``run_t10`` sorts before ``run_t2`` lexically, which a glob would otherwise
                stack in that order.
                Only a **dated** axis can be ordered this way. A file carrying no CF time
                units reports positional indices, which say nothing about where it belongs, so
                it keeps its given position even with ``sort=True``: for a set of undated
                files, pass an explicitly ordered sequence rather than a glob. In a mixed set
                the dated files are ordered among themselves and placed back into the slots
                they occupied, so the undated ones do not move.
                ``False`` always keeps the given order.
                Forwarded unchanged with ``chunks=``, so the same three clauses apply on the
                lazy path -- the backend opens every file to build its graph either way, and
                applies this ordering to those already-open handles.

        Returns:
            With ``chunks=None`` a ``MeshDataset`` -- one dataset on the shared topology, its
                time axis spanning every file in order. With ``chunks`` given, the backend's
                lazily-backed mesh object instead, whose type ``pyramids-lazy`` defines rather
                than core.

        Raises:
            ValueError: ``paths`` is empty, or a glob pattern matches nothing. Raised before the
                eager/lazy split, so it reports the same problem the same way on either path.
            FileNotFoundError: A resolved path does not exist. Also raised before the split.
            DriverNotExistError: ``driver`` names a driver that is not installed. Also raised
                before the split.
            UnsupportedMeshFormatError: ``driver`` names a results-only (``needs_mesh``) driver,
                which carries values but no topology and so cannot open a file on its own. Also
                raised before the split.
            AlignmentError: A file sits on a different topology than the first (eager path).
            OptionalPackageDoesNotExist: ``chunks`` was given but no compute backend is
                installed, the installed one does not provide ``open_mesh_mfdataset``, or it
                provides one whose signature cannot accept ``sort``. Both the name and the
                signature are checked before the call, so an older backend is refused with this
                branded error rather than a bare ``TypeError`` from the argument binder.

        See Also:
            read_all: Every topology within one file.
            concat: The time-axis join this performs, and its rules.

        Examples:
            - Stack a run written one file per day:
                ```python
                >>> ds = MeshDataset.open_mfdataset("out/*_map.nc")  # doctest: +SKIP
                >>> ds.time_values[0], ds.time_values[-1]  # doctest: +SKIP

                ```
            - Read the same run lazily, then materialise a variable:
                ```python
                >>> lazy = MeshDataset.open_mfdataset(  # doctest: +SKIP
                ...     "out/*_map.nc", chunks="auto"
                ... )
                >>> lazy["waterlevel"].compute().shape  # doctest: +SKIP

                ```
            - Paths are validated before the backend is consulted, so bad input fails the same
              way eagerly or lazily, and needs no backend installed:
                ```python
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> MeshDataset.open_mfdataset([], chunks="auto")
                Traceback (most recent call last):
                    ...
                ValueError: open_mfdataset() got an empty list of paths; there is nothing to open.

                ```
        """
        resolved = cls._resolved_mfdataset_inputs(paths, driver=driver, chunks=chunks)
        if chunks is not None:
            return cls._open_mfdataset_lazy(
                resolved, chunks=chunks, driver=driver, mesh_name=mesh_name, sort=sort
            )
        datasets = [
            cls.read_file(path, driver=driver, mesh_name=mesh_name) for path in resolved
        ]
        if sort:
            # A run's filenames need not sort chronologically (`run_t10` sorts before
            # `run_t2`), so order by what the data says instead. Only a *dated* axis can be
            # ordered: a file carrying no CF time units reports positional indices, which say
            # nothing about where it belongs.
            #
            # So the dated files are sorted *among themselves* and dropped back into the slots
            # they occupied, leaving every undated file exactly where it was given. Sorting the
            # whole list with undated files pushed to one end would move them, which is what
            # this method's own docstring promises not to do.
            dated_slots = [
                index
                for index, dataset in enumerate(datasets)
                if dataset.time_values
                and any(
                    TIME_UNITS_ATTR in var.attributes
                    for var in dataset._data_variables.values()
                )
            ]

            def first_stamp(dataset: MeshDataset) -> tuple:
                """The sort key of a dated dataset: its earliest stamp."""
                stamps = dataset.time_values
                # `dated_slots` holds only datasets whose `time_values` is non-empty, which
                # the type checker cannot see from the comprehension above.
                assert stamps is not None
                return _stamp_order(stamps[0])

            ordered = sorted(
                (datasets[index] for index in dated_slots), key=first_stamp
            )
            for slot, dataset in zip(dated_slots, ordered, strict=True):
                datasets[slot] = dataset
        head, *rest = datasets
        return head.concat(rest) if rest else head

    @classmethod
    def read_all(
        cls, path: str | Path, driver: str | None = None, lazy: bool = True
    ) -> dict[str, MeshDataset]:
        """Read **every** topology in a file, one dataset each.

        :meth:`read_file` answers a single mesh (the first, or the one named). This answers
        all of them, so nothing in a multi-mesh file is silently dropped. Dispatch follows
        :meth:`mesh_names` and :meth:`read_file`, so a third-party format works here too.

        Args:
            path: Path to the file.
            driver: Name of a registered **mesh** driver to use, bypassing detection.
                ``None`` (the default) detects.
            lazy: Defer reading variable values until first access. Forwarded to a driver,
                which may honour an eager read; the built-in UGRID reader has no eager mode, so
                ``lazy=False`` on a native read raises :class:`TypeError`, exactly as
                :meth:`read_file` does. Without it a multi-topology third-party format could
                only ever be read lazily.

        Returns:
            dict[str, MeshDataset]: Each topology name mapped to its dataset.

        Raises:
            FileNotFoundError: The file does not exist.
            DriverNotExistError: ``driver`` names a driver that is not installed.
            UnsupportedMeshFormatError: No installed driver reads the file and the built-in
                reader cannot either — raised from :meth:`mesh_names`, which this delegates
                to, so every error that method raises surfaces here too.
            IsADirectoryError: The path is a directory that no driver claims.
            RuntimeError: GDAL cannot open a file that reached the built-in reader by its own
                suffix. GDAL exceptions are enabled package-wide, so the open failure
                surfaces as GDAL's own error.
            TypeError: ``lazy=False`` on a native (built-in UGRID) read, which has no eager
                mode -- the same rejection :meth:`read_file` makes.
            ValueError: The file declares no UGRID topology.
        """
        # Resolve once, for real. Passing a driver *name* through only short-circuited the
        # route where detection had found something: when nothing claimed the suffix the name
        # was None, so `mesh_names` and every per-topology read re-ran full detection anyway
        # -- measured at two `can_open` calls for one `read_all`, growing per topology.
        # Threading the resolved object through covers the fallback route too.
        selected, native = _resolve_read_driver(path, driver)
        # Set in the native branch below; stays empty whenever a driver handles the file --
        # resolved up front, or picked by the fallback -- in which case it is never read.
        native_names: list[str] = []
        if selected is None:
            try:
                native_names = cls._ugrid_netcdf_mesh_names(
                    path, wrap_unreadable=not native
                )
            except NoMeshTopologyError:
                # The file opened as NetCDF but declares no UGRID topology; fall back to a
                # driver claiming the suffix, as read_file does. If none claims it, the
                # original error stands.
                selected = _netcdf_fallback_driver(path, driver)
                if selected is None:
                    raise
        if selected is None:
            result = {
                name: cls._read_ugrid_netcdf(
                    path, name, wrap_unreadable=not native, lazy=lazy
                )
                for name in native_names
            }
        else:
            names = _mesh_names_through_driver(selected, path)
            result = {
                name: _read_through_driver(selected, path, name, cls, lazy)
                for name in names
            }
        return result

    def _layer_axis(self, var: MeshVariable) -> int | None:
        """The axis holding the layer dimension, or ``None`` when the variable has none.

        A mesh variable's axes read ``[time?, layer?, element]`` — the element axis is the
        last and time is whichever :attr:`MeshVariable.time_axis_index` names, so a remaining
        middle axis is the layer. A 2-D variable whose dimension names mark it non-temporal
        is therefore ``(layer, element)`` with layer axis 0.

        Two cases answer ``None`` rather than guessing. A variable carrying no dimension
        names at all (an array-built one) falls back to "the leading axis of a >1-D array is
        time", so its 2-D form reads as ``(time, element)``; name the dimensions to have it
        read as layers. And a variable whose time dimension is *last* contradicts the layout
        above — the element axis is supposed to be last — so no axis can be claimed as the
        layer axis without mistaking the element axis for it.

        Args:
            var: The variable to inspect.

        Returns:
            int | None: The layer axis, or ``None``.
        """
        if not var.has_data_source:
            return None
        # The declared shape, not `var.data`: touching the data fires the lazy loader and
        # pulls every variable off disk just to learn how many axes it has -- the same defect
        # `info()` carried.
        rank = len(var.shape)
        if rank < 2:
            return None
        if var.time_axis_index == rank - 1:
            # Time sits where the element axis belongs, so the layout is not one we can read.
            return None
        axes = [axis for axis in range(rank - 1) if axis != var.time_axis_index]
        return axes[0] if axes else None

    @property
    def volume_topology(self) -> VolumeTopology | None:
        """The ragged (per-column) vertical layering, or ``None`` for a uniform / 2-D mesh.

        Present only for a genuinely 3-D mesh whose columns carry different layer counts
        (issue #1330): it maps each face to its slice of the flat ``volume`` axis that a
        ``location="volume"`` variable is indexed through. ``None`` means the dense
        ``(layer, element)`` path applies — the uniform-layer behaviour is unchanged.
        """
        return self._volume_topology

    @property
    def n_volume(self) -> int | None:
        """The total number of volumes, or ``None`` when the mesh has no volume topology.

        The flat ``volume``-axis length of a ragged 3-D mesh (the sum of every column's layer
        count). ``None`` for a 2-D or uniform-layer mesh, where :attr:`n_layer` applies
        instead.
        """
        return None if self._volume_topology is None else self._volume_topology.n_volume

    @property
    def n_layer(self) -> int | None:
        """The number of layers, or ``None`` when no variable is layered.

        A 3-D mesh output puts some variables on the ``n`` layer centres and others on the
        ``n + 1`` layer interfaces, so two counts differing by one is normal rather than a
        contradiction; this reports ``n``, the layer count proper. Any other disagreement is
        a real inconsistency and raises. :meth:`isel_layer` is unaffected either way — it
        indexes each variable on its own layer axis.

        Reporting ``n`` means the ``n + 1`` is not visible here, so a caller iterating
        ``range(n_layer)`` over an interface variable would skip its last interface. Read
        :attr:`n_layer_interface` for that count rather than inferring it: it answers ``None``
        unless a variable really sits on the interfaces.

        For a **ragged** 3-D mesh (:attr:`volume_topology` set) there is no single layer count:
        this reports the **maximum** over the columns — the number of layer *slots* — so
        ``range(n_layer)`` covers the deepest column, and :meth:`isel_layer` makes the absent
        shallow-column layers explicit (NaN) rather than padding them silently.

        Returns:
            int | None: The layer-centre count (the max, for a ragged mesh), or ``None`` when
            no variable is layered.

        Raises:
            ValueError: The layered variables carry counts that are not a layer-centre /
                layer-interface pair.
        """
        if self._volume_topology is not None:
            return self._volume_topology.max_layers
        counts = self._layer_counts()
        return min(counts) if counts else None

    @property
    def n_layer_interface(self) -> int | None:
        """The number of layer interfaces, or ``None`` when nothing sits on them.

        The companion to :attr:`n_layer`. A 3-D output's vertical velocities and fluxes live on
        the ``n + 1`` interfaces between the ``n`` layer centres, and ``n_layer`` deliberately
        reports the centre count -- so this is where the other half of the pair is readable.

        It is **not** simply ``n_layer + 1``, which is always computable: that would assert
        that something in the dataset is interface-located. Only a variable actually carrying
        that count makes it true, so a centres-only dataset answers ``None``.

        Returns:
            int | None: The layer-interface count, or ``None`` when no variable is layered or
            every layered variable sits on the centres.

        Raises:
            ValueError: The layered variables carry counts that are not a layer-centre /
                layer-interface pair -- the same guard :attr:`n_layer` applies, so the two
                properties agree about what is readable.
        """
        counts = self._layer_counts()
        if not counts:
            return None
        interfaces = min(counts) + 1
        return interfaces if interfaces in counts else None

    @property
    def n_layers(self) -> int | None:
        """Deprecated alias for :attr:`n_layer`.

        Deprecated: Renamed to ``n_layer`` (singular, matching ``n_node``/``n_face``/``n_edge``).

        Returns:
            int | None: The layer-centre count, or ``None`` when no variable is layered.
        """
        warnings.warn(
            "MeshDataset.n_layers is deprecated; use n_layer.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_layer

    @property
    def n_layer_interfaces(self) -> int | None:
        """Deprecated alias for :attr:`n_layer_interface`.

        Deprecated: Renamed to ``n_layer_interface`` (singular, matching the ``n_*`` family).

        Returns:
            int | None: The layer-interface count, or ``None`` when nothing sits on them.
        """
        warnings.warn(
            "MeshDataset.n_layer_interfaces is deprecated; use n_layer_interface.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_layer_interface

    def _layer_counts(self) -> set[int]:
        """The distinct layer counts across the layered variables, validated.

        Returns:
            set[int]: One entry per distinct count, empty when no variable is layered.

        Raises:
            ValueError: The counts are not a single value or a layer-centre / layer-interface
                pair. The test is ``counts - {smallest, smallest + 1}`` rather than "every gap
                is 1", so ``{20, 21, 22}`` is refused too: a mesh cannot have both 21 and 22
                interfaces for 20 centres.
        """
        counts = set()
        for var in self._data_variables.values():
            axis = self._layer_axis(var)
            if axis is not None:
                counts.add(int(var.shape[axis]))
        smallest = min(counts) if counts else 0
        if counts - {smallest, smallest + 1}:
            raise ValueError(
                f"layered variables disagree on their layer count: {sorted(counts)}."
            )
        return counts

    @property
    def layer_values(self) -> list | None:
        """The layer coordinate values, when a layered variable carries them.

        Read from a layered variable's ``layer_values`` attribute. The UGRID reader does not
        populate that attribute -- as with :attr:`time_values`, the coordinate is supplied by
        whoever builds or transforms the dataset -- so this is normally ``None`` for a mesh
        read straight from a file, and :meth:`sel_layer` then selects by layer position.
        """
        for var in self._data_variables.values():
            if self._layer_axis(var) is not None:
                stamps = var.attributes.get("layer_values")
                if stamps is not None:
                    return list(stamps)
        return None

    def isel_layer(self, index: int | slice | list[int] | np.ndarray) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.isel_layer`."""
        warnings.warn(
            "`mesh.isel_layer` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.isel_layer`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.isel_layer(index)

    def sel_layer(self, value: Any) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.sel_layer`."""
        warnings.warn(
            "`mesh.sel_layer` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.sel_layer`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.sel_layer(value)

    def _isel_layer_ragged(
        self, index: int | slice | list[int] | np.ndarray
    ) -> MeshDataset:
        """Select the ``index``-th layer per column for a ragged 3-D mesh (issue #1330).

        The per-column counterpart of :meth:`isel_layer`: a ``location="volume"`` variable's
        flat ``(time?, n_volume)`` data is gathered to a ``(time?, n_face)`` face field, each
        column contributing its ``index``-th volume where it has one and NaN where it does not,
        so the shallow columns' absent layers are explicit rather than padded. Non-volume
        variables are carried through untouched.

        Args:
            index: The 0-based layer position to take from every column. Only an ``int`` is
                meaningful for a ragged mesh.

        Returns:
            MeshDataset: A new dataset with the volume variables collapsed to face fields.

        Raises:
            ValueError: ``index`` is not an ``int``, or is negative.
        """
        if not isinstance(index, (int, np.integer)):
            raise ValueError(
                "isel_layer on a ragged (volume) mesh needs an int layer position; a slice "
                "or sequence is ambiguous when columns have different layer counts. Select "
                "one layer at a time."
            )
        if int(index) < 0:
            # A negative layer would make every column "valid" (`layer_count > -1`) and index
            # `volume_offset + layer` with a negative offset, wrapping into the previous
            # column's volumes — garbage, not a per-column top layer. Reject it, matching
            # VolumeTopology.volume_index, which guards negatives too (review L1).
            raise ValueError(
                f"isel_layer on a ragged (volume) mesh needs a non-negative layer position; "
                f"got {int(index)}. A column's top layer varies by column, so there is no "
                f"well-defined negative index."
            )
        topology = cast("VolumeTopology", self._volume_topology)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if var.location != "volume":
                new_vars[name] = var
                continue
            new_vars[name] = self._gather_layer_from_volume(var, int(index), topology)
        return self._rebuild_without_volume(new_vars)

    def _gather_layer_from_volume(
        self, var: MeshVariable, layer: int, topology: VolumeTopology
    ) -> MeshVariable:
        """Collapse one ragged ``volume`` variable to a ``(time?, n_face)`` face field.

        Face ``f`` takes volume ``volume_offset[f] + layer`` when ``layer < layer_count[f]``,
        else NaN. Vectorised: the valid faces gather in one fancy-index, the rest are left NaN.

        Args:
            var: The ``location="volume"`` variable (flat ``(time?, n_volume)`` data).
            layer: The 0-based layer position to take from each column.
            topology: The mesh's volume topology.

        Returns:
            MeshVariable: A ``location="face"`` variable of shape ``(time?, n_face)``.
        """
        data = np.asarray(var.data, dtype="float64")
        n_face = topology.n_face
        valid = topology.layer_count > layer
        volume_positions = topology.volume_offset[:-1][valid] + layer
        out_shape = (*data.shape[:-1], n_face)
        out = np.full(out_shape, np.nan, dtype="float64")
        out[..., valid] = data[..., volume_positions]
        # Drop the flat `volume` dimension name for the face dimension; keep a leading time
        # dimension name when the variable had one.
        if var.dimensions:
            new_dims = (*var.dimensions[:-1], f"{var.mesh_name}_nFaces")
        else:
            new_dims = ()
        return MeshVariable(
            name=var.name,
            location="face",
            mesh_name=var.mesh_name,
            shape=out.shape,
            attributes=dict(var.attributes),
            nodata=var.nodata,
            units=var.units,
            standard_name=var.standard_name,
            dimensions=new_dims,
            _data=out,
        )

    def _rebuild_without_volume(
        self, data_variables: dict[str, MeshVariable]
    ) -> MeshDataset:
        """A :meth:`_rebuild` that drops the volume topology — the layer axis is gone.

        Used by the ragged layer selectors: once a ``volume`` variable has been gathered to a
        face field, the result is an ordinary 2-D dataset and must not keep a volume topology
        that no variable is indexed through any more.
        """
        return type(self)(
            mesh=self._mesh,
            data_variables=data_variables,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    def collapse_layer(self, layer: int, *, location: str = "face") -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.collapse_layer`."""
        warnings.warn(
            "`mesh.collapse_layer` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.collapse_layer`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.collapse_layer(layer, location=location)

    def to_dataframe(
        self, variables: Sequence[str] | None = None, *, dropna: bool = False
    ) -> pd.DataFrame:
        """Facade -- delegates to :meth:`_MeshInterop.to_dataframe`."""
        warnings.warn(
            "`mesh.to_dataframe` is deprecated and will be removed in 0.90.0; use "
            "`mesh.interop.to_dataframe`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.interop.to_dataframe(variables, dropna=dropna)

    def _variable_frames(self, name: str, var: MeshVariable) -> list[pd.DataFrame]:
        """One tidy block per (time step, layer) of a single variable.

        Args:
            name: The variable's name, carried into the ``variable`` column.
            var: The variable to lay out.

        Returns:
            list: The blocks, in time-then-layer order.
        """
        if var.location == "volume" and self._volume_topology is not None:
            # A ragged volume variable is flat `(time?, n_volume)`: lay it out one row per
            # volume, carrying the column (`element`) and the layer position within it rather
            # than fabricating rows for the absent layers of shallow columns (issue #1330).
            return self._volume_variable_frames(name, var)
        raw = np.asarray(var.data)
        # Gap-masking casts to float64, which a classified / label field cannot survive --
        # and such a field carries no no-data value to mask anyway.
        values = (
            gaps_as_nan(raw, var.nodata) if np.issubdtype(raw.dtype, np.number) else raw
        )
        # Address the rows with the coordinates of the facet the variable actually sits on. Using
        # face centroids for every variable left a node- or edge-located one with a NaN
        # coordinate on every row, since its value count is n_node / n_edge.
        coords = self._element_coords(var.location)
        arranged, steps, layer_labels, temporal, layered = self._arrange_for_rows(
            var, values
        )
        blocks: list[pd.DataFrame] = []
        for step_position, step in enumerate(steps):
            per_step = arranged[step_position] if temporal else arranged
            for layer_position, layer in enumerate(layer_labels):
                slab = per_step[layer_position] if layered else per_step
                blocks.append(
                    self._tidy_rows(name, np.asarray(slab), coords, step, layer)
                )
        return blocks

    def _volume_variable_frames(
        self, name: str, var: MeshVariable
    ) -> list[pd.DataFrame]:
        """Tidy blocks for a ragged ``volume`` variable — one row per volume (issue #1330).

        Each row carries the column (``element`` = the face the volume belongs to), the
        0-based ``layer`` within that column, the face's ``x``/``y`` centroid, and the value.
        A temporal variable emits one block per step; a static one a single block. No row is
        emitted for an absent layer, since the flat volume axis holds only the layers that
        exist.

        Args:
            name: The variable name, carried into the ``variable`` column.
            var: The ``location="volume"`` variable.

        Returns:
            list: The per-step blocks.
        """
        topology = cast("VolumeTopology", self._volume_topology)
        raw = np.asarray(var.data)
        values = (
            gaps_as_nan(raw, var.nodata) if np.issubdtype(raw.dtype, np.number) else raw
        )
        face_of_volume = topology.face_of_volume()
        layer_of_volume = topology.layer_of_volume()
        face_coords = self._element_coords("face")
        have_coords = face_coords.shape[0] == topology.n_face
        coords = (
            face_coords[face_of_volume]
            if have_coords
            else np.full((topology.n_volume, 2), np.nan, dtype="float64")
        )
        if var.has_time:
            time_axis = cast("int", var.time_axis_index)
            stamps = var.attributes.get("time_values")
            steps = (
                list(stamps)
                if stamps is not None
                else list(range(values.shape[time_axis]))
            )
            per_step = [np.take(values, i, axis=time_axis) for i in range(len(steps))]
        else:
            steps = [None]
            per_step = [values]
        blocks: list[pd.DataFrame] = []
        for step, slab in zip(steps, per_step):
            flat = np.asarray(slab).ravel()
            blocks.append(
                pd.DataFrame(
                    {
                        "variable": np.repeat(name, flat.shape[0]),
                        "element": face_of_volume,
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "layer": layer_of_volume,
                        "time": np.repeat(step, flat.shape[0]),
                        "value": (
                            flat.astype("float64")
                            if np.issubdtype(flat.dtype, np.number)
                            else flat
                        ),
                    }
                )
            )
        return blocks

    def _element_axis(self, var: MeshVariable, values: np.typing.NDArray) -> int:
        """The axis of ``values`` that runs over the mesh elements.

        The element axis is conventionally last, and assuming so is what broke a variable whose
        dimension names put it first: the last axis was read as the elements, so a
        ``(n_faces, time)`` variable emitted one row per *time step*, dropping the values past
        the element count and leaving their coordinates NaN.

        The variable's declared element count decides it. The last axis is preferred when more
        than one matches -- a square variable is genuinely ambiguous, and last is the convention.

        Args:
            var: The variable, for its location and dimension names.
            values: Its values.

        Returns:
            int: The element axis. The last axis when nothing matches the element count, which
            keeps the previous behaviour for a variable whose shape says nothing.
        """
        if values.ndim < 2:
            return values.ndim - 1
        counts = {"face": self.n_face, "node": self.n_node, "edge": self.n_edge}
        expected = counts.get(var.location)
        matching = (
            [axis for axis in range(values.ndim) if values.shape[axis] == expected]
            if expected
            else []
        )
        return matching[-1] if matching else values.ndim - 1

    def _row_axes(
        self, var: MeshVariable, values: np.typing.NDArray
    ) -> tuple[int, int | None, int | None]:
        """The element, time and layer axes the tidy frame will walk.

        Resolves the three against each other and refuses anything left over, so
        :meth:`_arrange_for_rows` is left with the transpose and the labels.

        An axis can be claimed twice: `var.time_axis_index` may name the axis that holds the
        elements. :meth:`_element_axis` settles that by the declared element count, and the
        element reading wins -- so such an axis is *not* walked as a step, and the caller is
        told which axes were.

        Args:
            var: The variable being laid out.
            values: Its values.

        Returns:
            tuple: ``(element_axis, time_axis, layer_axis)``, the last two ``None`` when there
            is no such axis to walk.

        Raises:
            ValueError: An axis is neither walked nor addressed. The tidy layout is one row per
                element with a ``time`` and a ``layer`` column, so a fourth dimension has
                nowhere to go -- and folding it into the element slot emitted ``element``
                indices past the element count with NaN coordinates, a frame a caller joins on.
        """
        element_axis = self._element_axis(var, values)
        time_axis = cast("int | None", var.time_axis_index) if var.has_time else None
        time_axis = None if time_axis == element_axis else time_axis
        layer_axis = self._layer_axis(var)
        layer_axis = None if layer_axis == element_axis else layer_axis

        walked = {element_axis, *(a for a in (time_axis, layer_axis) if a is not None)}
        unaddressed = [axis for axis in range(values.ndim) if axis not in walked]
        if unaddressed:
            names = var.dimensions or ()
            labelled = [
                names[axis] if axis < len(names) else f"axis {axis}"
                for axis in unaddressed
            ]
            subject = "has" if len(labelled) == 1 else "have"
            pronoun = "it" if len(labelled) == 1 else "them"
            raise ValueError(
                f"to_dataframe cannot address variable {var.name!r}: the tidy frame carries one "
                f"row per element with a 'time' and a 'layer' column, and {labelled} "
                f"{subject} no column. Select {pronoun} away first, e.g. with isel()."
            )
        return element_axis, time_axis, layer_axis

    def _arrange_for_rows(
        self, var: MeshVariable, values: np.typing.NDArray
    ) -> tuple[np.typing.NDArray, list[Any], list[Any], bool, bool]:
        """Reorder a variable's axes to ``(time?, layer?, element)`` and label the leading ones.

        The tidy frame walks one row per element, so the axes above the element axis have to be
        walked in a known order with each position labelled. Flattening instead ran the element
        index past the element count and lost which layer a row came from.

        A missing axis is reported as the single label ``[None]``, so the caller's loop runs once
        over it and the frame still carries the column.

        Args:
            var: The variable whose axes are being arranged.
            values: Its values, already gap-masked.

        Returns:
            tuple: ``(arranged, steps, layer_labels, walked_time, walked_layer)`` -- the
            transposed array, one label per time step, one per layer, and whether a time and a
            layer axis were actually walked. The last two matter because they can disagree with
            `var.has_time` / `_layer_axis`: when the dimension names put time (or layer) in the
            element slot, the element reading wins and that axis is *not* walked. A caller that
            trusted `var.has_time` instead indexed `arranged[0]`, slicing the variable down to
            its first element and emitting one row for a two-face mesh.

        Raises:
            ValueError: The variable carries an axis the frame cannot address. The tidy layout is
                one row per element with a ``time`` and a ``layer`` column, so a fourth dimension
                has nowhere to go. Making the transpose a permutation stopped numpy raising, but
                the consumer then folded that axis into the element slot and emitted ``element``
                indices past the element count with NaN coordinates -- a frame a caller joins on,
                which is worse than the exception. Select the extra dimension away first.
        """
        element_axis, time_axis, layer_axis = self._row_axes(var, values)
        leading = [axis for axis in (time_axis, layer_axis) if axis is not None]
        order = [*leading, element_axis]
        arranged = np.transpose(values, order) if values.ndim > 1 else values

        steps: list[Any] = [None]
        if time_axis is not None:
            stamps = var.attributes.get("time_values")
            steps = (
                list(stamps) if stamps is not None else list(range(arranged.shape[0]))
            )

        layer_labels: list[Any] = [None]
        if layer_axis is not None:
            depth = arranged.shape[1 if time_axis is not None else 0]
            stored = var.attributes.get("layer_values")
            layer_labels = list(stored) if stored is not None else list(range(depth))

        return (
            arranged,
            steps,
            layer_labels,
            time_axis is not None,
            layer_axis is not None,
        )

    def _element_coords(self, location: str) -> np.typing.NDArray:
        """The ``(n_elements, 2)`` coordinate table of a mesh facet.

        Args:
            location: ``"face"``, ``"node"`` or ``"edge"``.

        Returns:
            np.ndarray: One ``(x, y)`` row per element, or an empty table when the facet's
            coordinates are not available (an edge-located variable on a mesh carrying no edge
            coordinates), which leaves those rows with a NaN coordinate.
        """
        try:
            table = _location_coords(
                self._require_mesh2d("location coordinates"), location
            )
        except ValueError:
            table = np.empty((0, 2), dtype="float64")
        return np.asarray(table, dtype="float64")

    def _tidy_rows(
        self,
        name: str,
        values: np.typing.NDArray,
        element_coords: np.typing.NDArray,
        step: Any,
        layer: Any = None,
    ) -> pd.DataFrame:
        """One tidy block: every element of ``values`` as a row."""
        flat = np.asarray(values).ravel()
        count = flat.shape[0]
        coords = (
            element_coords
            if element_coords.shape[0] == count
            else np.full((count, 2), np.nan, dtype="float64")
        )
        return pd.DataFrame(
            {
                "variable": np.repeat(name, count),
                "element": np.arange(count),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "layer": np.repeat(layer, count),
                "time": np.repeat(step, count),
                # Keep a non-numeric variable as it is. Forcing float64 raised
                # "could not convert string to float" on exactly the classified /
                # label field `polygonize` recommends.
                "value": (
                    flat.astype("float64")
                    if np.issubdtype(flat.dtype, np.number)
                    else flat
                ),
            }
        )

    def copy(self) -> MeshDataset:
        """An independent deep copy of this dataset.

        Every variable's values are copied, so writing into the copy never shows in the
        original — the counterpart of :meth:`pyramids.netcdf.NetCDF.copy`.

        Returns:
            MeshDataset: The copy.
        """
        copied: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            copied[name] = (
                var
                if var.data is None
                else var.with_data(np.array(var.data, copy=True))
            )
        return type(self)(
            mesh=self._mesh,
            data_variables=copied,
            global_attributes=dict(self._global_attributes),
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
            volume_topology=self._volume_topology,
        )

    @classmethod
    def from_bytes(cls, data: bytes, *, suffix: str = ".nc") -> MeshDataset:
        """Read a UGRID mesh from an in-memory buffer.

        Writes ``data`` into GDAL's in-memory filesystem and reads it as a file, mirroring
        :meth:`pyramids.netcdf.NetCDF.from_bytes` — for a mesh fetched over the network or
        held in a database blob. The values are materialised before the buffer is unlinked, so
        the result does not depend on it.

        Args:
            data: The bytes of a UGRID NetCDF file.
            suffix: The extension to present to GDAL. Defaults to ``".nc"``.

        Returns:
            MeshDataset: The mesh read from the buffer.

        Raises:
            RuntimeError: GDAL cannot read the buffer at all. It is written to ``/vsimem`` and
                read through :meth:`read_file`, so GDAL's own error comes back -- naming the
                in-memory path rather than the caller's bytes.
            ValueError: The buffer reads as a file but declares no UGRID topology.
        """
        name = f"/vsimem/pyramids_ugrid_{uuid4().hex}{suffix}"
        gdal.FileFromMemBuffer(name, data)
        try:
            result = cls.read_file(name).copy()
        finally:
            gdal.Unlink(name)
        return result

    def info(self, buf: Any = None, drivers: bool = False) -> None:
        """Print a summary of the mesh, its variables and its attributes.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.info`: one readable block naming
        the topology, the element counts, the CRS, and every data variable with its location,
        dtype and shape.

        Args:
            buf: Where to write. Defaults to ``sys.stdout``.
            drivers: Also list the mesh file-format drivers available in this interpreter —
                the built-in UGRID reader plus any installed plugin. Off by default because
                resolving them walks every installed distribution, which a summary of one
                dataset should not pay for.
        """
        stream = sys.stdout if buf is None else buf
        lines = [
            f"MeshDataset: {self.mesh_name}",
            f"  nodes: {self.n_node}  faces: {self.n_face}  edges: {self.n_edge}",
            f"  epsg: {self.epsg}",
            f"  variables ({len(self._data_variables)}):",
        ]
        for name, var in self._data_variables.items():
            # Metadata only: `var.data` would fire the lazy loader and pull every variable's
            # full array off disk, which is the opposite of what a summary is for. The declared
            # shape and `has_data_source` answer the same questions without a read.
            shape = str(tuple(var.shape)) if var.has_data_source else "unloaded"
            temporal = "temporal" if var.has_time else "static"
            lines.append(
                f"    {name}: location={var.location} dtype={var.dtype} "
                f"shape={shape} ({temporal})"
            )
        if self._global_attributes:
            lines.append(f"  global attributes ({len(self._global_attributes)}):")
            lines.extend(
                f"    {key}: {value}" for key, value in self._global_attributes.items()
            )
        if drivers:
            driver_lines = describe_drivers()
            # Count the plugins, not the lines: reporting "mesh drivers (1)" on a bare install
            # counted the built-in reader as a driver, while DriverNotExistError described the
            # same state as having none installed.
            plugins = len(driver_lines) - 1
            suffix = "" if plugins == 1 else "s"
            lines.append(f"  mesh drivers (built-in + {plugins} plugin{suffix}):")
            lines.extend(f"    {line}" for line in driver_lines)
        stream.write("\n".join(lines) + "\n")

    def set_global_attribute(self, name: str, value: Any) -> MeshDataset:
        """Return a new dataset carrying one more global attribute.

        A UGRID derivation is immutable here (as :meth:`with_variable` is), so this answers a
        new dataset rather than writing in place; :meth:`to_file` then persists the attributes
        through the shared CF writer.

        Args:
            name: The attribute name.
            value: The attribute value.

        Returns:
            MeshDataset: A new dataset whose global attributes include ``name``.
        """
        return type(self)(
            mesh=self._mesh,
            data_variables=dict(self._data_variables),
            global_attributes={**self._global_attributes, name: value},
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
            volume_topology=self._volume_topology,
        )

    def delete_global_attribute(self, name: str) -> MeshDataset:
        """Return a new dataset without the global attribute ``name``.

        Args:
            name: The attribute to drop. Dropping an absent attribute is not an error,
                mirroring the raster ``delete_global_attribute``.

        Returns:
            MeshDataset: A new dataset without that attribute.
        """
        remaining = {
            key: value for key, value in self._global_attributes.items() if key != name
        }
        return type(self)(
            mesh=self._mesh,
            data_variables=dict(self._data_variables),
            global_attributes=remaining,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
            volume_topology=self._volume_topology,
        )

    def active_mask(self, variable_name: str) -> np.typing.NDArray | None:
        """Facade -- delegates to :meth:`_MeshTopology.active_mask`."""
        warnings.warn(
            "`mesh.active_mask` is deprecated and will be removed in 0.90.0; use "
            "`mesh.topology.active_mask`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.topology.active_mask(variable_name)

    def masked_values(self, variable_name: str) -> np.typing.NDArray:
        """Facade -- delegates to :meth:`_MeshAnalysis.masked_values`."""
        warnings.warn(
            "`mesh.masked_values` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.masked_values`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.masked_values(variable_name)

    def _masked_values_impl(self, variable_name: str) -> np.typing.NDArray:
        """Implementation of :meth:`_MeshAnalysis.masked_values`; see that method for the contract."""
        var = self[variable_name]
        data = var.data
        if data is None:
            raise ValueError(f"Variable {variable_name!r} has no loaded data.")
        values = gaps_as_nan(np.asarray(data), var.nodata)
        if var.active is not None:
            values = _apply_active_mask(var, values)
        return cast("np.typing.NDArray", values)

    def _element_values(
        self, variable_name: str, time_index: int = 0, *, apply_active: bool = False
    ) -> tuple[MeshVariable, np.typing.NDArray]:
        """The per-element values of a variable, no-data blanked to NaN.

        Shared by the analysis members (:meth:`sample`, :meth:`weighted`,
        :meth:`zonal_stats`). A temporal variable is reduced to one
        step (``time_index``, the first by default) so the returned array is
        ``(n_elements,)``, matching the single-step contract :meth:`plot` and
        :meth:`to_dataset` already use. The variable's declared no-data value is turned
        into NaN through the shared :func:`pyramids.base._reductions.gaps_as_nan`, so
        every statistic leaves gaps out.

        Args:
            variable_name: Name of the data variable.
            time_index: Which step to take for a temporal variable. Defaults to 0.
            apply_active: Also blank dry elements (:attr:`MeshVariable.active` ``False``) to
                NaN, the mask reduced to the same step. Defaults to ``False`` — the mask is
                never applied unless a caller asks, so "dry" stays distinct from "missing"
                (issue #1331).

        Returns:
            tuple: The :class:`MeshVariable` and its ``(n_elements,)`` float64 values.

        Raises:
            ValueError: The variable has no loaded data, or is not per-element after the
                time collapse (e.g. a layered ``(n_layer, n_face)`` non-temporal variable).
        """
        var = self[variable_name]
        data = var.data
        if data is None:
            raise ValueError(f"Variable {variable_name!r} has no loaded data.")
        if var.has_time:
            # Take the step along the variable's own time axis, not blindly axis 0 — a
            # mesh variable may store time as a trailing axis (e.g. (nFaces, time)), and
            # the rest of this module keys off var.time_axis_index for exactly that reason.
            data = np.take(
                np.asarray(data), time_index, axis=cast("int", var.time_axis_index)
            )
        result = gaps_as_nan(np.asarray(data), var.nodata)
        if apply_active and var.active is not None and result.ndim == 1:
            # Reduce a per-step mask to the same step, then blank dry elements. A static mask
            # is used as-is. Only meaningful once `result` is the 1-D per-element vector.
            mask = np.asarray(var.active, dtype=bool)
            if mask.ndim >= 2:
                mask = mask[time_index]
            result = np.where(mask, result, np.nan)
        if result.ndim != 1:
            # A non-temporal multi-dimensional variable (e.g. layered (n_layer, n_face))
            # has no single per-element vector; reducing it over an arbitrary axis would
            # silently return a per-column scalar. Refuse rather than mislead.
            raise ValueError(
                f"Variable {variable_name!r} is not per-element after the time collapse "
                f"(shape {result.shape}); the analysis members need a 1-D variable. A "
                "layered/multi-dimensional non-temporal variable is not supported here."
            )
        return var, result

    def summary(
        self,
        variables: Sequence[str] | None = None,
        *,
        metrics: Sequence[str] = DEFAULT_METRICS,
        skipna: bool = True,
        ddof: int = 0,
    ) -> pd.DataFrame:
        """A per-variable summary table over the whole mesh.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.summary`, built on the same shared
        assembly (:func:`pyramids.base._summary.variable_summary`) so the two classes answer in an
        identical shape. Each variable is reduced over **all** of its samples — every element and,
        for a temporal variable, every step — with the declared no-data value left out.

        Args:
            variables: Which data variables to summarise, in output-row order. ``None`` (the
                default) takes every data variable. An unknown name raises ``KeyError``.
            metrics: Which statistics to report, in column order. Defaults to
                ``("count", "min", "max", "mean", "std")``; ``median`` / ``var`` / ``sum`` /
                ``prod`` are available too.
            skipna: Leave no-data out of the statistics (the default) or let it propagate.
            ddof: Delta degrees of freedom for ``std`` / ``var``. Defaults to 0.

        Returns:
            pandas.DataFrame: One row per variable (index name ``"variable"``), one column per
            metric. ``count`` is ``int64``; the rest are ``float64``. A variable with no valid
            element yields NaN statistics and a ``count`` of 0.

        Raises:
            KeyError: A requested variable is not a data variable of this dataset.
            TypeError: A requested variable is non-numeric.
            ValueError: A requested variable has no loaded data, or an unknown metric.

        Examples:
            - Per-variable summary of a two-triangle mesh:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([2.0, 4.0])},
                ... )
                >>> mesh.summary().loc["depth", ["count", "min", "max", "mean"]].tolist()
                [2.0, 2.0, 4.0, 3.0]

                ```
        """
        names = list(self._data_variables) if variables is None else list(variables)
        arrays: dict[str, np.typing.NDArray] = {}
        for name in names:
            if name not in self._data_variables:
                raise KeyError(name)
            var = self._data_variables[name]
            if var.data is None:
                raise ValueError(f"Variable {name!r} has no loaded data.")
            data = np.asarray(var.data)
            if not np.issubdtype(data.dtype, np.number):
                if variables is None:
                    warnings.warn(
                        f"skipping non-numeric variable {name!r} in summary",
                        stacklevel=2,
                    )
                    continue
                raise TypeError(f"variable {name!r} is non-numeric; cannot summarise")
            arrays[name] = gaps_as_nan(data, var.nodata)
        return variable_summary(arrays, metrics=metrics, skipna=skipna, ddof=ddof)

    def sample(
        self,
        variable_name: str,
        x: float | np.ndarray,
        y: float | np.ndarray,
        *,
        method: str = "contains",
        time_index: int = 0,
    ) -> np.typing.NDArray:
        """Facade -- delegates to :meth:`_MeshAnalysis.sample`."""
        warnings.warn(
            "`mesh.sample` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.sample`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.sample(
            variable_name, x, y, method=method, time_index=time_index
        )

    def weighted(
        self,
        variable_name: str,
        *,
        how: str = "mean",
        time_index: int = 0,
        q: float | None = None,
    ) -> float:
        """Facade -- delegates to :meth:`_MeshSelection.weighted`."""
        warnings.warn(
            "`mesh.weighted` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.weighted`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.weighted(
            variable_name, how=how, time_index=time_index, q=q
        )

    def zonal_stats(
        self,
        zones: Any,
        *,
        variable_name: str,
        stats: tuple[str, ...] = ("mean",),
        weighted: bool = True,
        time_index: int = 0,
    ) -> pd.DataFrame:
        """Facade -- delegates to :meth:`_MeshAnalysis.zonal_stats`."""
        warnings.warn(
            "`mesh.zonal_stats` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.zonal_stats`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.zonal_stats(
            zones,
            variable_name=variable_name,
            stats=stats,
            weighted=weighted,
            time_index=time_index,
        )

    def _temporal_names(self) -> list[str]:
        """Names of the variables that carry a real time dimension."""
        return [name for name, var in self._data_variables.items() if var.has_time]

    def _rebuild(self, data_variables: dict[str, MeshVariable]) -> MeshDataset:
        """A new dataset with the same mesh/metadata but different data variables.

        Every along-time member below is immutable — it derives a fresh dataset rather than
        mutating in place — mirroring :meth:`sel_time` / :meth:`crop`. The mesh topology,
        global attributes, topology info, CRS and source file name all carry over unchanged,
        and the result is of the receiver's own class so a subclass is not downgraded.
        """
        return type(self)(
            mesh=self._mesh,
            data_variables=data_variables,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
            volume_topology=self._volume_topology,
        )

    def _require_temporal(self, operation: str) -> list[str]:
        """The temporal variable names, or a clear error when there are none."""
        temporal = self._temporal_names()
        if not temporal:
            raise ValueError(
                f"{operation} needs at least one variable with a time dimension; this "
                f"dataset has none."
            )
        return temporal

    def _check_time_dim(self, operation: str, dim: str | None) -> None:
        """Refuse a `dim` that is not the time axis.

        The along-time members take `dim` for signature parity with the raster side, where a
        cube has several non-spatial dimensions to choose from. A mesh has exactly one: the
        element axis is the mesh's own, and carrying a value along it is meaningless. Accept
        ``"time"`` (or ``None``) and refuse anything else rather than silently using time.

        Args:
            operation: The calling method's name, for the message.
            dim: The dimension the caller named, or ``None``.

        Raises:
            ValueError: ``dim`` is neither ``None`` nor ``"time"``.
        """
        if dim is not None and dim != "time":
            raise ValueError(
                f"{operation}: a mesh has one along-able dimension, 'time'; got dim={dim!r}. "
                "The element axis is the mesh's own axis, which carries no ordering to run "
                "along."
            )

    def reduce(
        self,
        how: str,
        *,
        skipna: bool = True,
        q: float | None = None,
        groupby: Any = None,
        min_count: int = 0,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.reduce`."""
        warnings.warn(
            "`mesh.reduce` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.reduce`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.reduce(
            how, skipna=skipna, q=q, groupby=groupby, min_count=min_count
        )

    def _reduce_impl(
        self,
        how: str,
        *,
        skipna: bool = True,
        q: float | None = None,
        groupby: Any = None,
        min_count: int = 0,
    ) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.reduce`; see that method for the contract."""
        _check_reduce_how(how, q)
        validate_min_count(min_count)
        self._require_temporal("reduce")
        # all/any collapse to a uint8 flag band whose all-gap value is FLAG_NO_DATA (255);
        # the collapsed variable must declare that so a consumer does not read 255 as data.
        result_nodata = FLAG_NO_DATA if how in ("all", "any") else None
        grouped = (
            _group_positions(self._time_coords("reduce"), groupby)
            if groupby is not None
            else None
        )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to reduce.")
            arr = np.asarray(data)
            nodata = result_nodata if result_nodata is not None else var.nodata
            if grouped is None:
                reduced = reduce_axis(arr, axis, how, skipna, var.nodata, q, min_count)
                new_vars[name] = _static_from(
                    var, np.asarray(reduced), axis, nodata=nodata
                )
                continue
            labels, positions = grouped
            # Each group is reduced on its own slab and the results stacked back along the
            # time axis, so a grouped reduce keeps the dimension (shorter) instead of
            # collapsing it — the same shape contract the raster grouped reduce has.
            parts = [
                np.asarray(
                    reduce_axis(
                        np.take(arr, group, axis=axis),
                        axis,
                        how,
                        skipna,
                        var.nodata,
                        q,
                        min_count,
                    )
                )
                for group in positions
            ]
            # A grouped reduce re-bins the time axis, so a per-step mask no longer aligns —
            # even when n_groups == n_time keeps the shape, which would slip past the
            # length-only implicit guard (review L3). Drop it; a static mask is step-
            # independent and kept.
            new_var = var.with_data(
                np.stack(parts, axis=axis),
                active=None if var.active_has_time else _KEEP_ACTIVE,
            )
            new_var.nodata = nodata
            new_var.attributes = {**var.attributes, "time_values": list(labels)}
            new_vars[name] = new_var
        return self._rebuild(new_vars)

    def resample(
        self,
        freq: str,
        how: str = "mean",
        *,
        skipna: bool = True,
        q: float | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSpatial.resample`."""
        warnings.warn(
            "`mesh.resample` is deprecated and will be removed in 0.90.0; use "
            "`mesh.spatial.resample`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.spatial.resample(freq, how, skipna=skipna, q=q)

    def mean(self, *, skipna: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.mean`."""
        warnings.warn(
            "`mesh.mean` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.mean`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.mean(skipna=skipna)

    def sum(self, *, skipna: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.sum`."""
        warnings.warn(
            "`mesh.sum` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.sum`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.sum(skipna=skipna)

    def min(self, *, skipna: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.min`."""
        warnings.warn(
            "`mesh.min` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.min`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.min(skipna=skipna)

    def max(self, *, skipna: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.max`."""
        warnings.warn(
            "`mesh.max` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.max`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.max(skipna=skipna)

    def std(self, *, skipna: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.std`."""
        warnings.warn(
            "`mesh.std` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.std`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.std(skipna=skipna)

    def var(self, *, skipna: bool = True) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshAnalysis.var`."""
        warnings.warn(
            "`mesh.var` is deprecated and will be removed in 0.90.0; use "
            "`mesh.analysis.var`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analysis.var(skipna=skipna)

    def _transform_time(
        self, operation: str, fn: Any, active_fn: Any = None
    ) -> MeshDataset:
        """Apply a shape-preserving ``fn(float_data, axis)`` to each temporal variable.

        ``fn`` receives the variable's values as float64 with no-data blanked to NaN, and the
        integer time axis, and returns an array of the same shape. Static variables are kept
        unchanged. Shared by the cumulative, shift, fill and rolling members.

        ``active_fn`` transforms a **per-step** active mask alongside the data. It is needed
        only for an op that *relocates* snapshots along time (``shift``): the implicit mask
        carry keeps a shape-preserving mask as-is, which is right for the value-preserving
        per-step ops (``cumsum``/``cumprod``, ``ffill``/``bfill``, ``rolling`` — step ``t``
        still describes step ``t``) but wrong for ``shift``, where row ``t`` now holds another
        step's snapshot. When given, ``active_fn(mask)`` is applied to a per-step mask (time on
        axis 0) so it moves with the data; a static mask is step-independent and left unchanged.
        """
        self._require_temporal(operation)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(
                    f"Variable {name!r} has no loaded data for {operation}."
                )
            transformed = fn(gaps_as_nan(np.asarray(data), var.nodata), axis)
            if active_fn is not None and var.active_has_time and var.active is not None:
                moved = var.with_data(
                    np.asarray(transformed),
                    active=active_fn(np.asarray(var.active)),
                )
            else:
                moved = var.with_data(np.asarray(transformed))
            new_vars[name] = moved
        return self._rebuild(new_vars)

    def cumsum(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.cumsum`."""
        warnings.warn(
            "`mesh.cumsum` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.cumsum`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.cumsum()

    def cumprod(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.cumprod`."""
        warnings.warn(
            "`mesh.cumprod` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.cumprod`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.cumprod()

    def shift(self, periods: int = 1) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.shift`."""
        warnings.warn(
            "`mesh.shift` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.shift`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.shift(periods)

    def _require_variable_name(self, operation: str, variable_name: str | None) -> None:
        """Refuse a ``variable_name`` the dataset does not have.

        The members that sweep every variable skip the names that do not match, so an unknown
        name used to match nothing and return the dataset unchanged — a typo read as a
        successful fit or conversion.

        The wording comes from :func:`unknown_variable_message`, which the facade mixin's
        ``_facade_variable`` also uses, so both halves of the API say the same thing about the
        same mistake. They previously agreed only on the exception *type*: thirteen facade
        members raised a bare ``KeyError('nope')`` naming neither the operation nor the
        available variables.

        Args:
            operation: The calling method's name, for the message.
            variable_name: The requested name, or ``None`` to act on every variable.

        Raises:
            KeyError: ``variable_name`` is not one of the dataset's data variables.
        """
        if variable_name is None or variable_name in self._data_variables:
            return
        raise KeyError(
            unknown_variable_message(
                operation, variable_name, list(self._data_variables)
            )
        )

    def _alongtime_coord(self, operation: str) -> list:
        """The time coordinate the fitting / calculus members work against (issue #1377).

        Uses :attr:`time_values`, which carries real datetimes for a file whose CF time
        coordinate decoded (issue #1381) and positional indices otherwise. The kernels convert
        either to a numeric axis, so an unevenly spaced output interval gives the right rate.

        Args:
            operation: The calling method's name, for the message.

        Returns:
            list: The coordinate values along the time axis.

        Raises:
            ValueError: No variable has a time dimension.
        """
        self._require_temporal(operation)
        coord = self.time_values
        if coord is None:  # pragma: no cover - _require_temporal already guarantees one
            raise ValueError(
                f"{operation} needs a time coordinate; this dataset has none."
            )
        return coord

    def polyfit(self, deg: int, variable_name: str | None = None) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.polyfit`."""
        warnings.warn(
            "`mesh.polyfit` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.polyfit`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.polyfit(deg, variable_name)

    def curvefit(
        self, func: Any, p0: Any = None, variable_name: str | None = None
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.curvefit`."""
        warnings.warn(
            "`mesh.curvefit` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.curvefit`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.curvefit(func, p0, variable_name)

    def _fit_alongtime(
        self,
        variable_name: str | None,
        kernel: Any,
        axis_name: str,
    ) -> MeshDataset:
        """Shared body of :meth:`polyfit` / :meth:`curvefit`.

        Both replace the time axis with a fitted-parameter axis, which makes the result static
        — it is no longer indexed by time — so the time coordinate is dropped with it.

        Args:
            variable_name: Variable to fit, or ``None`` for every temporal one.
            kernel: ``f(values, axis, nodata, coord) -> fitted`` doing the fit. The
                sentinel and the coordinate are both passed per variable, so a declared
                ``-9999`` is excluded and each variable is fitted against its own time axis.
            axis_name: Name for the axis that replaces the time axis — ``"coefficient"`` for
                :meth:`polyfit`, ``"parameter"`` for :meth:`curvefit`.

        Returns:
            MeshDataset: A new dataset carrying the fitted values.

        Raises:
            ValueError: A selected variable has no loaded data to fit.
        """
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time or (
                variable_name is not None and name != variable_name
            ):
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to fit.")
            fitted = kernel(
                np.asarray(data), axis, var.nodata, _variable_time_coord(var)
            )
            dims = _fitted_dimensions(var, fitted.ndim, axis, axis_name)
            attributes = _without_time_metadata(var.attributes)
            new_vars[name] = MeshVariable(
                name=var.name,
                location=var.location,
                mesh_name=var.mesh_name,
                shape=fitted.shape,
                attributes=attributes,
                # Coefficients are a different quantity from the values they were fitted to,
                # so the source's sentinel describes nothing here - and the kernels mark a
                # missing result with NaN, so declaring `-9999` would claim a marker the data
                # does not use. Same reasoning as `_rewrap(keep_nodata=False)`.
                nodata=None,
                units=var.units,
                standard_name=var.standard_name,
                dimensions=dims,
                _data=fitted,
            )
        return self._rebuild(new_vars)

    def differentiate(self, variable_name: str | None = None) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.differentiate`."""
        warnings.warn(
            "`mesh.differentiate` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.differentiate`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.differentiate(variable_name)

    def cumulative_integrate(self, variable_name: str | None = None) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.cumulative_integrate`."""
        warnings.warn(
            "`mesh.cumulative_integrate` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.cumulative_integrate`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.cumulative_integrate(variable_name)

    def rank(
        self, variable_name: str | None = None, *, pct: bool = False
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.rank`."""
        warnings.warn(
            "`mesh.rank` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.rank`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.rank(variable_name, pct=pct)

    def _map_alongtime(
        self, operation: str, variable_name: str | None, kernel: Any
    ) -> MeshDataset:
        """Apply a shape-preserving along-time kernel to the selected temporal variables.

        The shared body of the members that keep the time axis — :meth:`differentiate`,
        :meth:`cumulative_integrate` and :meth:`rank`.

        Args:
            operation: The calling method's name, for messages.
            variable_name: Variable to transform, or ``None`` for every temporal one.
            kernel: ``f(values, axis, nodata, coord) -> transformed``, which must preserve
                the shape. The sentinel and the coordinate are both passed per variable, so
                each is transformed against its own time axis and its own missing-value
                convention.

        Returns:
            MeshDataset: A new dataset carrying the transformed values.

        Raises:
            ValueError: A selected variable has no loaded data.
        """
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time or (
                variable_name is not None and name != variable_name
            ):
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(
                    f"Variable {name!r} has no loaded data for {operation}."
                )
            transformed = var.with_data(
                kernel(np.asarray(data), axis, var.nodata, _variable_time_coord(var))
            )
            # A rate, a running total or a rank is not the input quantity: the kernels write
            # NaN where the value was missing, so carrying the input's sentinel would declare
            # a marker the result never uses.
            transformed.nodata = None
            new_vars[name] = transformed
        return self._rebuild(new_vars)

    def integrate(self, variable_name: str | None = None) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.integrate`."""
        warnings.warn(
            "`mesh.integrate` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.integrate`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.integrate(variable_name)

    def _integrate_impl(self, variable_name: str | None = None) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.integrate`; see that method for the contract."""
        self._require_variable_name("integrate", variable_name)
        self._alongtime_coord("integrate")
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time or (
                variable_name is not None and name != variable_name
            ):
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to integrate.")
            total = _timeseries.integrate_axis(
                np.asarray(data), _variable_time_coord(var), axis, var.nodata
            )
            # The implicit carry is what decides: a static mask survives the collapse (as it
            # does through `reduce`), while a per-step one cannot follow an axis that is gone.
            # Passing `active=None` here force-dropped both.
            # `_static_from` already strips the time coordinate, so no second strip is needed.
            collapsed = _static_from(var, total, axis)
            # The total is in value-seconds, not the input's units, so the input's sentinel
            # does not describe it.
            collapsed.nodata = None
            new_vars[name] = collapsed
        return self._rebuild(new_vars)

    def coarsen(
        self, window: int, how: str = "mean", variable_name: str | None = None
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.coarsen`."""
        warnings.warn(
            "`mesh.coarsen` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.coarsen`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.coarsen(window, how, variable_name)

    def _coarsen_impl(
        self, window: int, how: str = "mean", variable_name: str | None = None
    ) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.coarsen`; see that method for the contract."""
        if window < 1:
            raise ValueError(f"coarsen: window must be at least 1, got {window}.")
        if how not in _COARSEN_HOWS:
            raise ValueError(
                f"coarsen: how must be one of {sorted(_COARSEN_HOWS)}, got {how!r}."
            )
        self._require_variable_name("coarsen", variable_name)
        self._alongtime_coord("coarsen")
        sweeping = variable_name is None
        new_vars: dict[str, MeshVariable] = {}
        skipped: list[str] = []
        for name, var in self._data_variables.items():
            if not var.has_time or (not sweeping and name != variable_name):
                new_vars[name] = var
                continue
            binned = _coarsened_variable(name, var, window, how, sweeping)
            if binned is None:
                skipped.append(f"{name!r}({var.n_time_steps})")
                new_vars[name] = var
                continue
            new_vars[name] = binned
        if sweeping and len(skipped) == len(
            [v for v in self._data_variables.values() if v.has_time]
        ):
            raise ValueError(
                f"coarsen: no variable has as many as {window} step(s) "
                f"({', '.join(skipped)}), so a window of {window} would report nothing at "
                f"all. Use a smaller window."
            )
        return self._rebuild(new_vars)

    def interp_like(self, other: MeshDataset) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.interp_like`."""
        warnings.warn(
            "`mesh.interp_like` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.interp_like`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.interp_like(other)

    def _interp_like_impl(self, other: MeshDataset) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.interp_like`; see that method for the contract."""
        self._assert_same_topology(other, "interp_like")
        self._alongtime_coord("interp_like")
        target = other._alongtime_coord("interp_like")
        target_units, target_calendar = other._declared_time_metadata()
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(
                    f"Variable {name!r} has no loaded data to interpolate."
                )
            source_scale, target_scale = _interp_scale(
                _variable_time_coord(var), target
            )
            moved = var.with_data(
                _timeseries.interp_axis(
                    np.asarray(data), source_scale, target_scale, axis
                ),
                # A static mask describes the elements, not the steps, so re-stamping the time
                # axis leaves it valid; only a per-step mask cannot follow. This is the same
                # carry `integrate` and `coarsen` use.
                active=None if var.active_has_time else _KEEP_ACTIVE,
            )
            attributes = {**var.attributes, "time_values": list(target)}
            if target_units is None:
                # The target is positional, so the result's stamps are positions too. Leaving
                # the source's CF units behind would have them re-encoded as real dates that
                # were never in either dataset.
                attributes.pop(TIME_UNITS_ATTR, None)
                attributes.pop(TIME_CALENDAR_ATTR, None)
            else:
                attributes[TIME_UNITS_ATTR] = target_units
                attributes[TIME_CALENDAR_ATTR] = target_calendar
            moved.attributes = attributes
            new_vars[name] = moved
        return self._rebuild(new_vars)

    def _declared_time_metadata(self) -> tuple[str | None, str]:
        """The CF units and calendar this dataset declares, or ``(None, "standard")``.

        Returns:
            tuple: ``(units, calendar)``; ``units`` is ``None`` for a positional axis, which is
            how a caller tells a dated dataset from an undated one.
        """
        for var in self._data_variables.values():
            if var.has_time and TIME_UNITS_ATTR in var.attributes:
                return (
                    str(var.attributes[TIME_UNITS_ATTR]),
                    str(var.attributes.get(TIME_CALENDAR_ATTR) or "standard"),
                )
        return None, "standard"

    def _calendar_instants(self, operation: str) -> tuple[np.ndarray, str, str]:
        """Calendar-aware instants for the time axis, with its CF units and calendar.

        The calendar members need real dates, not positions: "move 29 February onto a
        ``noleap`` axis" is meaningless against a positional index. The stamps come from the CF
        time coordinate the reader decodes (issue #1381), and an axis carrying no CF ``units``
        is refused with a message that says why rather than inventing an epoch.

        Instants are normalised to ``cftime`` objects whatever the reader produced — it yields
        ``datetime64`` for the Gregorian family and ``cftime`` for model calendars, and only
        the latter answers calendar arithmetic.

        Args:
            operation: The calling method's name, for the message.

        Returns:
            tuple: ``(instants, units, calendar)``.

        Raises:
            ValueError: No variable has a time dimension, or the axis carries no CF time units
                and so cannot be re-calendared.
        """
        self._require_temporal(operation)
        stamps = self.time_values
        units = None
        calendar = "standard"
        for var in self._data_variables.values():
            if var.has_time and TIME_UNITS_ATTR in var.attributes:
                units = str(var.attributes[TIME_UNITS_ATTR])
                calendar = str(var.attributes.get(TIME_CALENDAR_ATTR) or "standard")
                break
        if units is None or stamps is None:
            raise ValueError(
                f"{operation}() needs a decoded CF time axis: this dataset's time coordinate "
                f"carries no 'units' attribute of the form '<interval> since <origin>', so its "
                f"steps are positions rather than dates and there is no calendar to convert "
                f"between."
            )
        instants = [_to_cftime(stamp, calendar) for stamp in stamps]
        return np.asarray(instants, dtype=object), units, calendar

    def convert_calendar(self, calendar: str, *, align_on: str = "date") -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.convert_calendar`."""
        warnings.warn(
            "`mesh.convert_calendar` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.convert_calendar`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.convert_calendar(calendar, align_on=align_on)

    def _convert_calendar_impl(
        self, calendar: str, *, align_on: str = "date"
    ) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.convert_calendar`; see that method for the contract."""
        instants, units, _source = self._calendar_instants("convert_calendar")
        # The five validations and the D8 `units` re-encode are the shared plan in
        # `netcdf/utils.py`, so the mesh refuses exactly where the cube's `convert_calendar`
        # does (issue #1381): an epoch the target calendar has no day for is caught here rather
        # than surviving to write, where `_write_time_coordinate` would warn and silently drop
        # the time axis. The mesh does not need the re-encoded offsets, only the refusal.
        carried, kept, _offsets = _plan_calendar_conversion(
            instants, units, calendar, align_on, caller="convert_calendar"
        )
        indices = np.asarray(kept, dtype=np.intp)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to restamp.")
            active = var.active
            if active is not None and var.active_has_time:
                # The mask is per step, so it loses exactly the steps the values lost.
                active = np.take(np.asarray(active), indices, axis=axis)
            moved = var.with_data(
                np.take(np.asarray(data), indices, axis=axis), active=active
            )
            moved.attributes = {
                **var.attributes,
                "time_values": list(carried),
                TIME_UNITS_ATTR: units,
                TIME_CALENDAR_ATTR: calendar,
            }
            new_vars[name] = moved
        return self._rebuild(new_vars)

    def interp_calendar(self, target: MeshDataset) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.interp_calendar`."""
        warnings.warn(
            "`mesh.interp_calendar` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.interp_calendar`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.interp_calendar(target)

    def _interp_calendar_impl(self, target: MeshDataset) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.interp_calendar`; see that method for the contract."""
        self._assert_same_topology(target, "interp_calendar")
        source_instants, _units, _calendar = self._calendar_instants("interp_calendar")
        target_instants, target_units, target_calendar = target._calendar_instants(
            "interp_calendar"
        )
        # The common-scale build and its two refusals (too few source steps, no overlap) are the
        # shared plan in `netcdf/utils.py`, the same one the cube's `interp_calendar` runs; the
        # no-overlap message now reports both decimal-year spans, as the cube's always has.
        source_years, target_years = _plan_calendar_interpolation(
            source_instants, target_instants, caller="interp_calendar", noun="dataset"
        )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(
                    f"Variable {name!r} has no loaded data to interpolate."
                )
            moved = var.with_data(
                _timeseries.interp_axis(
                    np.asarray(data), source_years, target_years, axis
                ),
                # As in `interp_like`: a static element mask is unaffected by re-stamping.
                active=None if var.active_has_time else _KEEP_ACTIVE,
            )
            moved.attributes = {
                **var.attributes,
                "time_values": list(target_instants),
                TIME_UNITS_ATTR: target_units,
                TIME_CALENDAR_ATTR: target_calendar,
            }
            new_vars[name] = moved
        return self._rebuild(new_vars)

    def to_image(
        self,
        variable_name: str | None = None,
        cell_size: float | None = None,
        cmap: str = "viridis",
        method: str = "nearest",
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshGlyph.to_image`."""
        warnings.warn(
            "`mesh.to_image` is deprecated and will be removed in 0.90.0; use "
            "`mesh.glyph.to_image`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glyph.to_image(
            variable_name, cell_size=cell_size, cmap=cmap, method=method
        )

    def convert_units(
        self, target: str, variable_name: str | None = None
    ) -> MeshDataset:
        """Convert variable values to ``target`` units (issue #1378).

        Mesh variables carry a ``units`` attribute, so the raster's conversion applies
        unchanged — and it is the same affine table
        (:func:`pyramids.dataset.ops.units.convert_array`), so the two sides cannot disagree
        about what a Kelvin is.

        No-data elements keep their sentinel rather than being converted as though they were
        data, which would otherwise turn ``-9999`` into a plausible-looking temperature.

        Args:
            target: Target unit label (e.g. ``"celsius"``, ``"hPa"``, ``"knots"``).
            variable_name: Variable to convert. ``None`` (default) converts every variable
                that declares a unit; one already in ``target`` units passes through unchanged.

        Returns:
            MeshDataset: A new dataset with converted values and the new unit recorded.

        Raises:
            ValueError: A selected variable declares no source unit, or the
                ``(source, target)`` pair is not in the conversion table.

        See Also:
            Dataset.convert_units: The raster counterpart, sharing this table.
        """
        self._require_variable_name("convert_units", variable_name)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            selected = variable_name is None or name == variable_name
            if not selected or var.units == target:
                new_vars[name] = var
                continue
            if not var.units:
                if variable_name is None:
                    # A sweep over every variable skips the ones with nothing to convert from;
                    # naming one explicitly is a request that must not pass silently.
                    new_vars[name] = var
                    continue
                raise ValueError(
                    f"convert_units() cannot convert {name!r}: it declares no units to "
                    f"convert from. Set its `units` first."
                )
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to convert.")
            values = np.asarray(data, dtype="float64")
            converted = convert_array(values, var.units, target)
            if var.nodata is not None:
                # The sentinel is not a measurement, so it keeps its value.
                converted = np.where(values == var.nodata, values, converted)
            moved = var.with_data(converted)
            moved.units = target
            moved.attributes = {**var.attributes, "units": target}
            new_vars[name] = moved
        return self._rebuild(new_vars)

    def to_dask_dataframe(
        self,
        variables: Sequence[str] | None = None,
        *,
        dropna: bool = False,
        npartitions: int = 1,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshInterop.to_dask_dataframe`."""
        warnings.warn(
            "`mesh.to_dask_dataframe` is deprecated and will be removed in 0.90.0; use "
            "`mesh.interop.to_dask_dataframe`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.interop.to_dask_dataframe(
            variables, dropna=dropna, npartitions=npartitions
        )

    def ffill(self, *, limit: int | None = None, dim: str | None = None) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.ffill`."""
        warnings.warn(
            "`mesh.ffill` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.ffill`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.ffill(limit=limit, dim=dim)

    def bfill(self, *, limit: int | None = None, dim: str | None = None) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.bfill`."""
        warnings.warn(
            "`mesh.bfill` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.bfill`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.bfill(limit=limit, dim=dim)

    def interpolate_na(
        self,
        *,
        method: str = "linear",
        limit: int | None = None,
        use_coordinate: bool = True,
        dim: str | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.interpolate_na`."""
        warnings.warn(
            "`mesh.interpolate_na` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.interpolate_na`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.interpolate_na(
            method=method, limit=limit, use_coordinate=use_coordinate, dim=dim
        )

    def rolling(
        self,
        window: int,
        how: str = "mean",
        *,
        center: bool = False,
        min_periods: int | None = None,
        q: float | None = None,
        dim: str | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.rolling`."""
        warnings.warn(
            "`mesh.rolling` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.rolling`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.rolling(
            window, how, center=center, min_periods=min_periods, q=q, dim=dim
        )

    def _rolling_impl(
        self,
        window: int,
        how: str = "mean",
        *,
        center: bool = False,
        min_periods: int | None = None,
        q: float | None = None,
        dim: str | None = None,
    ) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.rolling`; see that method for the contract."""
        if how not in _ROLLING_HOWS:
            # Restrict to the documented statistics. The counting reductions
            # (count/all/any) are excluded on purpose: a window's all/any flag carries the
            # 255 FLAG_NO_DATA sentinel, which would surface as a literal data value here.
            raise ValueError(
                f"rolling how must be one of {sorted(_ROLLING_HOWS)}, got {how!r}."
            )
        # Same guard as every other along-time member, through the shared check rather than
        # a second copy of it with its own wording.
        self._check_time_dim("rolling", dim)
        if how == "quantile" and q is None:
            raise ValueError("rolling(how='quantile') needs q in [0, 1].")
        if how != "quantile" and q is not None:
            raise ValueError(
                f"rolling(q=...) is only valid with how='quantile', not {how!r}."
            )
        # The shared window / min_periods guards (D10), the same the raster rolling uses, so
        # the mesh does not carry its own copy. `None` min_periods means the full window -- a
        # step whose window runs off the axis then has no statistic rather than a partial one.
        window = _check_window(window, caller="rolling")
        needed = _check_min_periods(min_periods, window)

        def _roll(data: np.ndarray, axis: int) -> np.ndarray:
            moved = np.moveaxis(data, axis, 0)
            size = moved.shape[0]
            out = np.full_like(moved, np.nan, dtype="float64")
            for step in range(size):
                members = window_members(step, size, window, center)
                block = moved[members]
                valid = np.sum(~np.isnan(block), axis=0)
                reduced = reduce_axis(block, 0, how, True, None, q)
                out[step] = np.where(valid >= needed, reduced, np.nan)
            return np.moveaxis(out, 0, axis)

        return self._transform_time("rolling", _roll)

    def _time_length(self, operation: str) -> int:
        """The shared time-axis length across all temporal variables.

        The positional/label selectors build one index set and apply it to every temporal
        variable, so they require a single shared length. Validate that here rather than let
        a mismatch leak a raw numpy ``IndexError`` (or silently trim against the wrong axis).

        Raises:
            ValueError: No variable has a time dimension, or temporal variables disagree on
                their time length.
        """
        temporal = self._require_temporal(operation)
        lengths = {name: self._data_variables[name].n_time_steps for name in temporal}
        distinct = set(lengths.values())
        if len(distinct) > 1:
            raise ValueError(
                f"{operation} needs every temporal variable to share one time length, but "
                f"they differ: {lengths}. Select per variable, or align them first."
            )
        return distinct.pop()

    def _select_steps(self, indices: np.typing.NDArray) -> MeshDataset:
        """Keep the time steps at ``indices`` (in order) for every temporal variable.

        Trims each temporal variable's data along its time axis and, when the variable
        carries a ``time_values`` coordinate attribute, trims that to match. Static variables
        are kept unchanged. The time dimension is retained (shorter), so the result stays
        temporal.
        """
        indices = np.asarray(indices, dtype=np.intp)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to select.")
            new_var = var.with_data(
                np.take(np.asarray(data), indices, axis=axis),
                active=var._time_indexed_active(indices),
            )
            times = var.attributes.get("time_values")
            if times is not None:
                new_var.attributes = {
                    **var.attributes,
                    "time_values": [times[int(i)] for i in indices],
                }
            new_vars[name] = new_var
        return self._rebuild(new_vars)

    def _select_elements(
        self, element: int | slice | list[int] | np.ndarray
    ) -> MeshDataset:
        """Keep the faces at ``element`` positions, renumbering the mesh around them.

        The element axis is the mesh's own axis, so selecting along it is a *topology* change,
        not a slice: the kept faces' nodes are renumbered compactly and the edges filtered, the
        way :meth:`clip` and :meth:`subset_by_bounds` already do. This shares their machinery
        (:func:`pyramids.netcdf.ugrid.spatial.subset_mesh_by_faces`) rather than re-deriving it.

        Args:
            element: An ``int`` face index, a ``slice``, or a sequence of face indices.

        Returns:
            MeshDataset: A new dataset whose mesh holds only the selected faces.

        Raises:
            IndexError: A face index is out of range for this mesh. A negative index counts
                from the end, so ``-1`` is the last face.
        """
        count = self._mesh.n_face
        if isinstance(element, slice):
            faces = [int(i) for i in np.arange(count)[element]]
        else:
            asked = [int(i) for i in np.atleast_1d(np.asarray(element, dtype=np.intp))]
            # A negative index counts from the end, as `isel(time=-1)` already does through
            # np.take. Refusing it here left one method with two index conventions.
            faces = [i + count if -count <= i < 0 else i for i in asked]
        out_of_range = [i for i in faces if i < 0 or i >= count]
        if out_of_range:
            raise IndexError(
                f"isel: face index/indices {out_of_range} out of range for a mesh with "
                f"{count} face(s)."
            )
        mesh, data_variables = subset_mesh_by_faces(self, faces)
        return self._wrap_subset(mesh, data_variables)

    def isel(
        self,
        time: int | slice | list[int] | np.ndarray | None = None,
        *,
        element: int | slice | list[int] | np.ndarray | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.isel`."""
        warnings.warn(
            "`mesh.isel` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.isel`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.isel(time, element=element)

    def _collapse_to_step(self, index: int) -> MeshDataset:
        """Take one time step from every temporal variable, dropping the time dimension.

        The single-step half of :meth:`isel`: each temporal variable becomes static, as
        :meth:`reduce` produces, while a static variable is carried through untouched. Shares
        the per-variable collapse (:func:`_collapse_variable_to_step`) with
        :meth:`sel_time`, so both read only the requested slab from disk for a leading-time
        file-backed variable (#982).

        Args:
            index: The time position to take.

        Returns:
            MeshDataset: A new dataset whose temporal variables hold that step.

        Raises:
            ValueError: A temporal variable has no loaded data.
            IndexError: ``index`` is out of range for the time axis.
        """
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if var.has_time:
                new_vars[name] = _collapse_variable_to_step(var, index)
            else:
                new_vars[name] = var
        return self._rebuild(new_vars)

    def head(self, n: int = 5) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.head`."""
        warnings.warn(
            "`mesh.head` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.head`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.head(n)

    def tail(self, n: int = 5) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.tail`."""
        warnings.warn(
            "`mesh.tail` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.tail`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.tail(n)

    def thin(self, step: int) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.thin`."""
        warnings.warn(
            "`mesh.thin` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.thin`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.thin(step)

    def drop_isel(self, indices: int | list[int] | np.ndarray) -> MeshDataset:
        """Drop the time steps at ``indices`` by position, keeping the rest.

        Args:
            indices: A step position, or a sequence of them, to drop (negative indices
                count from the end).

        Returns:
            MeshDataset: A new dataset without the dropped steps.

        Raises:
            ValueError: No variable has a time dimension.
        """
        length = self._time_length("drop_isel")
        drop = {int(i) % length for i in np.atleast_1d(np.asarray(indices))}
        keep = np.array([i for i in range(length) if i not in drop], dtype=np.intp)
        return self._select_steps(keep)

    def squeeze(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.squeeze`."""
        warnings.warn(
            "`mesh.squeeze` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.squeeze`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.squeeze()

    def _squeeze_impl(self) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.squeeze`; see that method for the contract."""
        # Decide per variable, not from the first temporal one: with mixed time lengths a
        # later single-step variable must still be squeezed even if the first has many steps.
        if not any(
            var.has_time and var.n_time_steps == 1
            for var in self._data_variables.values()
        ):
            return self
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time or var.n_time_steps != 1:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to squeeze.")
            # Squeezing the size-1 time axis is taking its single step, so collapse a per-step
            # mask to that step (`mask[0]`) rather than letting the implicit carry drop it as a
            # now-rank-mismatched 2-D mask (review L3). A static mask passes straight through.
            new_vars[name] = _static_from(
                var,
                np.squeeze(np.asarray(data), axis=axis),
                axis,
                active=var._time_sliced_active(0, None),
            )
        return self._rebuild(new_vars)

    def diff(self, n: int = 1) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.diff`."""
        warnings.warn(
            "`mesh.diff` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.diff`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.diff(n)

    def _extremum_along_time(
        self, operation: str, *, largest: bool, as_coordinate: bool
    ) -> MeshDataset:
        """Locate each element's extreme time step, as a position or as a coordinate.

        The time axis is dropped — the answer is one value per element — so the result is a
        static variable, as :meth:`reduce` produces. Gaps are left out: the search runs over the
        no-data-masked values, and an element with no valid step at all has no answer, which the
        position form declares as ``-1`` and the coordinate form as NaN (mirroring the raster
        ``argmin`` / ``idxmin``).

        Args:
            operation: The calling method's name, for error messages.
            largest: Search for the maximum rather than the minimum.
            as_coordinate: Answer with the time coordinate value instead of the step position.

        Returns:
            MeshDataset: A new dataset whose temporal variables became static; static
            variables are carried over unchanged.

        Raises:
            ValueError: No variable has a time dimension, a temporal variable has no loaded
                data, or the coordinate form was asked of date-like time coordinates, which
                cannot be expressed as the float64 the answer carries.
        """
        self._require_temporal(operation)
        coords = self._time_coords(operation) if as_coordinate else None
        if coords is not None and not all(
            isinstance(stamp, (int, float, np.integer, np.floating))
            and not isinstance(stamp, bool)
            for stamp in coords
        ):
            # The coordinate form answers float64, so a date-like coordinate cannot be
            # expressed. Say so, rather than letting the float() cast surface a raw
            # TypeError about Timestamp from inside numpy.
            raise ValueError(
                f"{operation}() answers the time coordinate as a float, so it needs numeric "
                "time coordinates; this dataset's are date-like. Use argmin()/argmax() for "
                "the step position instead."
            )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data.")
            values, nodata = _extremum_values(
                gaps_as_nan(np.asarray(data), var.nodata),
                axis,
                largest=largest,
                coords=coords if as_coordinate else None,
            )
            new_vars[name] = _static_from(var, values, axis, nodata=nodata)
        return self._rebuild(new_vars)

    def argmin(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.argmin`."""
        warnings.warn(
            "`mesh.argmin` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.argmin`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.argmin()

    def argmax(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.argmax`."""
        warnings.warn(
            "`mesh.argmax` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.argmax`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.argmax()

    def idxmin(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.idxmin`."""
        warnings.warn(
            "`mesh.idxmin` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.idxmin`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.idxmin()

    def idxmax(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.idxmax`."""
        warnings.warn(
            "`mesh.idxmax` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.idxmax`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.idxmax()

    def _time_coords(self, operation: str) -> list:
        """The time coordinate values, or a clear error when the dataset has no time."""
        coords = self.time_values
        if coords is None:
            raise ValueError(
                f"{operation} needs a time dimension; this dataset has none."
            )
        return list(coords)

    def sel(
        self,
        time: Any = None,
        *,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.sel`."""
        warnings.warn(
            "`mesh.sel` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.sel`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.sel(time, method=method, tolerance=tolerance)

    def _sel_impl(
        self,
        time: Any = None,
        *,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> MeshDataset:
        """Implementation of :meth:`_MeshSelection.sel`; see that method for the contract."""
        if time is None:
            # `time` is optional in the signature for parity with the raster `sel`, but a
            # missing value has no meaning here. Say so, as `isel` does, instead of
            # reporting that the coordinate [None] was not found.
            raise ValueError(
                "sel needs a time value to select; pass one, or use isel() to select by "
                "position."
            )
        coords = self._time_coords("sel")
        if method not in (None, "nearest"):
            raise ValueError(f"sel: method must be None or 'nearest', got {method!r}.")
        if tolerance is not None:
            if method != "nearest":
                raise ValueError("sel: tolerance is only valid with method='nearest'.")
            if float(tolerance) < 0:
                raise ValueError(
                    f"sel: tolerance must not be negative, got {tolerance!r}."
                )
        wanted = list(time) if isinstance(time, (list, tuple, np.ndarray)) else [time]
        positions = _resolve_time_positions(coords, wanted, method, tolerance)
        if isinstance(time, (list, tuple, np.ndarray)):
            result = self._select_steps(np.asarray(positions, dtype=np.intp))
        else:
            result = self.selection.isel(positions[0])
        return result

    def drop_sel(self, values: Any) -> MeshDataset:
        """Drop the time steps whose coordinate is in ``values``, keeping the rest.

        Args:
            values: A coordinate value, or a sequence of them, to drop.

        Returns:
            MeshDataset: A new dataset without the steps whose coordinate is in ``values``.

        Raises:
            ValueError: No variable has a time dimension.
        """
        coords = self._time_coords("drop_sel")
        drop = (
            set(values)
            if isinstance(values, (list, tuple, set))
            else {*np.atleast_1d(np.asarray(values)).tolist()}
        )
        keep = [i for i, coord in enumerate(coords) if coord not in drop]
        return self._select_steps(np.asarray(keep, dtype=np.intp))

    def sortby(self) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.sortby`."""
        warnings.warn(
            "`mesh.sortby` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.sortby`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.sortby()

    def drop_duplicates(self) -> MeshDataset:
        """Keep the first step of each distinct time coordinate value.

        Returns:
            MeshDataset: A new dataset with duplicate-coordinate steps removed, first kept.

        Raises:
            ValueError: No variable has a time dimension.
        """
        coords = self._time_coords("drop_duplicates")
        seen: set = set()
        keep: list[int] = []
        for i, coord in enumerate(coords):
            if coord not in seen:
                seen.add(coord)
                keep.append(i)
        return self._select_steps(np.asarray(keep, dtype=np.intp))

    def dropna(
        self, how: str = "any", thresh: int | None = None, *, dim: str | None = None
    ) -> MeshDataset:
        """Facade -- delegates to :meth:`_MeshSelection.dropna`."""
        warnings.warn(
            "`mesh.dropna` is deprecated and will be removed in 0.90.0; use "
            "`mesh.selection.dropna`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection.dropna(how, thresh, dim=dim)

    def _assert_same_topology(self, other: MeshDataset, operation: str) -> None:
        """Refuse ``operation`` unless ``other`` sits on the same mesh as ``self``.

        Same mesh means the same node/face/edge counts and the same node coordinates and
        face-node connectivity — combining data across different meshes is meaningless.
        """
        mine, theirs = self._mesh, other._mesh
        same = (
            mine.n_node == theirs.n_node
            and mine.n_face == theirs.n_face
            and mine.n_edge == theirs.n_edge
            and np.array_equal(mine.node_x, theirs.node_x)
            and np.array_equal(mine.node_y, theirs.node_y)
            and np.array_equal(
                np.asarray(mine.face_node_connectivity.data),
                np.asarray(theirs.face_node_connectivity.data),
            )
        )
        if not same:
            raise AlignmentError(
                f"{operation} requires datasets on the same mesh topology "
                f"(matching node/face/edge counts, coordinates and connectivity)."
            )

    @_joins_meshes
    def concat(self, others: MeshDataset | list[MeshDataset]) -> MeshDataset:
        """Concatenate same-topology datasets along time.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.concat`. Every dataset must sit
        on the same mesh. Each temporal variable's data is concatenated along its time axis
        (and its ``time_values`` coordinate, when present); a static variable is taken from
        ``self`` unchanged. The variable set must match across datasets.

        A **per-step** active mask (:attr:`MeshVariable.active` of shape ``(n_time, n_element)``)
        is **dropped** on concatenation — the parts' masks are not joined along time, and the
        concatenated time length no longer matches ``self``'s mask, so the implicit carry
        discards it (issue #1331). A **static** ``(n_element,)`` mask survives, since it applies
        to every step. Re-attach a per-step mask after concatenating if you need one.

        Args:
            others: One dataset, or a list of them, to append after ``self`` in order.

        Returns:
            MeshDataset: A new dataset spanning the concatenated time axis.

        Raises:
            AlignmentError: A dataset sits on a different mesh. Note this is **not** a
                `ValueError` subclass, so a caller catching `ValueError` around `concat` has
                to catch `AlignmentError` instead.
            ValueError: A dataset carries a different variable set, or a variable has no
                loaded data.
        """
        parts = [
            self,
            *([others] if isinstance(others, MeshDataset) else list(others)),
        ]
        for other in parts[1:]:
            self._assert_same_topology(other, "concat")
            if set(other._data_variables) != set(self._data_variables):
                raise ValueError(
                    "concat requires the same variables in every dataset; got "
                    f"{sorted(self._data_variables)} vs {sorted(other._data_variables)}."
                )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                # Static in self: refuse if any other part has it temporal, rather than
                # silently drop that part's time series (the mirror of the temporal-in-self
                # / static-in-other case _concat_temporal_variable rejects).
                if any(part._data_variables[name].has_time for part in parts[1:]):
                    raise ValueError(
                        f"concat: variable {name!r} is static in one dataset but temporal "
                        "in another; they cannot be joined along time."
                    )
                new_vars[name] = var  # a static variable is taken from self, as-is
            else:
                new_vars[name] = _concat_temporal_variable(name, var, parts)
        return self._rebuild(new_vars)

    @_joins_meshes
    def merge(
        self,
        others: MeshDataset | list[MeshDataset],
        *,
        compat: str = "no_conflicts",
    ) -> MeshDataset:
        """Merge the variables of same-topology datasets into one.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.merge`. Every dataset must sit
        on the same mesh; the union of their variables is returned. A variable name present
        in more than one dataset is a conflict and is refused.

        Args:
            others: One dataset, or a list of them, whose variables join ``self``'s. Under
                the class spelling ``MeshDataset.merge([a, b, c])`` this one argument is the
                whole join: its first element becomes ``self`` and the rest are merged onto it.
            compat: How to handle a variable name present in more than one dataset --
                ``"no_conflicts"`` refuses the collision, ``"override"`` keeps the value from
                the **first dataset in join order that carries the name**: ``self``, then each
                of ``others`` in turn. That is ``self``'s only when ``self`` has the variable;
                when it does not, the first of ``others`` carrying it wins. (Under the class
                spelling ``MeshDataset.merge([a, b, c])``, ``self`` is ``a``.) Defaults to
                ``"no_conflicts"``.

        Returns:
            MeshDataset: A new dataset carrying every variable.

        Raises:
            AlignmentError: A dataset sits on a different mesh.
            ValueError: An unknown ``compat``, or a variable name collides under
                ``compat="no_conflicts"``.
        """
        if compat not in ("no_conflicts", "override"):
            raise ValueError(
                f"merge: compat must be 'no_conflicts' or 'override', got {compat!r}."
            )
        parts = [others] if isinstance(others, MeshDataset) else list(others)
        new_vars: dict[str, MeshVariable] = dict(self._data_variables)
        for other in parts:
            self._assert_same_topology(other, "merge")
            for name, var in other._data_variables.items():
                if name in new_vars:
                    if compat == "no_conflicts":
                        raise ValueError(
                            f"merge: variable {name!r} is present in more than one dataset; "
                            "rename it first, or pass compat='override'."
                        )
                    # compat="override": the first dataset that carries the name wins, which
                    # is what already sits in new_vars, so the duplicate is simply skipped.
                    continue
                new_vars[name] = var
        return self._rebuild(new_vars)

    def _glyph_with_crs(self, glyph: Any) -> Any:
        """Tag a returned cleopatra ``MeshGlyph`` with the dataset CRS.

        cleopatra's ``MeshGlyph`` takes no ``crs`` in ``__init__`` and the ``Mesh2d``
        topology carries none, so the dataset is the only CRS source. Setting
        ``glyph.crs`` lets a downstream ``glyph.add_tiles()`` / ``add_features()`` /
        ``add_relief()`` default to the data CRS instead of cleopatra's EPSG:3857
        fallback. The EPSG code is used when the CRS has one; a projected CRS with no
        EPSG authority code (ESRI / custom projections) falls back to the CRS WKT,
        which cleopatra's ``crs`` setter also accepts. An already-set ``glyph.crs`` is
        left untouched.

        Args:
            glyph: The ``MeshGlyph`` returned by a cleopatra render call.

        Returns:
            The same glyph, with ``crs`` set to the dataset CRS (``self.epsg`` when it
            has an EPSG code, else ``self.crs_wkt``) when the dataset has a CRS and the
            glyph did not already carry one.
        """
        crs_source = self.epsg if self.epsg is not None else self.crs_wkt
        if crs_source is not None and hasattr(glyph, "crs") and glyph.crs is None:
            glyph.crs = crs_source
        return glyph

    def plot(
        self,
        variable_name: str,
        ax: Any = None,
        cmap: str = "viridis",
        title: str | None = None,
        basemap: bool | str | None = None,
        colorbar: bool | ColorBar | None = None,
        points: np.ndarray | PointOverlay | None = None,
        kind: str = "auto",
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        data_style: DataStyle | None = None,
        *,
        backend: str = "matplotlib",
        **kwargs: Any,
    ) -> Any:
        """Plot a mesh data variable.

        Facade -- delegates to :meth:`_MeshGlyph.plot`.
        """
        warnings.warn(
            "`mesh.plot` is deprecated and will be removed in 0.90.0; use "
            "`mesh.glyph.plot`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glyph.plot(
            variable_name,
            ax=ax,
            cmap=cmap,
            title=title,
            basemap=basemap,
            colorbar=colorbar,
            points=points,
            kind=kind,
            color=color,
            contour=contour,
            data_style=data_style,
            backend=backend,
            **kwargs,
        )

    def plot_vector_field(
        self,
        u: str,
        v: str,
        *,
        kind: str = "quiver",
        thin: int | None = None,
        time_index: int = 0,
        ax: Any = None,
        title: str | None = None,
        colorbar: bool | ColorBar | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot two mesh variables as a 2-component vector field (issue #1373).

        Facade -- delegates to :meth:`_MeshGlyph.plot_vector_field`.
        """
        warnings.warn(
            "`mesh.plot_vector_field` is deprecated and will be removed in 0.90.0; use "
            "`mesh.glyph.plot_vector_field`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glyph.plot_vector_field(
            u,
            v,
            kind=kind,
            thin=thin,
            time_index=time_index,
            ax=ax,
            title=title,
            colorbar=colorbar,
            **kwargs,
        )

    def _vector_component(
        self, variable_name: str, role: str, time_index: int
    ) -> tuple[np.ndarray, str]:
        """Read one vector component down to a single per-element array.

        Shared by both halves of :meth:`plot_vector_field` so ``u`` and ``v`` are validated
        and reduced identically -- the asymmetry of checking one and trusting the other is
        how a mismatched pair reaches the renderer and is drawn in the wrong place.

        The reduction itself is :meth:`_element_values`, the module's existing shared
        reducer, rather than a second copy of it: that is what turns the variable's declared
        no-data value into NaN (a fill reaching cleopatra as a real vector dominates the
        colour scale and hides the real arrows) and what refuses a layered, non-per-element
        variable by name. Only the checks the renderer needs on top -- the location, and the
        step range -- live here.

        Args:
            variable_name: The variable to read.
            role: ``"u"`` or ``"v"``, naming the component in any error.
            time_index: Step to take along a temporal component's own time axis. Negative
                indices count from the end, as elsewhere in numpy.

        Returns:
            tuple: The per-element values with gaps blanked to NaN, and the location they
                sit on.

        Raises:
            KeyError: No variable is named ``variable_name``.
            ValueError: The variable has no loaded data, sits on a location a vector cannot
                be anchored at, is not per-element after the time collapse, or
                ``time_index`` is outside its time axis.
        """
        var = self[variable_name]
        if var.data is None:
            raise ValueError(
                f"plot_vector_field: {role} component {variable_name!r} has no loaded data "
                f"to plot."
            )
        if var.location not in ("face", "node"):
            raise ValueError(
                f"plot_vector_field: {role} component {variable_name!r} is "
                f"{var.location}-located, and a vector can only be anchored at a face "
                f"centroid or a node. Supported locations: 'face', 'node'. Resolve an "
                f"edge-located component to faces first."
            )
        shape = np.asarray(var.data).shape
        if var.has_time:
            axis = cast("int", var.time_axis_index)
            if not -len(shape) <= axis < len(shape):
                # `time_index` is derived from `dimensions` by name, so a variable declaring
                # more axes than its data has yields an axis past the array's rank. Indexing
                # `shape` with it raised a bare IndexError out of the very pre-check that
                # exists to report this case well.
                raise ValueError(
                    f"plot_vector_field: {role} component {variable_name!r} declares a time "
                    f"axis at position {axis} but its data has {len(shape)} axis/axes "
                    f"{shape}; the variable's dimensions and its array disagree."
                )
            steps = shape[axis]
            if not -steps <= time_index < steps:
                raise ValueError(
                    f"plot_vector_field: time_index {time_index} is outside "
                    f"{variable_name!r}, which has {steps} step(s)."
                )
        # The rank is checked here rather than by catching the shared reducer's
        # ValueError: that reducer also funnels `gaps_as_nan`, which raises ValueError for a
        # dtype it cannot make float, and branding *that* as "not a single value per
        # element" was a confidently false diagnosis of a variable that is 1-D.
        collapsed_rank = len(shape) - 1 if var.has_time else len(shape)
        if collapsed_rank != 1:
            raise ValueError(
                f"plot_vector_field: {role} component {variable_name!r} is not a single "
                f"value per element, so it cannot be drawn as a vector field: it has "
                f"{collapsed_rank} axes once the time axis is collapsed. A layered or "
                f"otherwise multi-dimensional variable has no single per-element vector."
            )
        # `_element_values` resolves the name a second time rather than taking `var`. Left
        # as is: `get_data` is a mapping lookup, so the two calls cannot see different
        # objects, and threading `var` through would change the signature of a reducer five
        # other members share.
        _var, values = self._element_values(variable_name, time_index)
        return values, var.location

    def animate(
        self,
        variable_name: str | None = None,
        *,
        interval: int = 200,
        backend: str = "matplotlib",
        **kwargs: Any,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshGlyph.animate`."""
        warnings.warn(
            "`mesh.animate` is deprecated and will be removed in 0.90.0; use "
            "`mesh.glyph.animate`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glyph.animate(
            variable_name, interval=interval, backend=backend, **kwargs
        )

    def _animate_impl(
        self,
        variable_name: str | None = None,
        *,
        interval: int = 200,
        backend: str = "matplotlib",
        **kwargs: Any,
    ) -> Any:
        """Implementation of :meth:`_MeshGlyph.animate`.

        Kept on ``MeshDataset`` rather than moved onto the glyph engine: the body reads the
        per-variable time coordinate through the module-level ``_variable_time_coord`` helper
        in this module, which the engine cannot import without a cycle (``dataset`` imports
        the engine for its ``glyph`` accessor).
        """
        if backend != "matplotlib":
            # interval defaults to 200 ms; forward it only when the user changed it, so a backend
            # keeps its own default frame interval.
            forwarded = forward_kwargs(
                variable_name=variable_name,
                interval=None if interval == 200 else interval,
            )
            return render_via_backend(
                self, backend, animate=True, **forwarded, **kwargs
            )
        var = self._facade_variable("animate", variable_name)
        if not var.has_time:
            raise ValueError(
                f"animate() needs a time axis, but {var.name!r} has none — there would be a "
                f"single frame. Use plot() for a static variable."
            )
        data = np.asarray(var.data)
        axis = cast("int", var.time_axis_index)
        frames = np.moveaxis(data, axis, 0)
        if frames.ndim != 2:
            raise ValueError(
                f"animate() draws one value per element per frame, but {var.name!r} has "
                f"shape {data.shape} — a dimension besides time and the elements. Select a "
                f"layer first, e.g. with isel(layer=0) or collapse_layer()."
            )
        # The variable's own coordinate, not the dataset's: `self.time_values` is the FIRST
        # temporal variable's, so animating a second one of a different length failed inside
        # cleopatra ("time length (3) does not match frame count (6)"). Every other along-time
        # member moved onto this helper in 2c8a308; `animate` was missed.
        labels = _variable_time_coord(var)
        # animate_mesh_data returns a matplotlib FuncAnimation; no glyph is exposed and
        # no basemap is drawn, so there is no CRS to tag (unlike plot / plot_outline, see
        # _glyph_with_crs). If this ever hands back a glyph or composes a basemap, route
        # it through _glyph_with_crs or issue #1560 reopens on this surface.
        return _ugrid_plot.animate_mesh_data(
            self._require_mesh2d("animate"),
            frames,
            labels,
            location=var.location,
            interval=interval,
            **kwargs,
        )

    def plot_outline(
        self,
        ax: Any = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Facade -- delegates to :meth:`_MeshGlyph.plot_outline`."""
        warnings.warn(
            "`mesh.plot_outline` is deprecated and will be removed in 0.90.0; use "
            "`mesh.glyph.plot_outline`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glyph.plot_outline(ax=ax, basemap=basemap, **kwargs)

    def __str__(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"MeshDataset: {self._file_name or '(in-memory)'}",
            f"  Mesh: {self.mesh_name}",
            f"  Nodes: {self.n_node}, Faces: {self.n_face}, Edges: {self.n_edge}",
            f"  Bounds: {self.bounds}",
            # A mesh may carry a projected WKT with no EPSG code; reporting
            # 'unknown' for a CRS it plainly has was misleading.
            f"  CRS: {self.epsg or (self.crs.name if self.crs is not None else 'unknown')}",
            f"  Data variables ({len(self._data_variables)}):",
        ]
        for name, var in self._data_variables.items():
            lines.append(f"    {name}: location={var.location}, shape={var.shape}")
        result = "\n".join(lines)
        return result

    def __repr__(self) -> str:
        """Repr string for the dataset."""
        result = (
            f"MeshDataset(mesh='{self.mesh_name}', "
            f"n_node={self.n_node}, n_face={self.n_face}, n_edge={self.n_edge}, "
            f"variables={self.variable_names})"
        )
        return result


#: The statistics :meth:`MeshDataset.rolling` supports — the collapsing reducers that make
#: sense per window. The counting reducers (``count``/``all``/``any``) are excluded: their
#: flag bands carry the ``FLAG_NO_DATA`` sentinel, which must not surface as a data value.
_ROLLING_HOWS: frozenset[str] = frozenset(
    {"mean", "sum", "min", "max", "std", "var", "median", "quantile"}
)

#: The reducers :meth:`MeshDataset.coarsen` accepts — the set the retired ``coarsen_axis``
#: kernel hard-coded. Kept narrower than the full :data:`REDUCERS` registry on purpose: the
#: positional binner has always reported one float statistic per block, so the counting and
#: quantile reducers stay out of its surface.
_COARSEN_HOWS: frozenset[str] = frozenset(
    {"mean", "sum", "min", "max", "median", "std"}
)


def _check_reduce_how(
    how: str, q: float | None, *, allow_quantile: bool = True
) -> None:
    """Validate a reduction ``how`` / ``q`` pair, mirroring the raster path's guards.

    The raster ``Selection.reduce`` routes ``how`` through ``_check_how`` and ``q`` through
    ``_check_quantile``; the mesh members call this to get the same clear ``ValueError``
    instead of a raw numpy ``KeyError`` / ``TypeError``.

    Args:
        how: The requested reduction.
        q: The quantile, required for ``how="quantile"`` and rejected otherwise.
        allow_quantile: Whether ``"quantile"`` is a permitted ``how`` (``rolling`` has no
            ``q`` argument, so it forbids it). Defaults to True.

    Raises:
        ValueError: ``how`` is not a known reduction, ``q`` is given for a non-quantile
            ``how``, or ``how="quantile"`` is given without ``q``.
    """
    valid = set(REDUCERS) | set(COUNTING_REDUCERS)
    if not allow_quantile:
        valid.discard("quantile")
    if how not in valid:
        raise ValueError(f"how must be one of {sorted(valid)}, got {how!r}.")
    if how == "quantile" and q is None:
        raise ValueError("reduce(how='quantile') requires q, a float in [0, 1].")
    if how != "quantile" and q is not None:
        raise ValueError(f"q is only valid for how='quantile', not how={how!r}.")


def _extremum_values(
    masked: np.typing.NDArray,
    axis: int,
    *,
    largest: bool,
    coords: list | None,
) -> tuple[np.typing.NDArray, float]:
    """Locate the extreme step along `axis`, as a coordinate value or a step position.

    Gaps are left out of the search: they are parked at the losing end so the plain arg-search
    skips them without `nanargmin`'s all-NaN exception, and an element with no valid step at all
    is then blanked -- which the coordinate form declares as NaN and the position form as ``-1``,
    mirroring the raster `argmin` / `idxmin`.

    Args:
        masked: The values, with gaps already turned into NaN.
        axis: The time axis to search along.
        largest: Search for the maximum rather than the minimum.
        coords: The time coordinate values to answer with, or ``None`` for step positions.

    Returns:
        tuple: ``(values, nodata)`` -- one value per element, and the no-data value it declares.
    """
    valid = ~np.isnan(masked)
    anything = valid.any(axis=axis)
    filler = -np.inf if largest else np.inf
    picked = (np.argmax if largest else np.argmin)(
        np.where(valid, masked, filler), axis=axis
    )
    if coords is None:
        return np.where(anything, picked, -1).astype("int64"), -1
    stamps = np.asarray(coords, dtype="float64")
    return np.where(anything, stamps[picked], np.nan), float("nan")


def _bin_by_frequency(coords: list, freq: str) -> tuple[list, list[np.typing.NDArray]]:
    """Bin date-like time coordinates by a pandas offset alias, dropping empty bins.

    Args:
        coords: The time coordinate values, in step order.
        freq: A pandas offset alias naming the bin width (``"1MS"``, ``"YS"``, ...).

    Returns:
        tuple: ``(labels, positions)`` -- each non-empty bin's pandas label and its step
        positions. The label is the bin's start only for a start-anchored alias; an
        end-anchored one (``"ME"``, ``"YE"``, ``"QE"``, ``"W"``) labels the bin by its end.

    Raises:
        ValueError: The coordinates are step positions, or pandas cannot read them as dates.
    """
    # A mesh with no decoded time coordinate falls back to step positions. pandas reads plain
    # numbers as nanoseconds since the epoch, so `[0, 1, 2]` parses happily and every step lands
    # in the same 1970 month -- the axis collapses to one bin instead of raising. Numbers carry
    # no calendar, so refuse them here, as the docstring says.
    numeric = bool(coords) and all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        for value in coords
    )
    if numeric:
        raise ValueError(
            f"reduce(groupby={freq!r}) needs date-like time coordinates; this dataset's are "
            "step positions, which carry no calendar. Pass a per-step label sequence instead."
        )
    try:
        with warnings.catch_warnings():
            # pandas warns about falling back to dateutil per element before it decides the
            # values are not dates at all; the ValueError below is the real answer.
            warnings.simplefilter("ignore", UserWarning)
            index = pd.DatetimeIndex(pd.to_datetime(list(coords)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reduce(groupby={freq!r}) needs date-like time coordinates; this dataset's cannot "
            "be read as dates. Pass a per-step label sequence instead."
        ) from exc

    steps = pd.Series(np.arange(len(index)), index=index)
    labels: list = []
    positions: list[np.typing.NDArray] = []
    for label, group in steps.groupby(pd.Grouper(freq=freq)):
        if len(group) == 0:
            continue
        labels.append(label)
        positions.append(np.asarray(group.to_numpy(), dtype=np.intp))
    return labels, positions


def _topology_info_named(
    mesh_name: str, crs_wkt: str | None = None
) -> MeshTopologyInfo:
    """A minimal topology record carrying just the mesh name.

    The bridges rebuild a mesh from arrays rather than from a file's variables, so there are
    no variable names to record -- but the topology's name still has to survive, because it
    is what :attr:`MeshDataset.mesh_name` answers and what a re-export names its dimensions
    after.

    Args:
        mesh_name: The topology variable's name.
        crs_wkt: The CRS to record, or ``None`` for a record carrying none.

    Returns:
        MeshTopologyInfo: The record, with the 2-D topology dimension and the conventional
        node-coordinate variable names.
    """
    return MeshTopologyInfo(
        mesh_name=mesh_name,
        topology_dimension=2,
        node_x_var=f"{mesh_name}_node_x",
        node_y_var=f"{mesh_name}_node_y",
        crs_wkt=crs_wkt,
    )


def _group_positions(
    coords: list, groupby: Any
) -> tuple[list, list[np.typing.NDArray]]:
    """The time-step positions of each group, and each group's label.

    Two spellings, matching the raster grouped reduce: a **sequence of per-step labels** the
    same length as the time axis (grouped in first-appearance order, so the caller controls the
    order), or a **pandas offset alias** (``"1MS"``, ``"YS"``, …) binned against date-like
    coordinates. Empty bins are dropped, so a sparse year leaves no all-gap step behind.

    Args:
        coords: The time coordinate values, in step order.
        groupby: A per-step label sequence, or a pandas offset alias string.

    Returns:
        tuple: ``(labels, positions)`` — one label and one array of step positions per group.

    Raises:
        ValueError: A label sequence whose length is not the time length, or an offset alias
            asked of coordinates pandas cannot read as dates.
    """
    if isinstance(groupby, str):
        labels, positions = _bin_by_frequency(coords, groupby)
    else:
        given = list(groupby)
        if len(given) != len(coords):
            raise ValueError(
                f"reduce(groupby=...) needs one label per time step: got {len(given)} "
                f"label(s) for {len(coords)} step(s)."
            )
        buckets: dict = {}
        labels = []
        for position, label in enumerate(given):
            if label not in buckets:
                buckets[label] = []
                labels.append(label)
            buckets[label].append(position)
        positions = [np.asarray(buckets[label], dtype=np.intp) for label in labels]
    return labels, positions


def _resolve_time_positions(
    coords: list,
    wanted: list,
    method: str | None,
    tolerance: float | None,
) -> list[int]:
    """The step positions ``wanted`` names in ``coords``, exactly or by nearest.

    Exact matching uses ``list.index`` so a date-like or string coordinate works unchanged.
    ``"nearest"`` needs numeric coordinates, since snapping is a distance: it takes the closest
    coordinate and, when ``tolerance`` is set, refuses a snap further than that — raising
    ``KeyError`` for the breach, as the raster ``sel`` does, so a caller can tell "no coordinate
    is close enough" apart from "that value is not a coordinate at all" (``ValueError``).

    Args:
        coords: The time coordinate values, in step order.
        wanted: The requested coordinate values.
        method: ``None`` for exact, ``"nearest"`` to snap.
        tolerance: The furthest a snap may travel, or ``None`` for no cap.

    Returns:
        list[int]: One step position per requested value, in order.

    Raises:
        ValueError: A value is absent (exact), or ``"nearest"`` was asked of non-numeric
            coordinates.
        KeyError: A snap would exceed ``tolerance``.
    """
    if method is None:
        try:
            positions = [coords.index(value) for value in wanted]
        except ValueError as exc:
            raise ValueError(
                f"sel: time value(s) {wanted!r} not found in the time coordinate."
            ) from exc
        return positions
    try:
        axis = np.asarray(coords, dtype="float64")
        targets = np.asarray(wanted, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sel: method='nearest' needs numeric time coordinates; this dataset's are not."
        ) from exc
    positions = []
    for target in targets:
        gaps = np.abs(axis - target)
        index = int(np.argmin(gaps))
        if tolerance is not None and float(gaps[index]) > float(tolerance):
            raise KeyError(
                f"sel: no time coordinate within tolerance {tolerance!r} of {target!r}; "
                f"the nearest is {coords[index]!r}, {float(gaps[index])!r} away."
            )
        positions.append(index)
    return positions


def _apply_active_mask(
    var: MeshVariable, values: np.typing.NDArray
) -> np.typing.NDArray:
    """Blank ``values`` to NaN wherever ``var``'s active mask is ``False``.

    The on-demand application of the wet/dry mask (issue #1331): a dry element becomes NaN
    *in addition to* the ``nodata`` blanking, so a consumer that opts in sees dry and missing
    alike as NaN while the two stay distinguishable through :attr:`MeshVariable.active`. The
    mask is element-indexed (its last axis), optionally with a leading time axis; it is
    reshaped to broadcast against the data's element axis and, for a per-step mask, the
    variable's time axis — so a layer axis in between is spanned without the mask carrying one.

    The data's element axis is the last axis, except for a trailing-time variable (``time``
    last, e.g. ``(n_face, time)``), where it is the axis before time — the layout
    :attr:`MeshVariable.time_axis_index` already describes and the rest of the module keys off, so a
    mask applies correctly whichever way the variable stores time.

    Args:
        var: The variable supplying the mask (``var.active`` must not be ``None``).
        values: The already float, ``nodata``-blanked values to mask.

    Returns:
        np.ndarray: A new array (``np.where`` always allocates) with dry elements set to NaN.
    """
    masked = np.asarray(values, dtype="float64")
    mask = np.asarray(var.active, dtype=bool)
    ndim = masked.ndim
    time_axis = var.time_axis_index
    trailing_time = time_axis is not None and time_axis == ndim - 1 and ndim >= 2
    elem_axis = ndim - 2 if trailing_time else ndim - 1
    target = [1] * ndim
    target[elem_axis] = masked.shape[elem_axis]
    reshaped = mask
    if var.active_has_time and time_axis is not None and ndim >= 2:
        target[time_axis] = masked.shape[time_axis]
        if elem_axis < time_axis:
            # The mask is stored (time, element); this variable puts element before time, so
            # transpose it to (element, time) before reshaping into the broadcast target.
            reshaped = mask.T
    return np.where(reshaped.reshape(target), masked, np.nan)


def _coarsened_variable(
    name: str, var: MeshVariable, window: int, how: str, sweeping: bool
) -> MeshVariable | None:
    """One variable block-aggregated along time, or ``None`` when it is too short.

    Lifted out of :meth:`MeshDataset.coarsen` so that method reads as the loop it is: the
    per-variable work is where the skip-or-raise decision and the re-binning live, and keeping
    both inline took the method past the cognitive-complexity limit.

    Args:
        name: The variable's name, for messages.
        var: The variable to bin.
        window: Steps per block.
        how: Reducer name, one of :data:`_COARSEN_HOWS`.
        sweeping: True when every variable is being binned, which is what decides whether a
            variable with fewer steps than one window is skipped or refused.

    Returns:
        MeshVariable | None: The binned variable, or ``None`` when it has fewer steps than one
        window and the caller is sweeping — the caller then passes it through untouched.

    Raises:
        ValueError: The variable has no loaded data, or it is too short and was named
            explicitly rather than swept.
    """
    data = var.data
    if data is None:
        raise ValueError(f"Variable {name!r} has no loaded data to coarsen.")
    coord = _variable_time_coord(var)
    # The blocks==0 decision runs BEFORE window_positions: a sweep must skip a too-short
    # variable, where window_positions(boundary="trim") would raise instead, and the caller's
    # own "no variable has as many as N steps" aggregate error must be the one that fires.
    blocks = len(coord) // window
    if blocks == 0:
        if sweeping:
            # A sweep skips what it cannot bin, as `convert_units` skips the variables with no
            # units to convert from: one short variable should not lose the whole call. The
            # caller checks whether ANY variable was binned, since a sweep that did nothing
            # must not look like it succeeded.
            return None
        raise ValueError(
            f"coarsen: {name!r} has {len(coord)} step(s), fewer than one window of "
            f"{window}; there would be nothing to report."
        )
    axis = cast("int", var.time_axis_index)
    # One coarsen implementation (D6): the window positions and the grouped reduction are the
    # shared raster/mesh primitives. `boundary="trim"` keeps the mesh's historical silent-drop
    # of a trailing remainder. The sentinel is masked to NaN first and `ndv=None` is passed, so
    # a declared gap is skipped (not averaged in) and an all-missing block stays NaN rather than
    # declaring the sentinel — the positional binner's long-standing convention.
    _resized, positions = window_positions("time", len(coord), window, "trim")
    labels = [coord[int(group[0])] for group in positions]
    masked = gaps_as_nan(np.asarray(data), var.nodata)
    reduced = reduce_groups(masked, axis, positions, how, True, None)
    # The axis is re-binned, so a per-step mask no longer lines up with it.
    moved = var.with_data(reduced, active=None if var.active_has_time else _KEEP_ACTIVE)
    moved.attributes = {**var.attributes, "time_values": labels}
    return moved


def _without_time_metadata(attributes: dict) -> dict:
    """``attributes`` with the time-coordinate keys removed.

    A time-collapsing operation leaves a result that is no longer indexed by time, so its
    stamps, CF units and calendar describe nothing on the now-static variable and are dropped.

    Args:
        attributes: A variable's attributes.

    Returns:
        dict: A copy without ``time_values``, ``time_units`` or ``calendar``.
    """
    return {
        key: value
        for key, value in attributes.items()
        if key not in ("time_values", TIME_UNITS_ATTR, TIME_CALENDAR_ATTR)
    }


def _to_cftime(stamp: Any, calendar: str = "standard") -> Any:
    """``stamp`` as a calendar-aware ``cftime`` object.

    One shared conversion for the two places that need calendar arithmetic: a decoded axis
    carries ``datetime64`` for the Gregorian family and ``cftime`` objects for model calendars,
    and only the latter can be asked calendar questions.

    Args:
        stamp: A decoded stamp, either flavour.
        calendar: Calendar to attach when converting a Gregorian-family stamp.

    Returns:
        A ``cftime.datetime`` on ``calendar``, or ``stamp`` unchanged when it already carries
        one.
    """
    if hasattr(stamp, "calendar"):
        result = stamp
    else:
        # `.item()` on a datetime64[ns] hands back an int (nanoseconds since the epoch), not a
        # datetime; going through microsecond precision - also cftime's floor - yields the
        # real datetime whose fields are needed.
        moment = np.datetime64(stamp, "us").item()
        result = cftime.datetime(
            moment.year,
            moment.month,
            moment.day,
            moment.hour,
            moment.minute,
            moment.second,
            moment.microsecond,
            calendar=calendar,
        )
    return result


def _is_dated(stamps: Any) -> bool:
    """True when ``stamps`` are real dates rather than positional indices.

    What tells a decoded CF axis from the positional fallback, which is the distinction
    :func:`_interp_scale` needs: one counts elapsed time and the other counts steps.

    Args:
        stamps: A variable's time coordinate, or ``None``.

    Returns:
        bool: True when the first stamp is a ``cftime`` object, a ``datetime64`` or a
        ``datetime``; False for numbers, an empty coordinate or ``None``.
    """
    result = False
    if stamps is not None and len(stamps) > 0:
        first = stamps[0]
        result = hasattr(first, "calendar") or isinstance(
            first, (np.datetime64, datetime.datetime, datetime.date)
        )
    return result


def _interp_scale(source: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """``source`` and ``target`` on one numeric scale that spans calendars.

    Interpolating between two dated axes needs a scale that means the same thing on both. For
    two *dated* axes that scale is the **decimal year** — the year plus the fraction of it
    elapsed — which is calendar-independent: a 360-day and a 365-day year both run from ``Y.0``
    to ``Y+1.0``. Measuring the target from the source's first stamp instead cannot work across
    calendars at all, because ``cftime`` refuses to subtract two dates on different calendars.

    Two undated axes are positional on both sides, which is coherent — a step index means the
    same thing in each — so they are used as plain numbers. A **mixed** pair is refused: one
    side counting seconds and the other counting steps has no common scale, and answering
    anyway was wrong by whatever the step interval happens to be.

    Args:
        source: The coordinate the values sit on.
        target: The coordinate to interpolate onto.

    Returns:
        tuple: ``(source_scale, target_scale)`` as float arrays.

    Raises:
        ValueError: One axis is dated and the other positional.
    """
    source_dated, target_dated = _is_dated(source), _is_dated(target)
    if source_dated != target_dated:
        # One side counts seconds from its first stamp and the other counts steps, and there
        # is no defensible mapping between them: feeding both to the interpolator silently
        # answered an hourly source at "step 1" as though that were one second, out by 3600.
        dated, positional = (
            ("this dataset", "`other`") if source_dated else ("`other`", "this dataset")
        )
        raise ValueError(
            f"interp_like: {dated} has a dated time axis while {positional} is positional "
            f"(no CF time units), and a step index cannot be placed on a calendar. Give both "
            f"sides real stamps, or use interp_time() to resample onto evenly spaced steps."
        )
    if source_dated:
        return (
            _decimal_years(
                np.asarray([_to_cftime(stamp) for stamp in source], dtype=object)
            ),
            _decimal_years(
                np.asarray([_to_cftime(stamp) for stamp in target], dtype=object)
            ),
        )
    return (
        _timeseries.numeric_coordinate(source),
        _timeseries.numeric_coordinate(target),
    )


def _variable_time_coord(var: MeshVariable) -> list:
    """The time coordinate of one variable, falling back to its own positions.

    The along-time members used to take the coordinate from the dataset, which is the **first**
    temporal variable's. A mesh file may legitimately hold temporal variables of different
    lengths — a per-timestep map variable beside a per-output-interval one — and applying one
    variable's coordinate to another either failed with a raw numpy message naming nothing or,
    where the lengths happened to agree, silently transformed against the wrong stamps.

    A variable whose stored stamps do not match its own step count is treated as unstamped,
    since a mismatched coordinate is no more trustworthy than a missing one.

    Args:
        var: The variable whose coordinate to read.

    Returns:
        list: The variable's own stamps, or ``range(n_time_steps)`` when it carries none.
    """
    stamps = var.attributes.get("time_values")
    if stamps is not None and len(stamps) == var.n_time_steps:
        result = list(stamps)
    else:
        if stamps is not None:
            # Falling back is not a missing answer but a differently scaled one: a derivative
            # comes out per step instead of per second, which is the same number of orders of
            # magnitude wrong as the interval. Silence would make that undetectable.
            warnings.warn(
                f"{var.name!r} stores {len(stamps)} time stamp(s) for "
                f"{var.n_time_steps} step(s), so they cannot be trusted as the coordinate. "
                f"Falling back to step positions, which means rates come out per step rather "
                f"than per second.",
                UserWarning,
                stacklevel=3,
            )
        result = list(range(var.n_time_steps))
    return result


def _fitted_dimensions(
    var: MeshVariable, ndim: int, axis: int, axis_name: str
) -> tuple[str, ...]:
    """Dimension names for a fit result, with the time axis replaced by ``axis_name``.

    Names are synthesised when the source variable carries none, which an array-built variable
    does not. That matters for correctness rather than tidiness: :attr:`MeshVariable.time_axis_index`
    falls back to "the leading axis of a >1-D array" when there are no names, so a
    ``(deg + 1, n_face)`` fit would otherwise report ``has_time == True`` and be reduced over
    its coefficient axis as though that were time.

    Args:
        var: The variable being fitted.
        ndim: Dimension count of the fitted array.
        axis: The axis the fit ran along.
        axis_name: Name for the axis that replaces it.

    Returns:
        tuple: One name per axis, none of which matches the time-name patterns.
    """
    if var.dimensions and len(var.dimensions) == ndim:
        names = [
            axis_name if position == axis else dim
            for position, dim in enumerate(var.dimensions)
        ]
    else:
        names = []
        for position in range(ndim):
            if position == axis:
                names.append(axis_name)
            elif position == ndim - 1:
                names.append(var.location)
            else:
                # A middle axis of a layered variable is the layer axis, not an element one -
                # naming it for the location would claim it holds faces. `dim{n}` says only
                # "an axis", which is all that is known here, and matters because none of
                # these names may look like a time axis (see the note above).
                names.append(f"dim{position}")
    return tuple(names)


def _stamp_order(stamp: Any) -> tuple:
    """A sortable tuple for a time stamp, whatever flavour it is.

    The decoded stamps are ``datetime64`` for the Gregorian family and ``cftime`` objects for
    model calendars, and the two do not compare with each other — nor do two cftime objects on
    different calendars. Comparing the calendar fields instead keeps a mixed set of files
    sortable, which is the point: the sort exists to put a run in order, not to assert that
    every file uses the same calendar.

    Args:
        stamp: A decoded time stamp.

    Returns:
        tuple: ``(year, month, day, hour, minute, second, microsecond)``.
    """
    moment = stamp if hasattr(stamp, "year") else np.datetime64(stamp, "us").item()
    return (
        moment.year,
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        moment.second,
        moment.microsecond,
    )


def _static_from(
    var: MeshVariable,
    data: np.typing.NDArray,
    time_axis: int,
    nodata: float | None = None,
    active: np.ndarray | None = _KEEP_ACTIVE,
) -> MeshVariable:
    """A static (time-collapsed) copy of ``var`` carrying ``data``.

    Drops the time dimension (at ``time_axis``) from the variable's ``dimensions`` so the
    result reports ``has_time == False``, and takes its shape from ``data``. Used by
    :meth:`MeshDataset.reduce` when a reduction collapses the time axis.

    Args:
        var: The source temporal variable.
        data: The reduced values, with the time axis already gone.
        time_axis: The index of the time dimension in ``var.dimensions`` that was collapsed.
        nodata: The no-data value the result declares. Defaults to ``None``, which keeps the
            source's — a reduction that introduces its own sentinel (``all`` / ``any`` emit
            the ``255`` flag) passes it so the result does not hide that value as data.
        active: The active (wet/dry) mask the result carries. Left unset, the source's mask is
            carried *implicitly* and validated against the collapsed shape by
            :func:`_consistent_active` — so a static mask survives a time reduction while a
            per-step mask (which no longer aligns once the time axis is gone) is dropped. A
            single-step collapse passes the step's slice of the mask explicitly so it stays
            aligned; pass ``None`` to force-drop it.

    Returns:
        MeshVariable: The static result.
    """
    new_dims = (
        tuple(d for i, d in enumerate(var.dimensions) if i != time_axis)
        if var.dimensions
        else ()
    )
    # Copy the attributes (never alias — the same invariant MeshVariable.with_data keeps),
    # and drop the whole time coordinate (`time_values`, CF units, calendar): the time axis
    # is gone, so they describe nothing on the now-static variable.
    attributes = _without_time_metadata(var.attributes)
    return MeshVariable(
        name=var.name,
        location=var.location,
        mesh_name=var.mesh_name,
        shape=data.shape,
        attributes=attributes,
        nodata=var.nodata if nodata is None else nodata,
        units=var.units,
        standard_name=var.standard_name,
        dimensions=new_dims,
        # Implicit carry validated against the collapsed shape; an explicit mask trusted.
        active=(
            _consistent_active(var.active, data) if active is _KEEP_ACTIVE else active
        ),
        _data=data,
    )


def _collapse_variable_to_step(var: MeshVariable, index: int) -> MeshVariable:
    """One time step of a temporal variable, as a static (time-collapsed) variable.

    The shared body of the two single-step collapses — :meth:`MeshDataset.sel_time` and
    :meth:`MeshDataset._collapse_to_step` (reached by ``isel(time=index)``) — so they cannot
    drift. A leading-time variable reads only the requested slab from disk (the lazy
    single-step read of #982); a variable whose time axis is not leading is taken with
    :func:`numpy.take` on its own ``time_index``. The result drops the time dimension via
    :func:`_static_from`, which also strips the now-meaningless time coordinate.

    Args:
        var: The temporal variable to collapse (caller has checked ``var.has_time``).
        index: The time position to take (0-based; a negative index counts from the end).

    Returns:
        MeshVariable: The static variable holding that step.

    Raises:
        ValueError: The variable has neither eager data nor a loader to read from.
        IndexError: ``index`` is out of range for the time axis.
    """
    if not var.has_data_source:
        raise ValueError(f"Variable {var.name!r} has no loaded data to select.")
    axis = cast("int", var.time_axis_index)
    if axis == 0:
        # Leading time: the windowed read pulls only this slab from disk (#982).
        # `_read_time_window` is the raw-slab primitive `sel_time` wrapped; used directly so
        # `sel_time` has no internal caller and can be retyped/deprecated on its own.
        step = np.asarray(var._read_time_window(index, None))
    else:
        step = np.take(np.asarray(var.data), index, axis=axis)
    return _static_from(var, step, axis, active=var._time_sliced_active(index, None))


def _concat_temporal_variable(
    name: str, var: MeshVariable, parts: list[MeshDataset]
) -> MeshVariable:
    """Concatenate one temporal variable's data across ``parts`` along its time axis.

    The per-variable body of :meth:`MeshDataset.concat`. Joins each part's array along the
    variable's time axis and its ``time_values`` coordinate when every part has one; if any
    part lacks it, the result carries no (stale) coordinate.

    Args:
        name: The variable name (present in every part — concat validated that).
        var: ``self``'s copy of the variable, whose time axis and attributes seed the result.
        parts: The datasets to join, ``self`` first.

    Returns:
        MeshVariable: The concatenated temporal variable.

    Raises:
        ValueError: A part's same-named variable is static, or has no loaded data.
    """
    axis = cast("int", var.time_axis_index)
    arrays: list[np.typing.NDArray] = []
    times: list[Any] = []
    has_times = True
    for part in parts:
        pv = part._data_variables[name]
        if not pv.has_time:
            # `var` is temporal, so a static same-named variable in another part cannot join
            # along time. Say so with a domain message, not a raw np.concatenate shape error.
            raise ValueError(
                f"concat: variable {name!r} is temporal in one dataset but static in "
                "another; they cannot be joined along time."
            )
        data = pv.data
        if data is None:
            raise ValueError(f"Variable {name!r} has no loaded data to concat.")
        arrays.append(np.asarray(data))
        part_times = pv.attributes.get("time_values")
        if part_times is None:
            has_times = False
        else:
            times.extend(list(part_times))
    new_var = var.with_data(np.concatenate(arrays, axis=axis))
    if has_times:
        new_var.attributes = {**var.attributes, "time_values": times}
    elif "time_values" in new_var.attributes:
        # Some part had no time coordinate, so the joined axis has none; drop the stale,
        # too-short `time_values` carried over by with_data.
        new_var.attributes = {
            key: value
            for key, value in new_var.attributes.items()
            if key != "time_values"
        }
    return new_var


def _make_variable_loader(path: str, var_name: str):
    """Build a zero-arg loader that reads one variable's array on first access.

    The store opened in :meth:`MeshDataset.read_file` is closed before any
    :class:`MeshVariable` data is touched, so a lazy loader cannot capture the live
    root group — it re-opens ``path`` and reads ``var_name`` on demand instead. This
    keeps ``read_file`` metadata-only: variables the caller never touches are never read.

    Args:
        path: File path to re-open for the read.
        var_name: Name of the MDArray to read.

    Returns:
        Callable[[], np.ndarray | None]: A loader returning the variable's array (or
        ``None`` when it has no readable values).
    """

    def _load() -> np.typing.NDArray | None:
        # The whole-array variant of the shared mesh reader: re-open, resolve, read the
        # full array (no window). `ReadAsArray` already returns a fresh, numpy-owned array,
        # so no extra `.copy()` is needed (#982).
        return read_mesh_variable(path, var_name, context="lazy variable read")

    return _load


def _attach_active_masks(rg: gdal.Group, variables: dict[str, MeshVariable]) -> None:
    """Attach each variable's active (wet/dry) mask from its ``<var>_active`` companion.

    The read counterpart of :func:`pyramids.netcdf.ugrid.io._write_active_mask` (issue #1331).
    For every data variable it tries to open ``<name>_active``; when that array exists and
    carries the active-mask role, its values are read eagerly (the mask is small and boolean)
    and set on :attr:`MeshVariable.active`. A file with no companions leaves every ``active``
    at ``None``, so the read is unchanged for a mesh that carries no masks. Mutates
    ``variables`` in place.

    Args:
        rg: GDAL root group.
        variables: The data variables already built, keyed by name.
    """
    for name, var in variables.items():
        companion = open_mdarray(rg, f"{name}{ACTIVE_MASK_SUFFIX}")
        if companion is None:
            continue
        attrs = read_cf_attributes(companion)
        if attrs.get(ACTIVE_MASK_ATTR) != ACTIVE_MASK_ROLE:
            continue
        values = companion.ReadAsArray()
        if values is not None:
            var.active = np.asarray(values).astype(bool)


#: Attribute keys the decoded CF time coordinate is stored under on each temporal variable,
#: beside the ``time_values`` stamps themselves. Kept on the variable rather than threaded
#: through the dataset constructor because that is already where ``time_values`` lives, and a
#: derivation's ``with_data`` copies the attributes for free (issue #1381).
TIME_UNITS_ATTR = "time_units"
TIME_CALENDAR_ATTR = "calendar"

DEFAULT_TIME_DIM = "time"
"""The name the writer registers the time dimension under; see `_write_time_coordinate`."""


def _require_one_time_length(variables: dict[str, MeshVariable]) -> None:
    """Refuse a dataset whose temporal variables disagree about the number of steps.

    The writer registers a single ``time`` dimension, sized from the first temporal variable,
    so a ragged dataset cannot be written at all. Without this check it failed inside GDAL
    instead — ``arrayStartIdx[0] + (count[0]-1) * arrayStep[0] >= 2`` — which says nothing
    about which variables disagree or why.

    Such a dataset is easy to produce on purpose: coarsening or interpolating one named
    variable re-bins its axis and leaves the others alone. Reading one is legitimate (a mesh
    file may hold a per-timestep variable beside a per-interval one, which the along-time
    members handle per variable); it is only writing that cannot represent it.

    Args:
        variables: The data variables about to be written.

    Raises:
        ValueError: Two or more temporal variables report different step counts.
    """
    lengths: dict[str, int] = {
        name: var.n_time_steps for name, var in variables.items() if var.has_time
    }
    distinct = sorted(set(lengths.values()))
    if len(distinct) > 1:
        shown = ", ".join(
            f"{name!r}={steps}" for name, steps in sorted(lengths.items())
        )
        raise ValueError(
            f"to_file(): the temporal variables do not share the same number of time steps "
            f"({shown}), and the file format has one time dimension to write them against. "
            f"Apply the same along-time operation to every variable - call it without "
            f"variable_name - or write the variables to separate files."
        )


def _write_time_coordinate(
    rg: gdal.Group, dims: dict[str, Any], variables: dict[str, MeshVariable]
) -> None:
    """Write the CF ``time`` coordinate array beside its dimension (issue #1381).

    The writer created the ``time`` **dimension** but no ``time`` **array**, so a
    pyramids-written UGRID file carried no stamps at all and a read -> write -> read round trip
    silently degraded real dates to positional indices. This writes the coordinate whenever a
    temporal variable carries the units the reader decoded, re-encoding the stamps with
    :func:`pyramids.netcdf.utils.encode_cf_time` so the numbers on disk match the declared
    ``units``.

    Best-effort in the same way the reader is: a dataset whose time coordinate is positional
    (no ``time_units``) writes no coordinate, exactly as before. Re-encoding failure is skipped
    rather than raised, so writing a mesh never fails over its time metadata.

    Args:
        rg: GDAL root group.
        dims: Dimension registry; the time dimension was added when the data was written.
        variables: The dataset's data variables.
    """
    # `_time_coordinate_name` decides only WHETHER there is a time axis. The coordinate is
    # written as "time" because that is the only name the writer registers the dimension
    # under (see `to_file`); a variable whose own axis is named otherwise - `nmesh2d_time`,
    # say - is written against the "time" dimension, so its coordinate goes there too. The
    # name is therefore not round-tripped, which is a limitation of the data writer rather
    # than of this function.
    name = DEFAULT_TIME_DIM
    if _time_coordinate_name(variables) is None or name not in dims:
        # A dataset that declares stamps AND CF units plainly wanted a coordinate written, so
        # failing to place one is reported rather than returned from quietly - the three
        # warnings below all sit past this point, which left the most likely way to lose the
        # stamps as the one path that said nothing (round 2 review H4).
        if any(
            var.has_time
            and var.attributes.get("time_values") is not None
            and var.attributes.get(TIME_UNITS_ATTR)
            for var in variables.values()
        ):
            warnings.warn(
                f"to_file(): the dataset declares CF time units and stamps, but no "
                f"{name!r} dimension was registered to hang the coordinate on. The mesh and "
                f"its data are written, but the time axis will read back as positional "
                f"indices.",
                UserWarning,
                # `_write_time_coordinate` <- `to_file` <- the caller.
                stacklevel=3,
            )
        return
    source = next(
        (
            var
            for var in variables.values()
            if var.has_time
            and var.attributes.get("time_values") is not None
            and var.attributes.get(TIME_UNITS_ATTR)
        ),
        None,
    )
    if source is None:
        return
    units = str(source.attributes[TIME_UNITS_ATTR])
    calendar = str(source.attributes.get(TIME_CALENDAR_ATTR) or "standard")
    try:
        encoded = np.asarray(
            [
                encode_cf_time(stamp, units, calendar)
                for stamp in source.attributes["time_values"]
            ],
            dtype="float64",
        )
    except (ValueError, TypeError, OverflowError) as error:
        # A stamp that will not re-encode leaves the axis bare rather than failing the write -
        # but it says so, because the degradation this function exists to prevent is exactly
        # "the dates came back as positional indices and nobody knows why".
        warnings.warn(
            f"to_file(): could not write the time coordinate - re-encoding the stamps "
            f"against {units!r} failed ({error}). The mesh and its data are written, but the "
            f"time axis will read back as positional indices.",
            UserWarning,
            # `_write_time_coordinate` <- `to_file` <- the caller.
            stacklevel=3,
        )
        return
    # netCDF reserves the dimension's own name for its coordinate variable, so creating the
    # coordinate array can collide with one GDAL already materialised for the dimension
    # ("String match to name in use"). Reuse it when it is there, create it when it is not.
    array = open_mdarray(rg, name)
    if array is None:
        try:
            array = rg.CreateMDArray(
                name,
                [dims[name]],
                gdal.ExtendedDataType.Create(gdal.GDT_Float64),
            )
        except RuntimeError as error:
            warnings.warn(
                f"to_file(): could not create the 'time' coordinate array ({error}). The mesh "
                f"and its data are written, but the time axis will read back as positional "
                f"indices.",
                UserWarning,
                stacklevel=2,
            )
            return
    write_attributes_to_md_array(
        array,
        {"standard_name": "time", "units": units, TIME_CALENDAR_ATTR: calendar},
    )
    try:
        array.Write(encoded)
    except RuntimeError as error:
        warnings.warn(
            f"to_file(): the driver refused the time coordinate write ({error}). The mesh and "
            f"its data are written, but the time axis will read back as positional indices.",
            UserWarning,
            # `_write_time_coordinate` <- `to_file` <- the caller.
            stacklevel=3,
        )
        return


def _time_coordinate_name(variables: dict[str, MeshVariable]) -> str | None:
    """The name of the time dimension the temporal variables share, or ``None``.

    Taken from the first temporal variable's own ``dimensions`` at its
    :attr:`MeshVariable.time_axis_index`, so it follows whatever the file calls the axis rather than
    assuming ``"time"``. A CF coordinate variable is conventionally named after its dimension,
    which is what makes this usable as the array name to open.

    A variable that carries **no** dimension names — which is every variable built through
    :meth:`MeshDataset.from_arrays` — still reports ``has_time`` and a ``time_index``, so the
    axis exists and only its name is unknown. Those fall back to ``"time"``, the name the
    writer registers the dimension under. Returning ``None`` for them meant the whole CF
    coordinate write was skipped for any dataset a caller built in memory, silently.

    Args:
        variables: The data variables already built.

    Returns:
        str | None: The dimension name, or ``None`` when nothing is temporal at all.
    """
    result = None
    for var in variables.values():
        axis = var.time_axis_index
        if var.has_time and axis is not None:
            result = (
                var.dimensions[axis] if axis < len(var.dimensions) else DEFAULT_TIME_DIM
            )
            break
    return result


def _decode_time_coordinate(rg: gdal.Group, variables: dict[str, MeshVariable]) -> None:
    """Decode the file's CF time coordinate onto every temporal variable (issue #1381).

    The UGRID reader collects data variables by their ``location`` attribute, and a CF time
    coordinate carries none — so the time axis was read as a bare dimension and
    :attr:`MeshDataset.time_values` fell back to ``range(n_time_steps)``, positional indices,
    even for a file that states its stamps. Real D-Flow FM output does state them
    (``units = "seconds since 2017-01-01 00:00:00"``), and the decoding primitives already
    exist, so this reads the coordinate and stores the decoded stamps, units and calendar on
    each temporal variable's attributes.

    Deliberately best-effort: a file with no such coordinate, or one whose ``units`` is not a CF
    time unit, is left exactly as it read before. Decoding must not become a new way for
    :meth:`MeshDataset.read_file` to fail, so a malformed coordinate is skipped rather than
    raised on. Mutates ``variables`` in place.

    Args:
        rg: GDAL root group.
        variables: The data variables already built, keyed by name.
    """
    name = _time_coordinate_name(variables)
    if name is None:
        return
    array = open_mdarray(rg, name)
    if array is None:
        return
    attrs = read_cf_attributes(array)
    # GDAL exposes a CF `units` both as an attribute and through `GetUnit()`; prefer the
    # attribute and fall back, since which one is populated varies by driver.
    raw_units = attrs.get("units") or array.GetUnit() or None
    # `read_cf_attributes` types every attribute as the broad GDAL value union; a CF time unit
    # is a string, so narrow it before handing it to the decoder.
    units = str(raw_units) if raw_units is not None else None
    if not is_cf_time_units(units):
        return
    calendar = str(attrs.get(TIME_CALENDAR_ATTR) or "standard")
    try:
        values = array.ReadAsArray()
        stamps = decode_cf_time(np.asarray(values), units, calendar)
    except (RuntimeError, ValueError, TypeError, OverflowError):
        # A coordinate GDAL cannot read, or a units/calendar pair cftime rejects, leaves the
        # positional fallback in place rather than failing the whole read.
        return
    for var in variables.values():
        if not var.has_time or var.n_time_steps != len(stamps):
            continue
        var.attributes = {
            **var.attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: str(units),
            TIME_CALENDAR_ATTR: calendar,
        }


def _read_data_variables(
    rg: gdal.Group,
    topo_info: MeshTopologyInfo,
    path: str,
) -> dict[str, MeshVariable]:
    """Read every mesh data variable's metadata, deferring the array read.

    Creates a :class:`MeshVariable` per variable that references the mesh topology.
    Only metadata (attributes, shape, dtype, nodata, units, standard name) is read
    eagerly; the array itself loads lazily on first ``.data`` access via a re-opening
    loader, so ``read_file`` does not pull every variable into memory.

    Args:
        rg: GDAL root group (used for metadata only).
        topo_info: Parsed topology info with data variable names and locations.
        path: File path threaded into each variable's lazy loader.

    Returns:
        Dictionary mapping variable name to MeshVariable.
    """
    variables: dict[str, MeshVariable] = {}

    for var_name, location in topo_info.data_variables.items():
        md_arr = open_mdarray(rg, var_name)
        if md_arr is None:
            continue
        attrs = read_cf_attributes(md_arr)
        dims = md_arr.GetDimensions()
        shape = tuple(d.GetSize() for d in dims) if dims else ()
        dim_names = tuple(d.GetName() for d in dims) if dims else ()

        # Through the dedicated accessor, not `attrs`: GDAL's netCDF driver does not list
        # `_FillValue` among an array's attributes, so the attribute lookup found nothing even
        # for a file that declares one, and every sentinel read back as ordinary data.
        nodata = no_data_value_of(md_arr)
        units = cast("str | None", attrs.get("units"))
        standard_name = cast("str | None", attrs.get("standard_name"))
        try:
            dtype = np.dtype(_dtype_to_str(md_arr.GetDataType()))
        except (RuntimeError, TypeError):
            dtype = None

        variables[var_name] = MeshVariable(
            name=var_name,
            location=location,
            mesh_name=topo_info.mesh_name,
            shape=shape,
            attributes=attrs,
            nodata=nodata,
            units=units,
            standard_name=standard_name,
            dimensions=dim_names,
            _loader=_make_variable_loader(path, var_name),
            _dtype=dtype,
            _source_path=path,
        )

    return variables


#: The mesh class a driver is required to produce, so `_read_through_driver` types its
#: result precisely instead of widening every caller to `Any`. Bound rather than
#: unconstrained: the only two shapes a driver may return are a surface mesh and a 1D
#: network, and each entry point passes its own, so the bound documents the pair.
_MeshObject = TypeVar("_MeshObject", bound="MeshDataset | Mesh1d")


def _unknown_driver_message(driver: str) -> str:
    """Build the error text for a driver name that is not installed.

    One builder rather than the three near-duplicates the dispatch paths each had, which gave
    the same mistake a different message depending on the entry point.

    Args:
        driver: The name the caller asked for.

    Returns:
        A message naming what is installed and how to install more.
    """
    installed = mesh_drivers()
    available = ", ".join(sorted(installed)) or "none installed"
    result = (
        f"No mesh driver named {driver!r} is installed.\n"
        f"  Built-in reader: 'ugrid' (UGRID NetCDF)\n"
        f"  Plugin drivers : {available}\n"
        f"  Install a mesh driver plugin to read 2DM, Selafin, DFSU, XMDF, PLY and more."
    )
    return result


def _sniff(driver: MeshDriver, target: str) -> bool:
    """Ask a driver whether it recognises a file, treating a raising sniff as a veto.

    `can_open` is documented as never raising, but a plugin can still break it — and an
    exception there propagated out of `read_file`, so one broken driver broke reads of every
    other format sharing its suffix.

    Args:
        driver: The driver to ask.
        target: The path to offer it.

    Returns:
        True when the driver claims the file. False when it declines, or when its sniff
        raised — in which case a warning names the driver.
    """
    try:
        claimed = bool(driver.can_open(target))
    except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
        warnings.warn(
            f"The {driver.name!r} driver's can_open raised for {target!r} and was treated as "
            f"declining the file: {error!r}",
            RuntimeWarning,
            stacklevel=4,
        )
        claimed = False
    return claimed


def _require_existing_file(path: str | Path) -> None:
    """Raise if a path cannot be opened at all, before any driver is consulted.

    `read_file` has always documented `FileNotFoundError`, but the check lived inside the
    NetCDF reader, so a typo'd filename on a driver path was reported by the dispatch wrapper
    as an unsupported format -- sending the user to install a plugin they already had.

    A directory is deliberately *not* rejected here: Esri TIN is a directory format, so a
    driver must be allowed to claim one. An unclaimed directory is reported as a directory by
    the fallback path instead.

    Args:
        path: The target to check. A GDAL virtual-filesystem name is left to GDAL.

    Raises:
        FileNotFoundError: The path does not exist on disk.
    """
    target = str(path)
    if not target.startswith("/vsi") and not Path(target).exists():
        raise FileNotFoundError(f"File not found: {target}")


def _open_multidim(opened_name: str, reported: str, wrap_unreadable: bool) -> Any:
    """Open a file with GDAL's multidimensional driver, reporting failure usefully.

    Args:
        opened_name: The name handed to GDAL — an OS path or a virtual-filesystem name.
        reported: The path to name in an error, as the caller passed it.
        wrap_unreadable: True when this call is the fallback for a suffix no driver claimed.
            GDAL's "not recognized as a supported file format" would then send the caller
            looking for a corrupt file rather than a missing plugin, so the failure is
            re-reported in terms of the drivers available.

    Returns:
        The opened GDAL dataset.

    Raises:
        IsADirectoryError: The path is a directory that no driver claimed.
        UnsupportedMeshFormatError: GDAL refused a file that reached here as a fallback.
        RuntimeError: GDAL refused a file that reached here by its own NetCDF suffix.
        ValueError: GDAL answered ``None`` rather than raising, which needs
            ``gdal.DontUseExceptions()`` to have been called process-wide.
    """
    try:
        ds = gdal.OpenEx(
            opened_name,
            gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR,
        )
    except RuntimeError as error:
        if not wrap_unreadable:
            raise
        is_virtual = opened_name.startswith("/vsi")
        if not is_virtual and Path(opened_name).is_dir():
            # Reporting a directory as a missing plugin sends the user to install something
            # for what is really a wrong-path mistake. A directory-based format (Esri TIN) is
            # claimed by its driver earlier and never reaches here.
            raise IsADirectoryError(
                f"Not a mesh file: {opened_name} is a directory."
            ) from error
        suffix = "" if is_virtual else Path(opened_name).suffix
        raise UnsupportedMeshFormatError(
            unsupported_format_message(reported, suffix)
        ) from error
    if ds is None:  # pragma: no cover - needs gdal.DontUseExceptions() process-wide
        raise ValueError(f"GDAL cannot open file: {reported}")
    return ds


def _named_read_driver(driver: str) -> tuple[MeshDriver | None, bool]:
    """Resolve an explicitly requested driver name.

    Args:
        driver: The name the caller asked for.

    Returns:
        ``(driver, native)``. ``(None, True)`` for a reserved name, which means the built-in
        reader; otherwise the named plugin and ``False``.

    Raises:
        DriverNotExistError: The name is neither reserved nor installed.
    """
    if driver.strip().lower() in RESERVED_NAMES:
        # `ugrid` identifies the built-in reader -- that is why RESERVED_NAMES stops a plugin
        # taking it. Asking for it by name used to raise DriverNotExistError, telling the user
        # to install a plugin to reach a reader that ships with pyramids.
        result: tuple[MeshDriver | None, bool] = (None, True)
    else:
        installed = mesh_drivers()
        if driver not in installed:
            raise DriverNotExistError(_unknown_driver_message(driver))
        result = (installed[driver], False)
    return result


def _first_claiming_driver(suffix: str, target: str) -> MeshDriver | None:
    """Return the first registered driver claiming ``suffix`` whose ``can_open`` accepts ``target``.

    ``can_open`` is what decides between drivers sharing a suffix -- ``.dat`` and ``.mesh`` are
    each claimed by more than one format. Shared by :func:`_detected_read_driver` (detection) and
    :func:`_netcdf_fallback_driver` (the NetCDF fallback) so the two routes sniff identically and
    cannot drift.

    Args:
        suffix: The lowercased file suffix to match.
        target: The path handed to each candidate's ``can_open``.

    Returns:
        The first claiming driver, or ``None`` when none claims the suffix or accepts the file.
    """
    for candidate in drivers_for_suffix(suffix):
        if _sniff(candidate, target):
            return candidate
    return None


def _detected_read_driver(target: str) -> tuple[MeshDriver | None, bool]:
    """Choose a driver for an OS path by suffix and then by content.

    Args:
        target: The path to resolve, already known not to be a virtual-filesystem name.

    Returns:
        ``(driver, native)``. ``native`` is True only when the suffix is one the built-in
        reader claims, which is what keeps a NetCDF read from triggering driver discovery.
    """
    suffix = Path(target).suffix.lower()
    if suffix in NETCDF_SUFFIXES:
        result: tuple[MeshDriver | None, bool] = (None, True)
    else:
        # Only here does detection-time discovery run, and only for a suffix the built-in
        # reader does not claim.
        result = (_first_claiming_driver(suffix, target), False)
    return result


def _netcdf_fallback_driver(
    path: str | Path, driver: str | None, expected: str = "MeshDataset"
) -> MeshDriver | None:
    """Find a driver for a NetCDF-suffixed file the built-in reader could not read.

    Consulted **only** after :meth:`MeshDataset._read_ugrid_netcdf` raises
    :class:`NoMeshTopologyError` — a ``.nc`` that opened but carries no UGRID topology, as a
    3Di or TUFLOW FV result does. A NetCDF suffix otherwise short-circuits ahead of discovery
    (see :func:`_detected_read_driver`), so running discovery here, and nowhere earlier, is
    what lets such a format be dispatched to while a successful UGRID read still pays nothing:
    discovery runs only once the built-in read has already failed.

    The fallback honours the same short-circuits :func:`_resolve_read_driver` applies, so the two
    routes to the built-in reader cannot diverge: an explicit ``driver`` (including a reserved
    name such as ``ugrid``) never falls back, because the caller opted out of detection, and a
    ``/vsi`` path stays on GDAL, which is where detection keeps it.

    Args:
        path: The file the built-in reader rejected.
        driver: The name the caller passed to the entry point, or ``None`` to detect. A
            non-``None`` value means the caller chose the reader explicitly, so no fallback.
        expected: The class the calling entry point returns, forwarded to
            :func:`_needs_mesh_error` so a ``Mesh1d`` caller is not sent to an
            ``attach_data`` that only :class:`MeshDataset` has. Defaults to
            ``"MeshDataset"``.

    Returns:
        The first driver claiming the NetCDF suffix whose ``can_open`` accepts the file, or
        ``None`` — an explicit ``driver``, a ``/vsi`` path, no NetCDF suffix, or none claims it —
        in which case the caller re-raises the original :class:`NoMeshTopologyError`.

    Raises:
        UnsupportedMeshFormatError: The claiming driver is ``needs_mesh``, so it carries
            values but no topology and cannot open the file on its own. Built by
            :func:`_needs_mesh_error`, whose advice depends on ``expected``.
    """
    if driver is not None:
        return None
    target = str(path)
    if target.startswith("/vsi"):
        return None
    suffix = Path(target).suffix.lower()
    if suffix not in NETCDF_SUFFIXES:
        return None
    selected = _first_claiming_driver(suffix, target)
    if selected is not None and "needs_mesh" in selected.capabilities:
        # Mirror the detection path: a results-only driver cannot open a file on its own, so
        # refuse it with guidance the calling entry point can actually follow.
        raise _needs_mesh_error(selected, target, expected)
    return selected


def _resolve_read_driver(
    path: str | Path, driver: str | None, expected: str = "MeshDataset"
) -> tuple[MeshDriver | None, bool]:
    """Decide which reader opens a file.

    Implements steps 1-3 of :meth:`MeshDataset.read_file`'s resolution order. The second
    element of the result separates "this is a NetCDF path" from "no driver claimed it", which
    is what lets a genuine NetCDF failure report GDAL's error while an unclaimed suffix reports
    the installed drivers instead.

    Args:
        path: The file to read.
        driver: An explicitly requested driver name, or ``None`` to detect.
        expected: The class the calling entry point returns, forwarded to
            :func:`_needs_mesh_error`. Defaults to ``"MeshDataset"``. A
            :class:`Mesh1d` caller reaches this function for every non-NetCDF suffix,
            which is the route the ``needs_mesh`` formats actually use -- an SMS ``.dat``,
            an XMDF or an XDMF -- so without it the advice named an ``attach_data`` that
            :class:`Mesh1d` does not have.

    Returns:
        ``(driver, native)``. ``driver`` is the driver to use, or ``None`` for the built-in
        reader. ``native`` is True when the built-in reader was chosen because the path is a
        NetCDF one, and False when it was chosen only because nothing else claimed the file.

    Raises:
        DriverNotExistError: If ``driver`` names a driver that is not installed.
        UnsupportedMeshFormatError: If the chosen driver needs a mesh supplied, which
            :meth:`MeshDataset.attach_data` does and :meth:`read_file` cannot.
    """
    _require_existing_file(path)
    target = str(path)
    if driver is not None:
        selected, native = _named_read_driver(driver)
    elif target.startswith("/vsi"):
        selected, native = None, True
    else:
        selected, native = _detected_read_driver(target)

    if selected is not None and "needs_mesh" in selected.capabilities:
        raise _needs_mesh_error(selected, target, expected)
    return selected, native


def _needs_mesh_error(
    selected: MeshDriver, target: str, expected: str = "MeshDataset"
) -> UnsupportedMeshFormatError:
    """Build the error for a results-only driver asked to read a file on its own.

    A ``needs_mesh`` driver carries values but no topology, so it cannot open a file by
    itself. Shared by :func:`_resolve_read_driver` (the detection path) and
    :func:`_netcdf_fallback_driver` (the NetCDF fallback) so both refuse it with the same
    actionable message rather than letting the driver's own ``read`` raise whatever it likes.

    Args:
        selected: The driver, named in the message.
        target: The path the caller gave, shown in the suggested ``attach_data`` call.
        expected: The class the caller asked for. ``"MeshDataset"`` (the default) gets the
            ``attach_data`` recipe; a :class:`Mesh1d` caller cannot be given it, because
            ``Mesh1d`` has no ``attach_data``, so it is told that instead of being sent to
            a method that does not exist.

    Returns:
        An :class:`UnsupportedMeshFormatError` whose advice the caller can actually follow.
    """
    if expected != "MeshDataset":
        result = UnsupportedMeshFormatError(
            f"{selected.long_name} files carry values but no mesh, so {target!r} cannot be "
            f"opened on its own -- and {expected} has no attach_data to combine them with a "
            f"topology. Read the mesh this file's values belong to with "
            f"MeshDataset.read_file, then attach them there."
        )
        return result
    return UnsupportedMeshFormatError(
        f"{selected.long_name} files carry values but no mesh, so {target!r} cannot be "
        f"opened on its own. Read the mesh first, then attach these values to it:\n"
        f"    mesh = MeshDataset.read_file('<the mesh file>')\n"
        f"    result = mesh.variables.attach_data({target!r})"
    )


def _read_through_driver(
    driver: MeshDriver,
    path: str | Path,
    mesh_name: str | None,
    expected: type[_MeshObject],
    lazy: bool = True,
) -> _MeshObject:
    """Read a file through a third-party driver, attributing any failure to it.

    Args:
        driver: The driver to read with.
        path: The file to read.
        mesh_name: Which topology to read, for a multi-mesh format.
        expected: The class the caller requires, so a driver returning the other mesh type is
            reported rather than handed back.
        lazy: Forwarded to the driver, which honours it when it declares ``lazy_data``.

    Returns:
        Whatever the driver built, once its type has been checked.

    Raises:
        UnsupportedMeshFormatError: If the driver raised, or returned the wrong mesh type.
    """
    try:
        result = driver.read(str(path), mesh_name=mesh_name, lazy=lazy)
    except (UnsupportedMeshFormatError, DriverNotExistError, OSError):
        # An OSError from a driver is a real I/O failure, not a format one. A multi-file
        # format -- FLO-2D, ADCIRC -- opens siblings of its own, and a missing sibling must
        # report as missing rather than as "this format is not supported".
        raise
    except Exception as error:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver failed to read {str(path)!r}: {error!r}"
        ) from error

    if not isinstance(result, expected):
        produced = type(result).__name__
        # Decided by an isinstance check on what the driver produced, not by matching on a
        # class name: a subclass called `MyMesh1d` would fail an `endswith` test, and a
        # subclass of `MeshDataset` was previously told to use `MeshDataset.read_file` --
        # the call it had just made.
        from pyramids.netcdf.ugrid.mesh1d import Mesh1d

        other = "Mesh1d" if isinstance(result, Mesh1d) else "MeshDataset"
        # Name the driver in the suggested call. Without it this message and the one
        # `Mesh1d.read_file` raises pointed at each other, and a caller bounced between the
        # two without either naming `driver=` as the way through.
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver read {str(path)!r} as a {produced}, not a "
            f"{expected.__name__}. A file whose topology dimension is 1 is a network, and a "
            f"surface mesh is not: read it with "
            f"{other}.read_file(path, driver={driver.name!r}) instead."
        )
    return result


def _mesh_names_through_driver(driver: MeshDriver, path: str | Path) -> list[str]:
    """Ask a driver for a file's topology names, attributing and checking what comes back.

    `read` and `write` were attributed and type-checked; this call was not, so a driver's
    parser error reached the caller bare and a driver returning a string instead of a list
    would have been iterated into single characters downstream.

    Args:
        driver: The driver to ask.
        path: The file to inspect.

    Returns:
        The topology names the driver reports.

    Raises:
        UnsupportedMeshFormatError: The driver raised, or returned something other than a
            list of strings.
    """
    try:
        names = driver.mesh_names(str(path))
    except (UnsupportedMeshFormatError, DriverNotExistError, OSError):
        raise
    except Exception as error:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver failed to list the topologies in {str(path)!r}: "
            f"{error!r}"
        ) from error

    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) for name in names
    ):
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver returned {names!r} from mesh_names; expected a "
            f"list of topology names, each a string that read_file(mesh_name=...) accepts."
        )
    # `read_all` keys a dict by these names, so duplicates collapse and a topology is lost
    # without a word. An empty list means the driver claimed the file and then reported
    # nothing in it, which `read_all` would turn into an empty result rather than an error.
    duplicates = sorted({name for name in names if list(names).count(name) > 1})
    if duplicates:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver reported duplicate topology name(s) {duplicates} "
            f"for {str(path)!r}; read_all keys its result by name, so a duplicate would "
            f"silently drop a topology."
        )
    if not names:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver claims {str(path)!r} but reported no topologies in "
            f"it. A single-mesh format returns a one-item list."
        )
    return list(names)


def _read_data_through_driver(
    driver: MeshDriver, path: str | Path, dataset: MeshDataset, lazy: bool
) -> dict[str, MeshVariable]:
    """Read a results-only file through a driver and check the shape of what it returns.

    Args:
        driver: The `needs_mesh` driver to read with.
        path: The results file.
        dataset: The mesh the values attach to, passed to the driver for its element-count check.
        lazy: Whether to defer reading the values.

    Returns:
        The variables the driver read, keyed by name.

    Raises:
        UnsupportedMeshFormatError: The driver raised, or returned something other than a
            mapping of name to :class:`MeshVariable`.
    """
    try:
        produced = driver.read_data(str(path), dataset, lazy=lazy)
    except (UnsupportedMeshFormatError, DriverNotExistError, OSError):
        raise
    except ValueError as error:
        # `ValueError` is how a driver reports the element-count mismatch, and burying that
        # behind "the driver failed" would hide the one check standing between the user and a
        # silently wrong dataset. But it is also what a plugin bug raises, so the message is
        # named rather than the whole class waved through: anything mentioning the mesh is
        # passed on verbatim, and the rest is attributed like any other driver failure.
        if "element" in str(error).lower() or "mesh" in str(error).lower():
            raise
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver failed to read {str(path)!r}: {error!r}"
        ) from error
    except Exception as error:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver failed to read {str(path)!r}: {error!r}"
        ) from error

    _check_variable_mapping(driver, produced)
    return produced


def _check_variable_mapping(driver: MeshDriver, produced: Any) -> None:
    """Reject anything that is not a mapping of variable name to :class:`MeshVariable`.

    Checked at the boundary so a driver written against the old contract -- which returned a
    whole merged dataset -- gets a message naming the expected shape rather than failing deep
    inside the merge.

    Args:
        driver: The driver that produced it, named in the message.
        produced: Whatever ``read_data`` returned.

    Raises:
        UnsupportedMeshFormatError: It is not a mapping, a key is not a string, or a value is
            not a :class:`MeshVariable`.
    """
    if not isinstance(produced, dict):
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver returned {type(produced).__name__} from read_data; "
            f"expected a mapping of variable name to MeshVariable. A driver returns only the "
            f"variables it read -- attach_data merges them onto the mesh."
        )
    bad_keys = [name for name in produced if not isinstance(name, str)]
    if bad_keys:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver returned non-string key(s) {bad_keys!r} from "
            f"read_data; each key is a variable name."
        )
    wrong = sorted(
        name for name, value in produced.items() if not isinstance(value, MeshVariable)
    )
    if wrong:
        raise UnsupportedMeshFormatError(
            f"The {driver.name!r} driver returned non-MeshVariable value(s) for {wrong}; "
            f"each value must be a MeshVariable."
        )


def _resolve_attach_driver(path: str | Path, driver: str | None) -> MeshDriver:
    """Pick the driver that attaches a results-only file to an existing mesh.

    Args:
        path: The results file to attach.
        driver: An explicitly requested driver name, or ``None`` to detect.

    Returns:
        A driver declaring ``needs_mesh``.

    Raises:
        DriverNotExistError: If ``driver`` names a driver that is not installed.
        UnsupportedMeshFormatError: If the named driver does not declare ``needs_mesh``, or if
            detection finds none for this suffix.
    """
    target = str(path)
    if driver is not None:
        selected = _named_attach_driver(driver)
    else:
        selected = _detected_attach_driver(target)
    return selected


def _named_attach_driver(driver: str) -> MeshDriver:
    """Resolve an explicitly requested driver for an attach, checking it can do the job.

    Args:
        driver: The name the caller asked for.

    Returns:
        The named driver.

    Raises:
        DriverNotExistError: The name is not installed.
        UnsupportedMeshFormatError: It does not declare ``needs_mesh``, so it reads a mesh of
            its own and has nothing to attach.
    """
    installed = mesh_drivers()
    if driver not in installed:
        raise DriverNotExistError(_unknown_driver_message(driver))
    selected = installed[driver]
    if "needs_mesh" not in selected.capabilities:
        can_do = ", ".join(sorted(selected.capabilities))
        raise UnsupportedMeshFormatError(
            f"The {driver!r} driver reads a mesh of its own, so it has nothing to attach; "
            f"it declares: {can_do}. Use MeshDataset.read_file instead."
        )
    return selected


def _detected_attach_driver(target: str) -> MeshDriver:
    """Find a results-only driver that recognises a file.

    Args:
        target: The path to resolve.

    Returns:
        The first ``needs_mesh`` driver claiming the suffix whose sniff accepts the file.

    Raises:
        UnsupportedMeshFormatError: No installed driver can attach this file.
    """
    suffix = "" if target.startswith("/vsi") else Path(target).suffix.lower()
    # Stop at the first match rather than sniffing every candidate: the read path already
    # works this way, and a sniff is not guaranteed to be free.
    for candidate in drivers_for_suffix(suffix):
        if "needs_mesh" in candidate.capabilities and _sniff(candidate, target):
            return candidate
    raise UnsupportedMeshFormatError(unsupported_format_message(target, suffix))


def __getattr__(name: str):
    """Resolve the renamed `UgridDataset` / `UgridMetadata` names, warning once.

    Args:
        name: The attribute requested from this module.

    Returns:
        The replacement object for a deprecated name.

    Raises:
        AttributeError: Any name that is neither current nor a known old name.
    """
    import warnings

    renamed = {
        "UgridDataset": ("MeshDataset", MeshDataset),
        "UgridMetadata": ("MeshMetadata", MeshMetadata),
    }
    entry = renamed.get(name)
    if entry is not None:
        new_name, obj = entry
        warnings.warn(
            f"{name} is renamed to {new_name}; import {new_name} instead. "
            f"{name} will be removed in a future release (see the migration guide).",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
