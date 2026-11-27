"""Selection / reduction engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.selection``).

Owns the largest mesh capability family -- the label/position selectors (``sel`` / ``isel`` /
``sel_layer`` / ``isel_layer`` / ``collapse_layer``), the along-time reductions and scans
(``reduce`` / ``coarsen`` / ``rolling`` / ``cumsum`` / ``cumprod`` / ``diff`` / ``argmax`` /
``argmin`` / ``idxmax`` / ``idxmin`` / ``weighted`` / ``rank``), the calculus family
(``differentiate`` / ``integrate`` / ``cumulative_integrate``), the gap / reshape helpers
(``dropna`` / ``ffill`` / ``bfill`` / ``interpolate_na`` / ``shift`` / ``sortby`` / ``squeeze`` /
``head`` / ``tail`` / ``thin``), the fits (``polyfit`` / ``curvefit``) and the calendar /
time-axis relabelling (``convert_calendar`` / ``interp_calendar`` / ``interp_like`` /
``interp_time``). It mirrors the raster selection/reduction surface; the flat ``mesh.<name>``
methods remain as deprecating facades that delegate here.

Several methods lean on module-level helpers that live in ``ugrid.dataset`` (the reduction
grouping, the static-variable rebuild, the per-variable time coordinate, the calendar planners)
or on constants defined there; those methods cannot move their body into this module without a
cycle (``dataset`` imports this engine for its ``selection`` accessor), so they keep their body
on :class:`MeshDataset` as a ``_<name>_impl`` method and this engine delegates to it. The rest
carry their body here, pulling what they need from :mod:`pyramids.base._reductions`,
:mod:`pyramids.base._timeseries` and the mesh-local :mod:`pyramids.netcdf.ugrid._facades`
helpers, none of which imports ``ugrid.dataset``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.base import _timeseries
from pyramids.base._reductions import (
    INTERP_METHODS,
    WEIGHTED_HOWS,
    gaps_as_nan,
    interpolated,
    pushed,
    shifted,
    weighted_statistic,
)
from pyramids.netcdf.ugrid._facades import _axis_distances
from pyramids.netcdf.ugrid.engines._base import _MeshEngine

if TYPE_CHECKING:
    from pyramids.netcdf.ugrid.dataset import MeshDataset
    from pyramids.netcdf.ugrid.models import MeshVariable


class _MeshSelection(_MeshEngine):
    """Selection / reduction ops for a mesh (``mesh.selection``)."""

    __slots__ = ()

    def sel(
        self,
        time: Any = None,
        *,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> MeshDataset:
        """Select time steps by coordinate value, exactly or by nearest.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.sel` over time. A scalar selects
        one step and collapses the time dimension; a sequence keeps it. Values are matched
        against the dataset's :attr:`time_values` (which fall back to step positions when the
        file carries no explicit time coordinate). ``method="nearest"`` snaps each value to the
        closest coordinate instead of requiring an exact hit, and ``tolerance`` caps how far it
        may snap — the same contract the raster ``sel`` uses, including raising ``KeyError`` (not
        ``ValueError``) when a snap would exceed ``tolerance``.

        Args:
            time: A coordinate value, or a sequence of them.
            method: ``None`` (the default) matches exactly; ``"nearest"`` snaps to the closest
                coordinate, which needs numeric coordinates.
            tolerance: The furthest a ``method="nearest"`` snap may travel. Only valid with
                ``"nearest"``, and must not be negative.

        Returns:
            MeshDataset: A new dataset with the selected steps.

        Raises:
            ValueError: No time dimension; an unknown ``method``; ``tolerance`` without
                ``method="nearest"`` or negative; a value is not among the coordinates (exact
                match); or ``"nearest"`` was asked of non-numeric coordinates.
            KeyError: A ``"nearest"`` snap would exceed ``tolerance``.
        """
        return self._ds._sel_impl(time, method=method, tolerance=tolerance)

    def isel(
        self,
        time: int | slice | list[int] | np.ndarray | None = None,
        *,
        element: int | slice | list[int] | np.ndarray | None = None,
    ) -> MeshDataset:
        """Select by position along the time axis, the element axis, or both.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.isel`. Over **time** an integer
        selects a single step and collapses the time dimension (the variables become static);
        a slice or sequence keeps the time dimension, trimmed to the chosen steps. Over
        **element** the selection is a topology change: the chosen faces are kept and the mesh
        is renumbered around them (as :meth:`clip` does), so the result is a smaller mesh.
        Giving both applies the element selection first, then the time selection.

        Args:
            time: An ``int`` step, a ``slice``, or a sequence of integer step positions.
                ``None`` (the default) leaves the time axis alone.
            element: An ``int`` face index, a ``slice``, or a sequence of face indices.
                ``None`` (the default) leaves the mesh alone.

        Returns:
            MeshDataset: A new dataset with the selection applied.

        Raises:
            ValueError: Neither ``time`` nor ``element`` was given, or ``time`` was asked of a
                dataset with no time dimension.
            IndexError: A face index is out of range for this mesh.

        Examples:
            - Keep one face, renumbering the mesh around it:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([2.0, 4.0])},
                ... )
                >>> one = mesh.selection.isel(element=0)
                >>> one.n_face, one.n_node
                (1, 3)
                >>> one["depth"].data.tolist()
                [2.0]

                ```
        """
        if time is None and element is None:
            raise ValueError("isel needs `time`, `element`, or both.")
        if element is not None:
            selected = self._ds._select_elements(element)
            return selected if time is None else selected.isel(time)
        n = self._ds._time_length("isel")
        if isinstance(time, (int, np.integer)):
            return self._ds._collapse_to_step(int(time))
        if isinstance(time, slice):
            indices = np.arange(n)[time]
        else:
            indices = np.asarray(time, dtype=np.intp)
        return self._ds._select_steps(indices)

    def reduce(
        self,
        how: str,
        *,
        skipna: bool = True,
        q: float | None = None,
        groupby: Any = None,
        min_count: int = 0,
    ) -> MeshDataset:
        """Reduce the time dimension of every temporal variable, whole or in groups.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.reduce` over time. Each
        temporal variable is reduced along its time axis through the shared
        :func:`pyramids.base._reductions.reduce_axis` — the very kernel the raster reductions
        use. Non-temporal variables are kept unchanged.

        Without ``groupby`` the whole axis collapses to one statistic and the variable becomes
        static. With ``groupby`` each group is reduced on its own slab and the results are
        stacked back along the time axis, so the variable stays temporal with one step per
        group — the same shape contract the raster grouped reduce has. For a ``(4, 2)``
        variable over four monthly steps, ``reduce("mean")`` answers shape ``(2,)`` with
        ``has_time`` False, while ``reduce("mean", groupby="MS")`` answers shape ``(2, 2)``
        with ``has_time`` still True.

        Args:
            how: One of ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"std"``, ``"var"``,
                ``"median"``, ``"prod"``, ``"quantile"``, ``"count"``, ``"all"`` or ``"any"``.
            skipna: Skip the declared no-data value and NaN. Defaults to True.
            q: The quantile for ``how="quantile"``; must stay ``None`` otherwise.
            groupby: Reduce within groups of steps instead of over the whole axis. Two
                spellings, matching the raster grouped reduce: a **sequence of per-step
                labels** the same length as the time axis (grouped in first-appearance order,
                so the caller controls the order), or a **pandas offset alias** (``"MS"``,
                ``"YS"``, …) binned against the dataset's date-like time coordinates. Each
                group's label becomes the result's ``time_values`` entry, and empty bins are
                dropped. ``None`` reduces the whole axis. Defaults to ``None``.
            min_count: Minimum valid (non-gap — NaN and the declared no-data value are both
                skipped) cells a reduced slice needs for an element to carry a value; a slice
                with fewer holds the no-data value instead of a statistic from too few
                samples. ``0`` (default) gates nothing. Only meaningful with ``skipna=True``
                and the statistics (not ``count`` / ``all`` / ``any``); matches
                :meth:`pyramids.netcdf.NetCDF.reduce`.

        Returns:
            MeshDataset: A new dataset. Without ``groupby`` each temporal variable is
            collapsed over time into a static per-element variable; with ``groupby`` each one
            keeps its time dimension, at one step per group.

        Raises:
            ValueError: ``how`` is not a known reduction; ``q`` is given for a non-quantile
                ``how`` or missing for ``how="quantile"``; ``min_count`` is negative; no
                variable has a time dimension; a temporal variable has no loaded data; a
                ``groupby`` label sequence's length is not the time length; or a ``groupby``
                offset alias is asked of time coordinates that are step positions, or that
                pandas cannot read as dates.
            TypeError: ``min_count`` is not an integer.

        Examples:
            - Mean over two time steps of a per-face variable:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"d": np.array([[1.0, 2.0], [3.0, 4.0]])},
                ...     data_locations={"d": "face"},
                ... )
                >>> mesh["d"].has_time
                True
                >>> reduced = mesh.selection.reduce("mean")
                >>> reduced["d"].data.tolist()
                [2.0, 3.0]
                >>> reduced["d"].has_time
                False

                ```
        """
        return self._ds._reduce_impl(
            how, skipna=skipna, q=q, groupby=groupby, min_count=min_count
        )

    def coarsen(
        self, window: int, how: str = "mean", variable_name: str | None = None
    ) -> MeshDataset:
        """Block-aggregate the time axis into windows of ``window`` steps (#1377).

        **Not** a duplicate of :meth:`resample`. That one bins by a calendar frequency and so
        needs date-like coordinates; this bins by a **count of steps** and needs no coordinate
        at all, which makes it the only option for a mesh whose file carries no CF time units.
        Where both work, ``resample`` is the calendar-aware choice and this the positional one.

        A trailing remainder shorter than a full window is dropped, so every reported bin is a
        full one rather than a short bin that would read as a dip in the series.

        A variable with fewer steps than one window cannot be binned at all. In a sweep
        (``variable_name=None``) it is passed through untouched, so one short variable does not
        lose the whole call — the choice :meth:`convert_units` makes for a variable with no
        units. Naming such a variable explicitly raises, since that is a request rather than a
        sweep, and so does a sweep in which **no** variable could be binned: that call did
        nothing, and returning the dataset unchanged would look like success.

        Args:
            window: Steps per block, at least 1.
            how: ``"mean"`` (default), ``"sum"``, ``"min"``, ``"max"``, ``"median"`` or
                ``"std"``.
            variable_name: Variable to aggregate (default: every temporal variable).

        Returns:
            MeshDataset: A new dataset whose time axis holds one step per full block, stamped
            with each block's first coordinate value.

        Raises:
            ValueError: No variable has a time dimension, ``window`` is below 1, ``how`` is
                unknown, or the axis is shorter than one window.

        See Also:
            resample: Calendar-frequency binning, when the axis carries real dates.
            reduce: Collapse the whole axis instead of binning it.
        """
        return self._ds._coarsen_impl(window, how, variable_name)

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
        """Rolling-window reduction along time for every temporal variable.

        Each time step becomes a statistic of the window of steps it owns (ending at it, or
        centred on it), cut where the window runs off the axis. Gaps are skipped; a step whose
        window holds fewer than ``min_periods`` valid cells is NaN. The time dimension keeps
        its length.

        Args:
            window: Steps per window.
            how: The reduction over each window (``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
                ``"std"``, ``"var"``, ``"median"``, ``"quantile"``). Defaults to ``"mean"``.
            center: Centre each window on its step rather than ending at it. Defaults to False.
            min_periods: Valid cells a window needs before its step holds a value. ``None``
                means the full ``window``, matching the raster `rolling`, so a step whose
                window runs off the axis is NaN rather than a partial statistic. Defaults to
                ``None``.
            q: The quantile in ``[0, 1]``, required when ``how="quantile"`` and rejected
                otherwise. Defaults to ``None``.
            dim: The dimension to roll along. A mesh rolls along ``"time"`` only, so any
                other name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with each temporal variable replaced by its rolling
            statistic along time; static variables are unchanged.

        Raises:
            ValueError: ``how`` is not a known reduction; ``q`` is given for a non-quantile
                ``how`` or missing for ``how="quantile"``; ``window`` is below 1;
                ``min_periods`` falls outside ``[1, window]``; ``dim`` names anything but
                ``"time"``; no variable has a time dimension; or a temporal variable has no
                loaded data.
        """
        return self._ds._rolling_impl(
            window, how, center=center, min_periods=min_periods, q=q, dim=dim
        )

    def diff(self, n: int = 1) -> MeshDataset:
        """Discrete difference along time (``out[i] = x[i] - x[i-1]``), ``n`` times.

        The time dimension shortens by ``n``; a ``time_values`` coordinate, if present, drops
        its first ``n`` entries to stay aligned with the result.

        Args:
            n: The number of successive differences. Defaults to 1.

        Returns:
            MeshDataset: A new dataset with each temporal variable differenced along time
            (its time axis shorter by ``n``); static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.
        """
        self._ds._require_temporal("diff")
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._ds._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to diff.")
            diffed = np.diff(gaps_as_nan(np.asarray(data), var.nodata), n=n, axis=axis)
            new_var = var.with_data(diffed)
            times = var.attributes.get("time_values")
            if times is not None:
                new_var.attributes = {**var.attributes, "time_values": list(times[n:])}
            new_vars[name] = new_var
        return self._ds._rebuild(new_vars)

    def differentiate(self, variable_name: str | None = None) -> MeshDataset:
        """The derivative of every temporal variable along time (#1377).

        Honours the time coordinate's own spacing, so an unevenly spaced output interval gives
        the right rate rather than one assuming a uniform step. With a decoded datetime axis
        (issue #1381) the result is **per second**.

        Args:
            variable_name: Variable to differentiate (default: every temporal variable).

        Returns:
            MeshDataset: A new dataset of rates, the same shape as the input.

        Raises:
            ValueError: No variable has a time dimension, or the axis has fewer than two steps.

        See Also:
            integrate: The inverse operation, which consumes the axis.
        """
        self._ds._require_variable_name("differentiate", variable_name)
        self._ds._alongtime_coord("differentiate")
        return self._ds._map_alongtime(
            "differentiate",
            variable_name,
            lambda values, axis, nodata, coord: _timeseries.differentiate_axis(
                values, coord, axis, nodata
            ),
        )

    def integrate(self, variable_name: str | None = None) -> MeshDataset:
        """The definite integral of every temporal variable along time (#1377).

        The integral **consumes** the time axis, so the result is static — one value per
        element, as :meth:`reduce` produces. With a decoded datetime axis the units are
        value-seconds.

        Args:
            variable_name: Variable to integrate (default: every temporal variable).

        Returns:
            MeshDataset: A new dataset whose temporal variables are now static totals.

        Raises:
            ValueError: No variable has a time dimension, or the axis has fewer than two steps.

        See Also:
            cumulative_integrate: The running integral, which keeps the axis.
            differentiate: The inverse operation.
        """
        return self._ds._integrate_impl(variable_name)

    def cumulative_integrate(self, variable_name: str | None = None) -> MeshDataset:
        """The running integral of every temporal variable along time (#1377).

        Keeps the time axis, unlike :meth:`integrate`. The first step is ``0`` — nothing has
        accumulated yet.

        Args:
            variable_name: Variable to integrate (default: every temporal variable).

        Returns:
            MeshDataset: A new dataset of running integrals, the same shape as the input.

        Raises:
            ValueError: No variable has a time dimension, or the axis has fewer than two steps.

        See Also:
            integrate: The definite integral, which consumes the axis.
        """
        self._ds._require_variable_name("cumulative_integrate", variable_name)
        self._ds._alongtime_coord("cumulative_integrate")
        return self._ds._map_alongtime(
            "cumulative_integrate",
            variable_name,
            lambda values, axis, nodata, coord: _timeseries.cumulative_integrate_axis(
                values, coord, axis, nodata
            ),
        )

    def cumsum(self) -> MeshDataset:
        """Cumulative sum along time for every temporal variable.

        Returns:
            MeshDataset: A new dataset in which each temporal variable holds the running
            sum of its steps along the time axis; static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.

        Examples:
            - Running sum over three time steps of a per-face variable:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"d": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
                ...     data_locations={"d": "face"},
                ... )
                >>> mesh.selection.cumsum()["d"].data.tolist()
                [[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]]

                ```
        """
        return self._ds._transform_time(
            "cumsum", lambda d, axis: np.cumsum(d, axis=axis)
        )

    def cumprod(self) -> MeshDataset:
        """Cumulative product along time for every temporal variable.

        Returns:
            MeshDataset: A new dataset in which each temporal variable holds the running
            product of its steps along the time axis; static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.
        """
        return self._ds._transform_time(
            "cumprod", lambda d, axis: np.cumprod(d, axis=axis)
        )

    def argmax(self) -> MeshDataset:
        """The time-step position of each element's maximum.

        Returns:
            MeshDataset: Static ``int64`` variables holding the step index of the maximum,
            declaring ``-1`` for an element with no valid step.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds._extremum_along_time(
            "argmax", largest=True, as_coordinate=False
        )

    def argmin(self) -> MeshDataset:
        """The time-step position of each element's minimum.

        Returns:
            MeshDataset: Static ``int64`` variables holding the step index of the minimum,
            declaring ``-1`` for an element with no valid step.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._ds._extremum_along_time(
            "argmin", largest=False, as_coordinate=False
        )

    def idxmax(self) -> MeshDataset:
        """The time **coordinate** of each element's maximum.

        Returns:
            MeshDataset: Static ``float64`` variables holding the time coordinate of the
            maximum, NaN where an element has no valid step.

        Raises:
            ValueError: No variable has a time dimension, or the time coordinates are
                date-like rather than numeric (use `argmin` / `argmax` for the step
                position instead).
        """
        return self._ds._extremum_along_time("idxmax", largest=True, as_coordinate=True)

    def idxmin(self) -> MeshDataset:
        """The time **coordinate** of each element's minimum.

        Where :meth:`argmin` answers a step position, this answers the value of the time
        coordinate at that step, so it needs numeric time coordinates.

        Returns:
            MeshDataset: Static ``float64`` variables holding the time coordinate of the
            minimum, NaN where an element has no valid step.

        Raises:
            ValueError: No variable has a time dimension, or the time coordinates are
                date-like rather than numeric (use `argmin` / `argmax` for the step
                position instead).
        """
        return self._ds._extremum_along_time(
            "idxmin", largest=False, as_coordinate=True
        )

    def dropna(
        self, how: str = "any", thresh: int | None = None, *, dim: str | None = None
    ) -> MeshDataset:
        """Drop time steps that do not hold enough valid cells.

        A step is kept only when **every** temporal variable meets the threshold at that
        step, so the variables stay time-aligned. ``how="any"`` requires every cell of a
        variable to be valid (a single gap drops the step); ``how="all"`` requires at least
        one; ``thresh`` overrides ``how`` with an explicit valid-cell count per variable.

        Args:
            how: ``"any"`` or ``"all"``. Defaults to ``"any"``.
            thresh: Explicit minimum valid-cell count per variable, overriding ``how``.
            dim: The dimension to drop along. A mesh drops along ``"time"`` only, so any other
                name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with the surviving steps.

        Raises:
            ValueError: ``how`` is neither ``"any"`` nor ``"all"``; ``dim`` names anything but
                ``"time"``; no variable has a time dimension; a temporal variable has no
                loaded data; or the threshold would drop **every** step, which would leave a
                temporal dataset with no steps for anything downstream to read.
        """
        if how not in ("any", "all"):
            raise ValueError(f"dropna how must be 'any' or 'all', got {how!r}.")
        self._ds._check_time_dim("dropna", dim)
        temporal = self._ds._require_temporal("dropna")
        length = self._ds._time_length("dropna")
        keep = np.ones(length, dtype=bool)
        for name in temporal:
            var = self._ds._data_variables[name]
            axis = cast("int", var.time_axis_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to dropna.")
            moved = np.moveaxis(gaps_as_nan(np.asarray(data), var.nodata), axis, 0)
            flat = moved.reshape(moved.shape[0], -1)
            per_step_valid = np.sum(~np.isnan(flat), axis=1)
            needed = flat.shape[1] if how == "any" else 1
            if thresh is not None:
                needed = thresh
            keep &= per_step_valid >= needed
        if not keep.any():
            # Returning a zero-length time axis would leave a "temporal" dataset with no
            # steps, which nothing downstream can read; the raster dropna refuses too.
            raise ValueError(
                f"dropna(how={how!r}, thresh={thresh!r}) would drop every one of the "
                f"{length} time step(s); nothing would be left."
            )
        return self._ds._select_steps(np.flatnonzero(keep))

    def ffill(self, *, limit: int | None = None, dim: str | None = None) -> MeshDataset:
        """Forward-fill gaps along time from the last valid step.

        Args:
            limit: Maximum consecutive gaps one valid step may fill, or ``None`` for no limit.
            dim: The dimension to fill along. A mesh fills along ``"time"`` only, so any other
                name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with each temporal variable's gaps carried forward
            from the previous valid step; static variables are unchanged.

        Raises:
            ValueError: ``dim`` names anything but ``"time"``, no variable has a time
                dimension, or a temporal variable has no loaded data.
        """
        self._ds._check_time_dim("ffill", dim)
        return self._ds._transform_time(
            "ffill", lambda d, axis: pushed(d, axis, limit, False)
        )

    def bfill(self, *, limit: int | None = None, dim: str | None = None) -> MeshDataset:
        """Back-fill gaps along time from the next valid step.

        Args:
            limit: Maximum consecutive gaps one valid step may fill, or ``None`` for no limit.
            dim: The dimension to fill along. A mesh fills along ``"time"`` only, so any other
                name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with each temporal variable's gaps carried backward
            from the next valid step; static variables are unchanged.

        Raises:
            ValueError: ``dim`` names anything but ``"time"``, no variable has a time
                dimension, or a temporal variable has no loaded data.
        """
        self._ds._check_time_dim("bfill", dim)
        return self._ds._transform_time(
            "bfill", lambda d, axis: pushed(d, axis, limit, True)
        )

    def interpolate_na(
        self,
        *,
        method: str = "linear",
        limit: int | None = None,
        use_coordinate: bool = True,
        dim: str | None = None,
    ) -> MeshDataset:
        """Fill each interior time gap by interpolating between the steps on either side.

        Distance is measured along the time coordinate by default, so unevenly spaced steps
        weight their neighbours accordingly; pass ``use_coordinate=False`` to measure by step
        position instead. A leading or trailing gap has only one neighbour and is left alone.

        Args:
            method: One of ``"linear"`` (distance-weighted), ``"nearest"``, or the scipy spline
                kinds ``"slinear"`` / ``"quadratic"`` / ``"cubic"`` (each falling back to linear on
                a variable with too few valid steps). Defaults to ``"linear"``.
            limit: Maximum consecutive gaps a run may fill, or ``None`` for no limit.
            use_coordinate: Measure distance along the time coordinate rather than by step
                position. With time values ``[0, 1, 10]`` and a gap at the middle step, the
                coordinate reading fills ``1.0`` and the positional reading ``5.0``. Date-like
                coordinates count as distances too (converted to nanoseconds), so unevenly
                spaced dates weight their neighbours; a coordinate with no measurable spacing
                (plain labels) falls back to step positions. Defaults to ``True``.
            dim: The dimension to interpolate along. A mesh interpolates along ``"time"``
                only, so any other name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            MeshDataset: A new dataset with each temporal variable's interior gaps
            interpolated along time; static variables are unchanged.

        Raises:
            ValueError: ``method`` is not one of the accepted interpolations; no variable has a
                time dimension; or a temporal variable has no loaded data.
        """
        if method not in INTERP_METHODS:
            # Validate up front, before the time check, so a bogus method is reported as such
            # even on a mesh with no temporal variable — matching the raster interpolate_na.
            raise ValueError(
                f"interpolate_na() takes method="
                f"{' or '.join(repr(one) for one in INTERP_METHODS)}, got {method!r}."
            )

        self._ds._check_time_dim("interpolate_na", dim)
        stamps = self._ds.time_values if use_coordinate else None
        axis_values = None
        if stamps is not None:
            axis_values = _axis_distances(stamps)

        def _fill(data: np.ndarray, axis: int) -> np.ndarray:
            size = data.shape[axis]
            if axis_values is not None and axis_values.shape[0] == size:
                positions = axis_values
            else:
                positions = np.arange(size, dtype="float64")
            return np.asarray(interpolated(data, axis, positions, method, limit))

        return self._ds._transform_time("interpolate_na", _fill)

    def shift(self, periods: int = 1) -> MeshDataset:
        """Shift each temporal variable ``periods`` steps along time, vacated steps NaN.

        Args:
            periods: Steps to shift; negative shifts towards the start. Defaults to 1.

        Returns:
            MeshDataset: A new dataset with each temporal variable shifted along time, the
            vacated steps filled with NaN; static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.
        """
        return self._ds._transform_time(
            "shift",
            lambda d, axis: shifted(d, axis, periods, np.nan),
            # Move a per-step mask with the snapshots (time on axis 0), vacated steps inactive —
            # a shifted-in step holds NaN data, so False is right (review H1).
            active_fn=lambda m: shifted(m, 0, periods, False),
        )

    def sortby(self) -> MeshDataset:
        """Sort the time steps by their coordinate value (stable).

        Returns:
            MeshDataset: A new dataset with steps ordered by ascending time coordinate.

        Raises:
            ValueError: No variable has a time dimension.
        """
        coords = self._ds._time_coords("sortby")
        order = np.argsort(np.asarray(coords), kind="stable")
        return self._ds._select_steps(order)

    def squeeze(self) -> MeshDataset:
        """Drop the time dimension when it has length 1; otherwise return unchanged.

        Mirrors :meth:`pyramids.netcdf.NetCDF.squeeze` for the time axis: a single-step
        temporal variable becomes static, a multi-step one is left alone.

        Returns:
            MeshDataset: A new dataset with single-step temporal variables collapsed to
            static, or ``self`` when there is nothing to squeeze.

        Raises:
            ValueError: A single-step temporal variable has no loaded data.
        """
        return self._ds._squeeze_impl()

    def head(self, n: int = 5) -> MeshDataset:
        """Keep the first ``n`` time steps (fewer if the axis is shorter).

        Args:
            n: Number of leading steps to keep. Defaults to 5.

        Returns:
            MeshDataset: A new dataset trimmed to the first ``n`` steps.

        Raises:
            ValueError: No variable has a time dimension.
        """
        length = self._ds._time_length("head")
        return self._ds._select_steps(np.arange(min(n, length)))

    def tail(self, n: int = 5) -> MeshDataset:
        """Keep the last ``n`` time steps (fewer if the axis is shorter).

        Args:
            n: Number of trailing steps to keep. Defaults to 5.

        Returns:
            MeshDataset: A new dataset trimmed to the last ``n`` steps.

        Raises:
            ValueError: No variable has a time dimension.
        """
        length = self._ds._time_length("tail")
        return self._ds._select_steps(np.arange(max(length - n, 0), length))

    def thin(self, step: int) -> MeshDataset:
        """Keep every ``step``-th time step.

        Args:
            step: Stride; must be >= 1.

        Returns:
            MeshDataset: A new dataset keeping every ``step``-th step.

        Raises:
            ValueError: ``step`` is less than 1, or no variable is temporal.
        """
        if step < 1:
            raise ValueError(f"thin step must be >= 1, got {step}.")
        length = self._ds._time_length("thin")
        return self._ds._select_steps(np.arange(0, length, step))

    def weighted(
        self,
        variable_name: str,
        *,
        how: str = "mean",
        time_index: int = 0,
        q: float | None = None,
    ) -> float:
        """Area-weighted statistic of a face variable over the mesh.

        An irregular mesh has faces of different sizes, so the meaningful aggregate weights
        each face by its area. Delegates to the shared
        :func:`pyramids.base._reductions.weighted_statistic` with :attr:`Mesh2d.face_areas`
        as the weights — the same kernel the raster ``weighted`` reduction uses.

        Args:
            variable_name: Name of a **face-located** data variable.
            how: One of ``"mean"``, ``"sum"``, ``"sum_of_weights"``, ``"std"``, ``"var"``,
                ``"quantile"`` (which needs ``q``). Defaults to ``"mean"``.
            time_index: Step for a temporal variable. Defaults to 0.
            q: The quantile in ``[0, 1]`` when ``how="quantile"`` (the area-weighted quantile
                uses the Hazen convention and does not match ``reduce(how="quantile")``); must be
                ``None`` otherwise.

        Returns:
            float: The area-weighted statistic over all faces.

        Raises:
            ValueError: ``how`` is not one of ``WEIGHTED_HOWS``; the variable is not
                face-located; or its length does not match the number of faces.

        Examples:
            - Area-weighted mean: the larger face (area 4, value 10) outweighs the
              smaller (area 2, value 20), pulling the mean below the unweighted 15.0:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 4.0, 4.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 2.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"v": np.array([10.0, 20.0])},
                ... )
                >>> round(mesh.selection.weighted("v"), 3)
                13.333

                ```
            - The area-weighted quantile clamps to the data range at the extremes:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> mesh = MeshDataset.from_arrays(
                ...     node_x=np.array([0.0, 4.0, 4.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 2.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"v": np.array([10.0, 20.0])},
                ... )
                >>> mesh.selection.weighted("v", how="quantile", q=0.0)
                10.0
                >>> mesh.selection.weighted("v", how="quantile", q=1.0)
                20.0

                ```
        """
        if how not in WEIGHTED_HOWS:
            # Validate up front, mirroring the raster weighted path: the kernel's catch-all
            # else-branch would otherwise return the standard deviation for any unsupported
            # `how`, a silent wrong answer.
            raise ValueError(
                f"weighted: how must be one of {sorted(WEIGHTED_HOWS)}, got {how!r}."
            )
        if how == "quantile":
            if q is None or isinstance(q, bool) or not 0.0 <= float(q) <= 1.0:
                raise ValueError(
                    f"weighted(how='quantile') needs q in [0, 1], got {q!r}."
                )
        elif q is not None:
            raise ValueError(
                f"weighted(q=...) is only valid with how='quantile', not {how!r}."
            )
        var, arr = self._ds._element_values(variable_name, time_index=time_index)
        if var.location != "face":
            raise ValueError(
                "area-weighted statistics need a face-located variable; "
                f"{variable_name!r} is on {var.location!r}."
            )
        areas = self._ds._mesh.face_areas
        if arr.shape[-1] != areas.shape[0]:
            raise ValueError(
                f"variable {variable_name!r} has {arr.shape[-1]} face value(s) but the "
                f"mesh has {areas.shape[0]} face(s)."
            )
        # `arr` already carries NaN for gaps, so `ndv=None` and `skipna=True` reduce over
        # the single face axis; the kernel keeps the axis as length 1, squeezed off here.
        result = weighted_statistic(arr, areas, (0,), how, None, True, q)
        return float(np.asarray(result).ravel()[0])

    def polyfit(self, deg: int, variable_name: str | None = None) -> MeshDataset:
        """Fit a polynomial per element along time, returning its coefficients (#1377).

        The mesh counterpart of the raster ``polyfit``, and what produces a **trend map**: a
        degree-1 fit per face gives the slope of that face's series.

        The time axis is replaced by a ``coefficient`` axis of length ``deg + 1``, highest
        power first — the order :func:`numpy.polyfit` returns. The result is therefore no
        longer a time series, so it is marked static.

        Args:
            deg: Polynomial degree.
            variable_name: Variable to fit (default: every temporal variable).

        Returns:
            MeshDataset: A new dataset whose variables carry the fitted coefficients.

        Raises:
            ValueError: No variable has a time dimension, ``deg`` is negative, or the axis is
                too short for the degree.

        See Also:
            curvefit: Fit an arbitrary model instead of a polynomial.
            differentiate: The instantaneous rate rather than a fitted trend.
        """
        self._ds._require_variable_name("polyfit", variable_name)
        self._ds._alongtime_coord("polyfit")
        return self._ds._fit_alongtime(
            variable_name,
            lambda values, axis, nodata, coord: _timeseries.polyfit_axis(
                values, coord, deg, axis, nodata
            ),
            "coefficient",
        )

    def curvefit(
        self, func: Any, p0: Any = None, variable_name: str | None = None
    ) -> MeshDataset:
        """Fit an arbitrary model per element along time (#1377).

        The general form of :meth:`polyfit`. Each element is fitted in turn, since there is no
        closed form; an element whose fit does not converge yields ``NaN`` parameters rather
        than failing the whole call, so one bad face does not lose the other thousands.

        Args:
            func: The model ``f(x, *params)``. ``x`` is the time coordinate as seconds elapsed
                from the first stamp when the axis carries datetimes.
            p0: Initial parameter guess. Without one the parameter count is read from ``func``'s
                signature.
            variable_name: Variable to fit (default: every temporal variable).

        Returns:
            MeshDataset: A new dataset whose variables carry the fitted parameters along a
            ``parameter`` axis.

        Raises:
            ValueError: No variable has a time dimension.

        See Also:
            polyfit: The closed-form polynomial case, much faster.
        """
        self._ds._require_variable_name("curvefit", variable_name)
        self._ds._alongtime_coord("curvefit")
        return self._ds._fit_alongtime(
            variable_name,
            lambda values, axis, nodata, coord: _timeseries.curvefit_axis(
                values, coord, func, p0, axis, nodata
            ),
            "parameter",
        )

    def rank(
        self, variable_name: str | None = None, *, pct: bool = False
    ) -> MeshDataset:
        """Rank each element's values along time, ties averaged (#1377).

        Per face this is an **exceedance map**: where does each step sit among all the steps at
        that face. Needs no date coordinate, so it works on a positional axis too.

        The declared sentinel and ``NaN`` are excluded rather than ranked, as the cube's
        ``rank`` excludes them — ranking a ``-9999`` as the smallest value would shift every
        real rank by one.

        Args:
            variable_name: Variable to rank (default: every temporal variable).
            pct: Report each rank as a fraction of that element's usable values rather than a
                count, so its largest usable value is ``1.0``.

        Returns:
            MeshDataset: A new dataset of ranks, the same shape as the input, counting from 1
            (or in ``(0, 1]`` with ``pct=True``), with ``NaN`` where the value was missing.

        Raises:
            ValueError: No variable has a time dimension.

        See Also:
            idxmax: The step at which each element peaks.
        """
        self._ds._require_variable_name("rank", variable_name)
        self._ds._require_temporal("rank")
        return self._ds._map_alongtime(
            "rank",
            variable_name,
            lambda values, axis, nodata, _coord: _timeseries.rank_axis(
                values, axis, nodata, pct
            ),
        )

    def convert_calendar(self, calendar: str, *, align_on: str = "date") -> MeshDataset:
        """Restamp the time axis onto another CF calendar (issue #1381).

        The mesh counterpart of :meth:`NetCDF.convert_calendar`, and what lets a mesh run on a
        model calendar be compared or :meth:`concat`-enated with one on another.

        **Only the time coordinate changes.** No value is touched and nothing is interpolated:
        each step keeps its values and is given the stamp its date has in the target calendar.
        A date the target has no day for is **dropped, and its step with it** — 29 February
        moving to ``noleap``, 31 December moving to ``360_day`` — so the axis gets shorter.
        :meth:`interp_calendar` is the lossless counterpart.

        Args:
            calendar: The target CF calendar — ``"standard"`` / ``"gregorian"``,
                ``"proleptic_gregorian"``, ``"noleap"`` / ``"365_day"``, ``"all_leap"`` /
                ``"366_day"``, ``"360_day"`` or ``"julian"``.
            align_on: ``"date"`` (default) keeps the calendar date and drops what the target
                lacks. ``"year"`` keeps the *position in the year* instead, which is what a
                ``360_day`` axis of model bookkeeping wants; it loses less, but is not lossless
                either — when the target year is shorter two source days can land on one target
                day, and the duplicates are dropped to keep the axis strictly monotonic.

        Returns:
            MeshDataset: A new dataset whose time axis is restamped onto ``calendar``, carrying
            that calendar and the original units.

        Raises:
            ValueError: The axis carries no CF time units; ``calendar`` is not one ``cftime``
                knows; ``align_on`` is neither ``"date"`` nor ``"year"``; or every step would be
                dropped, which would leave nothing to return.

        See Also:
            interp_calendar: Keeps every step by interpolating instead of dropping.
        """
        return self._ds._convert_calendar_impl(calendar, align_on=align_on)

    def interp_calendar(self, target: MeshDataset) -> MeshDataset:
        """Interpolate the values onto another dataset's stamps, across calendars (#1381).

        :meth:`convert_calendar`'s lossless counterpart. Where that one restamps the steps it
        has and drops the dates the target calendar lacks, this one **interpolates the values**
        onto the target's own stamps, so no step is lost and the two line up exactly — which is
        what :meth:`concat` or an operator between two runs needs.

        Both axes are put on a common **decimal-year** scale — the year plus the fraction of it
        elapsed — which means the same thing in every calendar: a 360-day and a 365-day year
        both run from ``Y.0`` to ``Y+1.0``.

        Args:
            target: The dataset whose stamps, units and calendar the result takes. Must sit on
                the same topology.

        Returns:
            MeshDataset: A new dataset on ``target``'s stamps, carrying its units and calendar.

        Raises:
            AlignmentError: ``target`` sits on a different mesh.
            ValueError: Either side carries no CF time units; this side has fewer than two steps
                to interpolate between; or the two spans do not overlap at all, so every step
                would come back a gap.

        Note:
            A target stamp inside this dataset's span interpolates; one outside it is a **gap**,
            not an extrapolation. A partly overlapping target therefore comes back partly
            masked, and one that does not overlap at all is refused rather than answered as a
            fully-masked dataset.

        See Also:
            convert_calendar: Restamps without interpolating, dropping absent dates.
        """
        return self._ds._interp_calendar_impl(target)

    def interp_like(self, other: MeshDataset) -> MeshDataset:
        """Interpolate onto another dataset's time stamps (#1377).

        What lines two runs up on a common axis before :meth:`concat` or an operator. Linear,
        and deliberately **not** extrapolating: a target stamp outside this dataset's span comes
        back as a gap (NaN), which is the contract the raster ``interp`` already has.

        Args:
            other: The dataset whose time coordinate supplies the target stamps. Must sit on
                the same topology.

        Note:
            Two dated axes are put on a common **decimal-year** scale, which means the same
            thing in every calendar, so this works across calendars as well as within one. The
            result declares ``other``'s units and calendar, since those are the stamps it now
            carries; an undated ``other`` leaves the result positional, with no CF metadata at
            all.

        Returns:
            MeshDataset: A new dataset on ``other``'s time stamps, carrying ``other``'s time
            units and calendar (or none, when ``other`` is positional).

        Raises:
            AlignmentError: ``other`` sits on a different mesh.
            ValueError: Either side has no time dimension; this one has fewer than two steps to
                interpolate between; or one side is dated and the other positional, which have
                no common scale.

        See Also:
            interp_time: Interpolate onto a given number of evenly spaced steps.
        """
        return self._ds._interp_like_impl(other)

    def sel_layer(self, value: Any) -> MeshDataset:
        """Select along the layer axis by coordinate value.

        Matches ``value`` against :attr:`layer_values`, falling back to layer positions when
        no layer coordinate is present -- which is the usual case, since the reader does not
        populate `layer_values` (see :attr:`layer_values`). So on a mesh read from a file,
        ``sel_layer(1)`` selects layer position 1.

        Args:
            value: The layer coordinate value to select.

        Returns:
            MeshDataset: A new dataset with that layer selected.

        Raises:
            ValueError: No layer dimension, or the value is not among the layer coordinates.
        """
        count = self._ds.n_layer
        if count is None:
            raise ValueError(
                "sel_layer needs at least one variable with a layer dimension; this "
                "dataset has none."
            )
        # Read the property once: it walks every variable, so evaluating it twice walked
        # them twice.
        stored = self._ds.layer_values
        coords = stored if stored is not None else list(range(count))
        if value not in coords:
            raise ValueError(
                f"sel_layer: layer value {value!r} not found in the layer coordinate "
                f"{coords!r}."
            )
        return self.isel_layer(coords.index(value))

    def isel_layer(self, index: int | slice | list[int] | np.ndarray) -> MeshDataset:
        """Select along the layer axis by position.

        A layered variable — ``(layer, element)``, or ``(time, layer, element)`` — is indexed
        on its layer axis. An integer collapses that axis (the variable loses its layer
        dimension); a slice or sequence keeps it, trimmed. Variables with no layer axis are
        carried through untouched, as the time selectors carry static variables.

        For a **ragged** 3-D mesh (:attr:`volume_topology` set) a ``location="volume"``
        variable has no rectangular layer axis to index; the selection is per column instead.
        ``index`` must be an ``int`` ``k``: each column yields its ``k``-th layer where it has
        one, and a column shallower than ``k + 1`` layers yields NaN — the absent layers are
        made explicit, never padded silently. The volume variable becomes a plain
        ``location="face"`` field; other variables carry through as on the uniform path.

        Args:
            index: An ``int`` layer, a ``slice``, or a sequence of layer positions. A ragged
                mesh accepts only an ``int``.

        Returns:
            MeshDataset: A new dataset with the layer selection applied.

        Raises:
            ValueError: No variable has a layer dimension, or a non-``int`` ``index`` was given
                for a ragged mesh.
        """
        if self._ds._volume_topology is not None:
            return self._ds._isel_layer_ragged(index)
        new_vars: dict[str, MeshVariable] = {}
        layered = False
        for name, var in self._ds._data_variables.items():
            axis = self._ds._layer_axis(var)
            if axis is None:
                new_vars[name] = var
                continue
            layered = True
            data = np.asarray(var.data)
            if isinstance(index, (int, np.integer)):
                picked = np.take(data, int(index), axis=axis)
                moved = var.with_data(picked)
                moved.dimensions = tuple(
                    dim
                    for position, dim in enumerate(var.dimensions)
                    if position != axis
                )
            else:
                positions = (
                    np.arange(data.shape[axis])[index]
                    if isinstance(index, slice)
                    else np.asarray(index, dtype=np.intp)
                )
                moved = var.with_data(np.take(data, positions, axis=axis))
            new_vars[name] = moved
        if not layered:
            raise ValueError(
                "isel_layer needs at least one variable with a layer dimension; this "
                "dataset has none."
            )
        return self._ds._rebuild(new_vars)

    def collapse_layer(self, layer: int, *, location: str = "face") -> MeshDataset:
        """Collapse a ragged 3-D mesh's ``volume`` variables to a 2-D field (issue #1330).

        The explicit 2-D collapse of a ragged mesh: each column's ``layer``-th value becomes a
        plain ``(n_face,)`` (or ``(time, n_face)``) face field, absent layers NaN. It is
        :meth:`isel_layer` under a name that says "give me a 2-D slice", so a caller can ask for
        the surface/bed layer as a flat field without reaching for the per-column internals.
        (Named ``collapse_layer`` rather than ``to_location`` because the latter already moves a
        variable *between* mesh locations by graph interpolation — a different operation.)

        Args:
            layer: The 0-based layer position to take from each column.
            location: The target mesh location. Only ``"face"`` is supported — a ragged volume
                collapses onto its columns' faces. Defaults to ``"face"``.

        Returns:
            MeshDataset: A new dataset whose volume variables are 2-D face fields.

        Raises:
            ValueError: ``location`` is not ``"face"``, or the dataset has no volume topology
                to collapse.
        """
        if self._ds._volume_topology is None:
            raise ValueError(
                "collapse_layer needs a ragged volume topology; this dataset has none (it is "
                "2-D or uniformly layered). Use isel_layer for a uniform mesh."
            )
        if location != "face":
            raise ValueError(
                f"collapse_layer supports only 'face' for a ragged volume mesh, not "
                f"{location!r}."
            )
        return self._ds._isel_layer_ragged(int(layer))

    def interp_time(
        self, new_times: Any, *, method: str = "linear", bounds: str = "nan"
    ) -> Any:
        """Interpolate every temporal variable onto new time coordinates.

        Distinct from :meth:`interpolate_na`, which fills gaps on the existing axis: this
        *relabels* the axis, resampling the series onto the times you ask for.

        Args:
            new_times: The time coordinates to interpolate onto.
            method: ``"linear"`` or ``"nearest"``. Defaults to ``"linear"``. (Not every
                ``scipy.interpolate.interp1d`` kind: the kernel accepts these two.)
            bounds: What a label outside the existing axis answers -- ``"nan"`` for NaN, or
                ``"clamp"`` to hold the first/last step's value. Defaults to ``"nan"``.

        Returns:
            MeshDataset: A new dataset on the requested time coordinates.

        Raises:
            ValueError: No variable has a time dimension; ``method`` is neither ``"linear"``
                nor ``"nearest"``; ``bounds`` is neither ``"nan"`` nor ``"clamp"``; or the time
                coordinates are not numeric.
        """
        return self._ds._interp_time_impl(new_times, method=method, bounds=bounds)
