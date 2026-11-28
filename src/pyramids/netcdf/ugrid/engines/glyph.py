"""Glyph engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.glyph``).

Owns the plotting family -- ``plot`` / ``plot_histogram`` / ``plot_vector_field`` /
``plot_outline`` / ``animate`` / ``to_image``. Every method here builds a cleopatra glyph (via
the shared :func:`pyramids.dataset._plot_helpers.mesh_render` /
:func:`~pyramids.dataset._plot_helpers.nonnull_group_kwargs` helpers and the
:mod:`pyramids.netcdf.ugrid.plot` wrappers) -- pyramids owns no matplotlib drawing code of its
own, so this engine only marshals the mesh and its values and hands them to cleopatra, exactly
as the raster ``Dataset.glyph`` engine does.

``animate`` delegates back to ``MeshDataset._animate_impl`` rather than moving its body here: it
reads the per-variable time coordinate through ``pyramids.netcdf.ugrid.dataset._variable_time_coord``,
a module-level helper in ``dataset.py`` that this module cannot import without a cycle
(``dataset`` imports this engine for its ``glyph`` accessor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.base._plot_backends import (
    forward_kwargs,
    is_colorbar_spec,
    reject_mpl_only_params,
    render_via_backend,
)
from pyramids.base.crs import crs_spec
from pyramids.dataset._plot_helpers import mesh_render as _mesh_render
from pyramids.dataset._plot_helpers import nonnull_group_kwargs as _nonnull_group_kwargs
from pyramids.netcdf.ugrid import analysis as _analysis
from pyramids.netcdf.ugrid import plot as _ugrid_plot
from pyramids.netcdf.ugrid.engines._base import _MeshEngine

if TYPE_CHECKING:
    from cleopatra.glyphs.gridded.array_glyph import PointOverlay
    from cleopatra.styling.colorbar import ColorBar
    from cleopatra.styling.params import Contour, DataStyle
    from cleopatra.styling.scaling import ColorScaling


class _MeshGlyph(_MeshEngine):
    """Plotting for a mesh: plot, histogram, vector field, outline, animate, to_image (``mesh.glyph``)."""

    __slots__ = ()

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

        N-6 — this facade now goes through the same module-level
        helper as the raster path. The mesh-specific dispatch lives in
        :func:`pyramids.dataset._plot_helpers.mesh_render`; both
        ``Dataset.plot``/``NetCDF.plot`` and ``MeshDataset.plot`` share
        the "resolve data, hand to a single helper" contract so the
        two formats no longer maintain independent plotting code paths.

        Args:
            variable_name: Name of the data variable to plot.
            ax: matplotlib Axes. Created if None.
            backend: Renderer backend. ``"matplotlib"`` (default) draws the mesh through
                cleopatra and returns a ``MeshGlyph``. Any other name (e.g. ``"interactive"``
                / ``"3d"`` / ``"web"``) routes to a ``pyramids.plot_backends`` plugin (in
                practice ``digitalearth``) and returns that backend's own object. The
                matplotlib-only render groups (``color`` / ``contour`` / ``data_style`` /
                ``points`` / a ``ColorBar`` instance) then raise a :class:`ValueError` if set,
                and ``ax`` does not apply. Requires the backend's package installed.
            cmap: Colormap name.
            title: Plot title. Defaults to variable name.
            basemap: If True, add an OpenStreetMap basemap. If a string,
                use it as the tile provider name (e.g. "CartoDB.Positron").
                Default is None (no basemap). Requires the [viz] extra.
            colorbar (bool or ColorBar, optional): Colour-bar spec, part of the
                shared plot signature. A ``pyramids.plot.ColorBar(label=…, …)``
                configures the bar; ``False`` hides it and ``None`` (default) uses
                cleopatra's default (a bar is drawn). Only forwarded when set.
            points (np.ndarray or PointOverlay, optional): Accepted for signature
                symmetry with the raster plot family, but a **no-op here** — a mesh
                has no point-overlay layer (the mesh geometry is the data). Ignored.
            kind (str, optional): Accepted for signature symmetry with the raster
                plot family, but a **no-op here** — ``kind`` selects a raster
                renderer (``imshow``/``pcolormesh``); the mesh always renders via
                ``tripcolor``/``tricontour``. Ignored.
            color (ColorScaling, optional): Colour-scale spec
                ``pyramids.plot.ColorScaling`` (linear / power / sym-log / boundary /
                midpoint norm), e.g. ``ColorScaling.power(gamma=0.7)``. Default ``None``.
            contour (Contour, optional): Contour-line spec
                ``pyramids.plot.Contour(levels=…, label_kw=…)``. Default ``None``.
            data_style (DataStyle, optional): Data-style / relief spec
                ``pyramids.plot.DataStyle(style=…, hillshade=…)``. (A mesh has no
                cell-value overlay, so there is no ``cells`` param here.) Default ``None``.
            **kwargs: Additional arguments passed to mesh_render
                (forwarded to plot_mesh_data). Notably ``colorbar``
                (``bool``, default ``True``): pass ``colorbar=False`` to
                suppress the per-mesh colorbar when you want to attach a
                custom or shared one to ``glyph.ax``. Also ``style`` (name of
                a cleopatra ``DATA_STYLES`` preset, e.g. ``"flow_accumulation"``)
                and ``hillshade`` (``True`` or a params dict) to colour / relief-
                shade the mesh; both require cleopatra >= 0.24 (``hillshade``
                needs node-centered data). Distinct from
                :meth:`pyramids.dataset.Dataset.hillshade`, which *returns* a
                shaded-relief array.

        Returns:
            cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph instance with the plot
                rendered, carrying the dataset CRS (the EPSG code, or the CRS
                WKT when there is no EPSG) so a downstream ``glyph.add_tiles()``
                projects correctly. Use the returned object to access the
                matplotlib handles and the mappable:

                - ``glyph.fig`` / ``glyph.ax`` — Figure and Axes.
                - ``glyph.im`` — the mesh mappable (the
                  ``tripcolor``/``tricontour(f)`` artist) set by ``plot()``;
                  use it for a custom colorbar or ``glyph.im.set_clim(...)``.
                  It is ``None`` after :meth:`plot_outline` (an outline
                  carries no scalar mapping).
                - ``glyph.apply_style(style)`` (cleopatra >= 0.25) — re-apply a
                  ``DATA_STYLES`` preset by name in place, without re-plotting.

        Raises:
            ValueError: If the selected variable has no loaded data, or if
                `basemap` is requested while the dataset has no CRS (`epsg`).
        """
        if backend != "matplotlib":
            reject_mpl_only_params(
                backend,
                color=color,
                contour=contour,
                data_style=data_style,
                points=points,
                colorbar=colorbar if is_colorbar_spec(colorbar) else None,
            )
            forwarded = forward_kwargs(
                variable_name=variable_name,
                # Forward cmap only when the user changed it from the matplotlib default, so a
                # non-matplotlib backend keeps its own default colormap rather than pyramids'.
                cmap=None if cmap == "viridis" else cmap,
                title=title,
                basemap=basemap,
                colorbar=None if is_colorbar_spec(colorbar) else colorbar,
                kind=None if kind == "auto" else kind,
            )
            return render_via_backend(self._ds, backend, **forwarded, **kwargs)
        var = self._ds[variable_name]
        data = var.data
        if data is None:
            raise ValueError(f"Variable {variable_name!r} has no loaded data to plot.")
        if var.has_time:
            # First step along the variable's own time axis (see `_element_values`).
            data = np.take(np.asarray(data), 0, axis=cast("int", var.time_axis_index))
        if title is None:
            title = variable_name
        if basemap and crs_spec(self._ds.epsg, self._ds.crs_wkt) is None:
            raise ValueError("MeshDataset must have a CRS (epsg) to use basemap.")
        # ``points`` / ``kind`` are part of the shared raster-family plot signature
        # but have no meaning for a mesh (no point overlay; the renderer is fixed to
        # tripcolor/tricontour), so they are accepted and ignored. ``colorbar`` and the
        # typed render groups (``color`` / ``contour`` / ``data_style``) map onto the mesh
        # backend and are forwarded only when set (so cleopatra's backend defaults are
        # preserved otherwise).
        if colorbar is not None:
            kwargs["colorbar"] = colorbar
        kwargs.update(
            _nonnull_group_kwargs(color=color, contour=contour, data_style=data_style)
        )
        result = _mesh_render(
            mesh=self._ds._mesh,
            data=data,
            location=var.location,
            ax=ax,
            cmap=cmap,
            title=title,
            basemap=basemap,
            basemap_epsg=self._ds.epsg,
            basemap_crs_wkt=self._ds.crs_wkt,
            **kwargs,
        )
        return self._ds._glyph_with_crs(result)

    def plot_histogram(
        self,
        bins: int = 15,
        variable_name: str | None = None,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the distribution of a variable's values.

        Delegates to cleopatra's ``HistogramGlyph``.

        Args:
            bins: Number of histogram bins.
            variable_name: Variable to histogram (default: the only one).
            ax: Existing matplotlib axes to draw into.
            **kwargs: Forwarded to the glyph constructor.

        Returns:
            ``(fig, ax, hist)`` from ``HistogramGlyph.histogram``.

        See Also:
            pyramids.netcdf.ugrid.engines.vector._MeshVector.contour: Spatial contour
                extraction from the same variable.

        """
        var = self._ds._facade_variable("plot_histogram", variable_name)
        result = _analysis.plot_histogram(
            np.asarray(var.data), var.nodata, bins=bins, ax=ax, **kwargs
        )
        return result

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

        The mesh counterpart of :meth:`pyramids.dataset.Dataset.plot_vector_field`, sharing
        its ``kind`` / ``thin`` vocabulary so the two read the same. A hydrodynamic mesh
        writes its velocity as two per-face variables -- D-Flow FM names them ``mesh2d_ucx``
        and ``mesh2d_ucy`` -- and this draws them as arrows coloured by
        ``hypot(u, v)``.

        Both components must sit on the same location, and that location must be ``"face"``
        or ``"node"``: those are what a vector can be anchored at (face centroids, or the
        nodes themselves). An edge-located component such as D-Flow FM's ``mesh2d_u1`` is
        refused rather than drawn in the wrong place -- resolve it to faces first.

        A temporal component is reduced to one step, as :meth:`plot` does, except that the
        step is selectable here through ``time_index`` rather than fixed at the first.

        Two deliberate differences from :meth:`plot`, both inherited from matching the raster
        :meth:`~pyramids.dataset.Dataset.plot_vector_field` instead:

        - It returns matplotlib's ``(fig, ax, im)`` rather than the ``MeshGlyph``. The glyph
          is built internally and not handed back, so a follow-up ``apply_style`` or a second
          render onto the same glyph is not available here -- compose through ``ax=`` instead,
          passing the axes from a previous render.
        - There is no ``basemap=``. :meth:`plot` accepts one and validates the dataset's CRS
          first; here a ``basemap=`` keyword falls through to cleopatra, which refuses it as
          an unknown option.
        - ``kind="streamplot"`` is not available. The raster method documents it alongside
          ``quiver`` and ``barbs``; streamlines need a structured grid, so cleopatra refuses
          it with ``kind='streamplot' is not supported on an unstructured mesh``. The shared
          vocabulary is therefore ``quiver`` and ``barbs`` only.

        Args:
            u: Name of the variable holding the first vector component.
            v: Name of the variable holding the second vector component.
            kind: ``"quiver"`` for arrows or ``"barbs"`` for wind barbs. Defaults to
                ``"quiver"``.
            thin: Draw every nth element, keeping a large mesh legible. ``None`` (the
                default) draws every one.
            time_index: Which step to draw for a temporal component. Negative values
                count from the end, as they do in numpy. Defaults to ``0``.
            ax: Existing matplotlib axes to draw on. Defaults to ``None``.
            title: Plot title. Defaults to ``None``, which labels the plot ``"u / v"``.
            colorbar: ``True``/``False``, or a :class:`ColorBar` spec. Defaults to ``None``,
                which leaves cleopatra's default.
            **kwargs: Forwarded to cleopatra's ``MeshGlyph.plot_vector_field``.

        Returns:
            tuple: ``(fig, ax, im)``, as the raster ``plot_vector_field`` returns.

        Raises:
            KeyError: Either name is not a variable of this dataset.
            OptionalPackageDoesNotExist: cleopatra is not installed, or is older than the
                0.44.0 that provides ``MeshGlyph.plot_vector_field``.
            ValueError: A loose colour-bar keyword the :class:`ColorBar` spec replaces was
                passed (``cbar_label``, ``ticks_spacing``, ...); ``kind``, ``location`` or
                ``thin`` is not one cleopatra accepts, ``kind="streamplot"`` among them --
                the glyph validates all three, so that message is cleopatra's; a component
                has no loaded data; the two sit on different locations;
                the shared location is not ``"face"`` or ``"node"``; the two have different
                shapes; or ``time_index`` is outside a temporal component's time axis.

        Examples:
            - Draw a D-Flow FM velocity field, thinned so the arrows stay readable:
                ```python
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> ds = MeshDataset.read_file("FlowFM_map.nc")       # doctest: +SKIP
                >>> fig, ax, im = ds.glyph.plot_vector_field(          # doctest: +SKIP
                ...     "mesh2d_ucx", "mesh2d_ucy", kind="quiver", thin=4
                ... )
                >>> im.N < ds.n_face                                   # doctest: +SKIP
                True

                ```
            - Draw the last step of a temporal component as wind barbs:
                ```python
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> ds = MeshDataset.read_file("FlowFM_map.nc")       # doctest: +SKIP
                >>> fig, ax, im = ds.glyph.plot_vector_field(          # doctest: +SKIP
                ...     "mesh2d_ucx", "mesh2d_ucy", kind="barbs", time_index=-1
                ... )

                ```
            - Compose a vector field over an existing mesh plot by reusing its axes:
                ```python
                >>> from pyramids.netcdf.ugrid import MeshDataset
                >>> ds = MeshDataset.read_file("FlowFM_map.nc")       # doctest: +SKIP
                >>> glyph = ds.glyph.plot("mesh2d_waterdepth")         # doctest: +SKIP
                >>> fig, ax, im = ds.glyph.plot_vector_field(          # doctest: +SKIP
                ...     "mesh2d_ucx", "mesh2d_ucy", ax=glyph.ax
                ... )

                ```
        """
        u_values, location = self._ds._vector_component(u, "u", time_index)
        v_values, v_location = self._ds._vector_component(v, "v", time_index)
        if location != v_location:
            raise ValueError(
                f"plot_vector_field: {u!r} is {location}-located and {v!r} is "
                f"{v_location}-located; a vector needs both components at the same place."
            )
        if u_values.shape != v_values.shape:
            raise ValueError(
                f"plot_vector_field: {u!r} has shape {u_values.shape} and {v!r} has "
                f"{v_values.shape}; the two components must match."
            )
        result = _ugrid_plot.plot_vector_field_data(
            mesh=self._ds._mesh,
            u=u_values,
            v=v_values,
            location=location,
            kind=kind,
            thin=thin,
            ax=ax,
            title=f"{u} / {v}" if title is None else title,
            colorbar=colorbar,
            **kwargs,
        )
        # plot_vector_field_data returns the (fig, ax, im) tuple from
        # MeshGlyph.plot_vector_field; no glyph is exposed and no basemap is drawn, so
        # there is no CRS to tag (unlike plot / plot_outline, see _glyph_with_crs). If
        # this ever hands back a glyph or composes a basemap, route it through
        # _glyph_with_crs or issue #1560 reopens on this surface.
        return result

    def animate(
        self,
        variable_name: str | None = None,
        *,
        interval: int = 200,
        backend: str = "matplotlib",
        **kwargs: Any,
    ) -> Any:
        """Animate a temporal mesh variable over its time axis (issue #1380).

        The mesh had no animation at all, although a mesh time series is one of the most
        natural things to animate — a flood propagating across an unstructured grid. The
        frames come from cleopatra's :meth:`MeshGlyph.animate`, so this is wiring rather than
        new drawing code: the topology stays fixed and each step is drawn on it in turn.

        Args:
            variable_name: Variable to animate (default: the only temporal one).
            interval: Milliseconds between frames. Defaults to 200.
            backend: Renderer backend. ``"matplotlib"`` (default) animates through cleopatra;
                any other name routes to a ``pyramids.plot_backends`` plugin (in practice
                ``digitalearth``) with ``animate=True`` and returns that backend's own object.
                Requires the backend's package installed.
            **kwargs: Forwarded to cleopatra's ``MeshGlyph.animate`` — ``colorbar``, ``color``,
                ``contour``, ``data_style``, ``edgecolor``, ``text_loc``, …

        Returns:
            matplotlib.animation.FuncAnimation: The animation — save it with
            ``anim.save("out.gif")``. Note this differs from :meth:`Dataset.animate` and
            :meth:`NetCDF.animate`, which return a glyph whose ``anim`` attribute holds the
            animation: cleopatra's mesh and array glyph families return different objects and
            pyramids passes each through rather than wrapping one to look like the other.
            **Keep a reference to it** — matplotlib warns and stops an animation that is only
            a temporary, so ``anim = ds.glyph.animate()`` plays where a bare
            ``ds.glyph.animate()`` may not.

        Raises:
            ValueError: The variable has no time dimension (there would be one frame), or it
                has a dimension besides time and the elements — a layered variable needs a
                layer picked before it can be drawn.

        See Also:
            plot: A single frame.
            Dataset.animate: The raster counterpart, over bands.

        Examples:
            - Animate a water-depth series and save it:
                ```python
                >>> anim = ds.glyph.animate("mesh2d_waterdepth", interval=120)  # doctest: +SKIP
                >>> anim.save("depth.gif")  # doctest: +SKIP
                ```
        """
        # Delegates to the dataset: the body reads the per-variable time coordinate through
        # `pyramids.netcdf.ugrid.dataset._variable_time_coord`, a module-level helper this
        # engine cannot import without a cycle (`dataset` imports this engine for `glyph`).
        return self._ds._animate_impl(
            variable_name, interval=interval, backend=backend, **kwargs
        )

    def plot_outline(
        self,
        ax: Any = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot mesh wireframe.

        Args:
            ax: matplotlib Axes. Created if None.
            basemap: ``True`` or a web-tile provider name, drawing a tile backdrop under the
                wireframe — the same backdrop :meth:`plot` draws under a filled mesh. ``None``
                (default) draws none. The dataset's own ``epsg`` supplies the CRS unless
                ``basemap_epsg`` is passed explicitly, so a mesh with no CRS raises rather
                than mis-placing the tiles (issue #1354).
            **kwargs: Additional arguments passed to plot_mesh_outline.

        Returns:
            cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph instance with the wireframe
                rendered. ``glyph.fig`` / ``glyph.ax`` are the matplotlib
                handles; ``glyph.im`` is ``None`` (an outline carries no
                scalar mapping, so no mappable is produced). The glyph carries
                the dataset CRS (the EPSG code, or the CRS WKT when there is no
                EPSG) so a downstream ``glyph.add_tiles()`` projects correctly.
        """
        # `basemap_epsg` is a documented parameter of `plot_mesh_outline`, so passing it
        # here used to collide with the dataset's own CRS ("got multiple values for keyword
        # argument"). An explicit one wins; otherwise the dataset's CRS supplies it.
        kwargs.setdefault("basemap_epsg", self._ds.epsg)
        kwargs.setdefault("basemap_crs_wkt", self._ds.crs_wkt)
        result = _ugrid_plot.plot_mesh_outline(
            self._ds._mesh,
            ax=ax,
            basemap=basemap,
            **kwargs,
        )
        return self._ds._glyph_with_crs(result)

    def to_image(
        self,
        variable_name: str | None = None,
        cell_size: float | None = None,
        cmap: str = "viridis",
        method: str = "nearest",
    ) -> Any:
        """Export a mesh variable as a colour-mapped RGB image (issue #1378).

        An image is a grid of pixels, so the mesh is rasterised first and the raster's own
        :meth:`Dataset.analysis.to_image <pyramids.dataset.engines.analysis._Analysis.to_image>`
        does the colour mapping — which keeps the colours going
        through cleopatra, as all pyramids rendering does, and keeps one implementation of the
        mapping rather than two that can drift.

        Args:
            variable_name: Variable to render (default: the only one).
            cell_size: Pixel size in coordinate units. ``None`` (default) picks one that puts
                roughly 512 pixels across the mesh's longer side — a readable thumbnail
                without asking the caller to work out a resolution for a mesh whose units
                they may not know.
            cmap: Matplotlib colormap name.
            method: Rasterisation method, ``"nearest"`` (default) or ``"linear"``.

        Returns:
            PIL.Image.Image: The colour-mapped image. The pixels the rasterisation leaves
            empty — those outside the mesh footprint — are no-data, and so are rendered with
            the colormap's "bad" fill colour, exactly as the raster version renders them.
            Requires the ``viz`` extra.

        Raises:
            ValueError: The variable is not one value per element, or no variable is selected
                and the dataset holds more than one.

        See Also:
            pyramids.netcdf.ugrid.dataset.MeshDataset.to_dataset: The rasterisation this
                builds on, when you want the grid itself.
            plot: An annotated figure rather than a bare image.
        """
        var = self._ds._facade_variable("to_image", variable_name)
        if var.has_time:
            raise ValueError(
                f"to_image() needs one value per element, but {var.name!r} is a time series "
                f"of {var.n_time_steps} steps. Select a step first, e.g. with isel(time=0)."
            )
        if cell_size is None:
            xmin, ymin, xmax, ymax = self._ds.bounds
            # ~512 px across the longer side: a readable thumbnail without making the caller
            # reason about a resolution in coordinate units they may not know.
            cell_size = max(xmax - xmin, ymax - ymin) / 512.0
        grid = self._ds.io.to_dataset(var.name, cell_size, method=method)
        return grid.analysis.to_image(cmap=cmap)
