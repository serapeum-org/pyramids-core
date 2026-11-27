"""Variables engine for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset` (``mesh.variables``).

Owns the data-variable editing family -- ``with_variable`` / ``attach_data`` / ``rename_variable``
/ ``drop_variables``. A UGRID derivation is immutable, so each of these returns a *new* dataset
rather than mutating in place: this engine builds the replacement ``{name: variable}`` mapping and
hands it to ``MeshDataset._rebuild``.

``attach_data`` delegates back to ``MeshDataset._attach_data_impl`` rather than moving its body
here: that body reads a results-only file through the module-level ``_require_existing_file`` /
``_resolve_attach_driver`` / ``_read_data_through_driver`` helpers in ``dataset.py``, which this
module cannot import without a cycle (``dataset`` imports this engine for its ``variables``
accessor). The engine method delegates to it, exactly as ``mesh.glyph.animate`` delegates to
``MeshDataset._animate_impl``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from pyramids.netcdf.ugrid.engines._base import _MeshEngine
from pyramids.netcdf.ugrid.models import MeshVariable

if TYPE_CHECKING:
    from pathlib import Path

    from pyramids.netcdf.ugrid.dataset import MeshDataset


class _MeshVariables(_MeshEngine):
    """Data-variable editing for a mesh: with_variable, attach_data, rename_variable, drop_variables (``mesh.variables``)."""

    __slots__ = ()

    def with_variable(
        self,
        name: str,
        data: np.ndarray,
        *,
        location: str = "face",
        nodata: float | None = None,
    ) -> MeshDataset:
        """Return a new dataset with ``name`` added or replaced.

        UGRID derivations are immutable — unlike ``NetCDF.set_variable`` this does not mutate
        in place but returns a fresh dataset. A 2-D ``data`` array is treated as temporal
        (leading axis = time); a 1-D array is a static per-element variable.

        Args:
            name: Variable name to add or replace.
            data: The values; ``(n_elements,)`` static or ``(n_time, n_elements)`` temporal.
            location: Mesh location — ``"face"``, ``"node"`` or ``"edge"``. Defaults to
                ``"face"``.
            nodata: Optional no-data value for the new variable.

        Returns:
            MeshDataset: A new dataset carrying the variable.
        """
        arr = np.asarray(data)
        variable = MeshVariable(
            name=name,
            location=location,
            mesh_name=self._ds.mesh_name,
            shape=arr.shape,
            nodata=nodata,
            _data=arr,
        )
        new_vars = dict(self._ds._data_variables)
        new_vars[name] = variable
        return self._ds._rebuild(new_vars)

    def attach_data(
        self, path: str | Path, driver: str | None = None, lazy: bool = True
    ) -> MeshDataset:
        """Attach a results-only file's variables to this mesh.

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

        Delegates to :meth:`~pyramids.netcdf.ugrid.dataset.MeshDataset._attach_data_impl`: the
        body reads the file through module-level helpers in ``dataset.py`` that this engine
        cannot import without a cycle.

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
        return self._ds._attach_data_impl(path, driver, lazy)

    def drop_variables(self, names: str | list[str]) -> MeshDataset:
        """Return a new dataset without the named variable(s).

        Args:
            names: A variable name, or a list of them, to drop.

        Returns:
            MeshDataset: A new dataset without those variables.

        Raises:
            KeyError: A named variable is not present.
        """
        drop = [names] if isinstance(names, str) else list(names)
        new_vars = dict(self._ds._data_variables)
        for name in drop:
            if name not in new_vars:
                raise KeyError(
                    f"Variable {name!r} not found. Available: {self._ds.variable_names}"
                )
            del new_vars[name]
        return self._ds._rebuild(new_vars)

    def rename_variable(self, old: str, new: str) -> MeshDataset:
        """Return a new dataset with variable ``old`` renamed to ``new``.

        Args:
            old: The current variable name.
            new: The new name.

        Returns:
            MeshDataset: A new dataset with the variable renamed.

        Raises:
            KeyError: ``old`` is not present.
            ValueError: ``new`` already names a different variable.
        """
        if old not in self._ds._data_variables:
            raise KeyError(
                f"Variable {old!r} not found. Available: {self._ds.variable_names}"
            )
        if new in self._ds._data_variables and new != old:
            raise ValueError(f"Variable {new!r} already exists.")
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._ds._data_variables.items():
            if name == old:
                # `replace` keeps the lazy loader and every other field, only re-labelling
                # the variable, so a rename never forces a read.
                new_vars[new] = replace(var, name=new)
            else:
                new_vars[name] = var
        return self._ds._rebuild(new_vars)
