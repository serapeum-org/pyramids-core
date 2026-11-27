"""Engine accessors for :class:`~pyramids.netcdf.ugrid.dataset.MeshDataset`.

Mirrors the raster engine-accessor convention (issue #1631, implemented for
``Dataset``/``NetCDF``/``DatasetCollection`` in #1647) on the unstructured-mesh side: the
public mesh API is grouped onto ``mesh.<engine>.<method>`` accessors, each a fresh engine
instance bound by a strong reference to the owning :class:`MeshDataset`. The historical flat
methods remain as thin ``DeprecationWarning``-emitting facades that delegate here.

``MeshDataset`` is not a ``RasterBase`` subclass, so these engines cannot reuse the raster
``pyramids.dataset.engines._base._Engine`` (which is bound to ``RasterBase``); they share the
mesh-side :class:`~pyramids.netcdf.ugrid.engines._base._MeshEngine` base instead.
"""
