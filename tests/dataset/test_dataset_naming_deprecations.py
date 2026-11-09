"""Tests for the dataset-subpackage naming cleanup (PR #1497).

Covers two changes:

- **DS-2** ``to_raster`` -> ``to_file``. ``Dataset.to_raster`` and ``IO.to_raster`` were a redundant second name
  for :meth:`Dataset.to_file`; they keep working but emit a ``DeprecationWarning`` pointing at ``to_file``.
- **DS-4** ``DatasetCollection.first()`` / ``.last()`` now return a lazy :class:`Dataset` (matching ``iloc``)
  instead of a 2-D ndarray.
"""

import warnings

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.collection import DatasetCollection

_GEO = dict(top_left_corner=(0, 2), cell_size=1.0, epsg=4326)


def _value_raster() -> Dataset:
    """Return a 2x2 float32 value raster."""
    values = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
    return Dataset.from_array(values, geo_ref=GeoReference(**_GEO))


def _cube(time_length: int = 3, rows: int = 4, cols: int = 5):
    """Return a ``(DatasetCollection, values)`` pair with known per-timestep arrays.

    Args:
        time_length: Number of timesteps.
        rows: Raster height.
        cols: Raster width.

    Returns:
        tuple: The collection and the ``(time, rows, cols)`` value array it holds.
    """
    base = Dataset.from_array(
        np.zeros((rows, cols), dtype="float64"), geo_ref=GeoReference(**_GEO)
    )
    cube = DatasetCollection.from_dataset(base, time_length=time_length)
    values = np.arange(time_length * rows * cols, dtype="float64").reshape(
        time_length, rows, cols
    )
    cube.values = values
    return cube, values


class TestToRasterRenamedToToFile:
    """``to_raster`` is the deprecated alias; ``to_file`` is canonical."""

    def test_dataset_to_raster_warns_and_writes(self, tmp_path):
        """``Dataset.to_raster`` still writes the file but warns about ``to_file``.

        Test scenario:
            Calling the facade alias emits a ``DeprecationWarning`` whose message names
            ``to_file`` and still produces the output raster on disk.
        """
        out = tmp_path / "out.tif"
        with pytest.warns(DeprecationWarning, match="to_file"):
            _value_raster().to_raster(str(out))
        assert out.exists(), "to_raster must still write the raster"

    def test_io_engine_to_raster_warns_and_writes(self, tmp_path):
        """The engine-level ``IO.to_raster`` warns and writes too.

        Test scenario:
            A strong ref to the Dataset is kept (the ``io`` engine holds only a weakref);
            ``io.to_raster`` warns about ``to_file`` and writes the file.
        """
        out = tmp_path / "out2.tif"
        ds = _value_raster()
        with pytest.warns(DeprecationWarning, match="to_file"):
            ds.io.to_raster(str(out))
        assert out.exists(), "IO.to_raster must still write the raster"

    def test_to_file_does_not_warn(self, tmp_path):
        """The canonical ``to_file`` must not emit a ``DeprecationWarning``.

        Test scenario:
            With ``DeprecationWarning`` promoted to an error, ``to_file`` writes cleanly.
        """
        out = tmp_path / "clean.tif"
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _value_raster().to_file(str(out))
        assert out.exists(), "to_file must write the raster without warning"

    def test_to_raster_round_trips_the_values(self, tmp_path):
        """``to_raster`` writes the same data a round-trip through ``to_file`` would.

        Test scenario:
            Reading back a raster written via the deprecated ``to_raster`` yields the
            original array, identical to one written via ``to_file``.
        """
        src = _value_raster()
        expected = src.read_array(band=0)
        via_alias = tmp_path / "alias.tif"
        via_canonical = tmp_path / "canonical.tif"
        with pytest.warns(DeprecationWarning):
            src.to_raster(str(via_alias))
        _value_raster().to_file(str(via_canonical))
        got_alias = Dataset.read_file(str(via_alias)).read_array(band=0)
        got_canonical = Dataset.read_file(str(via_canonical)).read_array(band=0)
        np.testing.assert_array_equal(
            got_alias, expected, err_msg="to_raster round-trip changed the data"
        )
        np.testing.assert_array_equal(
            got_alias,
            got_canonical,
            err_msg="to_raster and to_file wrote different data",
        )

    def test_io_to_raster_forwards_kwargs_to_to_file(self, tmp_path, mocker):
        """``IO.to_raster`` forwards every argument to ``to_file`` unchanged.

        Test scenario:
            Patch ``IO.to_file`` and call ``io.to_raster`` with keyword arguments; the
            patched method must receive exactly those arguments.
        """
        ds = _value_raster()
        out = str(tmp_path / "fwd.tif")
        spy = mocker.patch.object(ds.io, "to_file", return_value="sentinel")
        with pytest.warns(DeprecationWarning):
            result = ds.io.to_raster(
                out, band=0, tile_length=256, creation_options=None, driver="GTiff"
            )
        assert result == "sentinel", "to_raster must return what to_file returns"
        spy.assert_called_once_with(
            out,
            band=0,
            tile_length=256,
            creation_options=None,
            driver="GTiff",
            compute=True,
            lock=None,
        )


class TestFirstLastReturnDataset:
    """``DatasetCollection.first()`` / ``.last()`` now return a ``Dataset`` (DS-4)."""

    def test_first_returns_a_dataset(self):
        """``first()`` returns a ``Dataset`` handle, not an ndarray.

        Test scenario:
            The returned object is a ``Dataset`` whose first band equals timestep 0.
        """
        cube, values = _cube()
        first = cube.first()
        assert isinstance(first, Dataset), (
            f"first() must return a Dataset, got {type(first)}"
        )
        np.testing.assert_array_equal(
            first.read_array(band=0),
            values[0],
            err_msg="first() does not expose timestep 0",
        )

    def test_last_returns_a_dataset(self):
        """``last()`` returns a ``Dataset`` handle for the final timestep.

        Test scenario:
            The returned object is a ``Dataset`` whose first band equals timestep -1.
        """
        cube, values = _cube()
        last = cube.last()
        assert isinstance(last, Dataset), (
            f"last() must return a Dataset, got {type(last)}"
        )
        np.testing.assert_array_equal(
            last.read_array(band=0),
            values[-1],
            err_msg="last() does not expose the final timestep",
        )

    def test_first_and_last_agree_with_iloc(self):
        """``first()``/``last()`` are consistent with ``iloc(0)``/``iloc(-1)``.

        Test scenario:
            The whole accessor family returns ``Dataset`` handles over the same
            timesteps, so ``first()`` matches ``iloc(0)`` and ``last()`` matches
            ``iloc(-1)`` cell for cell.
        """
        cube, _ = _cube()
        np.testing.assert_array_equal(
            cube.first().read_array(band=0),
            cube.iloc(0).read_array(band=0),
            err_msg="first() diverged from iloc(0)",
        )
        np.testing.assert_array_equal(
            cube.last().read_array(band=0),
            cube.iloc(-1).read_array(band=0),
            err_msg="last() diverged from iloc(-1)",
        )
