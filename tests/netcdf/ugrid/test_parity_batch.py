"""to_image / convert_units / to_dask_dataframe / open_mfdataset on a mesh (issue #1378).

The lower-priority parity batch: four members the raster and cube sides had and the mesh did
not. Each is built on something that already exists rather than reimplemented -- the
rasterisation for ``to_image``, the shared affine unit table for ``convert_units``, the tidy
frame for ``to_dask_dataframe``, and ``concat`` for ``open_mfdataset`` -- so these tests
concentrate on the mesh-specific wiring and the edges.
"""

from pathlib import Path

import numpy as np
import pytest

import pyramids.netcdf.ugrid.dataset as mesh_module
from pyramids.base._errors import DriverNotExistError, OptionalPackageDoesNotExist
from pyramids.netcdf.ugrid import MeshDataset
from pyramids.netcdf.ugrid._drivers import UnsupportedMeshFormatError
from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(values, units=None, nodata=None, location="face"):
    ds = MeshDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"t": np.asarray(values, dtype="float64")},
        data_locations={"t": location},
        epsg=32636,
    )
    if units is not None:
        ds["t"].units = units
    if nodata is not None:
        ds["t"].nodata = nodata
    return ds


class TestConvertUnits:
    """Values converted through the table the raster side already uses."""

    pytestmark = pytest.mark.core

    def test_kelvin_to_celsius(self):
        """273.15 K and 283.15 K are 0 and 10 degrees Celsius."""
        result = _mesh([273.15, 283.15], units="K").convert_units("celsius")
        assert result["t"].data.tolist() == pytest.approx([0.0, 10.0])

    def test_the_new_unit_is_recorded(self):
        result = _mesh([273.15, 283.15], units="K").convert_units("celsius")
        assert result["t"].units == "celsius"
        assert result["t"].attributes["units"] == "celsius"

    def test_the_source_is_left_untouched(self):
        source = _mesh([273.15, 283.15], units="K")
        source.convert_units("celsius")
        assert source["t"].data.tolist() == pytest.approx([273.15, 283.15])
        assert source["t"].units == "K"

    def test_nodata_keeps_its_sentinel(self):
        """Converting the sentinel would turn -9999 into a plausible-looking temperature."""
        ds = _mesh([273.15, -9999.0], units="K", nodata=-9999.0)
        result = ds.convert_units("celsius")
        assert result["t"].data.tolist() == pytest.approx([0.0, -9999.0])

    def test_a_variable_already_in_the_target_unit_passes_through(self):
        ds = _mesh([1.0, 2.0], units="celsius")
        assert ds.convert_units("celsius")["t"].data.tolist() == pytest.approx(
            [1.0, 2.0]
        )

    def test_naming_a_variable_with_no_unit_is_refused(self):
        """An explicit request must not pass silently."""
        subject = _mesh([1.0, 2.0])
        with pytest.raises(ValueError, match="declares no units"):
            subject.convert_units("celsius", variable_name="t")

    def test_a_sweep_skips_variables_with_no_unit(self):
        """Converting everything should not fail on a variable that has nothing to convert."""
        result = _mesh([1.0, 2.0]).convert_units("celsius")
        assert result["t"].data.tolist() == pytest.approx([1.0, 2.0])
        assert not result["t"].units

    def test_an_unsupported_pair_is_refused(self):
        subject = _mesh([1.0, 2.0], units="K")
        with pytest.raises(ValueError):
            subject.convert_units("metres")


class TestToImage:
    """Rasterise, then colour-map through the raster's own exporter."""

    pytestmark = pytest.mark.plot

    def test_it_returns_an_image_of_the_mesh(self):
        image = _mesh([1.0, 5.0]).to_image()
        assert image.size[0] > 1
        assert image.size[1] > 1

    def test_the_default_resolution_is_about_512_px_on_the_long_side(self):
        """So a caller need not reason about a cell size in unknown coordinate units."""
        image = _mesh([1.0, 5.0]).to_image()
        assert max(image.size) == pytest.approx(512, abs=2)

    def test_an_explicit_cell_size_sets_the_size(self):
        image = _mesh([1.0, 5.0]).to_image(cell_size=0.1)
        assert max(image.size) == pytest.approx(10, abs=1)

    def test_a_temporal_variable_is_refused_with_a_hint(self):
        """An image is one picture, so it needs one value per element."""
        ds = MeshDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"t": np.zeros((3, 2))},
            data_locations={"t": "face"},
            epsg=32636,
        )
        with pytest.raises(ValueError, match="isel"):
            ds.to_image()


class TestToDaskDataframe:
    """The lazy sibling of the tidy frame."""

    pytestmark = pytest.mark.core

    def test_it_returns_a_dask_dataframe(self):
        dd = pytest.importorskip("dask.dataframe")
        frame = _mesh([1.0, 2.0]).to_dask_dataframe()
        assert isinstance(frame, dd.DataFrame)

    def test_it_computes_to_the_same_rows_as_to_dataframe(self):
        """The lazy and eager paths must agree, or one of them is wrong."""
        pytest.importorskip("dask.dataframe")
        ds = _mesh([1.0, 2.0])
        eager = ds.to_dataframe()
        lazy = ds.to_dask_dataframe().compute()
        assert list(lazy.columns) == list(eager.columns)
        assert len(lazy) == len(eager)

    def test_partitions_are_honoured(self):
        pytest.importorskip("dask.dataframe")
        assert _mesh([1.0, 2.0]).to_dask_dataframe(npartitions=2).npartitions == 2

    def test_zero_partitions_is_refused(self):
        subject = _mesh([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 1"):
            subject.to_dask_dataframe(npartitions=0)


class TestOpenMfdataset:
    """Many files, one topology, stacked along time."""

    pytestmark = pytest.mark.core

    @staticmethod
    def _write_run(directory, dated):
        """Three single-step files whose names sort WRONG lexically: t10 before t2."""
        paths = []
        for label, day in (("t9", 9), ("t10", 10), ("t2", 2)):
            ds = MeshDataset.from_arrays(
                node_x=_NODE_X,
                node_y=_NODE_Y,
                face_node_connectivity=_TRIANGLES,
                data={"d": np.array([[float(day), float(day) * 2]])},
                data_locations={"d": "face"},
            )
            ds["d"].dimensions = ("time", "mesh2d_nFaces")
            if dated:
                ds["d"].attributes = {
                    **ds["d"].attributes,
                    "time_values": [np.datetime64(f"2001-01-{day:02d}")],
                    TIME_UNITS_ATTR: "days since 2001-01-01",
                    TIME_CALENDAR_ATTR: "standard",
                }
            path = directory / f"run_{label}.nc"
            ds.to_file(str(path))
            paths.append(path)
        return paths

    def test_a_glob_stacks_every_file(self, tmp_path):
        self._write_run(tmp_path, dated=True)
        stacked = MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"))
        assert stacked["d"].n_time_steps == 3

    def test_a_dated_run_comes_back_in_time_order_not_lexical(self, tmp_path):
        """The reason the sort exists: `run_t10` sorts before `run_t2` as text."""
        self._write_run(tmp_path, dated=True)
        stacked = MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"))
        assert stacked["d"].data[:, 0].tolist() == [2.0, 9.0, 10.0]

    def test_sort_false_keeps_the_given_order(self, tmp_path):
        self._write_run(tmp_path, dated=True)
        stacked = MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), sort=False)
        assert stacked["d"].data[:, 0].tolist() == [10.0, 2.0, 9.0]

    def test_undated_files_keep_their_given_position(self, tmp_path):
        """Positional indices say nothing about order, so the sort must not pretend they do -
        a caller with undated files has to pass an ordered sequence."""
        self._write_run(tmp_path, dated=False)
        stacked = MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"))
        assert stacked["d"].data[:, 0].tolist() == [10.0, 2.0, 9.0]

    def test_an_explicit_sequence_is_stacked_in_that_order(self, tmp_path):
        paths = self._write_run(tmp_path, dated=False)
        ordered = [paths[2], paths[0], paths[1]]  # t2, t9, t10
        stacked = MeshDataset.open_mfdataset([str(p) for p in ordered], sort=False)
        assert stacked["d"].data[:, 0].tolist() == [2.0, 9.0, 10.0]

    def test_a_single_path_is_accepted(self, tmp_path):
        paths = self._write_run(tmp_path, dated=True)
        assert MeshDataset.open_mfdataset(str(paths[0]))["d"].n_time_steps == 1

    def test_a_pattern_matching_nothing_is_refused(self, tmp_path):
        """Rather than answering an empty dataset that looks like a successful read."""
        with pytest.raises(ValueError, match="matches no file"):
            MeshDataset.open_mfdataset(str(tmp_path / "absent_*.nc"))

    def test_an_empty_sequence_is_refused(self):
        with pytest.raises(ValueError, match="nothing to open"):
            MeshDataset.open_mfdataset([])

    def test_a_file_on_a_different_topology_is_refused(self, tmp_path):
        self._write_run(tmp_path, dated=True)
        other = MeshDataset.from_arrays(
            node_x=np.array([0.0, 9.0, 9.0]),
            node_y=np.array([0.0, 0.0, 9.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"d": np.array([[1.0]])},
            data_locations={"d": "face"},
        )
        other["d"].dimensions = ("time", "mesh2d_nFaces")
        other.to_file(str(tmp_path / "other.nc"))
        with pytest.raises(Exception, match="(?i)mesh|topolog|align"):
            MeshDataset.open_mfdataset(str(tmp_path / "*.nc"))

    def test_a_mixed_set_keeps_undated_files_in_place(self, tmp_path):
        """Review M3: the docstring promises an undated file "keeps its given position even
        with sort=True", but every dated file sorted before every undated one.

        Passed as [undated, dated], the undated file must stay first.
        """
        undated = tmp_path / "zz_undated.nc"
        ds = MeshDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.array([[1.0, 2.0]])},
            data_locations={"d": "face"},
        )
        ds["d"].dimensions = ("time", "mesh2d_nFaces")
        ds.to_file(str(undated))

        dated = tmp_path / "aa_dated.nc"
        stamped = MeshDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.array([[9.0, 8.0]])},
            data_locations={"d": "face"},
        )
        stamped["d"].dimensions = ("time", "mesh2d_nFaces")
        stamped["d"].attributes = {
            **stamped["d"].attributes,
            "time_values": [np.datetime64("2001-01-01")],
            TIME_UNITS_ATTR: "days since 2001-01-01",
            TIME_CALENDAR_ATTR: "standard",
        }
        stamped.to_file(str(dated))

        stacked = MeshDataset.open_mfdataset([str(undated), str(dated)])
        assert stacked["d"].data[:, 0].tolist() == [1.0, 9.0], (
            "the undated file keeps its given position, as the docstring promises"
        )

    def test_a_mixed_set_still_orders_the_dated_files_among_themselves(self, tmp_path):
        """Sorting the dated subset is the point of sort=True; only the undated ones hold."""
        paths = []
        for label, day in (("t9", 9), ("t2", 2)):
            ds = MeshDataset.from_arrays(
                node_x=_NODE_X,
                node_y=_NODE_Y,
                face_node_connectivity=_TRIANGLES,
                data={"d": np.array([[float(day), 0.0]])},
                data_locations={"d": "face"},
            )
            ds["d"].dimensions = ("time", "mesh2d_nFaces")
            ds["d"].attributes = {
                **ds["d"].attributes,
                "time_values": [np.datetime64(f"2001-01-{day:02d}")],
                TIME_UNITS_ATTR: "days since 2001-01-01",
                TIME_CALENDAR_ATTR: "standard",
            }
            path = tmp_path / f"dated_{label}.nc"
            ds.to_file(str(path))
            paths.append(str(path))
        stacked = MeshDataset.open_mfdataset(paths)
        assert stacked["d"].data[:, 0].tolist() == [2.0, 9.0], (
            "dated files sort by stamp"
        )

    def test_chunks_none_matches_the_default_eager_read(self, tmp_path):
        """Passing ``chunks=None`` explicitly is the documented eager default.

        Test scenario:
            The default and the explicit ``None`` must take the same branch, so adding the
            keyword cannot change what existing callers get.
        """
        self._write_run(tmp_path, dated=True)
        pattern = str(tmp_path / "run_*.nc")

        default = MeshDataset.open_mfdataset(pattern)
        explicit = MeshDataset.open_mfdataset(pattern, chunks=None)

        assert type(explicit) is type(default), (
            f"chunks=None changed the return type: {type(explicit)!r} vs {type(default)!r}"
        )
        assert explicit["d"].n_time_steps == default["d"].n_time_steps, (
            "chunks=None returned a different number of time steps than the default"
        )

    def test_a_lazy_read_forwards_resolved_paths_and_options(
        self, tmp_path, monkeypatch
    ):
        """``chunks`` hands the *resolved* paths and every option to the backend.

        Test scenario:
            The glob is expanded in core so the backend never re-implements it, and the
            backend's return value is passed through untouched.
        """
        paths = self._write_run(tmp_path, dated=True)
        captured = {}
        sentinel = object()

        class _Backend:
            def open_mesh_mfdataset(self, resolved, **kwargs):
                captured["resolved"] = list(resolved)
                captured["kwargs"] = kwargs
                return sentinel

        monkeypatch.setattr(mesh_module, "require_compute_backend", lambda: _Backend())

        got = MeshDataset.open_mfdataset(
            str(tmp_path / "run_*.nc"), chunks="auto", mesh_name="mesh2d"
        )

        assert got is sentinel, "the backend's return value was not passed through"
        assert captured["resolved"] == sorted(str(p) for p in paths), (
            f"the backend got unresolved or unsorted paths: {captured['resolved']!r}"
        )
        assert captured["kwargs"] == {
            "chunks": "auto",
            "driver": None,
            "mesh_name": "mesh2d",
            "sort": True,
        }, f"options were not forwarded verbatim: {captured['kwargs']!r}"

    def test_a_lazy_read_without_a_backend_is_branded(self, tmp_path, monkeypatch):
        """No backend installed surfaces the seam's own branded error, not an AttributeError."""
        self._write_run(tmp_path, dated=True)

        def _absent():
            raise OptionalPackageDoesNotExist("no backend installed")

        monkeypatch.setattr(mesh_module, "require_compute_backend", _absent)

        with pytest.raises(OptionalPackageDoesNotExist, match="no backend installed"):
            MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks="auto")

    def test_a_backend_without_the_method_asks_for_an_upgrade(
        self, tmp_path, monkeypatch
    ):
        """A backend predating ``open_mesh_mfdataset`` is named as such, with the fix.

        Test scenario:
            Core probes the method with ``getattr`` rather than declaring it on the
            ``ComputeBackend`` protocol, so an older plugin stays usable for everything else
            and only this call fails -- with an actionable message.
        """
        self._write_run(tmp_path, dated=True)

        class _OldBackend:
            pass

        monkeypatch.setattr(
            mesh_module, "require_compute_backend", lambda: _OldBackend()
        )

        with pytest.raises(OptionalPackageDoesNotExist) as excinfo:
            MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks="auto")

        message = str(excinfo.value)
        assert "open_mesh_mfdataset" in message, (
            f"the missing member is not named: {message!r}"
        )
        assert "pip install -U pyramids-lazy" in message, (
            f"the upgrade command is missing: {message!r}"
        )

    def test_the_builtin_driver_name_is_accepted_on_the_lazy_path(
        self, tmp_path, monkeypatch
    ):
        """`ugrid` names the built-in reader, so validating must not reject it (review M4).

        Test scenario:
            RESERVED_NAMES exists precisely so asking for the built-in reader by name does not
            raise DriverNotExistError. The pre-branch validation has to honour that, or it
            breaks a name the eager path accepts.
        """
        self._write_run(tmp_path, dated=True)
        captured = {}

        class _Backend:
            def open_mesh_mfdataset(self, resolved, **kwargs):
                captured["driver"] = kwargs.get("driver")
                return "lazy"

        monkeypatch.setattr(mesh_module, "require_compute_backend", lambda: _Backend())

        got = MeshDataset.open_mfdataset(
            str(tmp_path / "run_*.nc"), chunks="auto", driver="ugrid"
        )

        assert got == "lazy", (
            f"the built-in driver name was rejected or not forwarded: {got!r}"
        )
        assert captured["driver"] == "ugrid", (
            f"driver was not forwarded verbatim: {captured.get('driver')!r}"
        )

    @pytest.mark.parametrize(
        ("bad_input", "error", "match"),
        [
            ({"paths": "absent_*.nc"}, ValueError, "matches no file"),
            ({"paths": []}, ValueError, "nothing to open"),
            ({"paths": "absent.nc"}, FileNotFoundError, "File not found"),
            (
                {"paths": "RUN_GLOB", "driver": "no_such"},
                DriverNotExistError,
                "no_such",
            ),
        ],
        ids=["bad-glob", "empty-sequence", "missing-file", "unknown-driver"],
    )
    def test_invalid_input_is_refused_without_consulting_the_backend(
        self, tmp_path, monkeypatch, bad_input, error, match
    ):
        """Every input core can judge is rejected before any backend is resolved.

        Args:
            bad_input: Keyword arguments naming one invalid-input shape.
            error: The exception the eager path already raises for that shape.
            match: A fragment of that exception's message.

        Test scenario:
            The resolver is replaced with one that fails loudly, so consulting it is an error
            rather than a different error. Two of these shapes (a glob matching nothing, an
            empty sequence) cannot reach the branch however it is written -- they are here to
            document the shape, not to discriminate. The other two can: before the paths and
            the driver were validated in the shared block, both reported "upgrade
            pyramids-lazy" for what is really a typo (review M3/M4).
        """
        self._write_run(tmp_path, dated=True)
        kwargs = dict(bad_input)
        if kwargs.get("paths") == "RUN_GLOB":
            kwargs["paths"] = str(tmp_path / "run_*.nc")
        elif isinstance(kwargs["paths"], str):
            kwargs["paths"] = str(tmp_path / kwargs["paths"])

        def _boom():
            raise AssertionError(
                "the backend was resolved for input core could already reject"
            )

        monkeypatch.setattr(mesh_module, "require_compute_backend", _boom)

        with pytest.raises(error, match=match):
            MeshDataset.open_mfdataset(chunks="auto", **kwargs)

    @pytest.mark.lazy
    def test_the_real_resolver_never_leaks_a_raw_attribute_error(self, tmp_path):
        """Through the real seam, the probe yields a mesh or a branded error -- never AttributeError.

        Test scenario:
            Every other lazy test replaces `require_compute_backend`, so none covers the
            integration with entry-point discovery and the api_version gate. Asserting
            "raises OptionalPackageDoesNotExist" would invert the day the plugin ships
            `open_mesh_mfdataset`, so this asserts the invariant that holds either way
            (review L4).
        """
        self._write_run(tmp_path, dated=True)

        try:
            result = MeshDataset.open_mfdataset(
                str(tmp_path / "run_*.nc"), chunks="auto"
            )
        except OptionalPackageDoesNotExist as exc:
            assert "open_mesh_mfdataset" in str(exc), (
                f"the branded error should name the missing member, got: {exc}"
            )
        else:
            assert result is not None, (
                "a backend that serves chunks= must return something"
            )

    @pytest.mark.parametrize(
        "bad", [False, 0, "", {}], ids=["False", "0", "empty-str", "empty-dict"]
    )
    def test_a_meaningless_chunks_value_is_refused(self, tmp_path, monkeypatch, bad):
        """A falsy `chunks` is rejected rather than silently routed to the backend.

        Args:
            bad: A falsy value that is not `None`.

        Test scenario:
            The switch is `chunks is not None`, so every falsy non-None value used to dispatch
            lazily -- `chunks=False` reads as "do not chunk" and did the opposite, and `0` /
            `False` are not even in the declared type (review N3).
        """
        self._write_run(tmp_path, dated=True)

        def _boom():
            raise AssertionError("a meaningless chunks value reached the backend")

        monkeypatch.setattr(mesh_module, "require_compute_backend", _boom)

        with pytest.raises(ValueError, match="chunks"):
            MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks=bad)

    @pytest.mark.parametrize(
        "awkward",
        [np.array([1, 2]), np.array([])],
        ids=["ndarray-many", "ndarray-empty"],
    )
    def test_an_argument_whose_truthiness_raises_is_forwarded_not_mangled(
        self, tmp_path, monkeypatch, awkward
    ):
        """A value whose `__bool__` raises reaches the backend instead of a numpy error.

        Args:
            awkward: A value that raises when its truth value is taken.

        Test scenario:
            The falsy guard was `if chunks is not None and not chunks`, which evaluates
            truthiness on whatever the caller passed. An ndarray therefore produced "The truth
            value of an array with more than one element is ambiguous. Use a.any() or a.all()" --
            a message naming neither `chunks` nor `open_mfdataset`, advising something the caller
            cannot act on. Before the guard existed the value was simply forwarded, which is what
            should still happen for anything the guard cannot judge (Round 2 finding L3).
        """
        self._write_run(tmp_path, dated=True)
        captured: dict = {}

        class _Backend:
            def open_mesh_mfdataset(self, resolved, **kwargs):
                captured.update(kwargs)
                return "LAZY"

        monkeypatch.setattr(mesh_module, "require_compute_backend", lambda: _Backend())

        got = MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks=awkward)

        assert got == "LAZY", (
            f"an unjudgeable chunks value should be forwarded, got {got!r}"
        )
        assert captured["chunks"] is awkward, (
            "the value should reach the backend unchanged for it to accept or reject"
        )

    @pytest.mark.parametrize("sort", [True, False], ids=["sort-true", "sort-false"])
    def test_sort_reaches_the_backend_so_it_can_honour_it(
        self, tmp_path, monkeypatch, sort
    ):
        """`sort` is forwarded, because the backend is the layer that can apply it.

        Args:
            sort: The value the caller passes.

        Test scenario:
            Core used to drop `sort` on the lazy path, on the premise that a backend could
            not reproduce its ordering. pyramids-lazy#26 disproves that premise: it opens
            every file and joins `time_values` whether or not it sorts, so reordering those
            already-open handles costs no extra read. Dropping the flag made
            `sort=False` a silent no-op -- the backend's own default re-sorted the files
            anyway -- so a caller asking for file order got chronological order.
        """
        self._write_run(tmp_path, dated=True)
        captured: dict = {}

        class _Backend:
            def open_mesh_mfdataset(self, paths, **kwargs):
                captured.update(kwargs)
                return "LAZY"

        monkeypatch.setattr(mesh_module, "require_compute_backend", lambda: _Backend())

        MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks="auto", sort=sort)

        assert captured.get("sort") is sort, (
            f"sort={sort} must reach the backend; got {captured.get('sort', '<dropped>')!r}"
        )

    def test_a_backend_that_cannot_take_sort_is_refused_branded(
        self, tmp_path, monkeypatch
    ):
        """A backend missing the `sort=` argument gets a branded error, not a raw TypeError.

        Test scenario:
            The probe is presence-only, so a backend written against the four-argument contract
            this branch published before `sort` was forwarded passes `getattr` and then dies in
            Python's argument binder: `TypeError: ... unexpected keyword argument 'sort'`. That
            is the unbranded failure the seam exists to prevent, and it contradicts the helper's
            documented promise of a branded error (Round 2 finding M2).
        """
        self._write_run(tmp_path, dated=True)

        class _FourArgBackend:
            def open_mesh_mfdataset(self, paths, *, chunks, driver, mesh_name):
                return "LAZY"

        monkeypatch.setattr(
            mesh_module, "require_compute_backend", lambda: _FourArgBackend()
        )

        with pytest.raises(OptionalPackageDoesNotExist, match="sort"):
            MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks="auto")

    @pytest.mark.parametrize(
        "signature",
        ["var_keyword", "explicit_sort"],
        ids=["accepts-**kwargs", "declares-sort"],
    )
    def test_a_backend_that_can_take_sort_is_accepted(
        self, tmp_path, monkeypatch, signature
    ):
        """Both ways of accepting `sort` are allowed through the signature check.

        Args:
            signature: Which accepting shape the fake backend uses.

        Test scenario:
            The released pyramids-lazy forwards through `open_mesh_mfdataset(self, paths,
            **kwargs)`, so a check that merely looked for a parameter named `sort` would reject
            the real backend. A `**kwargs` catch-all must count as accepting, as must an
            explicitly declared `sort`.
        """
        self._write_run(tmp_path, dated=True)

        class _VarKeyword:
            def open_mesh_mfdataset(self, paths, **kwargs):
                return "LAZY"

        class _ExplicitSort:
            def open_mesh_mfdataset(self, paths, *, chunks, driver, mesh_name, sort):
                return "LAZY"

        backend = _VarKeyword() if signature == "var_keyword" else _ExplicitSort()
        monkeypatch.setattr(mesh_module, "require_compute_backend", lambda: backend)

        got = MeshDataset.open_mfdataset(str(tmp_path / "run_*.nc"), chunks="auto")

        assert got == "LAZY", f"a backend accepting sort= must be used, got {got!r}"

    def test_a_results_only_driver_is_refused_on_the_lazy_path_too(
        self, tmp_path, monkeypatch
    ):
        """A `needs_mesh` driver is refused before the split, as the eager path refuses it.

        Test scenario:
            `_named_read_driver` answers only "reserved, or installed?" -- it does not apply the
            `needs_mesh` refusal that `_resolve_read_driver` adds on the eager path. So a
            results-only driver (SMS `.dat`, XMDF: values but no topology) raised
            UnsupportedMeshFormatError eagerly and was forwarded verbatim to the backend
            lazily, while the published contract told plugin authors core had validated the
            driver. The check is a capability lookup on the named driver, so it costs no file
            read (Round 2 finding L1).
        """
        self._write_run(tmp_path, dated=True)

        class _ResultsOnly:
            name = "results_only"
            long_name = "Results Only Format"
            suffixes = (".dat",)
            capabilities = frozenset({"read", "read_data", "needs_mesh"})

        monkeypatch.setattr(
            mesh_module, "mesh_drivers", lambda: {"results_only": _ResultsOnly()}
        )

        def _boom():
            raise AssertionError("a needs_mesh driver must not reach the backend")

        monkeypatch.setattr(mesh_module, "require_compute_backend", _boom)

        with pytest.raises(
            UnsupportedMeshFormatError, match="carry values but no mesh"
        ):
            MeshDataset.open_mfdataset(
                str(tmp_path / "run_*.nc"), chunks="auto", driver="results_only"
            )

    @pytest.mark.parametrize(
        "spelling", ["glob", "path-sequence", "single-path", "mixed-sequence"]
    )
    def test_the_backend_always_receives_plain_strings(
        self, tmp_path, monkeypatch, spelling
    ):
        """However the caller spelled the paths, the backend gets `str`.

        Args:
            spelling: How the caller expresses the same two files.

        Test scenario:
            `resolved` was built three ways -- the glob branch produced `str`, while the
            sequence and single-path branches passed the caller's objects straight through. So
            the backend saw `['str', 'str']`, `['WindowsPath', ...]` or even a mixed list
            depending only on how the call was written, while the contract promised "a list of
            concrete paths". A backend keying a dict on the value, or calling
            `.startswith("s3://")`, would behave differently per spelling (Round 2 finding L2).
        """
        paths = self._write_run(tmp_path, dated=True)
        captured: dict = {}

        class _Backend:
            def open_mesh_mfdataset(self, resolved, **kwargs):
                captured["resolved"] = list(resolved)
                return "LAZY"

        monkeypatch.setattr(mesh_module, "require_compute_backend", lambda: _Backend())

        if spelling == "glob":
            given = str(tmp_path / "run_*.nc")
        elif spelling == "path-sequence":
            given = [Path(p) for p in paths]
        elif spelling == "single-path":
            given = Path(paths[0])
        else:
            given = [str(paths[0]), Path(paths[1])]

        MeshDataset.open_mfdataset(given, chunks="auto")

        kinds = sorted({type(p).__name__ for p in captured["resolved"]})
        assert kinds == ["str"], (
            f"the backend should receive only str, got {kinds} for the {spelling} spelling"
        )
