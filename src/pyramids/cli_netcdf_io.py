"""NetCDF I/O export, construction, and combine commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`: export methods — round-trip to a
NetCDF (`convert`), tabular export (`to-dataframe`), a structured dump (`to-dict`), a
per-variable Zarr store (`to-zarr`) and kerchunk reference manifests (`to-kerchunk`,
`combine-kerchunk`); construction from a CSV / JSON dump (`from-dataframe`, `from-dict`);
a lazy multi-file stack report (`open-mf`); and multi-cube joins (`concat`, `merge`). No
xarray is imported here — the underlying methods own that; the CLI only marshals paths
and scalar options. The kerchunk and `open-mf` verbs need the ``[lazy]`` optional extra,
imported lazily by the methods, so ``import pyramids.cli`` stays extra-free.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from pyramids._cli_shared import (
    info_print,
    json_sanitize,
    needs_lazy_extra,
    refuse_existing,
)
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_INPUTS = "two or more NetCDF paths"


def _cmd_nc_convert(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc convert`` — re-write a file as a NetCDF.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    NetCDF.read_file(args.file).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_to_dataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-dataframe`` — export variables to a CSV table.

    Args:
        args: Parsed args with ``file``, ``output``, ``vars``, ``dropna``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    frame = nc.to_dataframe(variables=args.vars or None, dropna=args.dropna)
    frame.to_csv(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_to_dict(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-dict`` — print the file's structure as a dict.

    ``--data`` includes the variable arrays (large); the default is structure and
    metadata only.

    Args:
        args: Parsed args with ``file``, ``json``, ``data``.

    Returns:
        int: ``0`` on success.
    """
    nc = NetCDF.read_file(args.file)
    payload = nc.to_dict(data=args.data)
    if args.json:
        print(json.dumps(json_sanitize(payload)))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_nc_to_zarr(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-zarr`` — write one variable to a Zarr store.

    Zarr is written per variable (the container itself is not spatial), so ``--var``
    selects the gridded variable to store.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` names a variable that is not a gridded raster, or the
            store already exists without ``--overwrite``.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    variable = nc.get_variable(args.var)
    if not hasattr(variable, "to_zarr"):
        raise ValueError(
            f"variable {args.var!r} is not exposable as a raster plane "
            f"(GDAL returned a labeled array); it cannot be written to Zarr."
        )
    variable.to_zarr(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_to_kerchunk(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc to-kerchunk`` — write a kerchunk reference manifest.

    Emits a JSON reference manifest for the file (its chunks addressed in place), written
    to ``output``. Needs the ``[lazy]`` extra.

    Args:
        args: Parsed args with ``file``, ``output``, ``inline_threshold``,
            ``vlen_encode``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with needs_lazy_extra():
        NetCDF.read_file(args.file).to_kerchunk(
            args.output,
            inline_threshold=args.inline_threshold,
            vlen_encode=args.vlen_encode,
        )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_combine_kerchunk(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc combine-kerchunk`` — a combined manifest over many files.

    Emits one kerchunk manifest spanning ``--inputs``, concatenated along ``--concat-dims``
    with ``--identical-dims`` shared, written to ``output``. Needs the ``[lazy]`` extra.

    Args:
        args: Parsed args with ``output``, ``inputs``, ``concat_dims``,
            ``identical_dims``, ``inline_threshold``, ``overwrite``.

    Returns:
        int: ``0`` on success.
    """
    refuse_existing(args.output, args.overwrite)
    with needs_lazy_extra():
        NetCDF.combine_kerchunk(
            args.inputs,
            args.output,
            concat_dims=tuple(args.concat_dims),
            identical_dims=tuple(args.identical_dims),
            inline_threshold=args.inline_threshold,
        )
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_open_mf(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc open-mf`` — report the lazy stack of a variable over files.

    Opens ``--paths`` and stacks ``--var`` into one lazy dask array, then reports the
    stack's shape, dtype and chunk sizes (no data is computed). Needs the ``[lazy]`` extra.

    Args:
        args: Parsed args with ``paths``, ``var``, ``json``.

    Returns:
        int: ``0`` on success.
    """
    with needs_lazy_extra():
        stack = NetCDF.open_mfdataset(args.paths, args.var)
    payload = {
        "variable": args.var,
        "files": len(args.paths),
        "shape": list(stack.shape),
        "dtype": str(stack.dtype),
        "chunksize": list(stack.chunksize),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_nc_from_dataframe(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc from-dataframe`` — build a cube from a MultiIndex CSV.

    Reads ``file`` into a DataFrame indexed by ``--index-col`` (innermost last are the
    ``y`` / ``x`` grid axes; outer levels are band dimensions), builds a container with
    ``NetCDF.from_dataframe`` and writes it to ``output``. A CSV written by
    ``nc to-dataframe`` round-trips with its index columns named on ``--index-col``.

    Args:
        args: Parsed args with ``file``, ``output``, ``index_col``, ``crs``, ``x``,
            ``y``, ``vars``, ``no_data_value``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The frame is not a valid MultiIndex grid, or a requested column /
            axis is missing (raised by ``from_dataframe``).
    """
    refuse_existing(args.output, args.overwrite)
    frame = pd.read_csv(args.file, index_col=list(args.index_col))
    no_data_value = float("nan") if args.no_data_value is None else args.no_data_value
    result = NetCDF.from_dataframe(
        frame,
        crs=args.crs,
        x=args.x,
        y=args.y,
        variables=args.vars or None,
        no_data_value=no_data_value,
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_from_dict(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc from-dict`` — build a cube from a JSON structure dump.

    Reads ``file`` as the JSON payload produced by ``nc to-dict --json --data``, rebuilds
    the container with ``NetCDF.from_dict`` and writes it to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The payload is not a valid cube structure (raised by ``from_dict``).
    """
    refuse_existing(args.output, args.overwrite)
    with open(args.file, encoding="utf-8") as handle:
        payload = json.load(handle)
    NetCDF.from_dict(payload).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_concat(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc concat`` — join cubes end to end along a dimension.

    Reads every ``--inputs`` cube, joins them along ``--dim`` in the given order (same
    grid, same variables, agreeing on every other dimension) and writes the result to
    ``output``.

    Args:
        args: Parsed args with ``output``, ``inputs``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The cubes carry different variables or disagree on a dimension
            (raised by ``concat``).
        pyramids.base._errors.AlignmentError: The cubes are not on the same grid.
    """
    refuse_existing(args.output, args.overwrite)
    cubes = [NetCDF.read_file(path) for path in args.inputs]
    NetCDF.concat(cubes, args.dim).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_merge(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc merge`` — put the variables of several cubes side by side.

    Reads every ``--inputs`` cube (all on the same grid, agreeing on every dimension) and
    writes a single container holding the union of their variables to ``output``.
    ``--compat`` decides what to do with a variable more than one cube carries.

    Args:
        args: Parsed args with ``output``, ``inputs``, ``compat``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--compat`` is unknown, or two copies of a variable conflict (raised
            by ``merge``).
        pyramids.base._errors.AlignmentError: The cubes are not on the same grid.
    """
    refuse_existing(args.output, args.overwrite)
    cubes = [NetCDF.read_file(path) for path in args.inputs]
    NetCDF.merge(cubes, compat=args.compat).to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_io_commands(ncsub: argparse._SubParsersAction) -> None:
    """Register the ``nc`` I/O export commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_conv = ncsub.add_parser("convert", help="re-write a file as a NetCDF")
    nc_conv.add_argument("file", help=_HELP_NC_FILE)
    nc_conv.add_argument("output", help="destination NetCDF path")
    nc_conv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_conv.set_defaults(func=_cmd_nc_convert)

    nc_df = ncsub.add_parser("to-dataframe", help="export variables to a CSV table")
    nc_df.add_argument("file", help=_HELP_NC_FILE)
    nc_df.add_argument("output", help="destination CSV path")
    nc_df.add_argument("--vars", nargs="+", help="variables to export (default: all)")
    nc_df.add_argument(
        "--dropna", action="store_true", help="drop rows with any missing value"
    )
    nc_df.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_df.set_defaults(func=_cmd_nc_to_dataframe)

    nc_dict = ncsub.add_parser("to-dict", help="print the file's structure as a dict")
    nc_dict.add_argument("file", help=_HELP_NC_FILE)
    nc_dict.add_argument("--json", action="store_true", help="emit JSON")
    nc_dict.add_argument(
        "--data", action="store_true", help="include variable arrays (large)"
    )
    nc_dict.set_defaults(func=_cmd_nc_to_dict)

    nc_zarr = ncsub.add_parser("to-zarr", help="write one variable to a Zarr store")
    nc_zarr.add_argument("file", help=_HELP_NC_FILE)
    nc_zarr.add_argument("output", help="destination .zarr store path")
    nc_zarr.add_argument("--var", required=True, help="variable to store")
    nc_zarr.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_zarr.set_defaults(func=_cmd_nc_to_zarr)

    nc_k = ncsub.add_parser("to-kerchunk", help="write a kerchunk reference manifest")
    nc_k.add_argument("file", help=_HELP_NC_FILE)
    nc_k.add_argument("output", help="destination manifest JSON path")
    nc_k.add_argument(
        "--inline-threshold",
        type=int,
        default=500,
        dest="inline_threshold",
        help="embed chunks smaller than this many bytes (default: 500)",
    )
    nc_k.add_argument(
        "--vlen-encode",
        default="embed",
        dest="vlen_encode",
        help="variable-length string encoding mode (default: embed)",
    )
    nc_k.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_k.set_defaults(func=_cmd_nc_to_kerchunk)

    nc_ck = ncsub.add_parser(
        "combine-kerchunk", help="write a combined kerchunk manifest over many files"
    )
    nc_ck.add_argument("output", help="destination manifest JSON path")
    nc_ck.add_argument("--inputs", nargs="+", required=True, help=_HELP_INPUTS)
    nc_ck.add_argument(
        "--concat-dims",
        nargs="+",
        default=["time"],
        dest="concat_dims",
        help="dimension(s) to concatenate along (default: time)",
    )
    nc_ck.add_argument(
        "--identical-dims",
        nargs="+",
        default=["lat", "lon"],
        dest="identical_dims",
        help="dimensions expected identical across files (default: lat lon)",
    )
    nc_ck.add_argument(
        "--inline-threshold",
        type=int,
        default=500,
        dest="inline_threshold",
        help="embed chunks smaller than this many bytes (default: 500)",
    )
    nc_ck.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_ck.set_defaults(func=_cmd_nc_combine_kerchunk)

    nc_mf = ncsub.add_parser("open-mf", help="report a lazy variable stack over files")
    nc_mf.add_argument(
        "--paths", nargs="+", required=True, help="NetCDF paths to stack"
    )
    nc_mf.add_argument(
        "--var", required=True, help="variable to stack across the files"
    )
    nc_mf.add_argument("--json", action="store_true", help="emit JSON")
    nc_mf.set_defaults(func=_cmd_nc_open_mf)

    nc_fdf = ncsub.add_parser(
        "from-dataframe", help="build a cube from a MultiIndex CSV"
    )
    nc_fdf.add_argument("file", help="source CSV path")
    nc_fdf.add_argument("output", help="destination NetCDF path")
    nc_fdf.add_argument(
        "--index-col",
        nargs="+",
        required=True,
        dest="index_col",
        help="index columns, innermost last (y/x grid axes; outer are band dims)",
    )
    nc_fdf.add_argument(
        "--crs", default=None, help="CRS for the result (EPSG or string)"
    )
    nc_fdf.add_argument(
        "--x", default=None, help="index level holding the x coordinates"
    )
    nc_fdf.add_argument(
        "--y", default=None, help="index level holding the y coordinates"
    )
    nc_fdf.add_argument(
        "--vars", nargs="+", default=None, help="columns to use (default: all)"
    )
    nc_fdf.add_argument(
        "--no-data-value",
        type=float,
        default=None,
        dest="no_data_value",
        help="sentinel for gaps (default: NaN)",
    )
    nc_fdf.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_fdf.set_defaults(func=_cmd_nc_from_dataframe)

    nc_fd = ncsub.add_parser(
        "from-dict", help="build a cube from a JSON structure dump"
    )
    nc_fd.add_argument("file", help="source JSON path (as 'nc to-dict --json --data')")
    nc_fd.add_argument("output", help="destination NetCDF path")
    nc_fd.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_fd.set_defaults(func=_cmd_nc_from_dict)

    nc_cat = ncsub.add_parser("concat", help="join cubes end to end along a dimension")
    nc_cat.add_argument("output", help="destination NetCDF path")
    nc_cat.add_argument("--inputs", nargs="+", required=True, help=_HELP_INPUTS)
    nc_cat.add_argument("--dim", required=True, help="dimension to join along")
    nc_cat.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_cat.set_defaults(func=_cmd_nc_concat)

    nc_mg = ncsub.add_parser(
        "merge", help="put the variables of several cubes together"
    )
    nc_mg.add_argument("output", help="destination NetCDF path")
    nc_mg.add_argument("--inputs", nargs="+", required=True, help=_HELP_INPUTS)
    nc_mg.add_argument(
        "--compat",
        default="no_conflicts",
        choices=("no_conflicts", "override"),
        help="how to treat a shared variable (default: no_conflicts)",
    )
    nc_mg.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_mg.set_defaults(func=_cmd_nc_merge)
