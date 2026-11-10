"""NetCDF variable- and dimension-management commands for the ``pyramids nc`` group.

Thin wrappers over :class:`pyramids.netcdf.NetCDF`: rename or remove a variable
(`rename-var`/`remove-var`), copy a variable in from another NetCDF (`add-var`), write a
raster in as a variable (`set-var`), crop one variable to a vector mask (`crop-var`),
reproject or resample one variable in place (`reproject-var`/`resample-var`), lift one
variable onto a new length-one band dimension (`expand-dims`), restamp existing band
dimensions (`assign-coords`), and the xarray-style dimension/coordinate verbs — rename
band dimensions (`rename-dims`), drop them (`drop-dims`), promote or demote CF auxiliary
coordinates (`set-coords`/`reset-coords`) and drop length-one band dimensions
(`squeeze`). Every command writes a new file and never touches the source: the variable
mutators (`rename-var`/`remove-var`/`add-var`/`set-var`/`crop-var`/`reproject-var`/
`resample-var`) mutate a read-only read's in-memory copy and write it out, while the
dimension/coordinate verbs return a new cube that is written out. No xarray is imported
here — reads go through the ``NetCDF`` class.
"""

from __future__ import annotations

import argparse

from pyramids._cli_shared import info_print, refuse_existing
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

_HELP_NC_FILE = "NetCDF path"
_HELP_OUTPUT = "destination NetCDF path"
_HELP_OVERWRITE = "replace the output if it already exists"
_HELP_VAR = "variable name"
_HELP_METHOD = "resampling method (default: nearest neighbor)"


def _parse_pairs(items: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` tokens into a mapping, splitting on the first ``=``.

    Args:
        items: Tokens such as ``["time=t", "plev=level"]``.

    Returns:
        dict[str, str]: The ``{key: value}`` mapping, order preserved.

    Raises:
        ValueError: A token carries no ``=`` separator.
    """
    pairs: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        pairs[key] = value
    return pairs


def _parse_coord_lists(items: list[str]) -> dict[str, list[float]]:
    """Parse ``DIM=v1,v2,...`` tokens into a ``{dim: [float, ...]}`` mapping.

    Each token names a band dimension and the comma-separated coordinate values to
    restamp it with; the values are coerced to ``float`` (band-dimension coordinates
    are numeric, e.g. hours-since or pressure levels).

    Args:
        items: ``--coord`` tokens, each ``DIM=v1,v2,...`` (e.g. ``time=0,6,12,18``).

    Returns:
        dict[str, list[float]]: Dimension name mapped to its list of coordinate values.

    Raises:
        ValueError: A token has no ``=``, an empty dimension name, an empty value list,
            or a value that is not a float.
    """
    mapping: dict[str, list[float]] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep or not name:
            raise ValueError(f"expected DIM=v1,v2,..., got {item!r}")
        tokens = [piece for piece in raw.split(",") if piece != ""]
        if not tokens:
            raise ValueError(f"--coord token {item!r} lists no values")
        try:
            mapping[name] = [float(piece) for piece in tokens]
        except ValueError as exc:
            raise ValueError(
                f"--coord {item!r} has a non-numeric value in {raw!r}"
            ) from exc
    return mapping


def _cmd_nc_rename_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc rename-var`` — rename a variable into a new file.

    The source is read read-only; the rename mutates an in-memory copy (the class
    makes one), which is then written to ``output``, so the source is never modified.

    Args:
        args: Parsed args with ``file``, ``output``, ``old``, ``new``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--old`` does not resolve to a variable (raised by
            ``rename_variable``), or ``--new`` already exists.
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    nc.rename_variable(args.old, args.new)
    nc.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_remove_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc remove-var`` — remove a variable into a new file.

    The source is read read-only; the removal mutates an in-memory copy, which is
    then written to ``output``, so the source is never modified.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` is not found in the container (raised by
            ``remove_variable``).
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    nc.remove_variable(args.var)
    nc.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_reproject_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc reproject-var`` — reproject one variable into a new file.

    ``reproject_variable`` warps the variable and stores the result back on the same
    container, returning it; that container is written to ``output``. The source is
    read read-only and left untouched.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``epsg``, ``method``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` is not a valid variable name (raised by
            ``reproject_variable`` via ``get_variable``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).reproject_variable(
        args.var, args.epsg, method=args.method
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_resample_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc resample-var`` — resample one variable into a new file.

    ``resample_variable`` resamples the variable and stores the result back on the
    same container, returning it; that container is written to ``output``. The source
    is read read-only and left untouched.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``cell_size``,
            ``method``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` is not a valid variable name (raised by
            ``resample_variable`` via ``get_variable``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).resample_variable(
        args.var, args.cell_size, method=args.method
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_rename_dims(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc rename-dims`` — rename band dimensions into a new file.

    ``rename_dims`` is non-mutating and returns a new cube with the band dimensions
    re-labelled; that cube is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``map``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: A ``--map`` token is malformed (raised by :func:`_parse_pairs`),
            or an ``old`` name is not a band dimension (raised by ``rename_dims``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).rename_dims(_parse_pairs(args.map))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_drop_dims(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc drop-dims`` — drop band dimensions into a new file.

    ``drop_dims`` is non-mutating and returns a new cube without the named band
    dimensions; that cube is written to ``output``. ``--errors ignore`` makes an
    unknown dimension a no-op instead of an error.

    Args:
        args: Parsed args with ``file``, ``output``, ``dims``, ``errors``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: A dimension is not present and ``--errors`` is ``raise`` (raised
            by ``drop_dims``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).drop_dims(args.dims, errors=args.errors)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_set_coords(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc set-coords`` — promote variables to CF aux coordinates.

    ``set_coords`` is non-mutating and returns a new cube in which the named
    variables are CF auxiliary coordinates (so they leave ``variable_names``); that
    cube is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``names``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: A name is not a variable of the container, or names a dimension
            (raised by ``set_coords``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).set_coords(args.names)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_reset_coords(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc reset-coords`` — demote CF aux coordinates to variables.

    ``reset_coords`` is non-mutating and returns a new cube in which the named
    auxiliary coordinates (or all of them when ``--names`` is omitted) are data
    variables again; that cube is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``names``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: A name is not an auxiliary coordinate of the container (raised by
            ``reset_coords``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).reset_coords(args.names)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_squeeze(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc squeeze`` — drop length-one band dimensions into a file.

    ``squeeze`` is non-mutating and returns a new cube with the length-one band
    dimensions removed (one named dimension with ``--dim``, or every length-one band
    dimension otherwise); that cube is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``dim``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--dim`` is not a band dimension, or is longer than one (raised
            by ``squeeze``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).squeeze(args.dim)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_add_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc add-var`` — copy variables in from another NetCDF.

    The base file is read read-only; ``add_variable`` copies the requested variable(s)
    of ``--source`` into an in-memory copy of it, which is written to ``output`` (the
    source and base files are never modified). Omitting ``--var`` copies every variable
    of the source; a leaf name that collides with one already present is suffixed.

    Args:
        args: Parsed args with ``file``, ``output``, ``source``, ``var``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` does not resolve in the source, or either file is not a
            multidimensional container (raised by ``add_variable``).
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    source = NetCDF.read_file(args.source)
    nc.add_variable(source, args.var)
    nc.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_set_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc set-var`` — write a raster in as a NetCDF variable.

    The base file is read read-only; ``set_variable`` stores the ``--raster`` dataset as
    a variable named ``--var`` on an in-memory copy, which is written to ``output`` (a
    variable of that name already present is replaced). The source files are untouched.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``raster``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: The base file is not a multidimensional container (raised by
            ``set_variable``).
    """
    refuse_existing(args.output, args.overwrite)
    nc = NetCDF.read_file(args.file)
    nc.set_variable(args.var, Dataset.read_file(args.raster))
    nc.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_crop_var(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc crop-var`` — crop one variable to a vector mask.

    ``crop_variable`` crops the variable to the ``--mask`` polygons and stores the result
    back on an in-memory copy of the container, which is written to ``output``; the source
    is read read-only and left untouched. ``--no-touch`` excludes cells that only touch the
    mask boundary.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``mask``, ``no_touch``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` is not a valid variable, or the mask does not overlap the
            variable (raised by ``crop_variable``).
    """
    refuse_existing(args.output, args.overwrite)
    mask = FeatureCollection.read_file(args.mask)
    result = NetCDF.read_file(args.file).crop_variable(
        args.var, mask, touch=not args.no_touch
    )
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_expand_dims(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc expand-dims`` — lift one variable onto a new band dimension.

    ``expand_dims`` acts on a single variable's bands, so the command resolves ``--var``
    first and adds a new outermost band dimension of length one, with ``--value`` its sole
    coordinate (omitted leaves the dimension without coordinates). The lifted variable is
    written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``var``, ``dim``, ``value``,
            ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: ``--var`` is not a valid variable, or ``--dim`` is already a band
            dimension or names a spatial axis (raised by ``expand_dims``).
    """
    refuse_existing(args.output, args.overwrite)
    variable = NetCDF.read_file(args.file).get_variable(args.var)
    if not isinstance(variable, NetCDF):
        raise ValueError(
            f"variable {args.var!r} is not a raster plane, so a dimension cannot be "
            "added to it"
        )
    result = variable.expand_dims(args.dim, args.value)
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_nc_assign_coords(args: argparse.Namespace) -> int:
    """Handle ``pyramids nc assign-coords`` — restamp existing band dimensions.

    ``assign_coords`` is non-mutating and returns a new cube in which each named band
    dimension carries the given coordinate values (a pure re-label; no cells move); that
    cube is written to ``output``.

    Args:
        args: Parsed args with ``file``, ``output``, ``coord``, ``overwrite``.

    Returns:
        int: ``0`` on success.

    Raises:
        ValueError: A ``--coord`` token is malformed (raised by
            :func:`_parse_coord_lists`), or a name is not an existing band dimension or
            its value count does not match the dimension length (raised by
            ``assign_coords``).
    """
    refuse_existing(args.output, args.overwrite)
    result = NetCDF.read_file(args.file).assign_coords(_parse_coord_lists(args.coord))
    result.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def add_nc_vars_commands(
    ncsub: argparse._SubParsersAction,
    advanced: argparse._SubParsersAction | None = None,
) -> None:
    """Register the ``nc`` variable- and dimension-management commands on ``ncsub``.

    Args:
        ncsub: The ``nc`` group's ``add_subparsers`` action.
    """
    nc_rnv = ncsub.add_parser("rename-var", help="rename a variable")
    nc_rnv.add_argument("file", help=_HELP_NC_FILE)
    nc_rnv.add_argument("output", help=_HELP_OUTPUT)
    nc_rnv.add_argument("--old", required=True, help="current variable name")
    nc_rnv.add_argument("--new", required=True, help="new variable name")
    nc_rnv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rnv.set_defaults(func=_cmd_nc_rename_var)

    nc_rmv = ncsub.add_parser("remove-var", help="remove a variable")
    nc_rmv.add_argument("file", help=_HELP_NC_FILE)
    nc_rmv.add_argument("output", help=_HELP_OUTPUT)
    nc_rmv.add_argument("--var", required=True, help=_HELP_VAR)
    nc_rmv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rmv.set_defaults(func=_cmd_nc_remove_var)

    nc_rpv = ncsub.add_parser("reproject-var", help="reproject one variable")
    nc_rpv.add_argument("file", help=_HELP_NC_FILE)
    nc_rpv.add_argument("output", help=_HELP_OUTPUT)
    nc_rpv.add_argument("--var", required=True, help=_HELP_VAR)
    nc_rpv.add_argument("--epsg", type=int, required=True, help="target EPSG code")
    nc_rpv.add_argument("--method", default="nearest neighbor", help=_HELP_METHOD)
    nc_rpv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rpv.set_defaults(func=_cmd_nc_reproject_var)

    nc_rsv = ncsub.add_parser("resample-var", help="resample one variable")
    nc_rsv.add_argument("file", help=_HELP_NC_FILE)
    nc_rsv.add_argument("output", help=_HELP_OUTPUT)
    nc_rsv.add_argument("--var", required=True, help=_HELP_VAR)
    nc_rsv.add_argument("--cell-size", type=float, required=True, help="new cell size")
    nc_rsv.add_argument("--method", default="nearest neighbor", help=_HELP_METHOD)
    nc_rsv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rsv.set_defaults(func=_cmd_nc_resample_var)

    nc_rnd = (advanced or ncsub).add_parser(
        "rename-dims", help="rename band dimensions"
    )
    nc_rnd.add_argument("file", help=_HELP_NC_FILE)
    nc_rnd.add_argument("output", help=_HELP_OUTPUT)
    nc_rnd.add_argument(
        "--map",
        nargs="+",
        required=True,
        metavar="OLD=NEW",
        help="one or more OLD=NEW dimension renames",
    )
    nc_rnd.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rnd.set_defaults(func=_cmd_nc_rename_dims)

    nc_drd = (advanced or ncsub).add_parser("drop-dims", help="drop band dimensions")
    nc_drd.add_argument("file", help=_HELP_NC_FILE)
    nc_drd.add_argument("output", help=_HELP_OUTPUT)
    nc_drd.add_argument(
        "--dims", nargs="+", required=True, help="band dimensions to drop"
    )
    nc_drd.add_argument(
        "--errors",
        choices=("raise", "ignore"),
        default="raise",
        help="how to treat an unknown dimension (default: raise)",
    )
    nc_drd.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_drd.set_defaults(func=_cmd_nc_drop_dims)

    nc_stc = (advanced or ncsub).add_parser(
        "set-coords", help="promote variables to aux coordinates"
    )
    nc_stc.add_argument("file", help=_HELP_NC_FILE)
    nc_stc.add_argument("output", help=_HELP_OUTPUT)
    nc_stc.add_argument(
        "--names", nargs="+", required=True, help="variables to promote"
    )
    nc_stc.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_stc.set_defaults(func=_cmd_nc_set_coords)

    nc_rtc = (advanced or ncsub).add_parser(
        "reset-coords", help="demote aux coordinates to variables"
    )
    nc_rtc.add_argument("file", help=_HELP_NC_FILE)
    nc_rtc.add_argument("output", help=_HELP_OUTPUT)
    nc_rtc.add_argument(
        "--names", nargs="*", default=None, help="coordinates to demote (default: all)"
    )
    nc_rtc.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_rtc.set_defaults(func=_cmd_nc_reset_coords)

    nc_sqz = (advanced or ncsub).add_parser(
        "squeeze", help="drop length-one band dimensions"
    )
    nc_sqz.add_argument("file", help=_HELP_NC_FILE)
    nc_sqz.add_argument("output", help=_HELP_OUTPUT)
    nc_sqz.add_argument(
        "--dim",
        default=None,
        help="the one length-one dimension to drop (default: all)",
    )
    nc_sqz.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_sqz.set_defaults(func=_cmd_nc_squeeze)

    nc_adv = ncsub.add_parser("add-var", help="copy variables in from another NetCDF")
    nc_adv.add_argument("file", help=_HELP_NC_FILE)
    nc_adv.add_argument("output", help=_HELP_OUTPUT)
    nc_adv.add_argument("--source", required=True, help="NetCDF to copy variables from")
    nc_adv.add_argument(
        "--var", default=None, help="one variable to copy (default: all of the source)"
    )
    nc_adv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_adv.set_defaults(func=_cmd_nc_add_var)

    nc_sev = ncsub.add_parser("set-var", help="write a raster in as a variable")
    nc_sev.add_argument("file", help=_HELP_NC_FILE)
    nc_sev.add_argument("output", help=_HELP_OUTPUT)
    nc_sev.add_argument("--var", required=True, help="name for the new variable")
    nc_sev.add_argument(
        "--raster", required=True, help="raster to store as the variable"
    )
    nc_sev.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_sev.set_defaults(func=_cmd_nc_set_var)

    nc_cpv = ncsub.add_parser("crop-var", help="crop one variable to a vector mask")
    nc_cpv.add_argument("file", help=_HELP_NC_FILE)
    nc_cpv.add_argument("output", help=_HELP_OUTPUT)
    nc_cpv.add_argument("--var", required=True, help=_HELP_VAR)
    nc_cpv.add_argument("--mask", required=True, help="polygon vector mask to crop to")
    nc_cpv.add_argument(
        "--no-touch",
        action="store_true",
        help="exclude cells that only touch the mask boundary",
    )
    nc_cpv.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_cpv.set_defaults(func=_cmd_nc_crop_var)

    nc_exd = (advanced or ncsub).add_parser(
        "expand-dims", help="lift a variable onto a new band dim"
    )
    nc_exd.add_argument("file", help=_HELP_NC_FILE)
    nc_exd.add_argument("output", help=_HELP_OUTPUT)
    nc_exd.add_argument("--var", required=True, help=_HELP_VAR)
    nc_exd.add_argument(
        "--dim", required=True, help="name of the new length-one band dim"
    )
    nc_exd.add_argument(
        "--value",
        type=float,
        default=None,
        help="the new dimension's single coordinate value (default: none)",
    )
    nc_exd.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_exd.set_defaults(func=_cmd_nc_expand_dims)

    nc_asc = (advanced or ncsub).add_parser(
        "assign-coords", help="restamp existing band dimensions"
    )
    nc_asc.add_argument("file", help=_HELP_NC_FILE)
    nc_asc.add_argument("output", help=_HELP_OUTPUT)
    nc_asc.add_argument(
        "--coord",
        nargs="+",
        required=True,
        metavar="DIM=v1,v2,...",
        help="one or more DIM=v1,v2,... band-dimension restamps",
    )
    nc_asc.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    nc_asc.set_defaults(func=_cmd_nc_assign_coords)
