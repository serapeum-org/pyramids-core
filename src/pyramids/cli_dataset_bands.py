"""Raster band / mask / colour / sensor-model CLI commands.

Extends the raster command surface (:mod:`pyramids.cli_dataset`) with the
band-level and georeferencing-metadata operations of :class:`~pyramids.dataset.Dataset`:
``add-band``, ``select-bands``, ``get-band-by-color``, ``create-mask-band``,
``read-masks``, ``get-mask``, ``set-color-ramp``, ``set-rpcs`` and ``geolocate``.
Each handler is a thin 1:1 wrapper over the matching ``Dataset`` call; the
top-level parser wires the group in via :func:`add_dataset_bands_commands`.

The three commands that mutate georeference/colour metadata (``create-mask-band``,
``set-color-ramp``, ``set-rpcs``) edit an in-memory :meth:`Dataset.copy` and save
it to a new ``output`` path, leaving the input file untouched — the established
edit-in-memory-then-save workflow, so no update-mode driver support is required.

No commands are deferred: every listed method has a sensible file or stdout form.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from pyramids._cli_shared import info_print, json_sanitize, map_lookup_errors
from pyramids._cli_shared import refuse_existing as _refuse_existing
from pyramids.base._utils import DEFAULT_RESAMPLING
from pyramids.base.crs import crs_spec
from pyramids.base.georeference import GeoReference
from pyramids.cli_dataset import (
    _HELP_DST_RASTER,
    _HELP_INSPECT_RASTER,
    _HELP_JSON,
    _HELP_OVERWRITE,
    _HELP_SRC_RASTER,
)
from pyramids.dataset import Dataset

_HELP_BAND_0 = "0-based band index"


def _write_mask_raster(template: Dataset, mask: np.ndarray, output: str) -> None:
    """Write a mask array as a new raster on ``template``'s grid.

    Args:
        template: The dataset the mask was read from — supplies the geotransform
            and CRS carried onto the output.
        mask: The mask array, ``(rows, cols)`` for one band or
            ``(band_count, rows, cols)`` for several.
        output: Destination raster path.
    """
    # A mask has no sentinel of its own (0/255 are both meaningful), so suppress
    # from_array's default -9999, which is out of range for the uint8 mask dtype.
    Dataset.from_array(
        np.asarray(mask),
        geo_ref=GeoReference(
            geo=template.geotransform, epsg=crs_spec(template.epsg, template.crs)
        ),
        no_data_value=None,
    ).to_file(output)


def _cmd_add_band(args: argparse.Namespace) -> int:
    """Handle `pyramids add-band` — append a band read from another raster.

    Args:
        args: Parsed args with `input`, `output`, `source` (raster the new band
            is read from), `source_band` (0-based), `unit`, and `overwrite`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    source = Dataset.read_file(args.source)
    array = np.asarray(source.read_array(band=args.source_band))
    out = ds.add_band(array, unit=args.unit, inplace=False)
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_select_bands(args: argparse.Namespace) -> int:
    """Handle `pyramids select-bands` — subset bands into a new raster.

    Band selectors are **1-based** (matching the product band numbering), and
    order is preserved — duplicates are allowed.

    Args:
        args: Parsed args with `input`, `output`, `bands` (1-based indices),
            and `overwrite`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    out = ds.select_bands(list(args.bands))
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_get_band_by_color(args: argparse.Namespace) -> int:
    """Handle `pyramids get-band-by-color` — print the band for a colour.

    Prints the 0-based band index whose colour interpretation matches `--color`,
    or `none` / JSON `null` when no band carries it.

    Args:
        args: Parsed args with `file`, `color`, and `json`.

    Returns:
        int: `0` on success.
    """
    ds = Dataset.read_file(args.file)
    band = ds.get_band_by_color(args.color)
    if args.json:
        print(json.dumps(json_sanitize({"color": args.color, "band": band})))
    else:
        print(band if band is not None else "none")
    return 0


def _cmd_create_mask_band(args: argparse.Namespace) -> int:
    """Handle `pyramids create-mask-band` — add a mask band and save a copy.

    Args:
        args: Parsed args with `input`, `output`, `per_band`, and `overwrite`.
            `--per-band` creates a per-band mask; the default is a single mask
            shared by every band (`GMF_PER_DATASET`).

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    working = Dataset.read_file(args.input).copy()
    working.create_mask_band(per_dataset=not args.per_band)
    working.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_read_masks(args: argparse.Namespace) -> int:
    """Handle `pyramids read-masks` — write per-band mask array(s) as a raster.

    Writes a raster of mask values (`0` out-of-domain, `255` valid). With
    `--band` a single-band mask raster is written; without it, every band's mask
    is stacked into a multi-band raster.

    Args:
        args: Parsed args with `input`, `output`, `band` (0-based or `None`),
            and `overwrite`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    with map_lookup_errors():
        mask = ds.read_masks(band=args.band)
    _write_mask_raster(ds, mask, args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_get_mask(args: argparse.Namespace) -> int:
    """Handle `pyramids get-mask` — write one band's mask as a raster.

    Args:
        args: Parsed args with `input`, `output`, `band` (0-based), and
            `overwrite`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    with map_lookup_errors():
        mask = ds.get_mask(band=args.band)
    _write_mask_raster(ds, mask, args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_set_color_ramp(args: argparse.Namespace) -> int:
    """Handle `pyramids set-color-ramp` — attach an interpolated colour table.

    Fills a colour for every integer value in `[--start-value, --end-value]`,
    either as a two-colour linear ramp (`--start-color` + `--end-color`) or by
    sampling a named matplotlib `--colormap`. The ramp is applied to an in-memory
    copy and saved to `output`; the input is untouched.

    Args:
        args: Parsed args with `input`, `output`, `band` (1-based), `start_value`,
            `end_value`, `start_color`, `end_color`, `colormap`, and `overwrite`.

    Returns:
        int: `0` on success.
    """
    _refuse_existing(args.output, args.overwrite)
    working = Dataset.read_file(args.input).copy()
    working.set_color_ramp(
        band=args.band,
        start_value=args.start_value,
        end_value=args.end_value,
        start_color=args.start_color,
        end_color=args.end_color,
        colormap=args.colormap,
    )
    working.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_set_rpcs(args: argparse.Namespace) -> int:
    """Handle `pyramids set-rpcs` — attach RPC metadata and save a copy.

    Reads the rational-polynomial coefficients from a JSON object file
    (`--rpc-json`) — the five `*_OFF` offsets, the five `*_SCALE` scales, and the
    four coefficient vectors — attaches them to an in-memory copy and saves it.

    Args:
        args: Parsed args with `input`, `output`, `rpc_json` (path to the JSON
            coefficient object), and `overwrite`.

    Returns:
        int: `0` on success.

    Raises:
        ValueError: `--rpc-json` is not a JSON object, or required RPC keys are
            missing (`Dataset.set_rpcs` lists them).
    """
    _refuse_existing(args.output, args.overwrite)
    with open(args.rpc_json, encoding="utf-8") as handle:
        rpc = json.load(handle)
    if not isinstance(rpc, dict):
        raise ValueError(
            f"--rpc-json {args.rpc_json!r} must hold a JSON object mapping RPC "
            f"keys to values, not a {type(rpc).__name__}."
        )
    working = Dataset.read_file(args.input).copy()
    working.set_rpcs(rpc)
    working.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_geolocate(args: argparse.Namespace) -> int:
    """Handle `pyramids geolocate` — warp from geolocation arrays onto a grid.

    Resamples a swath / curvilinear raster whose georeferencing is carried by
    per-pixel lon/lat arrays (the GDAL `GEOLOCATION` domain) onto a north-up
    affine grid.

    Args:
        args: Parsed args with `input`, `output`, `to_crs`, `resampling`,
            `cell_size`, and `overwrite`.

    Returns:
        int: `0` on success.

    Raises:
        GeolocationArrayError: The input carries no `GEOLOCATION` domain.
    """
    _refuse_existing(args.output, args.overwrite)
    ds = Dataset.read_file(args.input)
    out = ds.geolocate(
        to_epsg=args.to_crs, method=args.resampling, cell_size=args.cell_size
    )
    out.to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _add_add_band(sub: argparse._SubParsersAction) -> None:
    """Register the ``add-band`` command."""
    parser = sub.add_parser("add-band", help="append a band read from another raster")
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help=_HELP_DST_RASTER)
    parser.add_argument(
        "--source", required=True, help="raster the new band's values are read from"
    )
    parser.add_argument(
        "--source-band",
        type=int,
        default=0,
        help="0-based band of --source to append (default 0)",
    )
    parser.add_argument("--unit", help="unit of the new band's values")
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_add_band)


def _add_select_bands(sub: argparse._SubParsersAction) -> None:
    """Register the ``select-bands`` command."""
    parser = sub.add_parser("select-bands", help="subset bands into a new raster")
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help=_HELP_DST_RASTER)
    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        required=True,
        help="1-based band indices to keep, in order (duplicates allowed)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_select_bands)


def _add_get_band_by_color(sub: argparse._SubParsersAction) -> None:
    """Register the ``get-band-by-color`` command."""
    parser = sub.add_parser(
        "get-band-by-color", help="print the band index for a colour interpretation"
    )
    parser.add_argument("file", help=_HELP_INSPECT_RASTER)
    parser.add_argument(
        "--color",
        required=True,
        help="colour interpretation (e.g. red, green, blue, alpha, gray_index)",
    )
    parser.add_argument("--json", action="store_true", help=_HELP_JSON)
    parser.set_defaults(func=_cmd_get_band_by_color)


def _add_create_mask_band(sub: argparse._SubParsersAction) -> None:
    """Register the ``create-mask-band`` command."""
    parser = sub.add_parser(
        "create-mask-band", help="add a mask band and save the result"
    )
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help=_HELP_DST_RASTER)
    parser.add_argument(
        "--per-band",
        action="store_true",
        help="create a per-band mask (default: one mask shared by every band)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_create_mask_band)


def _add_read_masks(sub: argparse._SubParsersAction) -> None:
    """Register the ``read-masks`` command."""
    parser = sub.add_parser(
        "read-masks", help="write per-band mask array(s) as a raster (0/255)"
    )
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help="destination mask raster path")
    parser.add_argument(
        "--band",
        type=int,
        default=None,
        help=f"{_HELP_BAND_0} to read (default: every band, stacked)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_read_masks)


def _add_get_mask(sub: argparse._SubParsersAction) -> None:
    """Register the ``get-mask`` command."""
    parser = sub.add_parser(
        "get-mask", help="write one band's mask as a raster (0/255)"
    )
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help="destination mask raster path")
    parser.add_argument(
        "--band", type=int, default=0, help=f"{_HELP_BAND_0} (default 0)"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_get_mask)


def _add_set_color_ramp(sub: argparse._SubParsersAction) -> None:
    """Register the ``set-color-ramp`` command."""
    parser = sub.add_parser(
        "set-color-ramp", help="attach an interpolated colour table and save a copy"
    )
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help=_HELP_DST_RASTER)
    parser.add_argument(
        "--band", type=int, default=1, help="1-based band to colour (default 1)"
    )
    parser.add_argument(
        "--start-value", type=int, required=True, help="first value in the ramp (>= 0)"
    )
    parser.add_argument(
        "--end-value",
        type=int,
        required=True,
        help="last value in the ramp (must exceed --start-value)",
    )
    parser.add_argument("--start-color", help="hex colour at --start-value")
    parser.add_argument("--end-color", help="hex colour at --end-value")
    parser.add_argument(
        "--colormap",
        help="named matplotlib colormap sampled across the range "
        "(instead of --start-color/--end-color)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_set_color_ramp)


def _add_set_rpcs(sub: argparse._SubParsersAction) -> None:
    """Register the ``set-rpcs`` command."""
    parser = sub.add_parser(
        "set-rpcs", help="attach RPC sensor-model metadata and save a copy"
    )
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help=_HELP_DST_RASTER)
    parser.add_argument(
        "--rpc-json",
        dest="rpc_json",
        required=True,
        help="path to a JSON object of RPC coefficients (offsets, scales, vectors)",
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_set_rpcs)


def _add_geolocate(sub: argparse._SubParsersAction) -> None:
    """Register the ``geolocate`` command."""
    parser = sub.add_parser(
        "geolocate", help="warp a raster from its geolocation arrays onto a grid"
    )
    parser.add_argument("input", help=_HELP_SRC_RASTER)
    parser.add_argument("output", help=_HELP_DST_RASTER)
    parser.add_argument("--to-crs", help="reproject the result to this CRS")
    parser.add_argument(
        "--resampling", default=DEFAULT_RESAMPLING, help="resampling method"
    )
    parser.add_argument(
        "--cell-size", type=float, help="output pixel size in target-CRS units"
    )
    parser.add_argument("--overwrite", action="store_true", help=_HELP_OVERWRITE)
    parser.set_defaults(func=_cmd_geolocate)


def add_dataset_bands_commands(sub: argparse._SubParsersAction) -> None:
    """Register the band / mask / colour / sensor-model commands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    _add_add_band(sub)
    _add_select_bands(sub)
    _add_get_band_by_color(sub)
    _add_create_mask_band(sub)
    _add_read_masks(sub)
    _add_get_mask(sub)
    _add_set_color_ramp(sub)
    _add_set_rpcs(sub)
    _add_geolocate(sub)
