"""CLI subcommands for the processing pipeline layer.

Registers the geoprocessing subcommands on the top-level ``pyramids`` parser:
``run`` (execute a pipeline YAML over inputs), ``apply`` (run one registered
tool over a single input), ``tools`` (list the registered tools), and
``tool <name>`` (print a tool's parameter schema). Help text is generated from
the registry, never hand-written.
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from pyramids._cli_shared import info_print, refuse_existing
from pyramids.processing.pipeline import Pipeline
from pyramids.processing.registry import resolve, tool_names
from pyramids.processing.runner import RunResult, run
from pyramids.processing.schema import ToolMetadata


def _manifest_payload(result: RunResult) -> dict[str, Any]:
    """Build a reproducible + timed manifest from a batch ``RunResult``.

    The ``pipeline`` block is the re-emitted, reloadable recipe (every input runs
    the same pipeline, so the first provenance supplies it); ``runs`` carries the
    per-input source, total wall time, and per-step timings.

    Args:
        result: The outcome of a :func:`pyramids.processing.runner.run` batch.

    Returns:
        dict: A YAML-ready mapping with ``pipeline`` and ``runs`` keys.
    """
    payload: dict[str, Any] = {}
    if result.provenance:
        payload["pipeline"] = result.provenance[0].to_pipeline().to_dict()["pipeline"]
    payload["runs"] = [
        {
            "source": prov.source,
            "total_seconds": prov.total_seconds,
            "steps": [
                {
                    "tool": step.tool,
                    "parameters": step.parameters,
                    "seconds": step.seconds,
                }
                for step in prov.steps
            ],
        }
        for prov in result.provenance
    ]
    return payload


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a pipeline YAML over ``--inputs`` and report the outcome.

    A single ``--inputs`` value keeps the scalar path (so a lone glob still
    expands); multiple values pass through as an explicit file list (B-2).
    """
    pipeline = Pipeline.from_yaml(args.pipeline)
    inputs: Any = args.inputs[0] if len(args.inputs) == 1 else args.inputs
    result = run(
        pipeline,
        inputs,
        on_error=args.on_error,
        out=args.out,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as handle:
            yaml.safe_dump(_manifest_payload(result), handle, sort_keys=False)
        info_print(args, f"wrote manifest {args.manifest}")
    # The count is informational chatter (the real outputs are the written files
    # + exit code), so --quiet silences it; the per-failure lines are error
    # reporting and always print (L4).
    info_print(args, f"{len(result.outputs)} succeeded, {len(result.failures)} failed")
    for source, exc in result.failures:
        print(f"  FAILED {source}: {exc}")
    return 0 if result.ok else 1


def _parse_set(tool: ToolMetadata, pairs: list[str] | None) -> dict[str, Any]:
    """Coerce ``--set KEY=VALUE`` pairs against ``tool``'s parameter schema.

    Args:
        tool: The resolved tool whose schema types the values.
        pairs: The raw ``KEY=VALUE`` strings (or ``None``).

    Returns:
        dict: The ``{name: typed-value}`` mapping to pass to the tool.

    Raises:
        ValueError: A pair is not ``KEY=VALUE``, names an unknown parameter, or
            carries a value that fails the parameter's type/choices.
    """
    params: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        parameter = tool.param(key)
        if parameter is None:
            valid = [p.name for p in tool.parameters]
            raise ValueError(
                f"tool {tool.name!r}: unknown parameter {key!r}; valid: {valid}"
            )
        params[key] = parameter.coerce(raw)
    return params


def _cmd_apply(args: argparse.Namespace) -> int:
    """Run one registered tool over a single input into an explicit output path.

    A one-step pipeline: reuses the runner so an ``Array``-output tool
    (slope/aspect/hillshade/focal_*) is materialised back into a georeferenced
    raster, then writes to ``output`` (F-1).
    """
    refuse_existing(args.output, args.overwrite)
    tool = resolve(args.name)  # ValueError (lists names) on an unknown tool
    params = _parse_set(tool, args.set)
    pipeline = Pipeline([(args.name, params)])  # schema-validates the parameters
    result = run(pipeline, [args.input], on_error="raise")
    result.outputs[0].to_file(args.output)
    info_print(args, f"wrote {args.output}")
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    """List every registered tool with its input/output types."""
    for name in tool_names():
        tool = resolve(name)
        print(f"{name:24} {tool.input_type} -> {tool.output_type}   {tool.description}")
    return 0


def _cmd_tool(args: argparse.Namespace) -> int:
    """Print one tool's parameter schema (its help block).

    An unknown name raises ``ValueError`` from ``resolve``; it propagates to
    ``main`` which reports it on stderr and exits non-zero, like every other
    subcommand.
    """
    print(resolve(args.name).help())
    return 0


def add_processing_commands(sub: argparse._SubParsersAction) -> None:
    """Register the ``run`` / ``apply`` / ``tools`` / ``tool`` subcommands on ``sub``.

    Args:
        sub: The top-level ``add_subparsers`` action from the main CLI parser.
    """
    run_p = sub.add_parser("run", help="run a pipeline YAML over inputs")
    run_p.add_argument("pipeline", help="pipeline YAML path")
    run_p.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        help="one or more input paths, or a glob (quote globs so the shell keeps them)",
    )
    run_p.add_argument("--out", required=True, help="output directory")
    run_p.add_argument(
        "--on-error",
        dest="on_error",
        choices=("skip", "raise"),
        default="skip",
        help="error policy (default: skip)",
    )
    run_p.add_argument(
        "--parallel",
        action="store_true",
        help="run the batch across a process pool (path inputs only)",
    )
    run_p.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="worker-process count for --parallel (default: ~CPU count)",
    )
    run_p.add_argument(
        "--manifest",
        help="write a provenance manifest (reloadable pipeline + per-input timings) here",
    )
    run_p.set_defaults(func=_cmd_run)

    apply_p = sub.add_parser(
        "apply", help="run one registered tool over a single input"
    )
    apply_p.add_argument("name", help="tool name (see 'pyramids tools')")
    apply_p.add_argument("input", help="source raster or vector path")
    apply_p.add_argument("output", help="destination path")
    apply_p.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="set a tool parameter; repeat for each (see 'pyramids tool NAME')",
    )
    apply_p.add_argument(
        "--overwrite", action="store_true", help="replace the output if it exists"
    )
    apply_p.set_defaults(func=_cmd_apply)

    tools_p = sub.add_parser("tools", help="list registered processing tools")
    tools_p.set_defaults(func=_cmd_tools)

    tool_p = sub.add_parser("tool", help="print a tool's parameter schema")
    tool_p.add_argument("name", help="tool name (see 'pyramids tools')")
    tool_p.set_defaults(func=_cmd_tool)
