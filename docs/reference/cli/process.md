# process — geoprocessing pipelines & tools

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids process --help` for the live
listing and `pyramids process <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?*
suggestion.

## Command tree

- `apply` — run one registered tool over a single input
- `run` — run a pipeline YAML over inputs
- `tool` …
  - `list` — list registered processing tools
  - `show` — print a tool's parameter schema

## Core verbs

Reached directly as `pyramids process <verb>`.

#### `pyramids process apply`

run one registered tool over a single input

```text
pyramids process apply [-h] [--set KEY=VALUE] [--overwrite] name input output
```

| argument | | description |
|---|---|---|
| `name` | required | tool name (see 'pyramids process tool list') |
| `input` | required | source raster or vector path |
| `output` | required | destination path |

| option | | description |
|---|---|---|
| `--set` | optional | set a tool parameter; repeat for each (see 'pyramids process tool show NAME') |
| `--overwrite` | optional | replace the output if it exists |

#### `pyramids process run`

run a pipeline YAML over inputs

```text
pyramids process run [-h] --inputs INPUTS [INPUTS ...] --out OUT [--on-error {skip,raise}] [--parallel] [--max-workers MAX_WORKERS] [--manifest MANIFEST] pipeline
```

| argument | | description |
|---|---|---|
| `pipeline` | required | pipeline YAML path |

| option | | description |
|---|---|---|
| `--inputs` | **required** | one or more input paths, or a glob (quote globs so the shell keeps them) |
| `--out` | **required** | output directory |
| `--on-error` | optional | error policy (default: skip) (choices: `skip`, `raise`) |
| `--parallel` | optional | run the batch across a process pool (path inputs only) |
| `--max-workers` | optional | worker-process count for --parallel (default: ~CPU count) |
| `--manifest` | optional | write a provenance manifest (reloadable pipeline + per-input timings) here |

## Subsystem `process tool`

Inspect the registered processing tools.

#### `pyramids process tool list`

list registered processing tools

```text
pyramids process tool list [-h]
```

#### `pyramids process tool show`

print a tool's parameter schema

```text
pyramids process tool show [-h] name
```

| argument | | description |
|---|---|---|
| `name` | required | tool name (see 'pyramids process tool list') |
