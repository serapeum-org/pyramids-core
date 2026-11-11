# plugins — third-party CLI plugin discovery

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids plugins --help` for the live
listing and `pyramids plugins <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?*
suggestion.

## Command tree

- `list` — list installed CLI plugins and their load status

## Core verbs

Reached directly as `pyramids plugins <verb>`.

#### `pyramids plugins list`

list installed CLI plugins and their load status

```text
pyramids plugins list [-h] [--json]
```

| option | | description |
|---|---|---|
| `--json` | optional | emit JSON |

