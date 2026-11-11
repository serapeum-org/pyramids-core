# stac — SpatioTemporal Asset Catalog access

Commands follow the `pyramids <domain> [<subsystem>] <verb>` convention. Run `pyramids stac --help` for the live
listing and `pyramids stac <subsystem> --help` for a subsystem. An unknown verb prints a *did you mean …?* suggestion.

## Command tree

- `download` — download assets from a STAC search
- `search` — search a STAC API

## Core verbs

Reached directly as `pyramids stac <verb>`.

#### `pyramids stac download`

download assets from a STAC search

```text
pyramids stac download [-h] --collections COLLECTIONS [COLLECTIONS ...] [--bbox MINX MINY MAXX MAXY] [--datetime DATETIME] [--assets ASSETS [ASSETS ...]] [--max-items MAX_ITEMS] --out OUT url
```

| argument | | description |
|---|---|---|
| `url` | required | STAC API root URL |

| option | | description |
|---|---|---|
| `--collections` | **required** | collection id(s) |
| `--bbox` | optional |  |
| `--datetime` | optional | RFC3339 instant or range |
| `--assets` | optional | asset keys to include (default: all) |
| `--max-items` | optional |  |
| `--out` | **required** | output directory |

#### `pyramids stac search`

search a STAC API

```text
pyramids stac search [-h] --collections COLLECTIONS [COLLECTIONS ...] [--bbox MINX MINY MAXX MAXY] [--datetime DATETIME] [--max-items MAX_ITEMS] [--json] url
```

| argument | | description |
|---|---|---|
| `url` | required | STAC API root URL |

| option | | description |
|---|---|---|
| `--collections` | **required** | collection id(s) |
| `--bbox` | optional |  |
| `--datetime` | optional | RFC3339 instant or range (e.g. 2020-01-01/2020-12-31) |
| `--max-items` | optional |  |
| `--json` | optional | emit JSON |
