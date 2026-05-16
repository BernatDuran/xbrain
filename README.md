# X Knowledge Base (`xkb`)

Extracts X (Twitter) bookmarks and own tweets into a structured JSON store and
generates an Obsidian markdown wiki from it. Built to feed teaching material.

## Setup

```bash
uv venv
uv pip install -e ".[dev]" --index-url https://pypi.org/simple
uv run playwright install chromium
```

Edit `config.toml` (vault path, X handle) and `courses.yaml` (course list).

## Pipeline

| Command | What it does |
|---------|--------------|
| `xkb login` | Open a browser to log in to X; saves the session |
| `xkb extract` | Extract bookmarks + own tweets (incremental) |
| `xkb import-archive <zip>` | Backfill own-tweet history from the X data archive |
| `xkb fetch` | Download linked article content + expand threads |
| `xkb enrich` | LLM enrichment — IN PAUSE (see spec §9) |
| `xkb generate` | Render markdown into the vault |
| `xkb sync` | `extract` + `fetch` + `generate` |
| `xkb status` | Counts and last-run info |

All stages accept `--since` / `--until` (ISO dates).

## Data

`data/items.json` is the source of truth (git-tracked). `auth/` holds the X
session and is never committed. Markdown is fully derived — safe to regenerate;
content you write below the `xkb:generated` marker in a note is preserved.

## Tests

```bash
uv run pytest -v
```

Design spec and implementation plan live in the Obsidian vault under
`zz-support-files/docs/`.
