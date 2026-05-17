# XBrain (`xbrain`)

> Your X bookmarks and posts, turned into a second brain.

Your second brain captures what you *write*. It never captures what you *consume* —
the things you bookmark, quote and reply to. XBrain closes that gap: it extracts
your X bookmarks and your own tweets, stores them as structured JSON, and generates
a layered Obsidian wiki you can actually navigate and search.

Built to feed teaching material. Runs locally, no paid API.

## What you get

A three-layer wiki in your Obsidian vault:

1. **Items** — one note per bookmark/tweet that links out, with the linked article
   fetched in full.
2. **Topics** — synthesis pages that distill dozens or hundreds of posts into a
   single readable essay, cross-linked to the items.
3. **Index** — the map.

## Setup

```bash
uv venv
uv pip install -e ".[dev]" --index-url https://pypi.org/simple
uv run playwright install chromium
```

Copy `config.toml.example` to `config.toml` and set your vault path and X handle.

## Authentication

XBrain needs a logged-in X session. The reliable path is importing cookies from
your real Chrome browser:

```bash
uv pip install browser-cookie3 --index-url https://pypi.org/simple
# Log in to X in Chrome first, then:
python scripts/import_chrome_session.py
```

`xbrain login` (an in-app Playwright login) also exists, but it is unreliable with
accounts that sign in through Google/SSO. The cookie import is recommended.

## Pipeline

| Command | What it does |
|---------|--------------|
| `xbrain extract` | Extract bookmarks + own tweets (incremental) |
| `xbrain import-archive <zip>` | Backfill own-tweet history from the X data archive |
| `xbrain fetch` | Download linked article content + expand threads |
| `xbrain enrich` | LLM enrichment of items (summary, topics) |
| `xbrain generate` | Render the wiki into your vault |
| `xbrain sync` | `extract` + `fetch` + `generate` |
| `xbrain status` | Counts and last-run info |

All stages accept `--since` / `--until` (ISO dates).

## Data

`data/items.json` is the source of truth; the markdown wiki is derived and safe to
regenerate. Content you write below the `xbrain:generated` marker in a note is
preserved across regenerations. `data/` is gitignored — your data never leaves
your machine.

## Tests

```bash
uv run pytest -v
```

## Responsible use

XBrain reads X through X's internal (non-public) endpoints. Use it for personal
purposes, with your own X account and your own data, at your own risk. It does not
use a paid API and it does not redistribute anyone else's content. Respect X's
Terms of Service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs written with AI agents are welcome — at
the same quality bar as any other code.

## License

MIT — see [LICENSE](LICENSE).
