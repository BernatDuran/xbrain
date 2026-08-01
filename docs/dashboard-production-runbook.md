# Dashboard Production Runbook

This runbook is for operating the served XBrain dashboard from `main`. The
current production dashboard is the forest light/dark workspace UI served behind
Basic Auth at `brainxterminal.duckdns.org`.

## Scope

- Source template: `src/xbrain/resources/dashboard.template.html`.
- Generated artifact: `<vault>/<output_subdir>/dashboard.html`.
- Local server: `uv run xbrain serve-dashboard --host 127.0.0.1 --port 8765`.
- Current production service: `/etc/systemd/system/xbrain-dashboard.service`.
- Current public proxy: `/etc/nginx/sites-available/brainxterminal`, with
  `brainxterminal.duckdns.org` protected by Basic Auth and proxied to
  `http://127.0.0.1:8765`.

The served dashboard exposes operational actions (`/api/refresh-all`,
`/api/retry-failed`, `/api/chat`, note repair paths). Do not bind it directly to
an unauthenticated public interface. Prefer `127.0.0.1` plus authenticated
reverse proxy access.

## Release Steps

1. Check out the production branch and pull the latest GitHub state.

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

2. Generate the latest dashboard artifact from the current XBrain data.

   ```bash
   uv run xbrain generate
   ```

3. Restart the local served dashboard.

   ```bash
   systemctl restart xbrain-dashboard.service
   ```

4. Confirm the production reverse proxy still points at
   `http://127.0.0.1:8765`.

   ```bash
   nginx -T | grep -A20 brainxterminal.duckdns.org
   ```

5. Open the protected production URL and verify these routes:

   - `/#overview`: workspace navigation, KPI strip, signal rail, and chart
     drill-down buttons.
   - `/#atlas`: Atlas Knowledge Graph Map/Timeline/List, topic explorer,
     source/topic charts, and storage overview.
   - `/#ops`: refresh, retry, and command surfaces.
   - `/#ask`: library chat surface.

## Smoke Checks

Run these on the production host:

```bash
curl -I http://127.0.0.1:8765/dashboard.html
curl http://127.0.0.1:8765/api/status
curl -I https://brainxterminal.duckdns.org
```

Expected result: the dashboard request returns `200`, and `/api/status` returns
JSON with pipeline state fields. If the public URL is intentionally protected,
an unauthenticated outside request may return `401`; verify through the
authorized path instead of disabling protection. A protected public response
should include `WWW-Authenticate: Basic realm="BrainX Dashboard"`.

## Validation Evidence

Before restarting production after code changes, run:

```bash
uv run pytest tests/test_dashboard.py -q
uv run pytest -q
```

The dashboard test suite covers workspace routing, theme tokens, compact KPI and
Ops layout, accessible drawer behavior, chart drill-down controls,
reduced-motion behavior, Atlas Knowledge Graph controls, and copy consistency.
Use a browser or Playwright against the served dashboard for final visual checks
at desktop and mobile widths. Include `/notes?path=...` in visual checks because
the web note viewer is served by the same process.

See `docs/atlas-knowledge-graph.md` for the graph data contract, controls,
local layout persistence, and served topic-action behavior.

## Rollback

1. Switch back to the previous known-good commit or tag on `main`.
2. Regenerate the dashboard with `uv run xbrain generate`.
3. Restart `xbrain-dashboard.service` and reload Nginx if proxy settings changed.
4. Re-run the smoke checks above.
