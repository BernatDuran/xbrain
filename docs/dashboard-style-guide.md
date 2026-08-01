# XBrain Dashboard Style Guide

## Product Direction

XBrain is a private intelligence terminal for a saved knowledge corpus. The UI should feel calm, precise, and premium: more command center than marketing page. It prioritizes scanning, triage, evidence, and confident navigation across the corpus.

## Theme

The dashboard uses a forest editorial system with paired light and dark themes. Every visible color must route through CSS custom properties in `dashboard.template.html`; chart colors are read from the computed theme at runtime.

Core roles:

- `--color-bg`: page background.
- `--color-panel`, `--color-panel-raised`, `--color-panel-soft`: surfaces.
- `--color-text`, `--color-text-muted`, `--color-text-faint`: text hierarchy.
- `--color-border`, `--color-border-strong`: borders and dividers.
- `--color-accent`: primary action and selected state.
- `--color-forest`, `--color-moss`, `--color-fern`, `--color-clay`, `--color-sky`, `--color-amber`, `--color-stone`, `--color-bark`: chart and status accents.
- `--color-danger`: failures and risk states.
- `--shadow-panel`, `--shadow-drawer`, `--shadow-focus`: elevation and focus.

## Type

- Display: `Fraunces` for brand and major numbers only.
- UI: `Hanken Grotesk` for labels, prose, and controls.
- Mono: `Spline Sans Mono` for timestamps, commands, metadata, and compact technical labels.

Scale:

- Hero/brand: 44-58px responsive via media queries, never viewport-scaled.
- Panel title: 18-24px.
- Body: 14-15px.
- Metadata/chips: 9-12px.

## Components

- Cards are used only for repeated metrics, panels, modals, and the drawer. No card-in-card nesting.
- Segmented controls express view or data range choices.
- Workspace navigation separates Overview, Atlas, Ops, and Ask so the app never exposes the whole analytical surface at once.
- Signal rail cards are compact command surfaces: they summarize one live metric, show proportional state, and navigate to the relevant detail.
- Icon buttons are square, token-sized, and include accessible labels/tooltips.
- Chips use the mono font, uppercase labels, and semantic colors.
- Charts must use the active CSS theme, not hardcoded JS colors.
- Drawers are used for drill-down detail so the primary dashboard remains dense.
- The served note viewer (`/notes`) uses the same forest tokens, theme persistence, compact controls, and editorial reading rhythm as the dashboard.

## Interaction Principles

- First screen: live product dashboard, not a landing page.
- Preserve all existing dashboard actions: refresh, retry failures, command copy, drill-down charts, note links, and library chat.
- Theme toggle persists in local storage and must not require a server.
- Controls need clear hover, focus, active, disabled, and loading states.
- Mobile layout keeps charts scannable with ranked fallbacks where a canvas chart is too dense.
- Production serving must stay local or behind authenticated reverse proxy access because the dashboard includes Ops and Ask actions.
- KPI summaries stay as a compact 4-column system so eight metrics render as two dense rows on the primary dashboard surface.
- Ops is a compact command surface: actions, pending counters, command copy, and triage signals should fit without turning the workspace into a second dashboard.

## Copy Tone

Copy is concise, operational, and specific. Avoid generic dashboard filler. Prefer verbs and precise nouns: "Refresh pipeline", "Evidence stream", "Review queue", "Captured articles".
