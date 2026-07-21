"""Command-line interface for XBrain."""

from __future__ import annotations

import enum
import functools
import html
import json
import logging
import os
import re
import threading
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import typer
from markdown_it import MarkdownIt

from xbrain import snapshot
from xbrain.archive import parse_archive
from xbrain.chat import MAX_QUESTION_CHARS, answer_question
from xbrain.config import Config, load_config
from xbrain.content_policy import prune_store, retained_store, should_keep_item
from xbrain.describe import apply_describe_worksheet, export_describe_worksheet
from xbrain.describe import describe_all as run_describe_all
from xbrain.describe import emit_summary_line as describe_emit_summary_line
from xbrain.diff import diff_snapshots, format_json, format_text
from xbrain.digest import VisualConfig, digest_videos, format_digest_summary
from xbrain.enrich import (
    apply_worksheet_judgments,
    enrich_selected_with_executor,
    enrich_with_executor,
    items_for_taxonomy_reenrichment,
    items_pending_enrichment,
)
from xbrain.executors.api import ApiExecutor
from xbrain.extract.browser import login as run_login
from xbrain.extract.browser import x_context
from xbrain.extract.extractor import RateLimitTruncated, extract_source
from xbrain.extract.threads import expand_threads
from xbrain.fetch import fetch_pending
from xbrain.fetch_x import fetch_x_articles
from xbrain.generate import generate as run_generate
from xbrain.llm_client import validate_llm_model
from xbrain.media import download_all as run_media_download
from xbrain.media import emit_summary_line as media_emit_summary_line
from xbrain.models import ArchiveImport, Author, ContentSourceFailure, Item, SourceName, Topic
from xbrain.notes_io import GEN_END, GEN_START
from xbrain.refresh import estimate_download_size, refresh_video_media
from xbrain.rubrics import load_vocab, save_vocab
from xbrain.store import (
    load_state,
    load_store,
    load_topic_pages,
    merge_items,
    save_state,
    save_store,
    save_topic_pages,
)
from xbrain.topic_synth import (
    apply_overview_judgments,
    export_topic_worksheet,
    import_topic_worksheet,
    synthesize_overviews_api,
)
from xbrain.topics import (
    build_topic_inputs,
    compute_topic_posts,
    merge_overviews,
    topics_needing_synth,
    write_topic_pages,
)
from xbrain.transcribe import Transcript, transcribe_media
from xbrain.video_fetch import (
    FetchReport,
    fetch_result_to_json,
    fetch_videos,
    format_fetch_summary,
)
from xbrain.video_frames import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_SCENE_THRESHOLD,
    KeyFrame,
    extract_key_frames,
)
from xbrain.video_media import (
    VideoDownloadPlan,
    VideoReport,
    emit_video_summary_line,
    format_size_gate,
    parse_size_to_bytes,
    plan_video_downloads,
)
from xbrain.video_media import download_videos as run_download_videos
from xbrain.video_select import format_video_table, list_video_entries, row_to_json
from xbrain.vision import describe_image, describe_image_with_llm
from xbrain.vocab import (
    apply_vocab_worksheet,
    export_vocab_worksheet,
    import_vocab_worksheet,
    induce_vocab,
)
from xbrain.worksheet import export_worksheet, import_worksheet

app = typer.Typer(help="XBrain — artículos y vídeos guardados en X a un wiki de Obsidian")

_BOOKMARKS_URL = "https://x.com/i/bookmarks"

_HEADLESS_HELP = (
    "Navegador oculto. Por defecto headful (visible) — más difícil de "
    "fingerprintear como bot. Usa --headless en runs desatendidos sin display."
)


@dataclass(frozen=True)
class ExtractReport:
    source: SourceName
    seen: int
    added: int
    already_known: int
    duplicates: int
    discarded: int = 0


@dataclass
class DashboardRunState:
    running: bool = False
    action: str | None = None
    last_action: str | None = None
    last_started: str | None = None
    last_finished: str | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class RetryFailedReport:
    candidates: int
    articles: int
    x_articles: int


@app.callback()
def _configure_logging() -> None:
    """Surface library `logging` warnings (e.g. the 429 backoff notice) cleanly.

    Without a configured handler these fall to Python's last-resort handler with
    an ugly `WARNING:logger:` prefix; route warnings through a plain stderr stream
    so the user sees the backoff message during a long pause.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")


class Source(str, enum.Enum):
    bookmarks = "bookmarks"
    tweets = "tweets"
    all = "all"


class VideoStatus(str, enum.Enum):
    """The `list-videos --status` filter values (mirrors the four `VideoState`s)."""

    downloaded = "downloaded"
    failed = "failed"
    pending = "pending"
    poster_era = "poster-era"


def _repo_root() -> Path:
    """Repo root — overridable via XBRAIN_REPO_ROOT for tests."""
    override = os.environ.get("XBRAIN_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def _config() -> Config:
    return load_config(_repo_root())


def _parse_date(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse an ISO date/datetime into a UTC-aware datetime.

    A *date-only* ``value`` (e.g. ``2025-12-31``) carries no time component,
    so it parses to that day's midnight. For a ``since`` bound that is the
    correct day start. For an ``until`` bound (``end_of_day=True``) midnight
    would exclude the whole final day, so we snap it to the last microsecond
    (``23:59:59.999999`` UTC) — the ``item.created_at > until`` filters then
    include every item created on that day. An explicit time
    (e.g. ``2025-12-31T09:00``) is respected as-is and never snapped.
    """
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if end_of_day and _is_date_only(value):
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


# A bare ISO date (``YYYY-MM-DD``) optionally carrying a tz offset (``+00:00``,
# ``-0500``, ``Z``) but NO time-of-day. A time-of-day is always introduced by a
# ``T``/space separator, so ``2025-12-31T09:00:00`` and ``2025-12-31 120000``
# never match — only whole-day bounds do.
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[Zz]|[+-]\d{2}:?\d{2})?")


def _is_date_only(value: str) -> bool:
    """True when an ISO string is a bare date (no time-of-day), so an ``until``
    bound should cover the whole day. See ``_DATE_ONLY_RE``."""
    return _DATE_ONLY_RE.fullmatch(value) is not None


_OPERATOR_ERRORS = (
    FileNotFoundError,
    ValueError,
    KeyError,
    RuntimeError,
    NotImplementedError,
    # OSError covers PermissionError, FileExistsError, IsADirectoryError, etc.
    # The snapshot module hits these on permission or disk issues — they should
    # surface as a clean exit-1, not a raw traceback.
    OSError,
    # NOTE: MemoryError is deliberately NOT here — a global catch would swallow
    # OOM stacks for every command and print an empty "Error: ". `download-videos`
    # handles a too-large body LOCALLY in `_download_one_video` (records the cause
    # + continues the batch); see `xbrain.video_media`.
)


def _handle_cli_errors(func: Callable) -> Callable:
    """Surface expected operator errors as a clean message + exit code 1."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except _OPERATOR_ERRORS as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


def _report_invalid(invalid: list[tuple[str, list[str]]]) -> None:
    if invalid:
        typer.echo(f"Rechazados por el validador: {len(invalid)}", err=True)
        for item_id, errors in invalid:
            typer.echo(f"  {item_id}: {'; '.join(errors)}", err=True)


def _format_extract_report(report: ExtractReport) -> str:
    text = (
        f"{report.source}: {report.added} nuevos, {report.already_known} ya existentes, "
        f"{report.duplicates} duplicados en lote ({report.seen} vistos)"
    )
    if report.discarded:
        text += f" · {report.discarded} descartados por política"
    return text


def _run_extract(
    cfg: Config,
    source: str,
    since: datetime | None,
    until: datetime | None,
    *,
    headless: bool = False,
) -> list[ExtractReport]:
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    targets = {
        "bookmark": _BOOKMARKS_URL,
        "own_tweet": f"https://x.com/{cfg.x_handle}",
    }
    source_sets: dict[str, list[SourceName]] = {
        "bookmarks": ["bookmark"],
        "tweets": [],
        "all": ["bookmark"],
    }
    chosen = source_sets[source]
    if not chosen:
        typer.echo("Extracción de tweets propios desactivada por política: 0 items.")
        return []
    known_ids = set(store)
    reports: list[ExtractReport] = []
    truncated: list[str] = []
    with x_context(cfg.storage_state_path, headless=headless) as context:
        for src in chosen:
            cursor = state.bookmarks if src == "bookmark" else state.own_tweets
            first_run = cursor.last_seen_id is None
            try:
                items = extract_source(context, src, targets[src], known_ids, since, until)
            except RateLimitTruncated as exc:
                # A truncated run is a partial, non-contiguous batch. Merging it
                # (and advancing the cursor) would seal a permanent gap in the
                # incremental store, so persist NOTHING for this source and fail
                # loud; the next run re-scrolls the window cleanly.
                typer.echo(f"ERROR: {exc} (no se guardó nada de {src})", err=True)
                truncated.append(src)
                continue
            if not items and first_run:
                typer.echo(
                    f"AVISO: {src} devolvió 0 items en una extracción inicial — "
                    "revisa la sesión de X o el parser GraphQL (spec §6).",
                    err=True,
                )
            kept_items = [item for item in items if should_keep_item(item)]
            discarded = len(items) - len(kept_items)
            before_ids = set(store)
            unique_ids = {item.id for item in kept_items}
            added = merge_items(store, kept_items)
            report = ExtractReport(
                source=src,
                seen=len(items),
                added=added,
                already_known=len(unique_ids & before_ids),
                duplicates=len(kept_items) - len(unique_ids),
                discarded=discarded,
            )
            reports.append(report)
            if items:
                cursor.last_seen_id = max(items, key=lambda i: int(i.id)).id
            cursor.last_run = datetime.now(timezone.utc)
            typer.echo(_format_extract_report(report))
    save_store(store, cfg.items_path)
    save_state(state, cfg.state_path)
    if truncated:
        raise RuntimeError(
            f"Extracción truncada por rate-limit/bloqueo de X en: {', '.join(truncated)}. "
            "Las fuentes completadas se guardaron; reanuda más tarde para el resto."
        )
    return reports


def _auto_snapshot(cfg: Config, command: str) -> None:
    """Snapshot data/ before a destructive op and echo the path + item count.

    Called from every destructive code path (vocab --regenerate, topics
    --resynth, fetch --force). The manifest's `command` field carries the
    destructive op name (e.g. `vocab-regenerate`); the directory label uses
    the `pre-<op>` prefix so the listing is self-describing.

    Any failure here propagates and aborts the destructive op — a snapshot we
    can't take must not be silently skipped.
    """
    path, manifest = snapshot.snapshot_create(
        cfg.data_dir,
        command=command,
        dir_label=f"pre-{command}",
    )
    typer.echo(f"Snapshot created: {path.name} ({manifest.item_count} items)")


def _format_size_estimate(estimated_bytes: int, n_estimable: int, n_unknown: int) -> str:
    """The human download-size line; never prints '~0.0 GB' when nothing is estimable.

    With at least one estimable video, reports the GB sum plus the unknown
    count. With none estimable, says the size is unknown for the N videos that
    carry no bitrate/duration (so a large unknown count never misreads as
    "~0.0 GB, nothing to download"), and reports "no videos" when there are none.
    """
    if n_estimable == 0:
        if n_unknown == 0:
            return "Estimated video download: no videos in the store."
        return (
            f"Estimated video download: size unknown for {n_unknown} videos "
            "(no bitrate/duration captured)."
        )
    gigabytes = estimated_bytes / 1_000_000_000
    return (
        f"Estimated video download: ~{gigabytes:.1f} GB across {n_estimable} videos; "
        f"{n_unknown} with unknown size."
    )


def _run_refresh_media(cfg: Config, source: str, *, force: bool, headless: bool = False) -> None:
    """Re-capture the FULL X history and backfill playable video media in place.

    Destructive — it overwrites the video entries on existing items — so it
    auto-snapshots `data/` first (label `pre-refresh-media`); a snapshot failure
    propagates and aborts before any capture or write (CONTRIBUTING §Safety).

    Then it scrolls each chosen source with an EMPTY `known_ids` set, so
    `extract_source` does NOT stop at the first known id and the whole timeline
    is walked. The freshly-parsed items are merged onto the store by
    `refresh_video_media` — video entries only; photos and every enrichment /
    description / fetch field are preserved. The store is saved and a
    download-size estimate is printed. Video DOWNLOAD is out of scope here.

    Empty-capture guard: `extract_source` returns `[]` (it does NOT raise) when
    the session is logged in but the GraphQL parser drifts or the scroll is
    interrupted. Re-seeing 0 known items against a NON-EMPTY store is therefore
    a likely-broken run, not success — it surfaces a loud warning and aborts
    non-zero WITHOUT saving (the merge was a no-op, so the store on disk is
    untouched and the pre-snapshot already fired). `--force` downgrades this to
    a warning and proceeds. An empty store (fresh project) and any non-zero
    capture (monotonic, re-runnable progress) are left to save normally.
    """
    _auto_snapshot(cfg, "refresh-media")
    store = load_store(cfg.items_path)
    # Mirrors `_run_extract` — the source → (target URL, GraphQL source) mapping.
    targets = {
        "bookmark": _BOOKMARKS_URL,
        "own_tweet": f"https://x.com/{cfg.x_handle}",
    }
    source_sets: dict[str, list[SourceName]] = {
        "bookmarks": ["bookmark"],
        "tweets": [],
        "all": ["bookmark"],
    }
    chosen = source_sets[source]
    typer.echo(
        "refresh-media scrolls the FULL X history with no skip-known — this is "
        "slow and human-paced and can take many minutes. Leave it running."
    )
    fresh: list[Item] = []
    with x_context(cfg.storage_state_path, headless=headless) as context:
        for src in chosen:
            # Empty known_ids disables the skip-known early-stop: the whole
            # history is returned, not just the items newer than the cursor.
            # Unlike `_run_extract`, the `state.json` cursors are intentionally
            # left untouched — this is a backfill of existing records, not an
            # incremental advance, so the next `extract` cursor must not move.
            fresh.extend(extract_source(context, src, targets[src], set()))
    report = refresh_video_media(store, fresh)

    if store and report.items_seen == 0:
        warning = (
            f"refresh-media re-vio 0 de los {len(store)} items ya conocidos — "
            "la sesión de X probablemente caducó o el parser GraphQL ha derivado "
            "(spec §6); no se actualizó nada."
        )
        if not force:
            # Nothing matched, so the store is unchanged — not saving is
            # byte-identical and the pre-snapshot already fired. Abort non-zero
            # so a broken capture never reports success.
            raise RuntimeError(f"{warning} Usa --force para guardar igualmente.")
        typer.echo(f"AVISO: {warning}", err=True)

    save_store(store, cfg.items_path)
    estimated_bytes, n_estimable, n_unknown = estimate_download_size(store)
    typer.echo(
        f"refresh-media: {report.items_seen} known items re-seen, "
        f"{report.items_refreshed} refreshed, {report.videos_updated} videos updated; "
        f"{report.items_with_video_not_seen} video items not re-seen (still poster-era)."
    )
    typer.echo(_format_size_estimate(estimated_bytes, n_estimable, n_unknown))


def _run_fetch(
    cfg: Config,
    since: datetime | None,
    until: datetime | None,
    force: bool,
    *,
    headless: bool = False,
) -> None:
    if force:
        _auto_snapshot(cfg, "fetch-force")
    store = load_store(cfg.items_path)
    snapshot_taken = force
    if any(not should_keep_item(item) for item in store.values()) and not snapshot_taken:
        _auto_snapshot(cfg, "prune-content-policy")
        snapshot_taken = True
    try:
        articles = fetch_pending(store, since, until, force)
        x_articles = fetch_x_articles(
            store, cfg.storage_state_path, force, since, until, headless=headless
        )
        threads = expand_threads(store, cfg.storage_state_path, force, headless=headless)
    finally:
        discarded = prune_store(store)
        # Persist whatever was fetched even if a later stage raised — a stage
        # error (e.g. an expired X session) must not discard in-memory work.
        save_store(store, cfg.items_path)
    suffix = f" · {len(discarded)} descartados por política" if discarded else ""
    typer.echo(
        f"Contenido descargado: {articles} artículos, {x_articles} de X, {threads} hilos{suffix}"
    )


_RETRY_SOURCE_SETS: dict[str, list[SourceName]] = {
    "bookmarks": ["bookmark"],
    "tweets": [],
    "all": ["bookmark"],
}


def _failed_article_kinds(item: Item) -> set[str]:
    """The failed article source kinds present on an item."""
    if item.content is None:
        return set()
    return {
        source.kind
        for source in item.content.sources
        if isinstance(source, ContentSourceFailure)
        and source.kind in ("external_article", "x_article")
    }


def _failed_article_items(
    store: dict[str, Item], source: str = "bookmarks"
) -> dict[str, tuple[Item, set[str]]]:
    """Items whose fetched linked content has failed, scoped by X source."""
    chosen = set(_RETRY_SOURCE_SETS[source])
    return {
        item_id: (item, kinds)
        for item_id, item in store.items()
        if item.source in chosen and (kinds := _failed_article_kinds(item))
    }


def _run_retry_failed_articles(
    cfg: Config,
    source: str = "bookmarks",
    *,
    headless: bool = True,
    regenerate: bool = True,
) -> RetryFailedReport:
    """Force-retry only items with failed linked article/X-article sources."""
    store = load_store(cfg.items_path)
    failed = _failed_article_items(store, source)
    if not failed:
        label = "bookmarks" if source == "bookmarks" else source
        typer.echo(f"No hay {label} fallidos que relanzar.")
        return RetryFailedReport(candidates=0, articles=0, x_articles=0)

    _auto_snapshot(cfg, f"retry-failed-{source}")
    external_scoped = {
        item_id: item for item_id, (item, kinds) in failed.items() if "external_article" in kinds
    }
    x_scoped = {item_id: item for item_id, (item, kinds) in failed.items() if "x_article" in kinds}

    try:
        articles = fetch_pending(external_scoped, force=True) if external_scoped else 0
        x_articles = (
            fetch_x_articles(x_scoped, cfg.storage_state_path, True, headless=headless)
            if x_scoped
            else 0
        )
    finally:
        save_store(store, cfg.items_path)

    report = RetryFailedReport(
        candidates=len(failed),
        articles=articles,
        x_articles=x_articles,
    )
    typer.echo(
        f"Fallidos relanzados: {report.candidates} items "
        f"({report.articles} artículos, {report.x_articles} de X)"
    )
    if regenerate:
        _run_generate(cfg, None, None)
    return report


def _run_generate(cfg: Config, since: datetime | None, until: datetime | None) -> None:
    store = load_store(cfg.items_path)
    store = retained_store(store)
    topic_pages = load_topic_pages(cfg.topics_path) if cfg.topics_path.exists() else {}
    run_generate(
        store,
        cfg.output_dir,
        since,
        until,
        cfg.output_language,
        cfg.topic_style,
        media_root=cfg.media_dir,
        topic_pages=topic_pages,
    )
    typer.echo(f"Markdown generado en {cfg.output_dir}")


def _run_enrich_api(
    cfg: Config,
    since: datetime | None,
    until: datetime | None,
    *,
    taxonomy_risk: bool = False,
) -> None:
    """Run API enrichment with the configured provider and persist the store."""
    store = load_store(cfg.items_path)
    library_store = retained_store(store)
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab_topics:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    executor = ApiExecutor(
        model=cfg.enrich_model,
        output_language=cfg.output_language,
        provider=cfg.llm_provider,
        base_url=cfg.llm_base_url,
    )
    if taxonomy_risk:
        candidates = items_for_taxonomy_reenrichment(library_store, since, until)
        if not candidates:
            typer.echo("No hay items con riesgo taxonómico para re-enriquecer.")
            return
        _auto_snapshot(cfg, "enrich-taxonomy-risk")
        enriched, invalid = enrich_selected_with_executor(
            library_store, executor, vocab_topics, candidates
        )
        typer.echo(f"Re-enriquecidos por riesgo taxonómico: {enriched}/{len(candidates)} items")
    else:
        enriched, invalid = enrich_with_executor(
            library_store, executor, vocab_topics, since, until
        )
        typer.echo(f"Enriquecidos: {enriched} items")
    save_store(store, cfg.items_path)
    _report_invalid(invalid)


def _run_topics_executor(cfg: Config, executor: str | None, *, resynth: bool = False) -> None:
    """Run topic-page update/synthesis through the existing topic pipeline."""
    store = load_store(cfg.items_path)
    store = retained_store(store)
    vocab = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    _topics_run(cfg, store, vocab, resynth, executor)


def _run_refresh_all(
    cfg: Config,
    *,
    source: str = "bookmarks",
    since: datetime | None = None,
    until: datetime | None = None,
    headless: bool = True,
    executor: str = "api",
    media_limit: int | None = None,
    describe_limit: int | None = None,
    describe_batch_size: int = 5,
    skip_media: bool = False,
    skip_describe: bool = False,
) -> None:
    """One-command daily ingestion pipeline for unattended/mobile runs."""
    if executor != "api":
        raise ValueError("refresh-all solo soporta --executor api por ahora.")
    typer.echo("refresh-all: 1/7 extract")
    _run_extract(cfg, source, since, until, headless=headless)
    typer.echo("refresh-all: 2/7 fetch")
    _run_fetch(cfg, since, until, False, headless=headless)
    if skip_media:
        typer.echo("refresh-all: 3/7 media saltado")
    else:
        typer.echo("refresh-all: 3/7 media")
        _run_media(cfg, force=False, limit=media_limit, items_filter=None)
    if skip_describe:
        typer.echo("refresh-all: 4/7 describe saltado")
    else:
        typer.echo("refresh-all: 4/7 describe")
        _run_describe(
            cfg,
            force=False,
            limit=describe_limit,
            items_filter=None,
            model=cfg.llm_vision_model,
            batch_size=describe_batch_size,
            verbose=False,
        )
    typer.echo("refresh-all: 5/7 enrich")
    _run_enrich_api(cfg, since, until)
    typer.echo("refresh-all: 6/7 topics")
    _run_topics_executor(cfg, executor, resynth=False)
    typer.echo("refresh-all: 7/7 generate")
    _run_generate(cfg, since, until)
    typer.echo("refresh-all: completado")


def _resolve_served_note(output_dir: Path, raw_path: str) -> Path:
    """Resolve one dashboard note path without allowing filesystem escape."""
    if not raw_path.strip():
        raise ValueError("missing note path")

    root = output_dir.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError("note not found") from exc

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError("note path is outside the generated vault") from exc

    if resolved.suffix.lower() != ".md":
        raise ValueError("only markdown notes can be opened")
    return resolved


_WEB_IMAGE_SUFFIXES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _resolve_served_media(output_dir: Path, raw_path: str) -> Path:
    """Resolve one generated media file for the web note viewer.

    The route intentionally serves only files below `<output>/_media` and only
    known image suffixes. Notes can reference local media, but arbitrary files in
    the generated vault or elsewhere must not become downloadable through the
    dashboard server.
    """
    if not raw_path.strip():
        raise ValueError("missing media path")
    root = (output_dir / "_media").resolve()
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise PermissionError("absolute media paths are not allowed")
    if candidate.parts and candidate.parts[0] == "_media":
        candidate = Path(*candidate.parts[1:])
    try:
        resolved = (root / candidate).resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError("media not found") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError("media path is outside generated media") from exc
    if resolved.suffix.lower() not in _WEB_IMAGE_SUFFIXES:
        raise ValueError("only generated image media can be opened")
    return resolved


def _note_title(markdown: str, path: Path) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or path.stem
    return path.stem.replace("-", " ")


def _note_view_markdown(markdown: str) -> str:
    """Hide machine-only note wrappers before rendering for the web viewer."""
    text = markdown.replace(GEN_START, "").replace(GEN_END, "").lstrip()
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)
    return text.strip()


def _web_media_url(target: str) -> str | None:
    """Return the safe `/media` URL for an Obsidian image embed target."""
    path = target.strip()
    if not path or path.startswith("/") or "\\" in path:
        return None
    if path.startswith("_media/"):
        rel = path
    elif path.startswith("./_media/"):
        rel = path[2:]
    else:
        return None
    suffix = Path(rel).suffix.lower()
    if suffix not in _WEB_IMAGE_SUFFIXES:
        return None
    return "/media?path=" + quote(rel, safe="")


def _obsidian_image_embed_to_markdown(match: re.Match[str]) -> str:
    """Convert one Obsidian image embed to CommonMark image syntax if safe."""
    raw = match.group(1).strip()
    target = raw.split("|", 1)[0].strip()
    media_url = _web_media_url(target)
    if media_url is None:
        return match.group(0)
    alt = html.escape(Path(target).name, quote=False)
    return f"![{alt}]({media_url})"


def _rewrite_obsidian_image_embeds(markdown: str) -> str:
    """Convert generated Obsidian image embeds so the web viewer can render them."""
    return re.sub(r"!\[\[([^\]]+)\]\]", _obsidian_image_embed_to_markdown, markdown)


def _render_note_markdown(markdown: str) -> str:
    """Render markdown for the web note viewer without allowing raw HTML."""
    markdown = _rewrite_obsidian_image_embeds(markdown)
    parser = MarkdownIt(
        "commonmark",
        {
            "html": False,
            "linkify": False,
            "typographer": False,
        },
    )
    return parser.render(markdown).replace("<a ", '<a target="_blank" rel="noopener" ')


def _render_note_page(note_path: Path, output_dir: Path) -> bytes:
    """Render a generated markdown note as a safe, mobile-readable HTML page."""
    markdown = note_path.read_text(encoding="utf-8")
    display_markdown = _note_view_markdown(markdown)
    root = output_dir.resolve()
    rel = note_path.resolve().relative_to(root)
    title = _note_title(markdown, note_path)
    escaped_title = html.escape(title)
    escaped_rel = html.escape(str(rel))
    body_html = _render_note_markdown(display_markdown)
    obsidian_href = "obsidian://open?path=" + quote(str(note_path.resolve()))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="content-security-policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline';">
<title>{escaped_title}</title>
<style>
:root {{
  color-scheme: dark;
  --bg:#0E0C08; --surface:#17130D; --surface-2:#211B12; --ink:#ECE6DA;
  --muted:#BEB4A3; --faint:#8F8373; --hair:rgba(236,230,218,.13); --accent:#E0A233;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.wrap {{ max-width:920px; margin:0 auto; padding:22px 16px 46px; }}
.top {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }}
a.btn {{
  color:var(--ink); text-decoration:none; border:1px solid var(--hair);
  background:var(--surface-2); border-radius:7px; padding:9px 11px; font-size:13px;
}}
a.btn.primary {{ border-color:rgba(224,162,51,.38); color:var(--accent); }}
article {{
  background:var(--surface); border:1px solid var(--hair); border-radius:10px;
  padding:22px; overflow:hidden;
}}
.eyebrow {{
  color:var(--accent); font-size:11px; text-transform:uppercase; letter-spacing:.12em;
  margin:0 0 8px;
}}
h1 {{ margin:0 0 8px; font-size:clamp(26px, 7vw, 46px); line-height:1.08; font-weight:650; }}
.path {{ color:var(--faint); font-size:12px; word-break:break-word; margin-bottom:22px; }}
.note-body {{ color:var(--muted); font-size:15.5px; line-height:1.68; overflow-wrap:anywhere; }}
.note-body > *:first-child {{ margin-top:0; }}
.note-body > *:last-child {{ margin-bottom:0; }}
.note-body h1,.note-body h2,.note-body h3,.note-body h4 {{
  color:var(--ink); line-height:1.18; margin:1.35em 0 .55em; font-weight:650;
}}
.note-body h1 {{ font-size:30px; }}
.note-body h2 {{ font-size:24px; border-top:1px solid var(--hair); padding-top:20px; }}
.note-body h3 {{ font-size:19px; }}
.note-body p {{ margin:.85em 0; }}
.note-body a {{ color:var(--accent); text-decoration-thickness:1px; text-underline-offset:3px; }}
.note-body ul,.note-body ol {{ padding-left:1.35em; margin:.85em 0; }}
.note-body li {{ margin:.35em 0; }}
.note-body blockquote {{
  margin:1em 0; padding:.1em 0 .1em 1em; border-left:3px solid rgba(224,162,51,.55);
  color:var(--ink);
}}
.note-body img {{
  display:block; max-width:100%; height:auto; margin:1.15em auto; border-radius:8px;
  border:1px solid var(--hair); background:#100E0A;
}}
.note-body code {{
  font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.92em;
  background:#100E0A; color:var(--ink); border:1px solid var(--hair); border-radius:5px;
  padding:.08em .32em;
}}
.note-body pre {{
  background:#100E0A; border:1px solid var(--hair); border-radius:8px; padding:13px;
  overflow:auto; white-space:pre; margin:1em 0;
}}
.note-body pre code {{ border:0; padding:0; background:transparent; }}
.note-body hr {{ border:0; border-top:1px solid var(--hair); margin:1.4em 0; }}
@media (max-width:640px) {{
  .wrap {{ padding:14px 10px 34px; }}
  article {{ padding:16px 14px; border-radius:8px; }}
  .top {{ position:sticky; top:0; z-index:2; background:rgba(14,12,8,.94); padding:8px 0; }}
  a.btn {{ flex:1 1 auto; text-align:center; }}
  .note-body {{ font-size:14.5px; line-height:1.64; }}
  .note-body h1 {{ font-size:25px; }}
  .note-body h2 {{ font-size:20px; }}
}}
</style>
</head>
<body>
<main class="wrap">
  <nav class="top">
    <a class="btn primary" href="/#chat">Ask XBrain</a>
    <a class="btn" href="/">Dashboard</a>
    <a class="btn" href="{html.escape(obsidian_href)}">Obsidian</a>
  </nav>
  <article>
    <p class="eyebrow">XBrain note</p>
    <h1>{escaped_title}</h1>
    <div class="path">{escaped_rel}</div>
    <div class="note-body">{body_html}</div>
  </article>
</main>
</body>
</html>
"""
    return page.encode("utf-8")


def _serve_dashboard(cfg: Config, host: str, port: int) -> None:
    """Serve the generated dashboard and a localhost-only refresh endpoint."""
    run_state = DashboardRunState()
    lock = threading.Lock()

    def state_payload() -> dict[str, object]:
        with lock:
            return {
                "running": run_state.running,
                "action": run_state.action,
                "last_action": run_state.last_action,
                "last_started": run_state.last_started,
                "last_finished": run_state.last_finished,
                "last_error": run_state.last_error,
            }

    def start_background(action: str, target: Callable[[], object]) -> bool:
        with lock:
            if run_state.running:
                return False
            run_state.running = True
            run_state.action = action
            run_state.last_action = action
            run_state.last_started = datetime.now(timezone.utc).isoformat()
            run_state.last_finished = None
            run_state.last_error = None

        def job() -> None:
            error: str | None = None
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - background job reports through status API
                error = f"{exc}\n{traceback.format_exc(limit=4)}"
                logging.getLogger(__name__).warning("dashboard %s failed", action, exc_info=True)
            finally:
                with lock:
                    run_state.running = False
                    run_state.action = None
                    run_state.last_finished = datetime.now(timezone.utc).isoformat()
                    run_state.last_error = error

        threading.Thread(target=job, daemon=True, name=f"xbrain-{action}").start()
        return True

    def start_refresh() -> bool:
        return start_background(
            "refresh-all",
            lambda: _run_refresh_all(cfg, source="bookmarks", headless=True, executor="api"),
        )

    def start_retry_failed() -> bool:
        return start_background(
            "retry-failed",
            lambda: _run_retry_failed_articles(
                cfg, source="bookmarks", headless=True, regenerate=True
            ),
        )

    def dashboard_bytes() -> bytes:
        dashboard = cfg.output_dir / "dashboard.html"
        if not dashboard.exists():
            _run_generate(cfg, None, None)
        return dashboard.read_bytes()

    def chat_payload(question: str) -> dict[str, object]:
        return answer_question(
            cfg.output_dir,
            question,
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
        ).to_payload()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib API name
            logging.getLogger(__name__).info("dashboard: " + fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _read_json(self, *, max_bytes: int = 4096) -> dict[str, object]:
            raw_length = self.headers.get("content-length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid content-length") from exc
            if length <= 0:
                raise ValueError("empty request body")
            if length > max_bytes:
                raise ValueError(f"request body too large; max {max_bytes} bytes")
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("request body must be JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self) -> None:  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/status":
                self._send_json(200, state_payload())
                return
            if path in ("/", "/dashboard.html"):
                try:
                    self._send(200, dashboard_bytes(), "text/html; charset=utf-8")
                except Exception as exc:  # noqa: BLE001 - HTTP handler should return cleanly
                    self._send_json(500, {"error": str(exc)})
                return
            if path == "/notes":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                try:
                    note_path = _resolve_served_note(cfg.output_dir, raw_path)
                    self._send(
                        200,
                        _render_note_page(note_path, cfg.output_dir),
                        "text/html; charset=utf-8",
                    )
                except FileNotFoundError as exc:
                    self._send_json(404, {"error": str(exc)})
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc)})
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - HTTP handler should return cleanly
                    logging.getLogger(__name__).warning("dashboard note failed", exc_info=True)
                    self._send_json(500, {"error": str(exc)})
                return
            if path == "/media":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                try:
                    media_path = _resolve_served_media(cfg.output_dir, raw_path)
                    self._send(
                        200,
                        media_path.read_bytes(),
                        _WEB_IMAGE_SUFFIXES[media_path.suffix.lower()],
                    )
                except FileNotFoundError as exc:
                    self._send_json(404, {"error": str(exc)})
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc)})
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - HTTP handler should return cleanly
                    logging.getLogger(__name__).warning("dashboard media failed", exc_info=True)
                    self._send_json(500, {"error": str(exc)})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib API name
            path = self.path.split("?", 1)[0]
            if path == "/api/chat":
                try:
                    payload = self._read_json()
                    raw_question = payload.get("question")
                    if not isinstance(raw_question, str):
                        raise ValueError("question must be a string")
                    question = raw_question.strip()
                    if not question:
                        raise ValueError("question must not be empty")
                    if len(question) > MAX_QUESTION_CHARS:
                        raise ValueError(
                            f"question is too long; max {MAX_QUESTION_CHARS} characters"
                        )
                    self._send_json(200, chat_payload(question))
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - HTTP handler should return cleanly
                    logging.getLogger(__name__).warning("dashboard chat failed", exc_info=True)
                    self._send_json(500, {"error": str(exc)})
                return
            if path == "/api/refresh-all":
                if not start_refresh():
                    self._send_json(409, {**state_payload(), "error": "refresh already running"})
                    return
                self._send_json(202, state_payload())
                return
            if path == "/api/retry-failed":
                if not start_retry_failed():
                    self._send_json(409, {**state_payload(), "error": "job already running"})
                    return
                self._send_json(202, state_payload())
                return
            self._send_json(404, {"error": "not found"})

    server = ThreadingHTTPServer((host, port), Handler)
    typer.echo(f"Dashboard local: http://{host}:{server.server_port}/")
    typer.echo(
        "POST /api/refresh-all y /api/retry-failed ejecutan tareas solo en este servidor local."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Dashboard detenido.")
    finally:
        server.server_close()


@app.command()
@_handle_cli_errors
def login() -> None:
    """Abre un navegador para iniciar sesión en X y guarda la sesión."""
    run_login(_config().storage_state_path)


@app.command()
@_handle_cli_errors
def extract(
    source: Source = typer.Option(Source.bookmarks, help="bookmarks | tweets | all"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Extrae bookmarks de X; tweets propios y posts simples se descartan."""
    _run_extract(
        _config(),
        source.value,
        _parse_date(since),
        _parse_date(until, end_of_day=True),
        headless=headless,
    )


@app.command(name="import-archive")
@_handle_cli_errors
def import_archive(zip_path: Path) -> None:
    """Backfill del histórico de tweets desde el archivo oficial de X."""
    cfg = _config()
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    author = Author(handle=cfg.x_handle, name=cfg.x_handle)
    added = merge_items(store, parse_archive(zip_path, author))
    state.archive_imported = ArchiveImport(file=zip_path.name, at=datetime.now(timezone.utc))
    save_store(store, cfg.items_path)
    save_state(state, cfg.state_path)
    typer.echo(f"Archivo importado: {added} tweets nuevos")


@app.command()
@_handle_cli_errors
def fetch(
    since: str = typer.Option(None),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    force: bool = typer.Option(False, help="Volver a descargar lo ya descargado"),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Descarga el contenido de los artículos enlazados."""
    _run_fetch(
        _config(), _parse_date(since), _parse_date(until, end_of_day=True), force, headless=headless
    )


@app.command(name="retry-failed")
@_handle_cli_errors
def retry_failed(
    source: Source = typer.Option(Source.bookmarks, help="bookmarks | tweets | all"),
    headless: bool = typer.Option(True, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Relanza solo los items cuyos artículos enlazados fallaron y regenera el vault."""
    _run_retry_failed_articles(_config(), source.value, headless=headless, regenerate=True)


def _run_media(
    cfg: Config,
    *,
    force: bool,
    limit: int | None,
    items_filter: list[str] | None,
    verbose: bool = False,
) -> None:
    """Run the photo downloader: snapshot, load, download, persist, summarise.

    Always snapshots `data/` first (the same recovery boundary as
    `vocab --regenerate` etc): a botched run can be undone with
    `xbrain snapshot restore`.

    Persistence happens twice: once after every photo transition (the
    `on_progress` callback writes the store atomically, so Ctrl-C mid-run
    leaves `items.json` coherent), and once unconditionally at the end so
    the elapsed timestamp on the last `MediaPhotoDownloaded` is captured
    even if no transition fired (e.g. a `--limit 0` no-op).

    Persistence failure semantics: if `save_store` raises inside the
    `on_progress` callback (e.g. disk full), the exception propagates and
    aborts the run. The state of `items.json` for the photo currently
    being processed is whatever the previous successful write captured;
    later items remain in their pre-run variant. The `finally` block
    below still attempts a final write, but on a disk-full condition that
    too may fail — in which case the in-memory transitions for the
    interrupted batch are lost. This is acceptable: a re-run after the
    operator clears the disk picks up every still-pending photo cleanly.
    """
    if items_filter:
        target = set(items_filter)
        store_ids = set(load_store(cfg.items_path))
        missing = target - store_ids
        if missing and not (target & store_ids):
            typer.echo(
                f"AVISO: --items {','.join(items_filter)} no coincide con ningún item "
                f"del store ({len(store_ids)} items). El run será un no-op.",
                err=True,
            )
    _auto_snapshot(cfg, "media")
    store = load_store(cfg.items_path)

    def _persist() -> None:
        save_store(store, cfg.items_path)

    try:
        report = run_media_download(
            store,
            cfg.media_dir,
            force=force,
            limit=limit,
            items_filter=items_filter,
            on_progress=_persist,
        )
    finally:
        # Persist whatever changed, even if `download_all` raised. A
        # RuntimeError on total failure must not discard the per-photo
        # MediaPhotoFailed records that landed before the raise.
        save_store(store, cfg.items_path)
    media_emit_summary_line(report)
    article_failed = report.article_images_failed_permanent + report.article_images_failed_transient
    typer.echo(
        f"Media: descargadas {report.photos_downloaded}, "
        f"fallidas {report.photos_failed_permanent + report.photos_failed_transient}, "
        f"saltadas {report.photos_skipped_already_downloaded} "
        f"(imágenes de artículo: descargadas {report.article_images_downloaded}, "
        f"fallidas {article_failed}, saltadas {report.article_images_skipped})"
    )
    if verbose and report.per_item_failures:
        typer.echo("Failed media:", err=True)
        for item_id, failures in sorted(report.per_item_failures.items()):
            for url, reason in failures:
                typer.echo(f"  {item_id}  {reason:<14}  {url}", err=True)


@app.command()
@_handle_cli_errors
def media(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-descargar todas las fotos, incluso las ya descargadas o permanentemente "
        "fallidas (HTTP 4xx, format_error).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Máximo número de descargas a intentar en esta ejecución.",
    ),
    items: str | None = typer.Option(
        None,
        "--items",
        help="IDs de items separados por comas para limitar el alcance del run.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Imprime cada foto fallida (item_id, motivo, URL) al final del run.",
    ),
) -> None:
    """Descarga las fotos de los X-posts referenciadas en `items.json`.

    Solo descarga fotos (`MediaPhotoPending` + reintentos transient). Los
    vídeos quedan en su variante `MediaVideoPending` para una fase posterior
    — la opción `--force` NO los descarga.
    """
    cfg = _config()
    items_filter = [s.strip() for s in items.split(",") if s.strip()] if items else None
    _run_media(cfg, force=force, limit=limit, items_filter=items_filter, verbose=verbose)


@app.command(name="refresh-media")
@_handle_cli_errors
def refresh_media(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Guardar aunque se re-vean 0 items conocidos (sesión caducada / "
        "drift de GraphQL). Por defecto ese caso aborta sin escribir.",
    ),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Re-captura X y refresca la URL/metadata de vídeo de items ya guardados.

    Recorre el histórico COMPLETO (sin saltarse ids conocidos) y reescribe las
    entradas de vídeo poster-era con el stream reproducible + bitrate +
    duración. No toca fotos ni el estado de enriquecimiento/descripción, y no
    degrada un vídeo bueno a su póster si X deja de servir el stream.

    Es destructivo (reescribe `items.json` in situ) → auto-snapshot antes de
    escribir. Si se re-ven 0 items conocidos sobre un store no vacío (probable
    sesión caducada o drift del parser), aborta sin guardar salvo `--force`.
    NO descarga vídeo (eso es una fase posterior): solo imprime una estimación
    del tamaño total de descarga. El scroll es lento y a ritmo humano; puede
    tardar varios minutos.
    """
    _run_refresh_media(_config(), source.value, force=force, headless=headless)


def _run_describe(
    cfg: Config,
    *,
    force: bool,
    limit: int | None,
    items_filter: list[str] | None,
    model: str,
    batch_size: int,
    verbose: bool,
) -> None:
    """Run the vision-describe orchestrator and persist after every batch.

    Always snapshots `data/` first (the same recovery boundary as
    `xbrain media`): a botched run — a wrong model, a runaway prompt
    — can be undone with `xbrain snapshot restore`. Coherence on a
    Ctrl-C mid-run is held by the outer `try/finally` below, which
    saves the store unconditionally even when the orchestrator raises;
    the `on_progress` callback is for incremental persistence between
    batches on a clean run (so a long describe run never loses more
    than one batch of work to a process death).
    """
    if items_filter:
        target = set(items_filter)
        store_ids = set(load_store(cfg.items_path))
        missing = target - store_ids
        if missing and not (target & store_ids):
            typer.echo(
                f"AVISO: --items {','.join(items_filter)} no coincide con ningún item "
                f"del store ({len(store_ids)} items). El run será un no-op.",
                err=True,
            )
    _auto_snapshot(cfg, "describe")
    store = load_store(cfg.items_path)

    def _persist() -> None:
        save_store(store, cfg.items_path)

    try:
        report = run_describe_all(
            store,
            cfg.media_dir,
            model=model,
            output_language=cfg.output_language,
            description_version=cfg.describe_version,
            force=force,
            limit=limit,
            items_filter=items_filter,
            batch_size=batch_size,
            provider=cfg.llm_provider,
            base_url=cfg.llm_base_url,
            on_progress=_persist,
        )
    finally:
        # Persist whatever transitioned, even if `describe_all` raised. A
        # RuntimeError on total failure must not discard the per-photo
        # MediaPhotoDescribed records that landed before the raise.
        save_store(store, cfg.items_path)
    describe_emit_summary_line(report)
    typer.echo(
        f"Describe: descritas {report.photos_described}, "
        f"fallidas {report.photos_failed}, "
        f"saltadas {report.photos_skipped_already_described}"
    )
    if verbose and report.per_item_failures:
        typer.echo("Failed photos:", err=True)
        for item_id, failures in sorted(report.per_item_failures.items()):
            for url, error in failures:
                typer.echo(f"  {item_id}  {url}  {error}", err=True)


@app.command()
@_handle_cli_errors
def describe(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-describir todas las fotos, incluso las ya descritas en la versión actual.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Máximo número de fotos a describir en esta ejecución.",
    ),
    items: str | None = typer.Option(
        None,
        "--items",
        help="IDs de items separados por comas para limitar el alcance del run.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Modelo API LLM de visión para este run. Si no se pasa, se usa [llm].vision_model.",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        min=1,
        help="Número de imágenes por llamada a la API. 5 es el sweet spot (12-15%% ahorro de tokens).",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="api | manual | claude-code (default: api). manual/claude-code exportan un "
        "worksheet para describir sin API key (como enrich/topics).",
    ),
    apply: Path | None = typer.Option(
        None,
        "--apply",
        help="Importa un worksheet de descripciones relleno y lo aplica (sin API key).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Imprime cada foto fallida (item_id, URL, error) al final del run.",
    ),
) -> None:
    """Describe las fotos descargadas con un LLM de visión.

    Solo describe fotos con bytes en disco (`MediaPhotoDownloaded`).
    Las entradas ya descritas en la versión actual se saltan; bumpear
    `[describe].version` en `config.toml` fuerza un re-describe
    automático sin `--force`. Las descripciones se persisten en
    `items.json` y son consumidas por `xbrain enrich` y `xbrain topics`
    en las llamadas LLM subsiguientes.
    """
    cfg = _config()
    items_filter = [s.strip() for s in items.split(",") if s.strip()] if items else None
    worksheet_path = cfg.data_dir / "describe-worksheet.json"
    if apply is not None:
        _auto_snapshot(cfg, "describe-apply")
        store = load_store(cfg.items_path)
        applied, invalid = apply_describe_worksheet(store, apply)
        save_store(store, cfg.items_path)
        typer.echo(f"Describe worksheet aplicada: {applied} fotos descritas")
        _report_invalid(invalid)
        return
    if executor is not None and executor not in ("api", "manual", "claude-code"):
        raise ValueError(f"Ejecutor desconocido: {executor!r}")
    if executor in ("manual", "claude-code"):
        store = load_store(cfg.items_path)
        n = export_describe_worksheet(
            store,
            cfg.media_dir,
            worksheet_path,
            version=cfg.describe_version,
            output_language=cfg.output_language,
            force=force,
            limit=limit,
            items_filter=items_filter,
        )
        typer.echo(
            f"{n} fotos exportadas a {worksheet_path}\n"
            "Rellena el array `judgments` (con Claude Code o a mano) y ejecuta:\n"
            f"  xbrain describe --apply {worksheet_path}"
        )
        return
    chosen_model = model or cfg.llm_vision_model
    validate_llm_model(cfg.llm_provider, chosen_model, setting="describe --model")
    _run_describe(
        cfg,
        force=force,
        limit=limit,
        items_filter=items_filter,
        model=chosen_model,
        batch_size=batch_size,
        verbose=verbose,
    )


def _warn_items_filter_no_match(cfg: Config, items_filter: list[str]) -> None:
    """Echo a no-op warning when `--items` matches nothing (shared by media/video)."""
    target = set(items_filter)
    store_ids = set(load_store(cfg.items_path))
    if (target - store_ids) and not (target & store_ids):
        typer.echo(
            f"AVISO: --items {','.join(items_filter)} no coincide con ningún item "
            f"del store ({len(store_ids)} items). El run será un no-op.",
            err=True,
        )


def _skip_only_report(plan: VideoDownloadPlan) -> VideoReport:
    """A `VideoReport` carrying only `plan`'s skip counts (no attempts).

    Lets the skip-only path emit the same `SUMMARY:` line as a real run, so a
    monitor grepping stderr sees `download-videos` and `media` consistently.
    """
    return VideoReport(
        videos_skipped_hls=plan.n_hls_skipped,
        videos_skipped_poster_era=plan.n_poster_skipped,
        videos_skipped_already_downloaded=plan.n_already_downloaded,
        videos_skipped_too_large=plan.n_too_large,
        videos_skipped_size_unknown=plan.n_size_unknown_skipped,
    )


def _run_download_videos(
    cfg: Config,
    source: str,
    *,
    force: bool,
    limit: int | None,
    items_filter: list[str] | None,
    yes: bool,
    max_size_bytes: int | None,
) -> None:
    """Download the mp4 bytes for `MediaVideoPending` entries; persist + summarise.

    Flow: load → plan (no network, no write) → print the size gate → confirm
    (unless `--yes`) → snapshot `data/` → download → persist. The snapshot is the
    same recovery boundary as `xbrain media`, but taken AFTER the confirm so a
    declined gate never leaves a stray snapshot; a snapshot failure still
    propagates and aborts before any write (CONTRIBUTING §Safety). A run with no
    downloadable mp4 (only HLS / poster-era / already-downloaded / over-cap /
    unknown-size) writes nothing, so it skips both the confirm and the snapshot —
    but still emits the `SUMMARY:` line for monitor parity with `media`.

    `--source` scopes the run to retained bookmark items (`tweets` is an empty
    backwards-compatible scope); `scoped` shares the
    same `Item` objects as `store`, so the in-place transitions are persisted by
    saving the full `store`. mp4 ONLY: HLS entries are reported as deferred to
    the ffmpeg follow-up, never downloaded here. `max_size_bytes` caps the
    per-video estimated size.
    """
    if items_filter:
        _warn_items_filter_no_match(cfg, items_filter)
    store = load_store(cfg.items_path)
    source_sets: dict[str, list[SourceName]] = {
        "bookmarks": ["bookmark"],
        "tweets": [],
        "all": ["bookmark"],
    }
    chosen = set(source_sets[source])
    scoped = {item_id: item for item_id, item in store.items() if item.source in chosen}

    plan = plan_video_downloads(
        scoped, force=force, limit=limit, items_filter=items_filter, max_size_bytes=max_size_bytes
    )
    if plan.n_to_download == 0:
        typer.echo(
            f"No hay vídeos mp4 que descargar "
            f"({plan.n_hls_skipped} HLS pendientes de ffmpeg, "
            f"{plan.n_poster_skipped} poster-era, "
            f"{plan.n_already_downloaded} ya descargados, "
            f"{plan.n_too_large} > --max-size, "
            f"{plan.n_size_unknown_skipped} sin tamaño)."
        )
        emit_video_summary_line(_skip_only_report(plan))
        return
    typer.echo(format_size_gate(plan))
    if not yes:
        typer.confirm("¿Continuar con la descarga?", abort=True)
    _auto_snapshot(cfg, "download-videos")

    def _persist() -> None:
        save_store(store, cfg.items_path)

    try:
        report = run_download_videos(
            scoped,
            cfg.media_dir,
            force=force,
            limit=limit,
            items_filter=items_filter,
            max_size_bytes=max_size_bytes,
            on_progress=_persist,
        )
    finally:
        # Persist whatever transitioned even if `download_videos` raised — a
        # total-failure RuntimeError must not discard the MediaVideoFailed
        # records that landed before the raise.
        save_store(store, cfg.items_path)
    emit_video_summary_line(report)
    typer.echo(
        f"Vídeos: descargados {report.videos_downloaded}, "
        f"fallidos {report.videos_failed_permanent + report.videos_failed_transient}, "
        f"HLS saltados {report.videos_skipped_hls}, "
        f"poster-era saltados {report.videos_skipped_poster_era}, "
        f"ya descargados {report.videos_skipped_already_downloaded}, "
        f"> --max-size {report.videos_skipped_too_large}, "
        f"sin tamaño {report.videos_skipped_size_unknown}"
    )


@app.command(name="download-videos")
@_handle_cli_errors
def download_videos(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Máximo número de vídeos a descargar en esta ejecución.",
    ),
    items: str | None = typer.Option(
        None,
        "--items",
        help="IDs de items separados por comas para limitar el alcance del run.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-descargar vídeos ya descargados y reintentar los fallos permanentes.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="No pedir confirmación del tamaño de descarga (modo no interactivo).",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Saltar vídeos cuyo tamaño estimado supere este cap. Acepta 500MB / 2GB "
        "(unidades decimales); un número sin unidad se interpreta como MB. Con el cap "
        "puesto, los vídeos de tamaño desconocido (sin bitrate/duración) también se saltan.",
    ),
) -> None:
    """Descarga los bytes mp4 de los vídeos referenciados en `items.json`.

    Solo descarga streams mp4 reproducibles (entradas `MediaVideoPending` con
    URL real, más reintentos transient). Antes de descargar imprime una
    estimación del tamaño total (~X.X GB) y pide confirmación salvo `--yes`. Los
    manifiestos HLS (`.m3u8`) necesitan ffmpeg y se posponen a un follow-up: se
    cuentan y se saltan, no se descargan aquí. Las entradas poster-era (sin
    backfill: usa antes `xbrain refresh-media`) también se saltan. `--max-size`
    (p.ej. `500MB` / `2GB`) salta los vídeos demasiado grandes por estimación.
    Es destructivo (reescribe `items.json`) → auto-snapshot antes de escribir.
    """
    cfg = _config()
    items_filter = [s.strip() for s in items.split(",") if s.strip()] if items else None
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    _run_download_videos(
        cfg,
        source.value,
        force=force,
        limit=limit,
        items_filter=items_filter,
        yes=yes,
        max_size_bytes=max_size_bytes,
    )


@app.command(name="list-videos")
@_handle_cli_errors
def list_videos(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    topic: str | None = typer.Option(None, "--topic", help="Filtra por el primary_topic del item."),
    status: VideoStatus | None = typer.Option(
        None,
        "--status",
        help="Filtra por estado: downloaded | failed | pending | poster-era.",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Solo vídeos con tamaño conocido <= cap (500MB / 2GB; sin unidad = MB).",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Máximo número de filas."),
    json_out: bool = typer.Option(
        False, "--json", help="Salida como array JSON estable en vez de tabla humana."
    ),
) -> None:
    """Cataloga (solo lectura) los vídeos referenciados en `items.json`.

    Una fila por entrada de vídeo, con estado (downloaded / failed / pending /
    poster-era), tamaño estimado (exacto si ya está descargado, "unknown" si no
    hay bitrate/duración), el `primary_topic` del item y un snippet del texto.
    NO escribe nada ni toma snapshot. Con `--json` emite un array estable con los
    campos `id, url, state, topic, size_bytes|null, mp4_url, text` que un agente
    puede parsear para elegir qué vídeos pasar a `fetch-video`.
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    rows = list_video_entries(
        store,
        topic=topic,
        status=status.value if status is not None else None,
        max_size_bytes=max_size_bytes,
        source=source.value,
        limit=limit,
    )
    if json_out:
        typer.echo(json.dumps([row_to_json(row) for row in rows], ensure_ascii=False, indent=2))
    else:
        typer.echo(format_video_table(rows))


def _resolve_fetch_ids(
    store: dict[str, Item], ids: str | None, topic: str | None, source: str
) -> list[str]:
    """Resolve `--ids` and/or `--topic` into a de-duplicated, ordered id list.

    Explicit `--ids` are taken verbatim; `--topic` is expanded via the read-only
    catalog (scoped by `--source`). At least one selector is required — an empty
    selection is an operator error, not a silent no-op.
    """
    id_list: list[str] = []
    if ids:
        id_list.extend(part.strip() for part in ids.split(",") if part.strip())
    if topic:
        id_list.extend(row.id for row in list_video_entries(store, topic=topic, source=source))
    if not id_list:
        raise ValueError("fetch-video: indica --ids y/o --topic para seleccionar vídeos.")
    return list(dict.fromkeys(id_list))


def _emit_fetch_report(report: FetchReport, *, json_out: bool) -> None:
    """Print the fetch outcomes: JSON array, or human lines + a SUMMARY."""
    if json_out:
        typer.echo(
            json.dumps(
                [fetch_result_to_json(result) for result in report.results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for result in report.results:
        if result.outcome == "fetched":
            typer.echo(f"{result.id}: {result.path}")
        elif result.outcome == "skipped":
            typer.echo(f"{result.id}: saltado ({result.reason})")
        else:
            typer.echo(
                f"{result.id}: fallo ({result.reason}) {result.error or ''}".rstrip(), err=True
            )
    typer.echo(format_fetch_summary(report))


@app.command(name="fetch-video")
@_handle_cli_errors
def fetch_video(
    to: Path = typer.Option(
        ..., "--to", help="Directorio destino (REQUERIDO). Escribe <dir>/<id>.mp4."
    ),
    ids: str | None = typer.Option(None, "--ids", help="IDs de items separados por comas."),
    topic: str | None = typer.Option(
        None, "--topic", help="Selecciona vídeos por el primary_topic del item."
    ),
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Salta vídeos cuyo tamaño estimado supere el cap (500MB / 2GB; sin unidad = MB).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Máximo número de descargas en esta ejecución."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Salida como array JSON estable en vez de líneas humanas."
    ),
) -> None:
    """Descarga (efímera) el mp4 real de los vídeos elegidos a `--to`/<id>.mp4.

    Selecciona por `--ids` y/o `--topic` (+ `--max-size`, `--limit`). Reutiliza
    las primitivas de `download-videos` (validación de contenido, clasificación
    de fallos, escritura atómica, discriminador mp4/HLS/poster). Los HLS y
    poster-era se saltan y se cuentan. Es DELIBERADAMENTE no persistente: NO muta
    `items.json`, NO toma snapshot y NO escribe en `data/media/` — solo escribe
    bajo `--to`. Pensado para que un agente transcriba/analice el vídeo y luego
    descarte los bytes.
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    id_list = _resolve_fetch_ids(store, ids, topic, source.value)
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    report = fetch_videos(store, id_list, to, max_size_bytes=max_size_bytes, limit=limit)
    _emit_fetch_report(report, json_out=json_out)
    if report.fetched == 0 and report.failed > 0:
        # Parity with download-videos: a run where every attempted download
        # failed must surface as a non-zero exit, not a silent empty run. A pure
        # all-skips run (nothing attempted) stays exit 0.
        raise RuntimeError(
            f"fetch-video: all {report.failed} download attempt(s) failed; "
            "check network / video.twimg.com availability and the warnings above."
        )


def _resolve_digest_ids(
    store: dict[str, Item],
    ids: str | None,
    topic: str | None,
    all_pending: bool,
    source: str,
    limit: int | None,
) -> list[str]:
    """Resolve the digest selection into a de-duplicated, ordered id list.

    `--all-pending` expands to every fetchable (`pending`) video via the
    read-only catalog; `--ids` are taken verbatim; `--topic` is expanded via the
    catalog (scoped by `--source`). At least one selector is required — an empty
    selection is an operator error, not a silent no-op. `--limit` caps the number
    of items after de-duplication.
    """
    id_list: list[str] = []
    if all_pending:
        id_list.extend(row.id for row in list_video_entries(store, status="pending", source=source))
    if ids:
        id_list.extend(part.strip() for part in ids.split(",") if part.strip())
    if topic:
        id_list.extend(row.id for row in list_video_entries(store, topic=topic, source=source))
    if not id_list:
        raise ValueError(
            "digest-video: indica --ids, --topic o --all-pending para seleccionar vídeos."
        )
    unique = list(dict.fromkeys(id_list))
    return unique[:limit] if limit is not None else unique


def _build_visual_config(cfg: Config, vision_model: str | None = None) -> VisualConfig:
    """Build the `--frames` visual-layer config (#44 PR4).

    Binds `extract_key_frames` (ffmpeg, threshold/max-frames defaults) and
    either the EXTERNAL `[vision].command` or the configured cloud/API vision LLM
    so `digest_videos` calls them with just a path. An unset `[vision].command`
    now falls back to `[llm].provider` + `[llm].vision_model`, which is the useful
    VPS path when NanoGPT is configured.

    `vision_model` overrides `[vision].model` for this run (the `--vision-model`
    flag). The global LLM provider/model is also passed in the subprocess
    environment so a multi-backend wrapper cannot silently choose a different
    API provider than the main app.
    """
    model = vision_model or cfg.vision_model or cfg.llm_vision_model
    env = {
        **os.environ,
        "XBRAIN_LLM_PROVIDER": cfg.llm_provider,
        "XBRAIN_LLM_MODEL": cfg.llm_model,
        "XBRAIN_LLM_VISION_MODEL": model,
    }
    if cfg.llm_base_url:
        env["XBRAIN_LLM_BASE_URL"] = cfg.llm_base_url
    if cfg.llm_provider == "nanogpt":
        env["NANOGPT_MODEL"] = cfg.llm_model
        env["NANOGPT_VISION_MODEL"] = model
        if cfg.llm_base_url:
            env["NANOGPT_BASE_URL"] = cfg.llm_base_url
    if cfg.llm_provider == "anthropic":
        env["ANTHROPIC_MODEL"] = cfg.llm_model
        env["ANTHROPIC_VISION_MODEL"] = model

    def _extract(path: Path) -> list[KeyFrame]:
        return extract_key_frames(
            path, threshold=DEFAULT_SCENE_THRESHOLD, max_frames=DEFAULT_MAX_FRAMES
        )

    def _describe(path: Path) -> str:
        if cfg.vision_command.strip():
            return describe_image(path, command=cfg.vision_command, model=model, env=env)
        return describe_image_with_llm(
            path,
            provider=cfg.llm_provider,
            model=model,
            output_language=cfg.output_language,
            base_url=cfg.llm_base_url or None,
        )

    return VisualConfig(
        media_root=cfg.media_dir,
        extract_fn=_extract,
        describe_fn=_describe,
        allow_visual_only=True,
    )


def _run_digest_video(
    cfg: Config,
    *,
    ids: str | None,
    topic: str | None,
    all_pending: bool,
    source: str,
    limit: int | None,
    force: bool,
    language: str | None,
    frames: bool,
    vision_model: str | None = None,
    max_size_bytes: int | None = None,
) -> None:
    """Digest selected videos into `x_video` transcript sources; persist + summarise.

    Flow: load → resolve selection → ephemeral fetch + EXTERNAL transcribe +
    attach (dedup by video identity, in memory) → snapshot → persist. The
    transcriber is invoked via `transcribe_media` bound to the `[transcribe]`
    config (command / model) + `--language`. `--frames` (opt-in, #44 PR4) also
    extracts slide key frames and describes them via optional `[vision].command`
    or the configured API vision model, attaching them to slide-heavy videos. It
    is destructive (rewrites `items.json`), so it auto-snapshots BEFORE the save
    — but only when something was attached (a pure already-digested / no-video
    run writes nothing, so it takes no snapshot). A snapshot failure propagates
    and aborts before any write.
    """
    store = load_store(cfg.items_path)
    id_list = _resolve_digest_ids(store, ids, topic, all_pending, source, limit)
    visual = _build_visual_config(cfg, vision_model) if frames else None

    def _transcribe(path: Path) -> Transcript:
        return transcribe_media(
            path,
            command=cfg.transcribe_command,
            model=cfg.transcribe_model,
            language=language,
        )

    def _fetch(
        selected_store: dict[str, Item], selected_ids: list[str], dest_dir: Path
    ) -> FetchReport:
        return fetch_videos(selected_store, selected_ids, dest_dir, max_size_bytes=max_size_bytes)

    report = digest_videos(
        store,
        id_list,
        force=force,
        fetch_fn=_fetch,
        transcribe_fn=_transcribe,
        visual=visual,
    )
    if report.changed > 0:
        _auto_snapshot(cfg, "digest-video")
        save_store(store, cfg.items_path)
    typer.echo(format_digest_summary(report))


@app.command(name="digest-video")
@_handle_cli_errors
def digest_video(
    ids: str | None = typer.Option(None, "--ids", help="IDs de items separados por comas."),
    topic: str | None = typer.Option(
        None, "--topic", help="Selecciona vídeos por el primary_topic del item."
    ),
    all_pending: bool = typer.Option(
        False, "--all-pending", help="Selecciona todos los vídeos en estado pending (fetchables)."
    ),
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    limit: int | None = typer.Option(
        None, "--limit", help="Máximo número de items a procesar en esta ejecución."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-transcribir items que ya tienen un source x_video."
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Idioma a registrar en el transcript si el transcriptor no lo reporta "
        "(p.ej. en, es). El transcriptor autodetecta; no se le pasa como flag.",
    ),
    frames: bool = typer.Option(
        False,
        "--frames",
        help="Capa visual (opt-in): extrae key-frames de slides, los describe con "
        "`[vision].command` si está configurado o `[llm].vision_model` vía API si no, "
        "y los embebe en la nota. "
        "Solo para vídeos slide-heavy; los talking-head se saltan (se registra).",
    ),
    vision_model: str | None = typer.Option(
        None,
        "--vision-model",
        help="Sobrescribe `[vision].model` / `[llm].vision_model` para este run. "
        "Requiere --frames.",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Saltar vídeos cuyo tamaño estimado supere este cap (500MB / 2GB; sin unidad = MB).",
    ),
) -> None:
    """Transcribe vídeos guardados y adjunta el transcript como source `x_video`.

    Para cada vídeo seleccionado: descarga efímera (reutiliza `fetch-video`) →
    transcribe con un transcriptor EXTERNO local (config `transcribe.command`,
    por defecto `parakeet-mlx`; la ML NO vive en xbrain) → adjunta el transcript al
    item como `ContentSourceSuccess(kind="x_video")` → descarta los bytes. Los
    vídeos se **deduplican por identidad** (el id estable del path del mp4, no la
    URL firmada): N bookmarks del mismo vídeo se descargan y transcriben UNA vez y
    todos reciben el mismo transcript. Un vídeo sin voz/audio se adjunta con texto
    vacío + `has_speech=False` (nunca es un fallo duro). Idempotente: salta items
    que ya tienen un source x_video salvo `--force`. Es destructivo (reescribe
    `items.json`) → auto-snapshot antes de escribir. Nunca hay más de un vídeo en
    disco a la vez (efímero). Selecciona con `--ids`, `--topic` o `--all-pending`.

    `--frames` (opt-in, capa visual PR4): para vídeos slide-heavy extrae
    key-frames con ffmpeg (EXTERNO), los describe con `[vision].command` si está
    configurado, o con el LLM de visión API (`[llm].provider` +
    `[llm].vision_model`) si no. Adjunta las descripciones al source `x_video` y
    embebe las slides en la nota como fotos. Los vídeos talking-head se saltan y
    se registra el motivo. Sin `--frames` el flujo es idéntico al de PR2/PR3.
    """
    cfg = _config()
    if vision_model and not frames:
        raise typer.BadParameter("--vision-model requires --frames (the visual layer is off)")
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    _run_digest_video(
        cfg,
        ids=ids,
        topic=topic,
        all_pending=all_pending,
        source=source.value,
        limit=limit,
        force=force,
        language=language,
        frames=frames,
        vision_model=vision_model,
        max_size_bytes=max_size_bytes,
    )


@app.command()
@_handle_cli_errors
def enrich(
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: the enrich executor set in config.toml)"
    ),
    apply: Path | None = typer.Option(
        None, "--apply", help="Import a filled worksheet and apply it"
    ),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    taxonomy_risk: bool = typer.Option(
        False,
        "--taxonomy-risk",
        help=(
            "Re-enriquecer items con riesgo taxonómico: misc, baja/desconocida "
            "confianza, suggested_new_topics o contenido stale."
        ),
    ),
) -> None:
    """Enriquece los items con resumen + topics."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab_topics:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")

    if apply is not None:
        executor_name, judgments = import_worksheet(apply)
        enriched, invalid = apply_worksheet_judgments(store, judgments, vocab_topics, executor_name)
        save_store(store, cfg.items_path)
        typer.echo(f"Worksheet aplicada: {enriched} items enriquecidos")
        _report_invalid(invalid)
        return

    chosen = executor or cfg.enrich_executor

    if chosen in ("manual", "claude-code"):
        since_dt = _parse_date(since)
        until_dt = _parse_date(until, end_of_day=True)
        pending = (
            items_for_taxonomy_reenrichment(store, since_dt, until_dt)
            if taxonomy_risk
            else items_pending_enrichment(store, since_dt, until_dt)
        )
        if not pending:
            typer.echo(
                "No hay items con riesgo taxonómico para re-enriquecer."
                if taxonomy_risk
                else "No hay items pendientes de enriquecer."
            )
            return
        worksheet = cfg.data_dir / "enrich-worksheet.json"
        export_worksheet(pending, vocab_topics, worksheet, chosen, cfg.output_language)
        typer.echo(
            f"{len(pending)} items exportados a {worksheet}\n"
            f"Rellena el array `judgments` (con Claude Code o a mano) y ejecuta:\n"
            f"  xbrain enrich --apply {worksheet}"
        )
        return

    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")

    _run_enrich_api(
        cfg,
        _parse_date(since),
        _parse_date(until, end_of_day=True),
        taxonomy_risk=taxonomy_risk,
    )


def _mark_for_regenerate(store: dict, cfg: Config, regenerate: bool) -> None:
    """When `--regenerate` is set, drop every item's enrichment and persist."""
    if regenerate:
        for item in store.values():
            item.enriched = None
        save_store(store, cfg.items_path)
        typer.echo("Todos los items marcados para re-enriquecer.")


def _vocab_apply(cfg: Config, store: dict, apply: Path, regenerate: bool) -> None:
    """`xbrain vocab --apply` — import a filled vocab worksheet."""
    topics, invalid = apply_vocab_worksheet(import_vocab_worksheet(apply))
    _report_invalid(invalid)
    if not topics:
        raise RuntimeError("La worksheet no produjo ningún topic válido.")
    if regenerate:
        _auto_snapshot(cfg, "vocab-regenerate")
    # Mark the store first: a crash here leaves items pending (a re-run re-marks
    # idempotently) — safer than vocab.yaml updated while items stay stale.
    _mark_for_regenerate(store, cfg, regenerate)
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    typer.echo(f"Vocabulario aplicado: {len(topics)} topics → {cfg.data_dir / 'vocab.yaml'}")


def _vocab_run(cfg: Config, store: dict, executor: str | None, regenerate: bool) -> None:
    """`xbrain vocab` — induce the taxonomy (worksheet export, or `api`)."""
    chosen = executor or cfg.enrich_executor
    if chosen in ("manual", "claude-code"):
        worksheet = cfg.data_dir / "vocab-worksheet.json"
        export_vocab_worksheet(store, cfg.vocab_target_count, worksheet, cfg.output_language)
        regen = " --regenerate" if regenerate else ""
        typer.echo(
            f"Corpus exportado a {worksheet}\n"
            f"Induce la taxonomía (con Claude Code o a mano) y ejecuta:\n"
            f"  xbrain vocab --apply {worksheet}{regen}"
        )
        return
    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")
    if regenerate:
        _auto_snapshot(cfg, "vocab-regenerate")
    topics = induce_vocab(
        store,
        cfg.vocab_target_count,
        cfg.enrich_model,
        cfg.output_language,
        provider=cfg.llm_provider,
        base_url=cfg.llm_base_url,
    )
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    _mark_for_regenerate(store, cfg, regenerate)
    typer.echo(f"Vocabulario inducido: {len(topics)} topics → {cfg.data_dir / 'vocab.yaml'}")


@app.command()
@_handle_cli_errors
def vocab(
    regenerate: bool = typer.Option(
        False, help="Marca todos los items para re-enriquecer contra la taxonomía nueva"
    ),
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: el de config.toml)"
    ),
    apply: Path | None = typer.Option(None, "--apply", help="Importar una vocab worksheet rellena"),
) -> None:
    """Induce el vocabulario de topics (data/vocab.yaml) desde el corpus."""
    cfg = _config()
    store = load_store(cfg.items_path)
    if not store:
        raise RuntimeError("El store está vacío — ejecuta `xbrain extract` antes.")
    if apply is not None:
        _vocab_apply(cfg, store, apply, regenerate)
    else:
        _vocab_run(cfg, store, executor, regenerate)


def _topics_apply(cfg: Config, store: dict, vocab: list, apply: Path) -> None:
    """`xbrain topics --apply` — import a filled overview worksheet."""
    pages = load_topic_pages(cfg.topics_path)
    posts = compute_topic_posts(store, vocab)
    valid, invalid = apply_overview_judgments(import_topic_worksheet(apply))
    merge_overviews(pages, valid, posts)
    save_topic_pages(pages, cfg.topics_path)
    written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
    typer.echo(f"Worksheet aplicada: {len(valid)} overviews · {written} páginas escritas")
    _report_invalid(invalid)


def _topics_run(cfg: Config, store: dict, vocab: list, resynth: bool, executor: str | None) -> None:
    """`xbrain topics` — update lists and (re)synthesize stale overviews."""
    if resynth:
        _auto_snapshot(cfg, "topics-resynth")
    pages = load_topic_pages(cfg.topics_path)
    posts = compute_topic_posts(store, vocab)
    stale = topics_needing_synth(vocab, posts, pages, cfg.topics_resynth_threshold, resynth)
    inputs = build_topic_inputs(stale, vocab, posts)

    if not inputs:
        written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
        typer.echo(f"Topic pages actualizadas: {written} páginas (sin overviews pendientes).")
        return

    chosen = executor or cfg.enrich_executor
    if chosen in ("manual", "claude-code"):
        worksheet = cfg.data_dir / "topic-worksheet.json"
        export_topic_worksheet(inputs, worksheet, cfg.output_language)
        written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
        typer.echo(
            f"{len(inputs)} topics exportados a {worksheet} · {written} páginas escritas\n"
            f"Rellena el array `judgments` y ejecuta:\n"
            f"  xbrain topics --apply {worksheet}"
        )
        return
    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")

    judgments = synthesize_overviews_api(
        inputs,
        cfg.enrich_model,
        cfg.output_language,
        provider=cfg.llm_provider,
        base_url=cfg.llm_base_url,
    )
    merge_overviews(pages, judgments, posts)
    save_topic_pages(pages, cfg.topics_path)
    written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
    typer.echo(f"Topics sintetizados: {len(judgments)}/{len(inputs)} · {written} páginas escritas")


@app.command()
@_handle_cli_errors
def topics(
    resynth: bool = typer.Option(False, help="Re-sintetizar todos los overviews obsoletos"),
    apply: Path | None = typer.Option(
        None, "--apply", help="Importar un worksheet de overviews relleno"
    ),
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: el de config.toml)"
    ),
) -> None:
    """Genera las páginas de topic: listas de posts + overviews sintetizados."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    if apply is not None:
        _topics_apply(cfg, store, vocab, apply)
    else:
        _topics_run(cfg, store, vocab, resynth, executor)


@app.command()
@_handle_cli_errors
def generate(
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
) -> None:
    """Genera las notas markdown en el vault."""
    _run_generate(_config(), _parse_date(since), _parse_date(until, end_of_day=True))


@app.command(name="refresh-all")
@_handle_cli_errors
def refresh_all(
    source: Source = typer.Option(Source.bookmarks, help="bookmarks | tweets | all"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    headless: bool = typer.Option(True, "--headless/--no-headless", help=_HEADLESS_HELP),
    executor: str = typer.Option("api", help="Actualmente solo api; usa la sección llm para NanoGPT."),
    media_limit: int | None = typer.Option(None, "--media-limit"),
    describe_limit: int | None = typer.Option(None, "--describe-limit"),
    describe_batch_size: int = typer.Option(5, "--describe-batch-size", min=1),
    skip_media: bool = typer.Option(False, "--skip-media"),
    skip_describe: bool = typer.Option(False, "--skip-describe"),
) -> None:
    """Ejecuta el flujo diario completo para alimentar el vault."""
    _run_refresh_all(
        _config(),
        source=source.value,
        since=_parse_date(since),
        until=_parse_date(until, end_of_day=True),
        headless=headless,
        executor=executor,
        media_limit=media_limit,
        describe_limit=describe_limit,
        describe_batch_size=describe_batch_size,
        skip_media=skip_media,
        skip_describe=skip_describe,
    )


@app.command()
@_handle_cli_errors
def sync(
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """extract + fetch + generate en orden."""
    cfg = _config()
    _run_extract(cfg, "all", None, None, headless=headless)
    _run_fetch(cfg, None, None, False, headless=headless)
    _run_generate(cfg, None, None)


@app.command(name="serve-dashboard")
@_handle_cli_errors
def serve_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Host local donde escuchar."),
    port: int = typer.Option(8765, "--port", help="Puerto local del dashboard."),
) -> None:
    """Sirve el dashboard HTML y permite lanzar refresh-all desde localhost."""
    _serve_dashboard(_config(), host, port)


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def _taxonomy_health_lines(
    store: dict[str, Item], vocab: list[Topic], *, top: int = 10
) -> list[str]:
    """Read-only diagnostics for taxonomy drift and weak assignments."""
    vocab_slugs = {topic.slug for topic in vocab}
    enriched_items = [item for item in store.values() if item.enriched is not None]
    enriched_total = len(enriched_items)
    topic_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    suggested_counts: Counter[str] = Counter()
    unknown_topics: Counter[str] = Counter()
    misc_count = 0
    single_topic_count = 0

    for item in enriched_items:
        assert item.enriched is not None
        enrichment = item.enriched
        topics = [topic for topic in enrichment.topics if topic]
        topic_counts.update(topics)
        if enrichment.primary_topic:
            primary_counts.update([enrichment.primary_topic])
        confidence_counts.update([enrichment.topic_confidence or "unknown"])
        suggested_counts.update(enrichment.suggested_new_topics)
        if enrichment.primary_topic == "misc" or "misc" in topics:
            misc_count += 1
        if len(topics) <= 1:
            single_topic_count += 1
        for topic in topics:
            if topic not in vocab_slugs:
                unknown_topics.update([topic])
        if enrichment.primary_topic and enrichment.primary_topic not in vocab_slugs:
            unknown_topics.update([enrichment.primary_topic])

    unused_topics = sorted(slug for slug in vocab_slugs if topic_counts[slug] == 0)
    lines = [
        "Taxonomy health",
        f"Items: {len(store)} total · {enriched_total} enriched · {len(store) - enriched_total} pending",
        f"Vocabulary: {len(vocab)} topics",
        "",
        "Assignment signals",
        f"- misc: {misc_count} ({_pct(misc_count, enriched_total)})",
        f"- single-topic items: {single_topic_count} ({_pct(single_topic_count, enriched_total)})",
        f"- confidence high/medium/low/unknown: "
        f"{confidence_counts['high']}/{confidence_counts['medium']}/"
        f"{confidence_counts['low']}/{confidence_counts['unknown']}",
        f"- unused topics: {len(unused_topics)}",
    ]

    if unknown_topics:
        lines += ["", "Unknown assigned topics"]
        lines += [f"- {slug}: {count}" for slug, count in unknown_topics.most_common(top)]

    lines += ["", "Top assigned topics"]
    if topic_counts:
        lines += [f"- {slug}: {count}" for slug, count in topic_counts.most_common(top)]
    else:
        lines.append("- none")

    lines += ["", "Suggested new topics"]
    if suggested_counts:
        lines += [f"- {slug}: {count}" for slug, count in suggested_counts.most_common(top)]
    else:
        lines.append("- none")

    lines += ["", "Recommendation"]
    if suggested_counts or confidence_counts["low"] or misc_count > max(3, enriched_total // 10):
        lines.append(
            "- Review high `misc`, low confidence and repeated `suggested_new_topics`. "
            "First run `xbrain enrich --taxonomy-risk`, then `xbrain topics --resynth` "
            "and `xbrain generate`. If the same suggestions repeat, run "
            "`xbrain vocab --regenerate`."
        )
    else:
        lines.append("- Taxonomy looks stable enough; keep using `refresh-all` normally.")
    if unused_topics:
        sample = ", ".join(unused_topics[: min(5, len(unused_topics))])
        lines.append(f"- Consider merging/removing unused topics: {sample}")
    return lines


@app.command(name="taxonomy-health")
@_handle_cli_errors
def taxonomy_health(
    top: int = typer.Option(10, "--top", min=1, help="Número de filas por ranking."),
) -> None:
    """Diagnóstico de misc, baja confianza y topics sugeridos."""
    cfg = _config()
    store = retained_store(load_store(cfg.items_path))
    vocab = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    typer.echo("\n".join(_taxonomy_health_lines(store, vocab, top=top)))


@app.command()
@_handle_cli_errors
def status() -> None:
    """Muestra contadores y última ejecución."""
    cfg = _config()
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    typer.echo(f"Items: {len(store)}")
    typer.echo(f"  con enlace: {sum(1 for i in store.values() if i.links)}")
    typer.echo(f"  con contenido: {sum(1 for i in store.values() if i.content)}")
    typer.echo(f"  enriquecidos: {sum(1 for i in store.values() if i.enriched)}")
    typer.echo(f"  última extracción bookmarks: {state.bookmarks.last_run}")
    typer.echo(f"  última extracción tweets: {state.own_tweets.last_run}")


snapshot_app = typer.Typer(help="Gestionar snapshots de data/")
app.add_typer(snapshot_app, name="snapshot")


@snapshot_app.command("create")
@_handle_cli_errors
def snapshot_create_cmd(
    name: str | None = typer.Option(None, help="Optional directory label (default: 'manual')"),
) -> None:
    """Create a snapshot of data/ right now."""
    cfg = _config()
    path, manifest = snapshot.snapshot_create(
        cfg.data_dir,
        command="manual",
        dir_label=name,
    )
    typer.echo(f"Snapshot created: {path.name} ({manifest.item_count} items)")


@snapshot_app.command("list")
@_handle_cli_errors
def snapshot_list_cmd() -> None:
    """List snapshots, newest first. Corrupt entries surface as CORRUPT."""
    cfg = _config()
    rows = snapshot.snapshot_list(cfg.data_dir)
    if not rows:
        typer.echo("No snapshots.")
        return
    for path, manifest in rows:
        if manifest is None:
            typer.echo(
                f"{path.name}  CORRUPT — manifest missing or unreadable",
                err=True,
            )
            continue
        typer.echo(
            f"{path.name}  {manifest.command:<28}  "
            f"items={manifest.item_count}  topics={manifest.topic_count}  "
            f"vocab={manifest.vocab_size}"
        )


@snapshot_app.command("show")
@_handle_cli_errors
def snapshot_show_cmd(name: str = typer.Argument(..., help="Snapshot directory name")) -> None:
    """Print the manifest of one snapshot."""
    cfg = _config()
    _, manifest = snapshot.snapshot_show(cfg.data_dir, name)
    typer.echo(manifest.model_dump_json(indent=2))


@snapshot_app.command("restore")
@_handle_cli_errors
def snapshot_restore_cmd(name: str = typer.Argument(..., help="Snapshot directory name")) -> None:
    """Restore data/ from a snapshot.

    The vault is NOT touched — run `xbrain generate` next to refresh it.
    Every per-artifact action is echoed so 'a file vanished' never happens
    silently.
    """
    cfg = _config()
    actions = snapshot.snapshot_restore(cfg.data_dir, name)
    for artifact, action in actions:
        typer.echo(f"  {artifact}: {action}")
    typer.echo(f"Restored {name}. Run `xbrain generate` to refresh the vault.")


@snapshot_app.command("prune")
@_handle_cli_errors
def snapshot_prune_cmd(
    keep_last: int = typer.Option(10, "--keep-last", help="Keep the N newest snapshots"),
) -> None:
    """Delete older snapshots, keeping the N newest."""
    cfg = _config()
    deleted = snapshot.snapshot_prune(cfg.data_dir, keep_last=keep_last)
    typer.echo(f"Snapshots deleted: {deleted}")


def _resolve_data_dir(cfg: Config, name: str | None) -> Path:
    """Resolve a snapshot name to its data dir, or `None` to the live `data/`.

    `xbrain diff` accepts a snapshot name (resolved via `snapshot_show`) OR
    `None` to mean "the current live `data/`" — the most common B-side of the
    comparison the user runs after a destructive op.
    """
    if name is None:
        return cfg.data_dir
    snapshot_dir, _ = snapshot.snapshot_show(cfg.data_dir, name)
    return snapshot_dir


@app.command()
@_handle_cli_errors
def diff(
    snapshot_a: str = typer.Argument(..., help="Snapshot name on the A side."),
    snapshot_b: str | None = typer.Argument(
        None,
        help="Snapshot name on the B side. Defaults to the live data/ directory.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: 'text' (default) or 'json'.",
    ),
) -> None:
    """Compare two snapshots and surface drift.

    Reports reassigned items, topic-membership shifts, topic-overview drift
    (TF cosine similarity) and vocab changes. The B side defaults to the live
    `data/` directory so `xbrain diff <pre-snapshot>` answers "what did the
    last destructive op move?" with no extra arguments.
    """
    cfg = _config()
    if output_format not in ("text", "json"):
        raise ValueError(f"--format must be 'text' or 'json', got {output_format!r}")
    a_dir = _resolve_data_dir(cfg, snapshot_a)
    b_dir = _resolve_data_dir(cfg, snapshot_b)
    report = diff_snapshots(a_dir, b_dir)
    if output_format == "json":
        typer.echo(format_json(report))
    else:
        b_label = snapshot_b if snapshot_b is not None else "live data/"
        typer.echo("Comparing:")
        typer.echo(f"  A: {snapshot_a}")
        typer.echo(f"  B: {b_label}")
        typer.echo("")
        typer.echo(format_text(report))


if __name__ == "__main__":
    app()
