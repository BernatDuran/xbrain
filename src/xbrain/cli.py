"""Command-line interface for XBrain."""

from __future__ import annotations

import enum
import functools
import html
import json
import logging
import os
import re
import shutil
import threading
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
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
from xbrain.drive import (
    authenticate as drive_authenticate,
    drive_sync_down,
    drive_sync_up,
    drive_write_session,
    login as drive_login,
)
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
from xbrain.mail import send_bookmark_update_email
from xbrain.media import download_all as run_media_download
from xbrain.media import emit_summary_line as media_emit_summary_line
from xbrain.models import ArchiveImport, Author, ContentSourceFailure, Item, SourceName, Topic
from xbrain.notes_io import GEN_END, GEN_START, note_filename
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
from xbrain.video_media import parse_size_to_bytes
from xbrain.video_select import format_video_table, list_video_entries, row_to_json
from xbrain.video_digest import (
    configured_summary_fn,
    digest_video_transcripts,
    format_video_digest_summary,
)
from xbrain.video_transcript import VideoTranscript, fetch_or_transcribe_video_transcript
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
_DEFAULT_REFRESH_VIDEO_MAX_SIZE = "4GB"


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


@dataclass(frozen=True)
class ReprocessNoteReport:
    item_id: str
    articles: int
    x_articles: int


@dataclass(frozen=True)
class DeleteBookmarkReport:
    item_id: str
    notes_deleted: int
    media_dirs_deleted: int
    video_artifacts_deleted: int
    topic_pages_touched: int


@dataclass(frozen=True)
class TopicActionReport:
    item_id: str
    action: str
    primary_topic: str | None
    topics: list[str]
    topic_confidence: str | None
    suggested_new_topics: list[str]
    promoted_topics: list[str] = field(default_factory=list)


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


def _notify_bookmark_update(cfg: Config, *, command: str, updated_item_ids: list[str]) -> None:
    if not (cfg.drive_enabled and cfg.email_enabled):
        return
    try:
        store = load_store(cfg.items_path)
        send_bookmark_update_email(
            cfg,
            command=command,
            store=store,
            updated_item_ids=updated_item_ids,
        )
        typer.echo(f"Email enviado a {cfg.email_recipient}")
    except Exception as exc:  # noqa: BLE001 - notification must not undo a completed update
        typer.echo(f"AVISO: no se pudo enviar el email de actualización: {exc}", err=True)


def _sort_item_ids(item_ids: set[str]) -> list[str]:
    return sorted(
        item_ids,
        key=lambda item_id: (
            not item_id.isdigit(),
            int(item_id) if item_id.isdigit() else item_id,
        ),
    )


def _updated_item_ids(before: dict[str, Item], after: dict[str, Item]) -> list[str]:
    return _sort_item_ids(
        {
            item_id
            for item_id, item in after.items()
            if item_id not in before or item.model_dump_json() != before[item_id].model_dump_json()
        }
    )


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
    deleted = snapshot.snapshot_prune_auto(
        cfg.data_dir,
        keep_last=cfg.snapshot_auto_prune_keep_last,
    )
    if deleted:
        typer.echo(
            f"Snapshots auto-pruned: {deleted} "
            f"(keeping last {cfg.snapshot_auto_prune_keep_last} automatic snapshots)"
        )


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


def _run_reprocess_note(cfg: Config, item_id: str, *, headless: bool = True) -> ReprocessNoteReport:
    """Force-refresh one item, then regenerate derived layers.

    This is the dashboard's per-note manual repair path. It deliberately reuses
    the existing commands' behavior instead of maintaining a separate workflow:
    force fetch for this item, digest its video if present, pick up pending media
    and descriptions for this item, force API enrichment, update topics and
    regenerate the vault/dashboard.
    """
    store = load_store(cfg.items_path)
    item = store.get(item_id)
    if item is None:
        raise ValueError(f"item not found: {item_id}")
    _auto_snapshot(cfg, f"reprocess-note-{item_id}")
    selected = {item_id: item}
    try:
        articles = fetch_pending(selected, force=True)
        x_articles = fetch_x_articles(
            selected, cfg.storage_state_path, force=True, headless=headless
        )
    finally:
        save_store(store, cfg.items_path)

    _run_digest_video(
        cfg,
        ids=item_id,
        topic=None,
        all_pending=False,
        source="all",
        limit=None,
        force=True,
        language=None,
        frames=False,
        vision_model=None,
        max_size_bytes=parse_size_to_bytes(_DEFAULT_REFRESH_VIDEO_MAX_SIZE),
        allow_empty=True,
    )
    _run_media(cfg, force=False, limit=None, items_filter=[item_id])
    _run_describe(
        cfg,
        force=False,
        limit=None,
        items_filter=[item_id],
        model=cfg.llm_vision_model,
        batch_size=1,
        verbose=False,
    )

    store = load_store(cfg.items_path)
    item = store.get(item_id)
    if item is None:
        raise ValueError(f"item disappeared during reprocess: {item_id}")
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab_topics:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    executor = ApiExecutor(
        model=cfg.llm_model,
        output_language=cfg.output_language,
        provider=cfg.llm_provider,
        base_url=cfg.llm_base_url,
    )
    enriched, invalid = enrich_selected_with_executor(store, executor, vocab_topics, [item])
    save_store(store, cfg.items_path)
    _report_invalid(invalid)
    if invalid or enriched != 1:
        raise RuntimeError(f"reprocess-note no pudo re-enriquecer {item_id}")
    _run_topics_executor(cfg, "api", resynth=False)
    _run_generate(cfg, None, None)
    typer.echo(f"reprocess-note: {item_id} completado")
    return ReprocessNoteReport(item_id=item_id, articles=articles, x_articles=x_articles)


def _remove_direct_child_dir(root: Path, name: str) -> int:
    """Remove `<root>/<name>` only when it is a direct child directory."""
    if Path(name).is_absolute() or len(Path(name).parts) != 1:
        raise ValueError(f"invalid item id for artifact cleanup: {name!r}")
    path = root / name
    if not path.exists():
        return 0
    root_resolved = root.resolve()
    resolved = path.resolve()
    if resolved.parent != root_resolved:
        raise PermissionError(f"artifact path escapes {root}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
        return 1
    resolved.unlink()
    return 1


def _note_file_belongs_to_item(path: Path, item_id: str) -> bool:
    if path.name.endswith(f"-{item_id}.md"):
        return True
    try:
        return _note_frontmatter_id(path.read_text(encoding="utf-8")) == item_id
    except OSError:
        return False


def _delete_item_notes(output_dir: Path, item: Item) -> int:
    items_dir = output_dir / "items"
    if not items_dir.exists():
        return 0
    candidates = {items_dir / note_filename(item)}
    candidates.update(
        path for path in items_dir.glob("*.md") if _note_file_belongs_to_item(path, item.id)
    )
    removed = 0
    for path in sorted(candidates):
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _video_artifact_belongs_to_item(path: Path, item_id: str) -> bool:
    if path.name.endswith(f"-{item_id}"):
        return True
    for artifact in (path / "summary.md", path / "transcript.md"):
        if not artifact.exists():
            continue
        try:
            if _note_frontmatter_id(artifact.read_text(encoding="utf-8")) == item_id:
                return True
        except OSError:
            continue
    return False


def _delete_video_artifacts(output_dir: Path, item_id: str) -> int:
    videos_dir = output_dir / "videos"
    if not videos_dir.exists():
        return 0
    removed = 0
    for path in videos_dir.iterdir():
        if path.is_dir() and _video_artifact_belongs_to_item(path, item_id):
            shutil.rmtree(path)
            removed += 1
    return removed


def _refresh_topic_pages_after_delete(cfg: Config, store: dict[str, Item], item_id: str) -> int:
    vocab_path = cfg.data_dir / "vocab.yaml"
    if not vocab_path.exists():
        return 0
    vocab_topics = load_vocab(vocab_path)
    if not vocab_topics:
        return 0
    pages = load_topic_pages(cfg.topics_path) if cfg.topics_path.exists() else {}
    posts = compute_topic_posts(store, vocab_topics)
    touched = write_topic_pages(cfg.output_dir, vocab_topics, posts, pages, cfg.output_language)
    topics_dir = cfg.output_dir / "topics"
    if not topics_dir.exists():
        return touched
    for topic in vocab_topics:
        topic_posts = posts.get(topic.slug)
        if topic_posts is None or topic_posts.total != 0:
            continue
        path = topics_dir / f"{topic.slug}.md"
        if path.exists() and item_id in path.read_text(encoding="utf-8"):
            path.unlink()
            touched += 1
    return touched


def _topic_state_payload(item: Item) -> dict[str, object]:
    enrichment = item.enriched
    if enrichment is None:
        return {
            "item_id": item.id,
            "enriched": False,
            "primary_topic": None,
            "topics": [],
            "topic_confidence": None,
            "suggested_new_topics": [],
        }
    return {
        "item_id": item.id,
        "enriched": True,
        "primary_topic": enrichment.primary_topic,
        "topics": enrichment.topics,
        "topic_confidence": enrichment.topic_confidence,
        "suggested_new_topics": enrichment.suggested_new_topics,
    }


def _topic_description_for_promoted_slug(slug: str) -> str:
    label = slug.replace("-", " ").strip()
    return f"Posts about {label}. Added from an operator-approved XBrain suggestion."


def _selected_topic_suggestions(item: Item, suggested_topics: list[str] | None) -> list[str]:
    if not suggested_topics:
        return []
    if item.enriched is None:
        raise RuntimeError(f"item has no topic suggestions: {item.id}")
    allowed = set(item.enriched.suggested_new_topics)
    selected: list[str] = []
    for raw_slug in suggested_topics:
        slug = raw_slug.strip()
        if not slug:
            continue
        if slug not in allowed:
            raise ValueError(f"topic suggestion is not available for item {item.id}: {slug}")
        if slug not in selected:
            selected.append(slug)
    return selected


def _topic_suggestion_hints(
    item: Item,
    selected: list[str],
    *,
    prioritize_suggestions: bool,
) -> list[str]:
    if selected:
        return selected
    if prioritize_suggestions and item.enriched is not None:
        return list(dict.fromkeys(item.enriched.suggested_new_topics))
    return []


def _promote_topic_suggestions(vocab_topics: list[Topic], selected: list[str]) -> list[str]:
    known = {topic.slug for topic in vocab_topics}
    promoted: list[str] = []
    for slug in selected:
        if slug in known:
            continue
        vocab_topics.append(
            Topic(slug=slug, description=_topic_description_for_promoted_slug(slug))
        )
        known.add(slug)
        promoted.append(slug)
    return promoted


def _run_topic_regenerate(
    cfg: Config,
    store: dict[str, Item],
    item: Item,
    *,
    prioritize_suggestions: bool,
    suggested_topics: list[str] | None,
    promote_suggestions: bool,
) -> list[str]:
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab_topics:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    selected = _selected_topic_suggestions(item, suggested_topics)
    if promote_suggestions and not selected:
        raise ValueError("select at least one suggested topic to promote")
    promoted_topics = (
        _promote_topic_suggestions(vocab_topics, selected) if promote_suggestions else []
    )
    hints = _topic_suggestion_hints(
        item,
        selected,
        prioritize_suggestions=prioritize_suggestions or promote_suggestions,
    )
    executor = ApiExecutor(
        model=cfg.llm_model,
        output_language=cfg.output_language,
        provider=cfg.llm_provider,
        base_url=cfg.llm_base_url,
        topic_hints={item.id: hints} if hints else None,
    )
    enriched, invalid = enrich_selected_with_executor(store, executor, vocab_topics, [item])
    save_store(store, cfg.items_path)
    _report_invalid(invalid)
    if invalid or enriched != 1:
        raise RuntimeError(f"topic regenerate no pudo re-enriquecer {item.id}")
    if promoted_topics:
        save_vocab(vocab_topics, cfg.data_dir / "vocab.yaml")
    _run_topics_executor(cfg, "api", resynth=False)
    _run_generate(cfg, None, None)
    return promoted_topics


def _run_topic_accept_or_reject(
    cfg: Config, store: dict[str, Item], item: Item, action: str
) -> None:
    if item.enriched is None:
        raise RuntimeError(f"item has no topic assignment: {item.id}")
    if action == "accept":
        item.enriched.topic_confidence = "high"
        item.enriched.suggested_new_topics = []
    else:
        item.enriched.primary_topic = "misc"
        item.enriched.topics = ["misc"]
        item.enriched.topic_confidence = "high"
        item.enriched.suggested_new_topics = []
    save_store(store, cfg.items_path)
    _refresh_topic_pages_after_delete(cfg, store, item.id)
    _run_generate(cfg, None, None)


def _run_topic_action(
    cfg: Config,
    item_id: str,
    action: str,
    *,
    prioritize_suggestions: bool = False,
    suggested_topics: list[str] | None = None,
    promote_suggestions: bool = False,
) -> TopicActionReport:
    """Apply a per-note taxonomy decision and refresh generated outputs."""
    if action not in {"accept", "reject", "regenerate"}:
        raise ValueError(f"unknown topic action: {action}")
    if action != "regenerate" and (
        prioritize_suggestions or promote_suggestions or suggested_topics
    ):
        raise ValueError("topic suggestions can only be used with regenerate")
    store = load_store(cfg.items_path)
    item = store.get(item_id)
    if item is None:
        raise ValueError(f"item not found: {item_id}")
    _auto_snapshot(cfg, f"topic-{action}-{item_id}")
    promoted_topics: list[str] = []

    if action == "regenerate":
        promoted_topics = _run_topic_regenerate(
            cfg,
            store,
            item,
            prioritize_suggestions=prioritize_suggestions or promote_suggestions,
            suggested_topics=suggested_topics,
            promote_suggestions=promote_suggestions,
        )
    else:
        _run_topic_accept_or_reject(cfg, store, item, action)

    store = load_store(cfg.items_path)
    item = store.get(item_id)
    if item is None or item.enriched is None:
        raise RuntimeError(f"topic action left item unenriched: {item_id}")
    typer.echo(
        "topic-action: "
        f"{action} {item_id} · primary {item.enriched.primary_topic or '—'} · "
        f"confidence {item.enriched.topic_confidence or 'unknown'}"
    )
    return TopicActionReport(
        item_id=item_id,
        action=action,
        primary_topic=item.enriched.primary_topic,
        topics=item.enriched.topics,
        topic_confidence=item.enriched.topic_confidence,
        suggested_new_topics=item.enriched.suggested_new_topics,
        promoted_topics=promoted_topics,
    )


def _run_delete_bookmark(cfg: Config, item_id: str) -> DeleteBookmarkReport:
    """Remove one bookmarked item and the generated artifacts tied to it."""
    store = load_store(cfg.items_path)
    item = store.get(item_id)
    if item is None:
        raise ValueError(f"item not found: {item_id}")
    _auto_snapshot(cfg, f"delete-bookmark-{item_id}")
    store.pop(item_id)
    save_store(store, cfg.items_path)

    notes_deleted = _delete_item_notes(cfg.output_dir, item)
    media_dirs_deleted = _remove_direct_child_dir(cfg.media_dir, item_id)
    media_dirs_deleted += _remove_direct_child_dir(cfg.output_dir / "_media", item_id)
    video_artifacts_deleted = _delete_video_artifacts(cfg.output_dir, item_id)

    _run_generate(cfg, None, None)
    topic_pages_touched = _refresh_topic_pages_after_delete(cfg, store, item_id)
    typer.echo(
        "delete-bookmark: "
        f"{item_id} eliminado · notas {notes_deleted} · media dirs {media_dirs_deleted} · "
        f"vídeo {video_artifacts_deleted} · topics {topic_pages_touched}"
    )
    return DeleteBookmarkReport(
        item_id=item_id,
        notes_deleted=notes_deleted,
        media_dirs_deleted=media_dirs_deleted,
        video_artifacts_deleted=video_artifacts_deleted,
        topic_pages_touched=topic_pages_touched,
    )


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
        data_dir=cfg.data_dir,
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
    video_max_size_bytes: int | None = parse_size_to_bytes(_DEFAULT_REFRESH_VIDEO_MAX_SIZE),
) -> list[str]:
    """One-command daily ingestion pipeline for unattended/mobile runs."""
    if executor != "api":
        raise ValueError("refresh-all solo soporta --executor api por ahora.")
    before_store = load_store(cfg.items_path)
    typer.echo("refresh-all: 1/8 extract")
    _run_extract(cfg, source, since, until, headless=headless)
    typer.echo("refresh-all: 2/8 fetch")
    _run_fetch(cfg, since, until, False, headless=headless)
    typer.echo("refresh-all: 3/8 digest-video")
    _run_digest_video(
        cfg,
        ids=None,
        topic=None,
        all_pending=True,
        source=source,
        limit=None,
        force=False,
        language=None,
        frames=False,
        vision_model=None,
        max_size_bytes=video_max_size_bytes,
        allow_empty=True,
    )
    if skip_media:
        typer.echo("refresh-all: 4/8 media saltado")
    else:
        typer.echo("refresh-all: 4/8 media")
        _run_media(cfg, force=False, limit=media_limit, items_filter=None)
    if skip_describe:
        typer.echo("refresh-all: 5/8 describe saltado")
    else:
        typer.echo("refresh-all: 5/8 describe")
        _run_describe(
            cfg,
            force=False,
            limit=describe_limit,
            items_filter=None,
            model=cfg.llm_vision_model,
            batch_size=describe_batch_size,
            verbose=False,
        )
    typer.echo("refresh-all: 6/8 enrich")
    _run_enrich_api(cfg, since, until)
    typer.echo("refresh-all: 7/8 topics")
    _run_topics_executor(cfg, executor, resynth=False)
    typer.echo("refresh-all: 8/8 generate")
    _run_generate(cfg, since, until)
    typer.echo("refresh-all: completado")
    after_store = load_store(cfg.items_path)
    return _updated_item_ids(before_store, after_store)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _item_id_from_note_filename(path: Path) -> str | None:
    match = re.search(r"-([^-/.]+)\.md$", path.name)
    return match.group(1) if match else None


def _resolve_stale_served_note(root: Path, missing_path: Path) -> Path | None:
    item_id = _item_id_from_note_filename(missing_path)
    if item_id is None:
        return None
    items_dir = root / "items"
    if not items_dir.exists():
        return None
    for path in sorted(items_dir.glob("*.md")):
        if _note_file_belongs_to_item(path, item_id):
            return path.resolve(strict=True)
    return None


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
        if _path_is_within(candidate, root):
            fallback = _resolve_stale_served_note(root, candidate)
            if fallback is not None:
                return fallback
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


def _note_frontmatter_id(markdown: str) -> str | None:
    """Return the item id from note frontmatter when present."""
    text = markdown.replace(GEN_START, "", 1).lstrip()
    match = re.match(r"^---\s*\n(?P<body>.*?)\n---\s*\n?", text, flags=re.S)
    if match is None:
        return None
    id_match = re.search(
        r"^id:\s*[\"']?(?P<id>[^\"'\n]+)[\"']?\s*$", match.group("body"), flags=re.M
    )
    if id_match is None:
        return None
    item_id = id_match.group("id").strip()
    return item_id or None


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
    item_id = _note_frontmatter_id(markdown)
    reprocess_control = ""
    delete_modal = ""
    topic_modal = ""
    note_actions_script = ""
    if item_id:
        escaped_item_id = html.escape(item_id, quote=True)
        js_item_id = json.dumps(item_id)
        reprocess_control = (
            '<div class="note-actions">'
            f'<button class="note-icon" id="topics-note" type="button" '
            f'data-item-id="{escaped_item_id}" title="Review topics" '
            'aria-label="Review topics">#</button>'
            f'<button class="note-icon" id="reprocess-note" type="button" '
            f'data-item-id="{escaped_item_id}" title="Reprocess note" '
            'aria-label="Reprocess note">↻</button>'
            f'<button class="note-icon danger" id="delete-note" type="button" '
            f'data-item-id="{escaped_item_id}" title="Delete bookmark and note" '
            'aria-label="Delete bookmark and note">×</button>'
            '<span class="action-status" id="note-action-status"></span>'
            "</div>"
        )
        delete_modal = """
<div class="modal" id="delete-modal" hidden>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="delete-title">
    <h2 id="delete-title">Delete bookmark</h2>
    <p>This removes the note, the store item, and generated files for this bookmark.</p>
    <div class="modal-actions">
      <button class="modal-btn" id="delete-cancel" type="button">Cancel</button>
      <button class="modal-btn danger" id="delete-confirm" type="button">Delete</button>
    </div>
  </div>
</div>"""
        topic_modal = """
<div class="modal" id="topic-modal" hidden>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="topic-title">
    <h2 id="topic-title">Topics</h2>
    <div class="topic-state" id="topic-state">Loading...</div>
    <div class="topic-suggestions" id="topic-suggestions" hidden></div>
    <div class="modal-actions">
      <button class="modal-btn" id="topic-cancel" type="button">Close</button>
      <button class="modal-btn" id="topic-accept" type="button">Accept</button>
      <button class="modal-btn" id="topic-reject" type="button">Reject</button>
      <button class="modal-btn" id="topic-regenerate" type="button">Regenerate</button>
      <button class="modal-btn" id="topic-prioritize" type="button">Prioritize</button>
      <button class="modal-btn primary" id="topic-promote" type="button">Promote + regenerate</button>
    </div>
  </div>
</div>"""
        note_actions_script = f"""
<script>
const itemId = {js_item_id};
const actionStatus = document.getElementById('note-action-status');
const topicButton = document.getElementById('topics-note');
const topicModal = document.getElementById('topic-modal');
const topicState = document.getElementById('topic-state');
const topicSuggestions = document.getElementById('topic-suggestions');
const topicCancel = document.getElementById('topic-cancel');
const topicAccept = document.getElementById('topic-accept');
const topicReject = document.getElementById('topic-reject');
const topicRegenerate = document.getElementById('topic-regenerate');
const topicPrioritize = document.getElementById('topic-prioritize');
const topicPromote = document.getElementById('topic-promote');
const reprocessButton = document.getElementById('reprocess-note');
const deleteButton = document.getElementById('delete-note');
const deleteModal = document.getElementById('delete-modal');
const deleteCancel = document.getElementById('delete-cancel');
const deleteConfirm = document.getElementById('delete-confirm');
const escapeHtml = value => String(value ?? '').replace(/[&<>"]/g, char => ({{
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;'
}}[char]));
const setStatus = text => {{
  if (actionStatus) actionStatus.textContent = text;
}};
const postItemAction = async (endpoint, extra = {{}}) => {{
  const response = await fetch(endpoint, {{
    method: 'POST',
    headers: {{'content-type': 'application/json'}},
    body: JSON.stringify({{item_id: itemId, ...extra}})
  }});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'request failed');
  return payload;
}};
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const waitForAction = async (actionName, onDone, progressText) => {{
  for (;;) {{
    await sleep(1000);
    const response = await fetch('/api/status');
    const state = await response.json();
    if (state.running) {{
      setStatus(progressText);
      continue;
    }}
    if (state.last_action === actionName && state.last_error) {{
      throw new Error(String(state.last_error).split('\\n')[0]);
    }}
    onDone();
    return;
  }}
}};
if (reprocessButton) {{
  const originalText = reprocessButton.textContent;
  reprocessButton.addEventListener('click', async () => {{
    reprocessButton.disabled = true;
    reprocessButton.textContent = '...';
    setStatus('reprocess started...');
    try {{
      await postItemAction('/api/reprocess-note');
      await waitForAction('reprocess-note:' + itemId, () => window.location.reload(), 'reprocessing...');
    }} catch (error) {{
      reprocessButton.disabled = false;
      reprocessButton.textContent = originalText;
      setStatus('could not start: ' + error.message);
    }}
  }});
}}
if (topicButton && topicModal && topicState) {{
  const topicControls = [
    topicAccept,
    topicReject,
    topicRegenerate,
    topicPrioritize,
    topicPromote
  ].filter(Boolean);
  const setTopicModal = open => {{
    topicModal.hidden = !open;
    if (open) topicButton.focus();
  }};
  const setTopicDisabled = disabled => topicControls.forEach(button => {{ button.disabled = disabled; }});
  const renderTopicState = payload => {{
    const topics = payload.topics && payload.topics.length ? payload.topics.join(', ') : 'none';
    const suggestions = payload.suggested_new_topics || [];
    topicState.textContent = payload.enriched
      ? 'Primary: ' + (payload.primary_topic || 'none') + ' · topics: ' + topics +
        ' · confidence: ' + (payload.topic_confidence || 'unknown') +
        ' · suggested: ' + (suggestions.length ? suggestions.join(', ') : 'none')
      : 'No topic assignment yet. Use regenerate to create one.';
    if (topicSuggestions) {{
      if (suggestions.length) {{
        topicSuggestions.hidden = false;
        topicSuggestions.innerHTML =
          '<div class="topic-suggestions-title">Suggestions to review</div>' +
          suggestions.map(slug =>
            '<label class="topic-suggestion">' +
            '<input type="checkbox" value="' + escapeHtml(slug) + '" checked>' +
            '<span>' + escapeHtml(slug) + '</span>' +
            '</label>'
          ).join('');
      }} else {{
        topicSuggestions.hidden = true;
        topicSuggestions.innerHTML = '';
      }}
    }}
    const hasSuggestions = suggestions.length > 0;
    if (topicPrioritize) topicPrioritize.disabled = !hasSuggestions;
    if (topicPromote) topicPromote.disabled = !hasSuggestions;
  }};
  const selectedSuggestions = () => Array.from(
    topicSuggestions ? topicSuggestions.querySelectorAll('input[type="checkbox"]:checked') : []
  ).map(input => input.value);
  const loadTopicState = async () => {{
    topicState.textContent = 'Loading...';
    if (topicSuggestions) {{
      topicSuggestions.hidden = true;
      topicSuggestions.innerHTML = '';
    }}
    const response = await fetch('/api/topic-state?item_id=' + encodeURIComponent(itemId));
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'request failed');
    renderTopicState(payload);
  }};
  const runTopicAction = async (action, extra = {{}}) => {{
    setTopicDisabled(true);
    setStatus('topic ' + action + ' started...');
    try {{
      await postItemAction('/api/topic-action', {{action, ...extra}});
      await waitForAction('topic-' + action + ':' + itemId, () => window.location.reload(), 'updating topics...');
    }} catch (error) {{
      setTopicDisabled(false);
      setStatus('could not update topics: ' + error.message);
    }}
  }};
  topicButton.addEventListener('click', async () => {{
    setTopicModal(true);
    setTopicDisabled(false);
    try {{
      await loadTopicState();
    }} catch (error) {{
      topicState.textContent = 'Could not load topics: ' + error.message;
    }}
  }});
  if (topicCancel) topicCancel.addEventListener('click', () => setTopicModal(false));
  topicModal.addEventListener('click', event => {{
    if (event.target === topicModal) setTopicModal(false);
  }});
  if (topicAccept) topicAccept.addEventListener('click', () => runTopicAction('accept'));
  if (topicReject) topicReject.addEventListener('click', () => runTopicAction('reject'));
  if (topicRegenerate) topicRegenerate.addEventListener('click', () => runTopicAction('regenerate'));
  if (topicPrioritize) topicPrioritize.addEventListener('click', () => runTopicAction('regenerate', {{
    prioritize_suggestions: true,
    suggested_topics: selectedSuggestions()
  }}));
  if (topicPromote) topicPromote.addEventListener('click', () => {{
    const selected = selectedSuggestions();
    if (!selected.length) {{
      setStatus('select at least one suggestion');
      return;
    }}
    runTopicAction('regenerate', {{
      prioritize_suggestions: true,
      promote_suggestions: true,
      suggested_topics: selected
    }});
  }});
}}
if (deleteButton && deleteModal && deleteCancel && deleteConfirm) {{
  const setDeleteModal = open => {{
    deleteModal.hidden = !open;
    if (open) deleteConfirm.focus();
  }};
  const waitForDelete = async () => {{
    await waitForAction('delete-bookmark:' + itemId, () => window.location.href = '/', 'deleting...');
  }};
  deleteButton.addEventListener('click', () => setDeleteModal(true));
  deleteCancel.addEventListener('click', () => setDeleteModal(false));
  deleteModal.addEventListener('click', event => {{
    if (event.target === deleteModal) setDeleteModal(false);
  }});
  document.addEventListener('keydown', event => {{
    if (event.key === 'Escape' && !deleteModal.hidden) setDeleteModal(false);
  }});
  deleteConfirm.addEventListener('click', async () => {{
    deleteConfirm.disabled = true;
    deleteCancel.disabled = true;
    deleteButton.disabled = true;
    setStatus('deleting...');
    try {{
      await postItemAction('/api/delete-bookmark');
      setDeleteModal(false);
      await waitForDelete();
    }} catch (error) {{
      deleteConfirm.disabled = false;
      deleteCancel.disabled = false;
      deleteButton.disabled = false;
      setStatus('could not delete: ' + error.message);
    }}
  }});
}}
</script>"""
    theme_script = """
<script>
try {
  const savedTheme = localStorage.getItem('xbrain.theme');
  const prefersLight = matchMedia('(prefers-color-scheme: light)').matches;
  document.documentElement.dataset.theme = savedTheme || (prefersLight ? 'light' : 'dark');
} catch (error) {
  document.documentElement.dataset.theme = 'dark';
}
</script>"""
    note_theme_script = """
<script>
document.getElementById('note-theme')?.addEventListener('click', () => {
  const root = document.documentElement;
  const next = root.dataset.theme === 'light' ? 'dark' : 'light';
  root.dataset.theme = next;
  try { localStorage.setItem('xbrain.theme', next); } catch (error) {}
});
</script>"""
    page = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="content-security-policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self';">
<title>{escaped_title}</title>
{theme_script}
<style>
:root {{
  color-scheme:dark;
  --radius-1:2px; --radius-2:4px; --radius-3:6px; --radius-4:8px; --radius-5:10px; --radius-panel:12px;
  --color-bg:#07110D; --color-bg-wash:#0A1711; --color-panel:#0F1C16; --color-panel-raised:#14241C; --color-panel-soft:#102018;
  --color-input:#08140F; --color-text:#EDF4E8; --color-text-muted:#AEBBAB; --color-text-faint:#6F806F;
  --color-border:rgba(237,244,232,.11); --color-border-soft:rgba(237,244,232,.06); --color-border-strong:rgba(237,244,232,.2);
  --color-accent:#D7B56D; --color-accent-soft:rgba(215,181,109,.15); --color-forest:#2E6B4E; --color-danger:#D66A57;
  --color-danger-soft:rgba(214,106,87,.15); --color-overlay:rgba(3,8,5,.68);
  --shadow-panel:0 1px 0 rgba(0,0,0,.45), 0 22px 60px -42px rgba(0,0,0,.95);
  --shadow-focus:0 0 0 3px rgba(215,181,109,.16);
  --bg:var(--color-bg); --surface:var(--color-panel); --surface-2:var(--color-panel-raised);
  --muted:var(--color-text-muted); --faint:var(--color-text-faint); --ink:var(--color-text);
  --hair:var(--color-border); --accent:var(--color-accent);
  --display:Georgia, "Times New Roman", serif; --ui:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
[data-theme="light"] {{
  color-scheme:light;
  --color-bg:#F4F6EF; --color-bg-wash:#E9EFE3; --color-panel:#FFFFFF; --color-panel-raised:#F9FBF5; --color-panel-soft:#EDF3E9;
  --color-input:#FBFDF8; --color-text:#142219; --color-text-muted:#52614F; --color-text-faint:#7B8977;
  --color-border:rgba(20,34,25,.12); --color-border-soft:rgba(20,34,25,.07); --color-border-strong:rgba(20,34,25,.2);
  --color-accent:#806226; --color-accent-soft:rgba(128,98,38,.12); --color-forest:#24563D; --color-danger:#B44D3E;
  --color-danger-soft:rgba(180,77,62,.13); --color-overlay:rgba(20,34,25,.22);
  --shadow-panel:0 1px 0 rgba(20,34,25,.04), 0 22px 60px -46px rgba(20,34,25,.35);
  --shadow-focus:0 0 0 3px rgba(128,98,38,.15);
  --bg:var(--color-bg); --surface:var(--color-panel); --surface-2:var(--color-panel-raised);
  --muted:var(--color-text-muted); --faint:var(--color-text-faint); --ink:var(--color-text);
  --hair:var(--color-border); --accent:var(--color-accent);
}}
* {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }}
body {{
  margin:0; background:var(--bg); color:var(--ink); min-height:100vh; overflow-x:hidden; position:relative;
  font-family:var(--ui); font-size:15px; line-height:1.55;
}}
.note-glow {{
  position:fixed; inset:0; pointer-events:none; z-index:0;
  background:
    linear-gradient(115deg, transparent 0 34%, color-mix(in srgb,var(--color-forest) 12%, transparent) 34% 35%, transparent 35% 100%),
    repeating-linear-gradient(90deg, color-mix(in srgb,var(--color-text) 2.5%, transparent) 0 1px, transparent 1px 72px),
    linear-gradient(180deg, var(--color-bg-wash), var(--color-bg) 42%);
}}
.wrap {{ position:relative; z-index:1; max-width:1040px; margin:0 auto; padding:38px 22px 56px; }}
.top {{
  display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:18px;
  padding-bottom:16px; border-bottom:1px solid var(--hair);
}}
.btn {{
  color:var(--muted); text-decoration:none; border:1px solid var(--hair);
  background:var(--color-panel-soft); border-radius:var(--radius-3); padding:8px 10px;
  font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  min-height:34px; display:inline-flex; align-items:center; justify-content:center;
}}
.btn.primary {{ border-color:var(--color-border-strong); color:var(--accent); background:var(--color-accent-soft); }}
.btn.subtle {{ color:var(--faint); background:transparent; }}
.btn.icon {{ width:34px; padding:0; margin-left:auto; font-size:14px; line-height:1; }}
button.btn {{ cursor:pointer; }}
.btn:hover {{ color:var(--ink); border-color:var(--color-border-strong); }}
.btn:focus-visible,.note-icon:focus-visible,.modal-btn:focus-visible {{ outline:0; box-shadow:var(--shadow-focus); }}
button.btn:disabled {{ opacity:.52; cursor:wait; }}
.note-actions {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin:-4px 0 18px; }}
.note-icon {{
  width:28px; height:28px; display:inline-flex; align-items:center; justify-content:center;
  color:var(--faint); background:var(--color-panel-soft); border:1px solid var(--hair);
  border-radius:var(--radius-3); font:600 13px/1 var(--mono);
  cursor:pointer; padding:0;
}}
.note-icon:hover {{ color:var(--accent); border-color:var(--color-border-strong); }}
.note-icon.danger:hover {{ color:var(--color-danger); border-color:color-mix(in srgb,var(--color-danger) 44%, transparent); }}
.note-icon:disabled {{ opacity:.45; cursor:wait; }}
.action-status {{ align-self:center; color:var(--faint); font-size:11px; min-height:1em; }}
.modal[hidden] {{ display:none; }}
.modal {{
  position:fixed; inset:0; z-index:10; display:flex; align-items:center; justify-content:center;
  padding:18px; background:var(--color-overlay); backdrop-filter:blur(2px);
}}
.modal-card {{
  width:min(360px, 100%); background:var(--surface); border:1px solid var(--hair);
  border-radius:var(--radius-4); padding:18px; box-shadow:var(--shadow-panel);
}}
.modal-card h2 {{ margin:0 0 8px; font-size:18px; line-height:1.2; }}
.modal-card p {{ margin:0; color:var(--muted); font-size:13.5px; line-height:1.5; }}
.modal-actions {{ display:flex; justify-content:flex-end; flex-wrap:wrap; gap:8px; margin-top:16px; }}
.modal-btn {{
  color:var(--ink); background:var(--color-panel-soft); border:1px solid var(--hair);
  border-radius:var(--radius-3); padding:8px 10px; font:inherit; font-size:13px; cursor:pointer;
}}
.modal-btn.primary {{ color:var(--accent); border-color:var(--color-border-strong); }}
.modal-btn.danger {{ color:var(--color-danger); border-color:color-mix(in srgb,var(--color-danger) 44%, transparent); }}
.modal-btn:disabled {{ opacity:.52; cursor:wait; }}
.topic-state {{
  color:var(--muted); font-size:12.5px; line-height:1.5; border:1px solid var(--hair);
  background:var(--color-input); border-radius:var(--radius-3); padding:10px; overflow-wrap:anywhere;
}}
.topic-suggestions {{
  margin-top:10px; color:var(--muted); font-size:12.5px; line-height:1.4;
  border:1px solid var(--hair); background:var(--color-input); border-radius:var(--radius-3); padding:10px;
  max-height:190px; overflow:auto;
}}
.topic-suggestions-title {{ color:var(--ink); font-weight:650; margin-bottom:8px; }}
.topic-suggestion {{ display:flex; align-items:center; gap:8px; margin:6px 0; overflow-wrap:anywhere; }}
.topic-suggestion input {{ margin:0; accent-color:var(--accent); flex:0 0 auto; }}
article {{
  background:var(--surface); border:1px solid var(--hair); border-radius:var(--radius-panel);
  padding:28px 32px; overflow:hidden; box-shadow:var(--shadow-panel);
}}
.eyebrow {{
  color:var(--accent); font-family:var(--mono); font-size:10px; text-transform:uppercase; letter-spacing:.16em;
  margin:0 0 8px;
}}
h1 {{ margin:0 0 10px; font-family:var(--display); font-size:clamp(28px, 7vw, 48px); line-height:1.04; font-weight:650; }}
.path {{ color:var(--faint); font-family:var(--mono); font-size:11px; letter-spacing:.04em; word-break:break-word; margin-bottom:20px; }}
.note-body {{ color:var(--muted); font-size:15.5px; line-height:1.68; overflow-wrap:anywhere; }}
.note-body > *:first-child {{ margin-top:0; }}
.note-body > *:last-child {{ margin-bottom:0; }}
.note-body h1,.note-body h2,.note-body h3,.note-body h4 {{
  color:var(--ink); font-family:var(--display); line-height:1.18; margin:1.35em 0 .55em; font-weight:650;
}}
.note-body h1 {{ font-size:30px; }}
.note-body h2 {{ font-size:24px; border-top:1px solid var(--hair); padding-top:20px; }}
.note-body h3 {{ font-size:19px; }}
.note-body p {{ margin:.85em 0; }}
.note-body a {{ color:var(--accent); text-decoration-thickness:1px; text-underline-offset:3px; }}
.note-body ul,.note-body ol {{ padding-left:1.35em; margin:.85em 0; }}
.note-body li {{ margin:.35em 0; }}
.note-body blockquote {{
  margin:1em 0; padding:.1em 0 .1em 1em; border-left:3px solid var(--accent);
  color:var(--ink);
}}
.note-body img {{
  display:block; max-width:100%; height:auto; margin:1.15em auto; border-radius:var(--radius-4);
  border:1px solid var(--hair); background:var(--color-input);
}}
.note-body code {{
  font-family:var(--mono); font-size:.92em;
  background:var(--color-input); color:var(--ink); border:1px solid var(--hair); border-radius:var(--radius-2);
  padding:.08em .32em;
}}
.note-body pre {{
  background:var(--color-input); border:1px solid var(--hair); border-radius:var(--radius-4); padding:13px;
  overflow:auto; white-space:pre; margin:1em 0;
}}
.note-body pre code {{ border:0; padding:0; background:transparent; }}
.note-body hr {{ border:0; border-top:1px solid var(--hair); margin:1.4em 0; }}
@media (max-width:640px) {{
  .wrap {{ padding:18px 18px 38px; }}
  article {{ padding:18px 16px; border-radius:var(--radius-4); }}
  .top {{ position:sticky; top:0; z-index:2; background:color-mix(in srgb,var(--bg) 92%, transparent); padding:8px 0; backdrop-filter:blur(8px); }}
  .btn {{ flex:1 1 auto; text-align:center; }}
  .btn.icon {{ flex:0 0 34px; margin-left:0; }}
  .action-status {{ flex:1 0 100%; }}
  .note-body {{ font-size:14.5px; line-height:1.64; }}
  .note-body h1 {{ font-size:25px; }}
  .note-body h2 {{ font-size:20px; }}
}}
</style>
</head>
<body>
<div class="note-glow"></div>
<main class="wrap">
  <nav class="top">
    <a class="btn primary" href="/#chat">Ask XBrain</a>
    <a class="btn" href="/">Dashboard</a>
    <a class="btn" href="{html.escape(obsidian_href)}">Obsidian</a>
    <button class="btn icon" id="note-theme" type="button" aria-label="Toggle light and dark theme" title="Toggle theme">◐</button>
  </nav>
  <article>
    <p class="eyebrow">XBrain note</p>
    <h1>{escaped_title}</h1>
    <div class="path">{escaped_rel}</div>
    {reprocess_control}
    <div class="note-body">{body_html}</div>
  </article>
</main>
{topic_modal}
{delete_modal}
{note_theme_script}
{note_actions_script}
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
                updated_item_ids: list[str] = []
                with drive_write_session(cfg):
                    result = target()
                    if isinstance(result, list) and all(
                        isinstance(item_id, str) for item_id in result
                    ):
                        updated_item_ids = result
                if action == "refresh-all":
                    _notify_bookmark_update(
                        cfg, command="refresh-all", updated_item_ids=updated_item_ids
                    )
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

    def start_reprocess_note(item_id: str) -> bool:
        return start_background(
            f"reprocess-note:{item_id}",
            lambda: _run_reprocess_note(cfg, item_id, headless=True),
        )

    def start_topic_action(
        item_id: str,
        action: str,
        *,
        prioritize_suggestions: bool = False,
        suggested_topics: list[str] | None = None,
        promote_suggestions: bool = False,
    ) -> bool:
        return start_background(
            f"topic-{action}:{item_id}",
            lambda: _run_topic_action(
                cfg,
                item_id,
                action,
                prioritize_suggestions=prioritize_suggestions,
                suggested_topics=suggested_topics,
                promote_suggestions=promote_suggestions,
            ),
        )

    def start_delete_bookmark(item_id: str) -> bool:
        return start_background(
            f"delete-bookmark:{item_id}",
            lambda: _run_delete_bookmark(cfg, item_id),
        )

    def dashboard_bytes() -> bytes:
        dashboard = cfg.output_dir / "dashboard.html"
        if not dashboard.exists():
            with drive_write_session(cfg):
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

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            *,
            include_body: bool = True,
        ) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def _send_json(
            self,
            status: int,
            payload: dict[str, object],
            *,
            include_body: bool = True,
        ) -> None:
            self._send(
                status, json.dumps(payload).encode(), "application/json", include_body=include_body
            )

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

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/status":
                self._send_json(200, state_payload(), include_body=False)
                return
            if path in ("/", "/dashboard.html"):
                try:
                    self._send(
                        200,
                        dashboard_bytes(),
                        "text/html; charset=utf-8",
                        include_body=False,
                    )
                except Exception as exc:  # noqa: BLE001 - HTTP handler should return cleanly
                    self._send_json(500, {"error": str(exc)}, include_body=False)
                return
            self._send_json(404, {"error": "not found"}, include_body=False)

        def do_GET(self) -> None:  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/status":
                self._send_json(200, state_payload())
                return
            if path == "/api/topic-state":
                item_id = parse_qs(parsed.query).get("item_id", [""])[0].strip()
                if not item_id:
                    self._send_json(400, {"error": "missing item_id"})
                    return
                item = load_store(cfg.items_path).get(item_id)
                if item is None:
                    self._send_json(404, {"error": f"item not found: {item_id}"})
                    return
                self._send_json(200, _topic_state_payload(item))
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
            if path == "/api/reprocess-note":
                try:
                    payload = self._read_json()
                    raw_item_id = payload.get("item_id")
                    if not isinstance(raw_item_id, str):
                        raise ValueError("item_id must be a string")
                    item_id = raw_item_id.strip()
                    if not item_id:
                        raise ValueError("item_id must not be empty")
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                if not start_reprocess_note(item_id):
                    self._send_json(409, {**state_payload(), "error": "job already running"})
                    return
                self._send_json(202, state_payload())
                return
            if path == "/api/topic-action":
                try:
                    payload = self._read_json()
                    raw_item_id = payload.get("item_id")
                    raw_action = payload.get("action")
                    if not isinstance(raw_item_id, str):
                        raise ValueError("item_id must be a string")
                    if not isinstance(raw_action, str):
                        raise ValueError("action must be a string")
                    item_id = raw_item_id.strip()
                    action = raw_action.strip()
                    if not item_id:
                        raise ValueError("item_id must not be empty")
                    if action not in {"accept", "reject", "regenerate"}:
                        raise ValueError("action must be accept, reject or regenerate")
                    raw_prioritize_suggestions = payload.get("prioritize_suggestions", False)
                    raw_promote_suggestions = payload.get("promote_suggestions", False)
                    raw_suggested_topics = payload.get("suggested_topics", [])
                    if not isinstance(raw_prioritize_suggestions, bool):
                        raise ValueError("prioritize_suggestions must be a boolean")
                    if not isinstance(raw_promote_suggestions, bool):
                        raise ValueError("promote_suggestions must be a boolean")
                    if not isinstance(raw_suggested_topics, list) or not all(
                        isinstance(topic, str) for topic in raw_suggested_topics
                    ):
                        raise ValueError("suggested_topics must be a list of strings")
                    if action != "regenerate" and (
                        raw_prioritize_suggestions
                        or raw_promote_suggestions
                        or raw_suggested_topics
                    ):
                        raise ValueError("topic suggestions can only be used with regenerate")
                    if item_id not in load_store(cfg.items_path):
                        self._send_json(404, {"error": f"item not found: {item_id}"})
                        return
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                if not start_topic_action(
                    item_id,
                    action,
                    prioritize_suggestions=raw_prioritize_suggestions,
                    suggested_topics=raw_suggested_topics,
                    promote_suggestions=raw_promote_suggestions,
                ):
                    self._send_json(409, {**state_payload(), "error": "job already running"})
                    return
                self._send_json(202, state_payload())
                return
            if path == "/api/delete-bookmark":
                try:
                    payload = self._read_json()
                    raw_item_id = payload.get("item_id")
                    if not isinstance(raw_item_id, str):
                        raise ValueError("item_id must be a string")
                    item_id = raw_item_id.strip()
                    if not item_id:
                        raise ValueError("item_id must not be empty")
                    if item_id not in load_store(cfg.items_path):
                        self._send_json(404, {"error": f"item not found: {item_id}"})
                        return
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                if not start_delete_bookmark(item_id):
                    self._send_json(409, {**state_payload(), "error": "job already running"})
                    return
                self._send_json(202, state_payload())
                return
            self._send_json(404, {"error": "not found"})

    server = ThreadingHTTPServer((host, port), Handler)
    typer.echo(f"Dashboard local: http://{host}:{server.server_port}/")
    typer.echo(
        "POST /api/refresh-all, /api/retry-failed, /api/reprocess-note, "
        "/api/topic-action y /api/delete-bookmark ejecutan tareas solo en este servidor local."
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
    cfg = _config()
    with drive_write_session(cfg):
        _run_extract(
            cfg,
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
    with drive_write_session(cfg):
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
    cfg = _config()
    with drive_write_session(cfg):
        _run_fetch(
            cfg, _parse_date(since), _parse_date(until, end_of_day=True), force, headless=headless
        )


@app.command(name="retry-failed")
@_handle_cli_errors
def retry_failed(
    source: Source = typer.Option(Source.bookmarks, help="bookmarks | tweets | all"),
    headless: bool = typer.Option(True, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Relanza solo los items cuyos artículos enlazados fallaron y regenera el vault."""
    cfg = _config()
    with drive_write_session(cfg):
        _run_retry_failed_articles(cfg, source.value, headless=headless, regenerate=True)


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
    with drive_write_session(cfg):
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
    cfg = _config()
    with drive_write_session(cfg):
        _run_refresh_media(cfg, source.value, force=force, headless=headless)


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
    with drive_write_session(cfg):
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
    """Deshabilitado: XBrain no ofrece una descarga persistente de MP4."""
    raise RuntimeError(
        "download-videos está deshabilitado por política de almacenamiento: "
        "XBrain no guarda MP4. Usa digest-video para captions/transcripts."
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

    Una fila por entrada de vídeo, con estado del media (downloaded / failed /
    pending / poster-era), estado del digest (done / pending), tamaño estimado
    (exacto si ya está descargado, "unknown" si no hay bitrate/duración), el
    `primary_topic` del item y un snippet del texto. NO escribe nada ni toma
    snapshot. Con `--json` emite un array estable con los campos `id, url, state,
    digest, topic, size_bytes|null, mp4_url, text`.
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


@app.command(name="fetch-video")
@_handle_cli_errors
def fetch_video(
    to: Path | None = typer.Option(
        None, "--to", help="Deshabilitado: XBrain no escribe MP4 en disco."
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
    """Deshabilitado: XBrain no ofrece una descarga manual persistente."""
    raise RuntimeError(
        "fetch-video está deshabilitado por política de almacenamiento: "
        "XBrain no deja MP4 en disco. Usa digest-video para captions/transcripts."
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
    allow_empty: bool = False,
) -> None:
    """Digest selected videos from captions or a temporary transcription fallback."""
    if frames:
        raise ValueError(
            "digest-video --frames está deshabilitado: XBrain no almacena frames ni "
            "contenido visual derivado de vídeos."
        )
    if vision_model:
        raise ValueError("--vision-model solo tenía sentido con --frames, ahora deshabilitado.")
    store = load_store(cfg.items_path)
    try:
        id_list = _resolve_digest_ids(store, ids, topic, all_pending, source, limit)
    except ValueError:
        if not allow_empty:
            raise
        typer.echo("Vídeos: no hay vídeos seleccionados para digest.")
        return

    def _transcript(item: Item, entry) -> VideoTranscript:
        return fetch_or_transcribe_video_transcript(
            item,
            entry,
            language=language,
            transcribe_command=cfg.transcribe_command,
            transcribe_model=cfg.transcribe_model,
            transcribe_timeout_seconds=cfg.transcribe_timeout_seconds,
            max_size_bytes=max_size_bytes,
        )

    report = digest_video_transcripts(
        store,
        id_list,
        force=force,
        transcript_fn=_transcript,
        summary_fn=configured_summary_fn(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            output_language=cfg.output_language,
            base_url=cfg.llm_base_url or None,
        ),
    )
    if report.changed > 0:
        _auto_snapshot(cfg, "digest-video")
        save_store(store, cfg.items_path)
    typer.echo(format_video_digest_summary(report))


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
        False, "--force", help="Reprocesar items que ya tienen un resumen x_video."
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Idioma a registrar si la pista de captions no lo reporta (p.ej. en, es).",
    ),
    frames: bool = typer.Option(
        False,
        "--frames",
        help="Deshabilitado: XBrain no almacena frames ni contenido visual derivado de vídeos.",
    ),
    vision_model: str | None = typer.Option(
        None,
        "--vision-model",
        help="Deshabilitado junto a --frames.",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help=(
            "Límite para el fallback temporal a MP4 cuando no hay captions "
            "(500MB / 2GB; sin unidad = MB). Las captions no descargan vídeo."
        ),
    ),
) -> None:
    """Procesa vídeos sin persistir media: captions o ASR temporal → resumen ejecutivo.

    Para cada vídeo seleccionado, XBrain busca una pista textual de captions o
    transcript capturada en el payload de X. Si existe, descarga solo ese texto,
    genera un resumen ejecutivo con el LLM configurado y adjunta el resumen como
    `ContentSourceSuccess(kind="x_video")`. La transcripción raw queda guardada
    en el store y se renderiza en `videos/<video>/transcript.md`, pero no alimenta
    `enrich`, `topics`, dashboard ni Ask XBrain. Si X no expone transcript textual,
    XBrain descarga el MP4 solo dentro de un directorio temporal, lo transcribe con
    `[transcribe].command`, guarda solo el texto y elimina siempre los bytes de
    vídeo/audio temporales.
    """
    cfg = _config()
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    with drive_write_session(cfg):
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
    with drive_write_session(cfg):
        store = load_store(cfg.items_path)
        vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
        if not vocab_topics:
            raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")

        if apply is not None:
            executor_name, judgments = import_worksheet(apply)
            enriched, invalid = apply_worksheet_judgments(
                store, judgments, vocab_topics, executor_name
            )
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
    with drive_write_session(cfg):
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
    with drive_write_session(cfg):
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
    cfg = _config()
    with drive_write_session(cfg):
        _run_generate(cfg, _parse_date(since), _parse_date(until, end_of_day=True))


@app.command(name="refresh-all")
@_handle_cli_errors
def refresh_all(
    source: Source = typer.Option(Source.bookmarks, help="bookmarks | tweets | all"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    headless: bool = typer.Option(True, "--headless/--no-headless", help=_HEADLESS_HELP),
    executor: str = typer.Option(
        "api", help="Actualmente solo api; usa la sección llm para NanoGPT."
    ),
    media_limit: int | None = typer.Option(None, "--media-limit"),
    describe_limit: int | None = typer.Option(None, "--describe-limit"),
    describe_batch_size: int = typer.Option(5, "--describe-batch-size", min=1),
    skip_media: bool = typer.Option(False, "--skip-media"),
    skip_describe: bool = typer.Option(False, "--skip-describe"),
    video_max_size: str | None = typer.Option(
        _DEFAULT_REFRESH_VIDEO_MAX_SIZE,
        "--video-max-size",
        help=(
            "Límite para el fallback temporal de digest-video. "
            "Usa 4GB por defecto; pasa '' para desactivar el límite."
        ),
    ),
) -> None:
    """Ejecuta el flujo diario completo para alimentar el vault."""
    cfg = _config()
    with drive_write_session(cfg):
        updated_item_ids = _run_refresh_all(
            cfg,
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
            video_max_size_bytes=parse_size_to_bytes(video_max_size) if video_max_size else None,
        )
    _notify_bookmark_update(cfg, command="refresh-all", updated_item_ids=updated_item_ids)


@app.command()
@_handle_cli_errors
def sync(
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """extract + fetch + generate en orden."""
    cfg = _config()
    with drive_write_session(cfg):
        before_store = load_store(cfg.items_path)
        _run_extract(cfg, "all", None, None, headless=headless)
        _run_fetch(cfg, None, None, False, headless=headless)
        _run_generate(cfg, None, None)
        after_store = load_store(cfg.items_path)
    updated_item_ids = _updated_item_ids(before_store, after_store)
    _notify_bookmark_update(cfg, command="sync", updated_item_ids=updated_item_ids)


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


@dataclass
class _TaxonomyHealth:
    enriched_total: int
    topic_counts: Counter[str] = field(default_factory=Counter)
    confidence_counts: Counter[str] = field(default_factory=Counter)
    suggested_counts: Counter[str] = field(default_factory=Counter)
    unknown_topics: Counter[str] = field(default_factory=Counter)
    misc_count: int = 0
    single_topic_count: int = 0


def _taxonomy_health_summary(store: dict[str, Item], vocab_slugs: set[str]) -> _TaxonomyHealth:
    enriched_items = [item for item in store.values() if item.enriched is not None]
    summary = _TaxonomyHealth(enriched_total=len(enriched_items))
    for item in enriched_items:
        assert item.enriched is not None
        enrichment = item.enriched
        topics = [topic for topic in enrichment.topics if topic]
        summary.topic_counts.update(topics)
        summary.confidence_counts.update([enrichment.topic_confidence or "unknown"])
        summary.suggested_counts.update(enrichment.suggested_new_topics)
        summary.misc_count += int(enrichment.primary_topic == "misc" or "misc" in topics)
        summary.single_topic_count += int(len(topics) <= 1)
        unknown = [topic for topic in topics if topic not in vocab_slugs]
        if enrichment.primary_topic and enrichment.primary_topic not in vocab_slugs:
            unknown.append(enrichment.primary_topic)
        summary.unknown_topics.update(unknown)
    return summary


def _rank_section(title: str, counts: Counter[str], top: int) -> list[str]:
    lines = ["", title]
    lines += [f"- {slug}: {count}" for slug, count in counts.most_common(top)] or ["- none"]
    return lines


def _taxonomy_recommendations(summary: _TaxonomyHealth, unused_topics: list[str]) -> list[str]:
    lines = ["", "Recommendation"]
    if (
        summary.suggested_counts
        or summary.confidence_counts["low"]
        or summary.misc_count > max(3, summary.enriched_total // 10)
    ):
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


def _taxonomy_health_lines(
    store: dict[str, Item], vocab: list[Topic], *, top: int = 10
) -> list[str]:
    """Read-only diagnostics for taxonomy drift and weak assignments."""
    vocab_slugs = {topic.slug for topic in vocab}
    summary = _taxonomy_health_summary(store, vocab_slugs)
    unused_topics = sorted(slug for slug in vocab_slugs if summary.topic_counts[slug] == 0)
    lines = [
        "Taxonomy health",
        f"Items: {len(store)} total · {summary.enriched_total} enriched · "
        f"{len(store) - summary.enriched_total} pending",
        f"Vocabulary: {len(vocab)} topics",
        "",
        "Assignment signals",
        f"- misc: {summary.misc_count} ({_pct(summary.misc_count, summary.enriched_total)})",
        f"- single-topic items: {summary.single_topic_count} "
        f"({_pct(summary.single_topic_count, summary.enriched_total)})",
        f"- confidence high/medium/low/unknown: "
        f"{summary.confidence_counts['high']}/{summary.confidence_counts['medium']}/"
        f"{summary.confidence_counts['low']}/{summary.confidence_counts['unknown']}",
        f"- unused topics: {len(unused_topics)}",
    ]
    if summary.unknown_topics:
        lines += ["", "Unknown assigned topics"]
        lines += [f"- {slug}: {count}" for slug, count in summary.unknown_topics.most_common(top)]
    lines += _rank_section("Top assigned topics", summary.topic_counts, top)
    lines += _rank_section("Suggested new topics", summary.suggested_counts, top)
    return lines + _taxonomy_recommendations(summary, unused_topics)


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


drive_app = typer.Typer(help="Gestionar Google Drive como biblioteca remota")
app.add_typer(drive_app, name="drive")


def _selection_path(cfg: Config) -> Path:
    return cfg.repo_root / "auth" / "google_drive_selection.json"


def _write_drive_selection(cfg: Config, folder_id: str) -> None:
    path = _selection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"root_folder_id": folder_id}, indent=2), encoding="utf-8")


def _format_sync_report(prefix: str, report) -> str:
    return (
        f"{prefix}: downloaded={report.downloaded} uploaded={report.uploaded} "
        f"folders_created={report.folders_created} trashed={report.trashed}"
    )


def _local_legacy_drive_paths(cfg: Config) -> list[Path]:
    return [
        path
        for path in (
            cfg.drive_cache_dir / "data" / "media",
            cfg.drive_cache_dir / "vault" / "x-knowledge",
        )
        if path.exists()
    ]


@drive_app.command("login")
@_handle_cli_errors
def drive_login_cmd(
    port: int = typer.Option(
        8766,
        "--port",
        help="Puerto localhost para el callback OAuth. Útil con túnel SSH en VPS.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open-browser/--no-open-browser",
        help="Intentar abrir navegador en la máquina que ejecuta XBrain.",
    ),
) -> None:
    """Authenticate the current user with Google Drive."""
    cfg = _config()
    drive_login(cfg, port=port, open_browser=open_browser)
    typer.echo(f"Google Drive token guardado en {cfg.drive_token_path}")


@drive_app.command("logout")
@_handle_cli_errors
def drive_logout_cmd() -> None:
    """Remove the local Google Drive token."""
    cfg = _config()
    cfg.drive_token_path.unlink(missing_ok=True)
    typer.echo(f"Google Drive token eliminado: {cfg.drive_token_path}")


@drive_app.command("roots")
@_handle_cli_errors
def drive_roots(
    create: str | None = typer.Option(None, "--create", help="Crear una carpeta raíz."),
) -> None:
    """List available Drive folders or create a new root folder."""
    cfg = _config()
    client = drive_authenticate(cfg)
    if create:
        folder = client.create_folder(create)
        typer.echo(f"{folder.id}\t{folder.name}")
        return
    folders = client.list_folders()
    if not folders:
        typer.echo("No hay carpetas disponibles.")
        return
    for folder in folders:
        marker = " *" if folder.id == cfg.drive_root_folder_id else ""
        typer.echo(f"{folder.id}\t{folder.name}{marker}")


@drive_app.command("select-root")
@_handle_cli_errors
def drive_select_root(
    folder_id: str = typer.Argument(..., help="ID de carpeta de Google Drive"),
) -> None:
    """Select the Drive folder XBrain will use as its remote library root."""
    cfg = _config()
    _write_drive_selection(cfg, folder_id)
    typer.echo(f"Google Drive root seleccionado: {folder_id}")
    typer.echo("Activa [drive].enabled = true en config.toml para usarlo.")


@drive_app.command("sync")
@_handle_cli_errors
def drive_sync() -> None:
    """Synchronize the configured Drive root with the local cache."""
    cfg = _config()
    down = drive_sync_down(cfg)
    up = drive_sync_up(cfg)
    typer.echo(_format_sync_report("sync-down", down))
    typer.echo(_format_sync_report("sync-up", up))


def _copy_bootstrap_tree(source: Path, destination: Path, *, force: bool) -> int:
    if not source.exists():
        return 0
    if destination.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {destination}. Use --force.")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return sum(1 for path in destination.rglob("*") if path.is_file())


def _drive_cache_vault_output_dir(cfg: Config) -> Path:
    try:
        relative_output = cfg.output_dir.resolve().relative_to(cfg.vault.resolve())
    except ValueError:
        relative_output = Path(cfg.output_dir.name)
    return cfg.drive_cache_dir / "vault" / relative_output


@drive_app.command("bootstrap-local")
@_handle_cli_errors
def drive_bootstrap_local(
    force: bool = typer.Option(False, "--force", help="Sobrescribe la caché Drive local."),
) -> None:
    """Copy the current local library into the Drive cache and upload it."""
    cfg = _config()
    if not cfg.drive_root_folder_id:
        raise RuntimeError("Selecciona una carpeta con `xbrain drive select-root <folder-id>`.")
    cache_data = cfg.drive_cache_dir / "data"
    cache_vault = _drive_cache_vault_output_dir(cfg)
    if cfg.drive_enabled and (cfg.data_dir == cache_data or cfg.output_dir == cache_vault):
        raise RuntimeError(
            "bootstrap-local debe ejecutarse antes de activar [drive].enabled=true, "
            "o con una configuración local temporal."
        )
    data_files = _copy_bootstrap_tree(cfg.data_dir, cache_data, force=force)
    vault_files = _copy_bootstrap_tree(cfg.output_dir, cache_vault, force=force)
    report = drive_authenticate(cfg).sync_up(cfg.drive_root_folder_id, cfg.drive_cache_dir)
    typer.echo(
        f"Bootstrap local: data_files={data_files} vault_files={vault_files}\n"
        f"{_format_sync_report('sync-up', report)}"
    )


@drive_app.command("status")
@_handle_cli_errors
def drive_status() -> None:
    """Show Drive mode and local cache paths."""
    cfg = _config()
    typer.echo(f"enabled: {cfg.drive_enabled}")
    typer.echo(f"root_folder_id: {cfg.drive_root_folder_id or 'not selected'}")
    typer.echo(f"cache_dir: {cfg.drive_cache_dir}")
    typer.echo(f"data_dir: {cfg.data_dir}")
    typer.echo(f"vault: {cfg.vault}")
    typer.echo(f"media_dir: {cfg.media_dir}")
    typer.echo(f"token: {cfg.drive_token_path}")
    legacy_paths = _local_legacy_drive_paths(cfg)
    if legacy_paths:
        typer.echo("legacy local paths still present:")
        for path in legacy_paths:
            typer.echo(f"  {path}")


snapshot_app = typer.Typer(help="Gestionar snapshots de data/")
app.add_typer(snapshot_app, name="snapshot")


@snapshot_app.command("create")
@_handle_cli_errors
def snapshot_create_cmd(
    name: str | None = typer.Option(None, help="Optional directory label (default: 'manual')"),
) -> None:
    """Create a snapshot of data/ right now."""
    cfg = _config()
    with drive_write_session(cfg):
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
    with drive_write_session(cfg):
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
    with drive_write_session(cfg):
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
