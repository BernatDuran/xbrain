"""Build and render the self-contained interactive HTML metrics dashboard.

`compute_dashboard_data` is pure — it turns the store (plus topic overviews and
an id→note map) into the JSON blob the vendored template consumes.
`collect_thumbnails` does the photo file IO. `render_dashboard_html` injects the
blob and the vendored ECharts library into the template. `generate` wires these
together and writes `<output_dir>/dashboard.html`; nothing here touches a browser.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from xbrain.content_policy import has_video
from xbrain.models import (
    ArticleImageBlock,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
    TopicPage,
)

logger = logging.getLogger(__name__)

# Slug words rendered upper-case (acronyms) or as an ampersand in topic labels.
_ACRONYMS = frozenset({"ai", "ml", "llm", "mcp", "api", "ux", "ui", "gpu", "seo", "vc", "3d", "x"})
_THUMB_LIMIT = 18
# The three video media variants share `type="video"`, `thumbnail_url` and
# `duration_millis`; discriminating on the union (not a `getattr("type")`
# sniff) keeps the typed access mypy-checked, matching `generate._render_media_lines`.
_VIDEO_TYPES = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)
# A described photo (`MediaPhotoDescribed`) IS a downloaded photo — it carries
# the same `local_path`/bytes and only adds a vision caption. `xbrain describe`
# transitions Downloaded -> Described in place, so every downloaded-photo count,
# thumbnail source, and photo-post filter must accept BOTH variants or the whole
# corpus of described photos vanishes from the dashboard. Mirrors the isinstance
# grouping in `generate._render_media_lines`.
_DOWNLOADED_PHOTO_TYPES = (MediaPhotoDownloaded, MediaPhotoDescribed)


def _article_image_blocks(item: Item) -> list[ArticleImageBlock]:
    """Return inline image blocks from X Articles on an item."""
    if item.content is None:
        return []
    return [
        block
        for source in item.content.sources
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_article"
        for block in source.blocks
        if isinstance(block, ArticleImageBlock)
    ]


def _downloaded_image_entries(item: Item) -> list[tuple[MediaPhotoDownloaded | MediaPhotoDescribed, str]]:
    """Downloaded images for thumbnailing: post photos first, then article images."""
    entries: list[tuple[MediaPhotoDownloaded | MediaPhotoDescribed, str]] = [
        (entry, "post")
        for entry in item.media
        if isinstance(entry, _DOWNLOADED_PHOTO_TYPES)
    ]
    entries.extend(
        (block.media, "article")
        for block in _article_image_blocks(item)
        if isinstance(block.media, _DOWNLOADED_PHOTO_TYPES)
    )
    return entries


def _article_media_counts(item: Item) -> tuple[int, int, int]:
    """Return downloaded, pending and undescribed counts for inline Article images."""
    downloaded = pending = undescribed = 0
    for block in _article_image_blocks(item):
        entry = block.media
        if isinstance(entry, _DOWNLOADED_PHOTO_TYPES):
            downloaded += 1
        if isinstance(entry, MediaPhotoPending):
            pending += 1
        if isinstance(entry, MediaPhotoDownloaded):
            undescribed += 1
    return downloaded, pending, undescribed


def humanize_topic(slug: str) -> str:
    """Turn a topic slug into a display label (``ai-coding`` → ``AI Coding``)."""
    out: list[str] = []
    for word in slug.split("-"):
        if word in _ACRONYMS:
            out.append(word.upper())
        elif word == "and":
            out.append("&")
        else:
            out.append(word.capitalize())
    return " ".join(out)


def _summary(item: Item) -> str:
    """The item's Spanish enrichment summary, or a text fallback."""
    if item.enriched and item.enriched.summary:
        return item.enriched.summary
    return item.text[:200]


def _date(item: Item) -> str:
    return item.created_at.date().isoformat()


def _month(item: Item) -> str:
    return item.created_at.strftime("%Y-%m")


def _primary(item: Item) -> str | None:
    return item.enriched.primary_topic if item.enriched else None


def _confidence(item: Item) -> str | None:
    return item.enriched.topic_confidence if item.enriched else None


def _suggested_topics(item: Item) -> list[str]:
    return item.enriched.suggested_new_topics if item.enriched else []


def _recent(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The `n` most recent rows by their ``date`` field."""
    return sorted(rows, key=lambda r: r["date"], reverse=True)[:n]


def _row(item: Item, id2note: dict[str, str], slug2label: dict[str, str]) -> dict[str, Any]:
    """A drill-down post row: who/when/what plus deep links to X and the vault."""
    topic = _primary(item)
    suggestions = _suggested_topics(item)
    return {
        "handle": item.author.handle,
        "name": item.author.name,
        "source": item.source,
        "content_type": _content_type(item),
        "date": _date(item),
        "topic": slug2label.get(topic or "", topic or "—"),
        "topic_slug": topic,
        "confidence": _confidence(item) or "unknown",
        "suggested_topics": suggestions,
        "summary": _summary(item),
        "url": item.url,
        "note": id2note.get(item.id),
    }


def _article_sources(item: Item) -> list[ContentSourceSuccess | ContentSourceFailure]:
    """Article-like fetched sources attached to an item."""
    if item.content is None:
        return []
    return [
        source
        for source in item.content.sources
        if source.kind in ("external_article", "x_article")
        and isinstance(source, (ContentSourceSuccess, ContentSourceFailure))
    ]


def _saved_article_sources(item: Item) -> list[ContentSourceSuccess]:
    return [
        source for source in _article_sources(item) if isinstance(source, ContentSourceSuccess)
    ]


def _failed_article_sources(item: Item) -> list[ContentSourceFailure]:
    return [
        source for source in _article_sources(item) if isinstance(source, ContentSourceFailure)
    ]


def _content_type(item: Item) -> str:
    """Dashboard content bucket: article, video, article_failed or post_only."""
    if _saved_article_sources(item):
        return "article"
    if any(
        isinstance(source, ContentSourceSuccess) and source.kind == "x_video"
        for source in (item.content.sources if item.content else [])
    ) or has_video(item):
        return "video"
    if _failed_article_sources(item):
        return "article_failed"
    return "post_only"


def collect_thumbnails(
    items: list[Item], media_root: Path | None, id2note: dict[str, str], limit: int = _THUMB_LIMIT
) -> list[dict[str, Any]]:
    """Base64-encode a sample of downloaded images with their post metadata.

    Reads at most `limit` downloaded images from `media_root`, downscaling each
    to a small JPEG data URI so the dashboard stays self-contained. Returns an
    empty list when `media_root` is None. Unreadable files are skipped.
    """
    if media_root is None:
        return []
    thumbs: list[dict[str, Any]] = []
    for item in items:
        for entry, location in _downloaded_image_entries(item):
            if len(thumbs) >= limit:
                return thumbs
            path = media_root / entry.local_path
            if not path.exists():
                continue
            try:
                with Image.open(path) as opened:
                    rgb = opened.convert("RGB")
                rgb.thumbnail((200, 200))
                buffer = io.BytesIO()
                rgb.save(buffer, "JPEG", quality=78)
            except Exception:  # noqa: BLE001 - a bad image file must not break the dashboard
                logger.debug("Skipping unreadable thumbnail %s", path, exc_info=True)
                continue
            # Surface the vision caption on the thumbnail so the photo drawer says
            # what each image actually is. Only `MediaPhotoDescribed` carries a
            # caption, and the model validator forces a decorative one to "" —
            # so a single typed isinstance suffices (a plain `MediaPhotoDownloaded`
            # yields ""). Typed access, not a `getattr` sniff, per the `_VIDEO_TYPES`
            # note above.
            desc = entry.description if isinstance(entry, MediaPhotoDescribed) else ""
            thumbs.append(
                {
                    "thumb": "data:image/jpeg;base64,"
                    + base64.b64encode(buffer.getvalue()).decode(),
                    "url": item.url,
                    "note": id2note.get(item.id),
                    "handle": item.author.handle,
                    "date": _date(item),
                    "summary": _summary(item),
                    "desc": desc,
                    "kind": location,
                }
            )
    return thumbs


def _growth(items: list[Item]) -> dict[str, Any]:
    """Monthly new counts and cumulative totals (all / bookmarks / own posts)."""
    per_month: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        per_month[_month(item)].append(item)
    months = sorted(per_month)
    new_total, cum_total, cum_bm, cum_own = [], [], [], []
    run_t = run_b = run_o = 0
    for month in months:
        group = per_month[month]
        run_t += len(group)
        run_b += sum(1 for i in group if i.source == "bookmark")
        run_o += sum(1 for i in group if i.source == "own_tweet")
        new_total.append(len(group))
        cum_total.append(run_t)
        cum_bm.append(run_b)
        cum_own.append(run_o)
    return {
        "months": months,
        "new_total": new_total,
        "cum_total": cum_total,
        "cum_bm": cum_bm,
        "cum_own": cum_own,
        "_per_month": per_month,
    }


def _longform(items: list[Item], id2note: dict[str, str]) -> dict[str, Any]:
    """Captured long-form counts (external vs X) plus a recent article list."""
    counts = {"ext_saved": 0, "ext_failed": 0, "x_saved": 0, "x_failed": 0}
    articles: list[dict[str, Any]] = []
    for item in items:
        for source in _article_sources(item):
            prefix = "ext" if source.kind == "external_article" else "x"
            if isinstance(source, ContentSourceSuccess):
                counts[f"{prefix}_saved"] += 1
                articles.append(
                    {
                        "title": source.title or source.url[:80],
                        "url": source.url,
                        "source": "External" if source.kind == "external_article" else "X Article",
                        "handle": item.author.handle,
                        "date": _date(item),
                        "summary": _summary(item),
                        "topic": humanize_topic(_primary(item) or "misc"),
                        "confidence": _confidence(item) or "unknown",
                        "suggested_topics": _suggested_topics(item),
                        "post": item.url,
                        "note": id2note.get(item.id),
                    }
                )
            else:
                counts[f"{prefix}_failed"] += 1
    saved = counts["ext_saved"] + counts["x_saved"]
    total = saved + counts["ext_failed"] + counts["x_failed"]
    return {
        **counts,
        "saved": saved,
        "total": total,
        "saved_pct": round(saved / total * 100, 1) if total else 0.0,
        "items": _recent(articles, 60),
    }


def _article_failures(items: list[Item], id2note: dict[str, str]) -> list[dict[str, Any]]:
    """Recent failed article fetches with enough context to retry/debug."""
    rows: list[dict[str, Any]] = []
    for item in items:
        if not item.content:
            continue
        for source in item.content.sources:
            if not (
                isinstance(source, ContentSourceFailure)
                and source.kind in ("external_article", "x_article")
            ):
                continue
            rows.append(
                {
                    "handle": item.author.handle,
                    "date": _date(item),
                    "kind": source.kind,
                    "reason": source.failure_reason,
                    "url": source.url,
                    "post": item.url,
                    "note": id2note.get(item.id),
                    "summary": _summary(item),
                }
            )
    return _recent(rows, 20)


def _article_failure_count(items: list[Item]) -> int:
    """Total failed linked article sources across the corpus."""
    return sum(
        1
        for item in items
        if item.content is not None
        for source in item.content.sources
        if isinstance(source, ContentSourceFailure)
        and source.kind in ("external_article", "x_article")
    )


def _failed_bookmark_count(items: list[Item]) -> int:
    """Bookmarks with at least one failed linked article source."""
    return sum(
        1
        for item in items
        if item.source == "bookmark"
        and item.content is not None
        and any(
            isinstance(source, ContentSourceFailure)
            and source.kind in ("external_article", "x_article")
            for source in item.content.sources
        )
    )


def _media_counts(items: list[Item]) -> dict[str, int]:
    """Downloaded/pending photo counts and captured-video count across the store."""
    downloaded = pending = videos = article_downloaded = article_pending = 0
    for item in items:
        for entry in item.media:
            if isinstance(entry, _DOWNLOADED_PHOTO_TYPES):
                downloaded += 1
            elif isinstance(entry, MediaPhotoPending):
                pending += 1
            elif isinstance(entry, _VIDEO_TYPES):
                videos += 1
        item_article_downloaded, item_article_pending, _ = _article_media_counts(item)
        article_downloaded += item_article_downloaded
        article_pending += item_article_pending
    return {
        "photos_downloaded": downloaded,
        "photos_pending": pending,
        "article_images_downloaded": article_downloaded,
        "article_images_pending": article_pending,
        "videos": videos,
    }


def _ops(items: list[Item], id2note: dict[str, str], rows: _Rows) -> dict[str, Any]:
    """Mobile-first operational status for the ingestion workflow."""
    downloaded_undescribed = sum(
        1
        for item in items
        for entry in item.media
        if isinstance(entry, MediaPhotoDownloaded)
    )
    article_pending = article_undescribed = 0
    for item in items:
        _, item_article_pending, item_article_undescribed = _article_media_counts(item)
        article_pending += item_article_pending
        article_undescribed += item_article_undescribed
    pending_fetch = sum(1 for item in items if item.links and item.content is None)
    pending_enrich = sum(1 for item in items if item.enriched is None)
    failures = _article_failures(items, id2note)
    failure_count = _article_failure_count(items)
    failed_bookmarks = _failed_bookmark_count(items)
    recent_bookmarks = _recent(rows([item for item in items if item.source == "bookmark"]), 8)
    return {
        "command": "uv run xbrain refresh-all --headless",
        "retry_failed_command": "uv run xbrain retry-failed --source bookmarks --headless",
        "serve_command": "uv run xbrain serve-dashboard --host 127.0.0.1 --port 8765",
        "retry_failed_bookmarks": failed_bookmarks,
        "pending": {
            "fetch": pending_fetch,
            "media": sum(
                1 for item in items for entry in item.media if isinstance(entry, MediaPhotoPending)
            )
            + article_pending,
            "describe": downloaded_undescribed + article_undescribed,
            "enrich": pending_enrich,
            "article_failures": failure_count,
        },
        "recent_bookmarks": recent_bookmarks,
        "article_failures": failures,
    }


def _videos(items: list[Item], id2note: dict[str, str]) -> list[dict[str, Any]]:
    """A recent sample of video posts with poster, duration and deep links."""
    rows: list[dict[str, Any]] = []
    for item in items:
        for entry in item.media:
            if not isinstance(entry, _VIDEO_TYPES):
                continue
            dur = entry.duration_millis
            rows.append(
                {
                    "handle": item.author.handle,
                    "date": _date(item),
                    "summary": _summary(item),
                    "dur": round(dur / 1000) if dur else None,
                    "poster": entry.thumbnail_url,
                    "url": item.url,
                    "note": id2note.get(item.id),
                }
            )
            break
    return _recent(rows, 12)


_Rows = Callable[[list[Item]], list[dict[str, Any]]]


def _primaries(items: list[Item]) -> list[str]:
    """The primary-topic slugs of the enriched items (drops items without one)."""
    return [p for item in items if (p := _primary(item)) is not None]


def _topics_section(
    items: list[Item], topic_freq: "Counter[str]", topic_pages: dict[str, TopicPage], rows: _Rows
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The topics bar chart data plus per-topic drill-down (overview + posts)."""
    topics_sorted = [
        {"slug": s, "label": humanize_topic(s), "count": c} for s, c in topic_freq.most_common()
    ]
    by_topic: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if (p := _primary(item)) is not None:
            by_topic[p].append(item)
    topic_data = {
        s: {
            "label": humanize_topic(s),
            "count": topic_freq[s],
            "overview": topic_pages[s].overview if s in topic_pages else "",
            "samples": _recent(rows(by_topic[s]), 5),
        }
        for s in topic_freq
    }
    return topics_sorted, topic_data


def _authors_section(items: list[Item], rows: _Rows) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Top bookmarked authors plus per-author drill-down."""
    by_author: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.source == "bookmark":
            by_author[item.author.handle].append(item)
    top = Counter({h: len(v) for h, v in by_author.items()}).most_common(10)
    authors = [{"handle": h, "name": by_author[h][0].author.name, "count": c} for h, c in top]
    author_data = {
        h: {
            "name": by_author[h][0].author.name,
            "count": len(by_author[h]),
            "samples": _recent(rows(by_author[h]), 8),
        }
        for h, _ in top
    }
    return authors, author_data


def _domains_section(items: list[Item], rows: _Rows) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Top linked domains (x.com excluded) plus per-domain drill-down."""
    by_domain: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        for link in item.links:
            by_domain[link.domain].append(item)
    top = Counter({d: len(v) for d, v in by_domain.items() if d != "x.com"}).most_common(10)
    domains = [{"domain": d, "count": c} for d, c in top]
    domain_data = {d: {"count": c, "samples": _recent(rows(by_domain[d]), 7)} for d, c in top}
    return domains, domain_data


def _months_section(per_month: dict[str, list[Item]], rows: _Rows) -> dict[str, Any]:
    """Per-month drill-down: counts, top topics, top authors and sample posts."""
    out: dict[str, Any] = {}
    for month, group in per_month.items():
        tt = Counter(_primaries(group))
        ta = Counter(i.author.handle for i in group if i.source == "bookmark")
        out[month] = {
            "count": len(group),
            "bm": sum(1 for i in group if i.source == "bookmark"),
            "own": sum(1 for i in group if i.source == "own_tweet"),
            "top_topics": [{"label": humanize_topic(s), "count": c} for s, c in tt.most_common(6)],
            "top_authors": [{"handle": h, "count": c} for h, c in ta.most_common(6)],
            "samples": _recent(rows(group), 6),
        }
    return out


_META_LONGFORM_KEYS = (
    "ext_saved",
    "ext_failed",
    "x_saved",
    "x_failed",
    "saved",
    "total",
    "saved_pct",
)


def _library_section(items: list[Item]) -> dict[str, Any]:
    """Counts by retained content type for the article/video-first product."""
    counts = Counter(_content_type(item) for item in items)
    return {
        "articles": counts["article"],
        "videos": counts["video"],
        "article_failed": counts["article_failed"],
        "post_only": counts["post_only"],
    }


def _taxonomy_section(items: list[Item], rows: _Rows) -> dict[str, Any]:
    """Topic-assignment quality signals exposed in the dashboard."""
    confidence = Counter(_confidence(item) or "unknown" for item in items)
    misc_items = [item for item in items if _primary(item) == "misc"]
    low_items = [item for item in items if _confidence(item) == "low"]
    suggested = Counter(
        suggestion for item in items for suggestion in _suggested_topics(item)
    )
    review_items = {
        item.id: item
        for item in sorted(
            [*misc_items, *low_items],
            key=lambda i: i.created_at,
            reverse=True,
        )
    }
    return {
        "confidence": {
            "high": confidence["high"],
            "medium": confidence["medium"],
            "low": confidence["low"],
            "unknown": confidence["unknown"],
        },
        "misc": len(misc_items),
        "low": len(low_items),
        "suggested": [
            {"slug": slug, "count": count} for slug, count in suggested.most_common(20)
        ],
        "review_items": rows(list(review_items.values()))[:12],
    }


def _meta(
    items: list[Item],
    topic_freq: "Counter[str]",
    longform: dict[str, Any],
    media: dict[str, int],
    library: dict[str, Any],
    taxonomy: dict[str, Any],
    updated: str,
    bookmarks: int,
    own: int,
) -> dict[str, Any]:
    """The KPI header block (totals, enrichment, long-form, media, timestamp)."""
    return {
        "total": len(items),
        "bookmarks": bookmarks,
        "own": own,
        "enriched": sum(1 for i in items if i.enriched),
        "topics_count": len(topic_freq),
        "longform": {k: longform[k] for k in _META_LONGFORM_KEYS},
        "media": media,
        "library": library,
        "taxonomy": taxonomy,
        "updated": updated,
    }


def compute_dashboard_data(
    items: list[Item],
    topic_pages: dict[str, TopicPage],
    id2note: dict[str, str],
    thumbs: list[dict[str, Any]],
    updated: str,
) -> dict[str, Any]:
    """Assemble the full JSON blob the dashboard template consumes.

    Pure: no file or network IO (photo thumbnails are computed by
    `collect_thumbnails` and injected via `thumbs`). `updated` is a display
    string (the caller stamps the generation date).
    """
    growth = _growth(items)
    per_month: dict[str, list[Item]] = growth.pop("_per_month")
    topic_freq: "Counter[str]" = Counter(_primaries(items))
    slug2label = {s: humanize_topic(s) for s in topic_freq}

    def rows(group: list[Item]) -> list[dict[str, Any]]:
        return [_row(i, id2note, slug2label) for i in group]

    topics_sorted, topic_data = _topics_section(items, topic_freq, topic_pages, rows)
    authors, author_data = _authors_section(items, rows)
    domains, domain_data = _domains_section(items, rows)
    months_data = _months_section(per_month, rows)
    longform = _longform(items, id2note)
    media = _media_counts(items)
    library = _library_section(items)
    taxonomy = _taxonomy_section(items, rows)
    bookmark_items = [i for i in items if i.source == "bookmark"]
    own_items = [i for i in items if i.source == "own_tweet"]
    photo_posts = [i for i in items if _downloaded_image_entries(i)]

    return {
        "meta": _meta(
            items,
            topic_freq,
            longform,
            media,
            library,
            taxonomy,
            updated,
            len(bookmark_items),
            len(own_items),
        ),
        **growth,
        "topics_sorted": topics_sorted,
        "topic_data": topic_data,
        "authors": authors,
        "author_data": author_data,
        "domains": domains,
        "domain_data": domain_data,
        "months_data": months_data,
        "longform_full": longform,
        "taxonomy": taxonomy,
        "photos": {
            "downloaded": media["photos_downloaded"],
            "pending": media["photos_pending"],
            "article_downloaded": media["article_images_downloaded"],
            "article_pending": media["article_images_pending"],
            "thumbs": thumbs,
            "samples": _recent(rows(photo_posts), 6),
        },
        "videos": {"count": media["videos"], "items": _videos(items, id2note)},
        "sources": {
            "bookmark": {"count": len(bookmark_items), "samples": _recent(rows(bookmark_items), 6)},
            "own_tweet": {"count": len(own_items), "samples": _recent(rows(own_items), 6)},
        },
        "ops": _ops(items, id2note, rows),
    }


def _resource(name: str) -> str:
    return (resources.files("xbrain") / "resources" / name).read_text(encoding="utf-8")


def _escape_for_script(payload: str) -> str:
    """Make a JSON payload safe to inline inside a ``<script>`` block.

    ``json.dumps(ensure_ascii=False)`` leaves ``<``/``>``/``&`` raw and lets the
    JS line terminators U+2028/U+2029 and lone UTF-16 surrogates (from mangled
    emoji in scraped X text) through. Un-escaped, a ``</script>`` in any post
    summary/title/handle would close the tag at HTML-parse time — a stored-XSS
    break-out — and a lone surrogate would crash the UTF-8 write. The escaped
    forms ``JSON.parse`` back to the originals, so displayed content is identical.
    """
    escaped = (
        payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(chr(0x2028), "\\u2028")
        .replace(chr(0x2029), "\\u2029")
    )
    # Round-trip through UTF-8 to drop lone surrogates (illegal in UTF-8) so
    # writing the dashboard can never abort the whole `generate` run.
    return escaped.encode("utf-8", "replace").decode("utf-8")


def render_dashboard_html(
    data: dict[str, Any], template: str | None = None, echarts: str | None = None
) -> str:
    """Inject the data blob and the ECharts library into the vendored template.

    Loads the template and library from `xbrain/resources/` when not supplied.
    The result is a self-contained HTML document (no external scripts except the
    Google Fonts stylesheet).
    """
    template = template if template is not None else _resource("dashboard.template.html")
    echarts = echarts if echarts is not None else _resource("echarts.min.js")
    payload = _escape_for_script(json.dumps(data, ensure_ascii=False))
    # Inject the trusted (fixed) library first and the user-derived payload LAST,
    # so a summary/title containing the literal `/*__ECHARTS__*/` sentinel can
    # never splice the library into the JSON on a re-scan.
    return template.replace("/*__ECHARTS__*/", echarts).replace("/*__DATA__*/", payload)
