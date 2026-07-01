# tests/test_dashboard.py
from datetime import datetime, timezone

from xbrain.dashboard import compute_dashboard_data, humanize_topic, render_dashboard_html
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDownloaded,
    MediaPhotoPending,
    MediaVideoPending,
)

DT = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _item(
    item_id,
    source="bookmark",
    topic="ai-coding",
    handle="alice",
    name="Alice",
    links=None,
    media=None,
    content=None,
):
    return Item(
        id=item_id,
        source=source,
        url=f"https://x.com/{handle}/status/{item_id}",
        author=Author(handle=handle, name=name),
        text=f"text {item_id}",
        created_at=DT,
        captured_at=DT,
        links=links or [],
        media=media or [],
        content=content,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="resumen",
            primary_topic=topic,
            topics=[topic],
        ),
    )


def test_humanize_topic_acronyms_and_ampersand():
    assert humanize_topic("ai-coding") == "AI Coding"
    assert humanize_topic("agentic-engineering") == "Agentic Engineering"
    assert humanize_topic("ai-and-jobs") == "AI & Jobs"
    assert humanize_topic("llm-foundations") == "LLM Foundations"


def test_compute_counts_topics_authors_and_deep_links():
    items = [
        _item("100", "bookmark", "ai-coding", "alice"),
        _item(
            "101",
            "bookmark",
            "ai-coding",
            "bob",
            "Bob",
            links=[Link(url="https://ex.com/a", domain="ex.com")],
        ),
        _item("102", "own_tweet", "claude-code", "vgonpa"),
    ]
    id2note = {"100": "/v/items/100.md", "101": "/v/items/101.md", "102": "/v/items/102.md"}
    data = compute_dashboard_data(items, {}, id2note, [], "JUN 1, 2026")

    m = data["meta"]
    assert (m["total"], m["bookmarks"], m["own"], m["enriched"], m["topics_count"]) == (
        3,
        2,
        1,
        3,
        2,
    )
    assert data["topics_sorted"][0] == {"slug": "ai-coding", "label": "AI Coding", "count": 2}
    # own_tweet authors are excluded from the "bookmarked authors" chart
    assert {a["handle"] for a in data["authors"]} == {"alice", "bob"}
    assert data["domains"][0]["domain"] == "ex.com"
    assert "2026-06" in data["months_data"]

    row = data["topic_data"]["ai-coding"]["samples"][0]
    assert row["url"].startswith("https://x.com/")
    assert row["note"].endswith(".md")


def test_long_form_and_media_counts():
    items = [
        _item(
            "1",
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceSuccess(
                        kind="external_article", url="https://ex.com/x", text="body", title="T"
                    )
                ],
            ),
        ),
        _item(
            "2",
            content=Content(
                fetched_at=DT,
                sources=[
                    ContentSourceFailure(
                        kind="external_article", url="https://ex.com/y", failure_reason="paywall"
                    )
                ],
            ),
        ),
        _item(
            "3",
            media=[
                MediaPhotoDownloaded(
                    url="https://p",
                    local_path="3/0.png",
                    width=10,
                    height=10,
                    bytes_size=99,
                    downloaded_at=DT,
                ),
                MediaVideoPending(url="https://v"),
            ],
        ),
        _item("4", media=[MediaPhotoPending(url="https://p2")]),
    ]
    data = compute_dashboard_data(items, {}, {}, [], "JUN 1, 2026")

    lf = data["meta"]["longform"]
    assert (lf["ext_saved"], lf["ext_failed"], lf["saved"], lf["total"]) == (1, 1, 1, 2)
    assert data["longform_full"]["items"][0]["title"] == "T"

    md = data["meta"]["media"]
    assert (md["photos_downloaded"], md["photos_pending"], md["videos"]) == (1, 1, 1)


def test_render_injects_data_and_library_and_leaves_no_placeholder():
    html = render_dashboard_html(
        {"meta": {"total": 7}}, template="A /*__DATA__*/ B /*__ECHARTS__*/ C", echarts="LIB"
    )
    assert '"total": 7' in html
    assert "LIB" in html
    assert "__DATA__" not in html and "__ECHARTS__" not in html


def test_render_uses_vendored_resources():
    html = render_dashboard_html({"meta": {"total": 1}})
    assert '"total": 1' in html
    assert "/*__DATA__*/" not in html
    assert "echarts" in html.lower()


def test_generate_writes_dashboard_and_links_it_from_index(tmp_path):
    from xbrain.generate import generate

    store = {"1": _item("1"), "2": _item("2", "own_tweet", "claude-code", "vgonpa")}
    generate(store, tmp_path, output_language="Spanish")

    dashboard = tmp_path / "dashboard.html"
    assert dashboard.exists()
    assert "/*__DATA__*/" not in dashboard.read_text(encoding="utf-8")  # data injected
    assert "dashboard.html" in (tmp_path / "_index.md").read_text(encoding="utf-8")
