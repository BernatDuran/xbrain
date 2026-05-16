# tests/test_archive.py
import json
import zipfile
from pathlib import Path

from xkb.archive import parse_archive
from xkb.models import Author


def _make_archive(tmp_path: Path, filename: str = "data/tweets.js") -> Path:
    tweets = [
        {
            "tweet": {
                "id_str": "555",
                "created_at": "Wed May 10 14:23:00 +0000 2026",
                "full_text": "hello https://t.co/x",
                "entities": {
                    "urls": [{"expanded_url": "https://example.com/post"}]
                },
            }
        }
    ]
    body = "window.YTD.tweets.part0 = " + json.dumps(tweets)
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(filename, body)
    return zip_path


def test_parse_archive_extracts_own_tweets(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    items = parse_archive(_make_archive(tmp_path), author)
    assert len(items) == 1
    item = items[0]
    assert item.id == "555"
    assert item.source == "own_tweet"
    assert item.url == "https://x.com/vgonpa/status/555"
    assert item.links[0].domain == "example.com"


def test_parse_archive_handles_legacy_tweet_js_name(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    items = parse_archive(_make_archive(tmp_path, "data/tweet.js"), author)
    assert items[0].id == "555"
