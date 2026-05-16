"""Parse the official X data archive (tweets.js) into Item objects."""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from xkb.extract.graphql import X_DATE_FORMAT
from xkb.models import Author, Item, Link, Media

_TWEET_FILES = ("data/tweets.js", "data/tweet.js")


def parse_archive(zip_path: Path, author: Author) -> list[Item]:
    """Extract all own tweets from an X data archive ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        name = _find_tweets_file(archive)
        raw = archive.read(name).decode("utf-8")
    payload = raw[raw.index("["):]
    entries = json.loads(payload)
    return [_archive_tweet_to_item(entry["tweet"], author) for entry in entries]


def _find_tweets_file(archive: zipfile.ZipFile) -> str:
    names = set(archive.namelist())
    for candidate in _TWEET_FILES:
        if candidate in names:
            return candidate
    raise ValueError(f"No tweets file in archive (looked for {_TWEET_FILES})")


def _archive_tweet_to_item(tweet: dict, author: Author) -> Item:
    rest_id = str(tweet["id_str"])
    links = [
        Link(url=u["expanded_url"], domain=urlparse(u["expanded_url"]).netloc)
        for u in tweet.get("entities", {}).get("urls", [])
        if u.get("expanded_url")
    ]
    media_entries = (
        tweet.get("extended_entities", {}).get("media")
        or tweet.get("entities", {}).get("media", [])
    )
    media = [
        Media(
            type="video" if m.get("type") in ("video", "animated_gif") else "photo",
            url=m.get("media_url_https") or m["expanded_url"],
        )
        for m in media_entries
        if m.get("media_url_https") or m.get("expanded_url")
    ]
    return Item(
        id=rest_id,
        source="own_tweet",
        url=f"https://x.com/{author.handle}/status/{rest_id}",
        author=author,
        text=tweet.get("full_text", ""),
        created_at=datetime.strptime(tweet["created_at"], X_DATE_FORMAT),
        captured_at=datetime.now(timezone.utc),
        media=media,
        links=links,
        quoted_id=None,
    )
