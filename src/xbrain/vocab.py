"""The `vocab` stage — induce the topic taxonomy from the corpus.

Map-reduce: each chunk of posts proposes candidate topics (map); one
consolidation call merges all candidates into exactly `target_count` topics
(reduce). The Anthropic client is injected so tests run offline.
"""
from __future__ import annotations

import json
import re

from xbrain.models import Item, Topic
from xbrain.rubrics import load_rubric

_MAP_MAX_TOKENS = 1000
_REDUCE_MAX_TOKENS = 2000
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    return json.loads(match.group(0))


def _chunks(items: list[Item], size: int) -> list[list[Item]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _call(client, model: str, max_tokens: int, system: str, user: str) -> dict:
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _extract_json(response.content[0].text)


def induce_vocab(
    store: dict[str, Item],
    target_count: int,
    model: str,
    client=None,
    chunk_size: int = 80,
) -> list[Topic]:
    """Induce `target_count` topics from the items in `store`."""
    if client is None:
        from anthropic import Anthropic  # lazy: tests inject a fake

        client = Anthropic()

    system = load_rubric("vocab")
    items = list(store.values())

    # --- Map: each chunk proposes candidate topics ---
    candidates: list[dict] = []
    for chunk in _chunks(items, chunk_size):
        posts = "\n".join(f"- {it.text}" for it in chunk)
        user = (
            "MAP STEP. Propose candidate topics for these posts. Respond with "
            'JSON: {"candidates": [{"slug": "...", "description": "..."}]}\n\n'
            + posts
        )
        candidates.extend(
            _call(client, model, _MAP_MAX_TOKENS, system, user).get(
                "candidates", []))

    # --- Reduce: consolidate into exactly target_count topics ---
    cand_block = "\n".join(
        f"- {c.get('slug')}: {c.get('description')}" for c in candidates)
    user = (
        f"REDUCE STEP. Consolidate these candidate topics into exactly "
        f"{target_count} final topics. Respond with JSON: "
        '{"topics": [{"slug": "...", "description": "..."}]}\n\n'
        + cand_block
    )
    final = _call(client, model, _REDUCE_MAX_TOKENS, system, user)
    return [Topic(**entry) for entry in final.get("topics", [])]
