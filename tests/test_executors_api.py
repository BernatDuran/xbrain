# tests/test_executors_api.py
import json
from datetime import datetime, timezone

from xbrain.executors.api import ApiExecutor, _user_prompt
from xbrain.models import Author, Item, Link, Topic


def _item(item_id: str, **extra) -> Item:
    return Item(
        id=item_id, source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"), text="un post sobre LLMs",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc), **extra,
    )


VOCAB = [Topic(slug="ai-coding", description="LLMs writing software."),
         Topic(slug="misc", description="Posts that do not fit a topic.")]


class _FakeMessages:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Block:
            type = "text"
            text = json.dumps(self._payload)

        class _Resp:
            content = [_Block()]

        return _Resp()


class _FakeClient:
    def __init__(self, payload: dict):
        self.messages = _FakeMessages(payload)


def test_api_executor_returns_one_judgment_per_item():
    client = _FakeClient({"summary": "r", "primary_topic": "ai-coding",
                          "topics": ["ai-coding"]})
    ex = ApiExecutor(model="claude-haiku-4-5-20251001", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"1", "2"}
    assert len(client.messages.calls) == 2


def test_api_executor_sends_the_configured_model():
    client = _FakeClient({"summary": "r", "primary_topic": "misc",
                          "topics": ["misc"]})
    ApiExecutor(model="claude-sonnet-4-6", client=client).enrich_items(
        [_item("1")], VOCAB)
    assert client.messages.calls[0]["model"] == "claude-sonnet-4-6"


def test_user_prompt_includes_link_domains_and_folder():
    item = _item("1", links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")],
                 bookmark_folder="AI papers")
    prompt = _user_prompt(item, VOCAB)
    assert "arxiv.org" in prompt
    assert "AI papers" in prompt


def test_user_prompt_includes_folder_when_no_links():
    item = _item("1", bookmark_folder="AI papers")
    prompt = _user_prompt(item, VOCAB)
    assert "AI papers" in prompt
    assert not item.links


def test_user_prompt_includes_link_domains_when_no_folder():
    item = _item("1", links=[Link(url="https://arxiv.org/abs/1",
                                  domain="arxiv.org")])
    prompt = _user_prompt(item, VOCAB)
    assert "arxiv.org" in prompt
    assert not item.bookmark_folder


class _SequencedMessages:
    """A fake `messages` that returns a different payload per call."""

    def __init__(self, payloads: list):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload

        class _Block:
            type = "text"
            text = json.dumps(payload)

        class _Resp:
            content = [_Block()]

        return _Resp()


class _SequencedClient:
    def __init__(self, payloads):
        self.messages = _SequencedMessages(payloads)


def test_api_executor_skips_wrong_shape_response(capsys):
    # A response that is valid JSON but not a judgment object must be skipped
    # with a warning, not silently become an empty enrichment.
    client = _SequencedClient([
        {"not": "a judgment"},
        {"summary": "r", "primary_topic": "misc", "topics": ["misc"]},
    ])
    ex = ApiExecutor(model="m", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"2"}  # item 1 skipped
    err = capsys.readouterr().err
    assert "enrichment failed for item 1" in err


def test_api_executor_skips_item_on_api_failure(capsys):
    # A transient API failure on one item must not abort the whole batch.
    client = _SequencedClient([
        RuntimeError("503 service unavailable"),
        {"summary": "r", "primary_topic": "misc", "topics": ["misc"]},
    ])
    ex = ApiExecutor(model="m", client=client)
    out = ex.enrich_items([_item("1"), _item("2")], VOCAB)
    assert {j.item_id for j in out} == {"2"}
    assert "enrichment failed for item 1" in capsys.readouterr().err
