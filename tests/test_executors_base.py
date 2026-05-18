# tests/test_executors_base.py
from xbrain.executors.base import EnrichmentJudgment, EnrichmentExecutor


def test_judgment_holds_summary_and_topics():
    j = EnrichmentJudgment(item_id="1", summary="s",
                           primary_topic="ai-coding", topics=["ai-coding"])
    assert j.item_id == "1"
    assert j.primary_topic == "ai-coding"


def test_a_minimal_executor_satisfies_the_protocol():
    class Fake:
        def enrich_items(self, items, vocab):
            return [EnrichmentJudgment(item_id=i.id, summary="s",
                                       primary_topic="misc", topics=["misc"])
                    for i in items]

    executor: EnrichmentExecutor = Fake()
    assert hasattr(executor, "enrich_items")
