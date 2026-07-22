from tests.conftest import FakeLLMClient

from xbrain.models import MediaVideoPending
from xbrain.video_transcript import (
    VideoTranscript,
    fetch_video_transcript,
    parse_transcript_text,
    summarize_video_transcript,
)


def test_parse_transcript_text_cleans_vtt_cues_and_duplicates():
    raw = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Speaker>Hello &amp; welcome</v>

00:00:03.000 --> 00:00:05.000
Hello &amp; welcome

00:00:05.000 --> 00:00:07.000
Second idea.
"""

    assert parse_transcript_text(raw, "vtt") == "Hello & welcome\nSecond idea."


def test_parse_transcript_text_reads_srt():
    raw = """1
00:00:01,000 --> 00:00:03,000
Primera idea.

2
00:00:03,000 --> 00:00:05,000
Segunda idea.
"""

    assert parse_transcript_text(raw, "srt") == "Primera idea.\nSegunda idea."


def test_parse_transcript_text_reads_nested_json_text():
    raw = '{"events":[{"segs":[{"utf8":"Hello "},{"utf8":"world"}]},{"text":"Next"}]}'

    assert parse_transcript_text(raw, "json") == "Hello\nworld\nNext"


def test_fetch_video_transcript_requests_caption_url_only():
    class _Response:
        status_code = 200
        headers = {"content-type": "text/vtt"}
        text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nCaption line"

    class _Session:
        def __init__(self):
            self.calls: list[tuple[str, int]] = []

        def get(self, url, *, timeout):
            self.calls.append((url, timeout))
            return _Response()

    entry = MediaVideoPending(
        url="https://video.twimg.com/amplify_video/900/vid/720/a.mp4",
        thumbnail_url="https://pbs.twimg.com/poster.jpg",
        transcript_url="https://video.twimg.com/amplify_video/900/captions/en.vtt",
        transcript_language="en",
    )
    session = _Session()

    transcript = fetch_video_transcript(entry, session=session, timeout_seconds=12)

    assert session.calls == [("https://video.twimg.com/amplify_video/900/captions/en.vtt", 12)]
    assert transcript.text == "Caption line"
    assert transcript.language == "en"
    assert transcript.format == "vtt"


def test_summarize_video_transcript_uses_configured_text_llm_and_formats_markdown():
    client = FakeLLMClient(
        [
            {
                "title": "Video Strategy",
                "summary": "A useful executive summary.",
                "main_ideas": ["Idea one", "Idea two"],
                "first_order_conclusions": ["Direct conclusion"],
                "second_order_conclusions": ["Deeper implication"],
                "didactic_use": ["Teach it as a workflow"],
                "practical_applications": ["Apply it to the vault"],
            }
        ]
    )
    transcript = VideoTranscript(
        text="Original transcript content.",
        language="en",
        source_url="https://video.twimg.com/captions/en.vtt",
        format="vtt",
    )

    summary = summarize_video_transcript(
        "bookmark text",
        "alice",
        transcript,
        provider="nanogpt",
        model="zai-org/glm-5.2",
        output_language="Spanish",
        client=client,
    )

    assert summary.title == "Video Strategy"
    assert "### Executive Summary" in summary.markdown
    assert "A useful executive summary." in summary.markdown
    assert "- Idea one" in summary.markdown
    call = client.messages.calls[0]
    assert call["model"] == "zai-org/glm-5.2"
    assert "Write in Spanish" in call["system"]
    assert "Original video transcript:" in call["messages"][0]["content"]
