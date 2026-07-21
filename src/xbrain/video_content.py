"""Helpers for consuming `x_video` sources across enrichment and synthesis."""

from __future__ import annotations

from xbrain.models import ContentSourceSuccess, VideoFrame
from xbrain.rubrics import truncate_transcript


def format_timestamp(seconds: float) -> str:
    """Return a compact mm:ss / hh:mm:ss timestamp for video frame notes."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def frame_descriptions(frames: list[VideoFrame]) -> list[str]:
    """Return content-bearing frame descriptions with stable timestamps."""
    descriptions: list[str] = []
    for frame in frames:
        description = " ".join(frame.description.split())
        if description:
            descriptions.append(f"{format_timestamp(frame.timestamp)}: {description}")
    return descriptions


def video_content_text(source: ContentSourceSuccess, limit: int | None = None) -> str | None:
    """Return the text signal an `x_video` source contributes downstream.

    Audio transcripts are used when `has_speech` is not explicitly false. Visual
    frame descriptions are also included, because a screen recording or slide deck
    may carry its useful content in frames even when no ASR transcript is
    available. A stale non-empty no-speech transcript with no frames is ignored.
    """
    if source.kind != "x_video":
        return None
    parts: list[str] = []
    if source.text and source.has_speech is not False:
        parts.append(source.text)
    frame_lines = frame_descriptions(source.frames)
    if frame_lines:
        parts.append("Video key frames:\n" + "\n".join(f"- {line}" for line in frame_lines))
    if not parts:
        return None
    text = "\n\n".join(parts)
    return truncate_transcript(text, limit) if limit is not None else text
