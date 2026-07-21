from xbrain.models import ContentSourceSuccess, VideoFrame
from xbrain.video_content import video_content_text


def test_video_content_includes_transcript_and_frame_descriptions():
    source = ContentSourceSuccess(
        kind="x_video",
        url="https://x.com/v",
        text="Spoken transcript.",
        has_speech=True,
        frames=[
            VideoFrame(timestamp=12.0, local_path="1/frames/0.png", description="A title slide."),
            VideoFrame(timestamp=95.0, local_path="1/frames/1.png", description="A code demo."),
        ],
    )

    text = video_content_text(source)

    assert "Spoken transcript." in text
    assert "0:12: A title slide." in text
    assert "1:35: A code demo." in text


def test_video_content_uses_frames_for_visual_only_video():
    source = ContentSourceSuccess(
        kind="x_video",
        url="https://x.com/v",
        text="",
        has_speech=False,
        frames=[VideoFrame(timestamp=3.0, local_path="1/frames/0.png", description="A workflow.")],
    )

    assert video_content_text(source) == "Video key frames:\n- 0:03: A workflow."


def test_video_content_ignores_stale_no_speech_text_without_frames():
    source = ContentSourceSuccess(
        kind="x_video",
        url="https://x.com/v",
        text="stale transcript",
        has_speech=False,
    )

    assert video_content_text(source) is None
