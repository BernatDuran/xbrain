# `digest-video` - videos without storing video bytes

`digest-video` turns bookmarked X videos into text for the normal
`enrich -> topics -> generate` pipeline without downloading or storing MP4,
audio, frames or thumbnails as local video artifacts.

The only supported input is a caption/text-track URL already exposed by X and
captured in `items.json`. If X does not expose captions for a video, XBrain marks
that item as `sin transcript` and stops. There is no MP4 fallback.

## What It Stores

For each video with captions:

1. XBrain fetches only the small caption/text file, usually VTT/SRT/JSON.
2. It parses that file into the original-language raw transcript.
3. It asks the configured text LLM for a medium-depth executive summary.
4. It stores the executive summary as the `x_video` content source used by
   `enrich`, `topics`, the dashboard and Ask XBrain.
5. It renders two nearby vault files under `videos/<video>/`:
   - `summary.md`: dashboard-ready executive summary.
   - `transcript.md`: raw transcript reference, marked `xbrain_exclude: true`.

The raw transcript is retained for audit/reading, but it is not indexed by the
dashboard or Ask XBrain and is not fed into topic synthesis.

## Run It

```bash
uv run xbrain digest-video --all-pending
uv run xbrain generate
```

Useful selectors:

```bash
uv run xbrain digest-video --ids 123,456
uv run xbrain digest-video --topic ai-coding
uv run xbrain digest-video --all-pending --limit 10
uv run xbrain digest-video --ids 123 --force
```

Output example:

```text
Videos: resumidos 6, ya digeridos 2, sin transcript 4, fallidos 0, ...
Dedup: 12 items <- 9 videos (6 procesados este run).
```

## Disabled Paths

These commands/options are intentionally disabled by storage policy:

```bash
uv run xbrain download-videos
uv run xbrain fetch-video
uv run xbrain digest-video --frames
uv run xbrain digest-video --vision-model xiaomi/mimo-v2.5
```

They do not write MP4, audio or frame files. `--max-size` on `digest-video` is
accepted for compatibility but ignored, because no video bytes are downloaded.

## Limitations

This only works when X exposes a caption/text-track URL in the video payload.
Many X videos do not include captions. For those, XBrain keeps the bookmark and
video metadata but does not manufacture a transcript from audio, because doing so
would require downloading media bytes.
