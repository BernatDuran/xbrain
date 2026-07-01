# tests/test_describe_worksheet.py
import json
from datetime import datetime, timezone

from xbrain.describe import apply_describe_worksheet, export_describe_worksheet
from xbrain.generate import generate
from xbrain.models import (
    Author,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
)

DT = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _photo(local_path="1/0.png"):
    return MediaPhotoDownloaded(
        url="https://p/" + local_path,
        local_path=local_path,
        width=4,
        height=4,
        bytes_size=9,
        downloaded_at=DT,
    )


def _item(item_id="1", media=None):
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"text {item_id}",
        created_at=DT,
        captured_at=DT,
        media=media or [],
    )


def test_export_lists_eligible_photos_with_image_paths(tmp_path):
    store = {
        "1": _item("1", [_photo("1/0.png")]),
        "2": _item("2", [_photo("2/0.png"), _photo("2/1.png")]),
    }
    ws_path = tmp_path / "ws.json"
    n = export_describe_worksheet(
        store, tmp_path / "media", ws_path, version="v1", output_language="Spanish"
    )
    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    assert n == 3
    assert {(p["item_id"], p["index"]) for p in ws["photos"]} == {("1", 0), ("2", 0), ("2", 1)}
    assert ws["photos"][0]["image_path"].endswith("1/0.png")
    assert ws["rubric"] and ws["judgments"] == []


def test_apply_transitions_to_described_and_enforces_decorative_empty(tmp_path):
    store = {"1": _item("1", [_photo("1/0.png")]), "2": _item("2", [_photo("2/0.png")])}
    ws_path = tmp_path / "ws.json"
    export_describe_worksheet(
        store, tmp_path / "media", ws_path, version="v1", output_language="Spanish"
    )
    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    ws["judgments"] = [
        {
            "item_id": "1",
            "index": 0,
            "is_decorative": False,
            "description": "Un gráfico de barras.",
        },
        {"item_id": "2", "index": 0, "is_decorative": True, "description": "ignored by contract"},
    ]
    ws_path.write_text(json.dumps(ws), encoding="utf-8")

    assert apply_describe_worksheet(store, ws_path) == 2
    d1, d2 = store["1"].media[0], store["2"].media[0]
    assert isinstance(d1, MediaPhotoDescribed) and d1.description == "Un gráfico de barras."
    assert not d1.is_decorative
    assert isinstance(d2, MediaPhotoDescribed) and d2.is_decorative and d2.description == ""


def test_apply_skips_unknown_id_and_index(tmp_path):
    store = {"1": _item("1", [_photo("1/0.png")])}
    ws_path = tmp_path / "ws.json"
    ws_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "language": "Spanish",
                "judgments": [
                    {"item_id": "1", "index": 9, "is_decorative": False, "description": "x"},
                    {"item_id": "nope", "index": 0, "is_decorative": False, "description": "y"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert apply_describe_worksheet(store, ws_path) == 0
    assert isinstance(store["1"].media[0], MediaPhotoDownloaded)  # unchanged


def _described(local_path, description, *, decorative=False):
    return MediaPhotoDescribed(
        url="https://p/" + local_path,
        local_path=local_path,
        width=4,
        height=4,
        bytes_size=9,
        downloaded_at=DT,
        is_decorative=decorative,
        description=description,
        description_lang="Spanish",
        description_version="v1",
        described_at=DT,
    )


def test_generate_renders_photo_description_as_caption(tmp_path):
    store = {
        "1": _item("1", [_described("1/0.png", "Un diagrama de flujo.")]),
        "2": _item("2", [_described("2/0.png", "", decorative=True)]),
    }
    generate(store, tmp_path, output_language="Spanish")

    note1 = next((tmp_path / "items").glob("*-1.md")).read_text(encoding="utf-8")
    assert "> Un diagrama de flujo." in note1  # described photo → searchable caption

    note2 = next((tmp_path / "items").glob("*-2.md")).read_text(encoding="utf-8")
    assert "\n> " not in note2  # decorative photo → no caption line
