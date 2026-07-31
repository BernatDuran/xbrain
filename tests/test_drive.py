from pathlib import Path

from xbrain.config import load_config
from xbrain.drive import DriveClient, drive_write_session


class _Call:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _Files:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return _Call(
            {
                "files": [
                    {"id": "1", "name": "xbrain-root", "modifiedTime": "2026-07-30T00:00:00Z"}
                ]
            }
        )

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        body = kwargs["body"]
        return _Call({"id": "created", "name": body["name"], "modifiedTime": "now"})


class _Service:
    def __init__(self):
        self._files = _Files()

    def files(self):
        return self._files


class _TrashCall:
    def execute(self):
        return {}


class _TrashFiles:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return _Call({"files": [{"id": "child", "name": "child", "mimeType": "text/plain"}]})

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return _TrashCall()


class _TrashService:
    def __init__(self):
        self._files = _TrashFiles()

    def files(self):
        return self._files


def _write_config(root: Path) -> None:
    (root / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )


def test_drive_client_lists_folders():
    service = _Service()
    client = DriveClient(service)

    folders = client.list_folders()

    assert folders[0].id == "1"
    assert folders[0].name == "xbrain-root"
    _, kwargs = service.files().calls[0]
    assert "mimeType='application/vnd.google-apps.folder'" in kwargs["q"]


def test_drive_client_creates_folder():
    service = _Service()
    client = DriveClient(service)

    folder = client.create_folder("xbrain-root", parent_id="parent")

    assert folder.id == "created"
    _, kwargs = service.files().calls[0]
    assert kwargs["body"]["parents"] == ["parent"]


def test_drive_write_session_is_noop_when_drive_disabled(tmp_path: Path):
    _write_config(tmp_path)
    cfg = load_config(tmp_path)

    with drive_write_session(cfg):
        (tmp_path / "ran").write_text("yes", encoding="utf-8")

    assert (tmp_path / "ran").read_text(encoding="utf-8") == "yes"


def test_drive_trashes_folder_without_recursing_into_children():
    service = _TrashService()
    client = DriveClient(service)

    trashed = client._trash_tree(  # noqa: SLF001 - verifies the Drive sync primitive directly
        {
            "id": "folder",
            "name": "old-snapshot",
            "mimeType": "application/vnd.google-apps.folder",
        }
    )

    assert trashed == 1
    assert service.files().calls == [
        ("update", {"fileId": "folder", "body": {"trashed": True}, "supportsAllDrives": True})
    ]
