"""HfHub logic over a fake HfApi; OfflineHub behaviour."""

from pathlib import Path
from typing import cast

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from inat_landscape.adapters.hub import HfHub, OfflineHub


class FakeApi:
    def __init__(self, files=None, missing=False):
        self.files = files or []
        self.missing = missing
        self.calls = []

    def list_repo_files(self, repo_id, repo_type):
        if self.missing:
            raise RepositoryNotFoundError(
                "missing",
                response=httpx.Response(404, request=httpx.Request("GET", "https://x")),  # ty: ignore[invalid-argument-type]
            )
        return self.files

    def hf_hub_download(self, repo_id, path, repo_type, local_dir):
        self.calls.append(("download", path))
        return str(Path(local_dir) / path)

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.calls.append(("create", repo_id, repo_type, private, exist_ok))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw["path_in_repo"], kw["repo_type"]))

    def delete_folder(self, **kw):
        self.calls.append(("delete_folder", kw["path_in_repo"]))
        if kw["path_in_repo"] == "missing":
            raise EntryNotFoundError("missing")

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw["allow_patterns"], kw["delete_patterns"]))


def _hub(api: FakeApi) -> HfHub:
    return HfHub("u/r", api=cast("HfApi", api))


def test_list_files_filters_by_prefix_and_sorts():
    hub = _hub(FakeApi(["b/2", "README.md", "b/1"]))
    assert hub.list_files("b/") == ["b/1", "b/2"]


def test_list_files_of_missing_repo_is_empty():
    assert _hub(FakeApi(missing=True)).list_files("") == []


def test_download_returns_local_path(tmp_path):
    assert _hub(FakeApi()).download("a/b", tmp_path) == tmp_path / "a/b"


def test_uploads_create_public_dataset_repo_once(tmp_path):
    api = FakeApi()
    hub = _hub(api)
    hub.upload_file(tmp_path, "x", "m")
    hub.upload_folder(tmp_path, "m", ["data/**"], ["data/**"])
    assert api.calls == [
        ("create", "u/r", "dataset", False, True),
        ("upload_file", "x", "dataset"),
        ("upload_folder", ["data/**"], ["data/**"]),
    ]


def test_offline_hub_has_no_remote(tmp_path):
    hub = OfflineHub()
    assert hub.list_files("") == []
    hub.upload_file(tmp_path, "x", "m")
    hub.upload_folder(tmp_path, "m", [], [])
    with pytest.raises(FileNotFoundError):
        hub.download("x", tmp_path)


def test_private_hub_creates_a_private_repo(tmp_path):
    api = FakeApi()
    HfHub("u/state", api=cast("HfApi", api), private=True).upload_file(tmp_path, "x", "m")
    assert api.calls[0] == ("create", "u/state", "dataset", True, True)


def test_delete_folder_ignores_a_missing_folder():
    api = FakeApi()
    hub = _hub(api)
    hub.delete_folder("data/x", "m")
    hub.delete_folder("missing", "m")
    assert ("delete_folder", "data/x") in api.calls
    OfflineHub().delete_folder("data/x", "m")


def test_read_ids_reads_the_photo_id_column_of_a_remote_parquet(tmp_path, monkeypatch):
    path = tmp_path / "datasets" / "o" / "r" / "e" / "part-1.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"photo_id": [5, 6], "embedding": [[1.0], [2.0]]}), path)
    monkeypatch.setattr(
        "inat_landscape.adapters.hub.HfFileSystem",
        lambda **_: DirFileSystem(path=str(tmp_path), fs=LocalFileSystem()),
    )
    assert HfHub("o/r", api=cast(HfApi, FakeApi())).read_ids("e/part-1.parquet") == [5, 6]


def test_offline_hub_cannot_read_ids():
    with pytest.raises(FileNotFoundError, match="offline"):
        OfflineHub().read_ids("e/part-1.parquet")
