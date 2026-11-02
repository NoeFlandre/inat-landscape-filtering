"""Hugging Face dataset repository access, and an offline stand-in."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

REPO_ID = "NoeFlandre/inat-landscape-filtering"


class Hub(Protocol):
    """The remote store that the pipeline reads from and writes to."""

    def list_files(self, prefix: str) -> list[str]: ...
    def download(self, path: str, root: Path) -> Path: ...
    def upload_file(self, local: Path, path: str, message: str) -> None: ...
    def upload_folder(
        self, root: Path, message: str, allow: Sequence[str], delete: Sequence[str]
    ) -> None: ...


class HfHub:
    """A public dataset repository on the Hugging Face Hub."""

    def __init__(self, repo_id: str = REPO_ID, api: HfApi | None = None) -> None:
        self.repo_id = repo_id
        self.api = api or HfApi()
        self._created = False

    def list_files(self, prefix: str) -> list[str]:
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            return []
        return sorted(f for f in files if f.startswith(prefix))

    def download(self, path: str, root: Path) -> Path:
        return Path(
            self.api.hf_hub_download(self.repo_id, path, repo_type="dataset", local_dir=root)
        )

    def upload_file(self, local: Path, path: str, message: str) -> None:
        self._ensure_repo()
        self.api.upload_file(
            path_or_fileobj=local,
            path_in_repo=path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=message,
        )

    def upload_folder(
        self, root: Path, message: str, allow: Sequence[str], delete: Sequence[str]
    ) -> None:
        self._ensure_repo()
        self.api.upload_folder(
            folder_path=root,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=message,
            allow_patterns=list(allow),
            delete_patterns=list(delete),
        )

    def _ensure_repo(self) -> None:
        if not self._created:
            self.api.create_repo(self.repo_id, repo_type="dataset", private=False, exist_ok=True)
            self._created = True


class OfflineHub:
    """No remote: everything stays in the local work directory."""

    def list_files(self, prefix: str) -> list[str]:  # noqa: ARG002 - same interface as HfHub
        return []

    def download(self, path: str, root: Path) -> Path:
        raise FileNotFoundError(f"offline: cannot download {path} into {root}")

    def upload_file(self, local: Path, path: str, message: str) -> None:
        pass

    def upload_folder(
        self, root: Path, message: str, allow: Sequence[str], delete: Sequence[str]
    ) -> None:
        pass
