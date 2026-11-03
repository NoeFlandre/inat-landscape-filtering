"""Append-only embedding shards, kept locally and mirrored to the hub.

The shards are the only record of which photos have an embedding. Before a run, the
store reads the photo ids of the remote shards, so no photo is embedded twice, even on a new
machine. A worker slot tags its shards (`part-w1of4-…`) and reads the ids of its own shards
and of the untagged ones only: this is the work that can overlap with its own.
"""

import hashlib
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.adapters.hub import Hub


class EmbeddingStore:
    """Embedding shards under `root / remote_dir`, mirrored at `remote_dir` on the hub."""

    def __init__(self, root: Path, remote_dir: str, hub: Hub, *, tag: str = "") -> None:
        self.root = root
        self.remote_dir = remote_dir.strip("/")
        self.hub = hub
        self.tag = tag

    @property
    def local_dir(self) -> Path:
        return self.root / self.remote_dir

    def push(self) -> int:
        """Upload local shards that are not on the hub yet. Return how many."""
        remote = set(self._remote_shards())
        local = [str(p.relative_to(self.root)) for p in self._shards()]
        missing = [p for p in local if p not in remote]
        for path in missing:
            self.hub.upload_file(self.root / path, path, "Add embeddings made offline")
        return len(missing)

    def _remote_shards(self) -> list[str]:
        return [p for p in self.hub.list_files(self.remote_dir + "/") if p.endswith(".parquet")]

    def remote_ids(self) -> set[int]:
        """Return the ids of the remote shards of this tag and of the untagged shards.

        Only the `photo_id` column is read: no shard is stored.
        """
        return {
            i for path in self._remote_shards() if self._mine(path) for i in self.hub.read_ids(path)
        }

    def _mine(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return not self.tag or not name.startswith("part-w") or name.startswith(f"part-{self.tag}-")

    def stream(self) -> Iterator[tuple[list[int], np.ndarray]]:
        """Yield the ids and float16 embeddings of each shard, one shard at a time.

        Local shards are read in place. A remote shard with no local copy is downloaded, read
        and deleted, so the disk keeps one remote shard at most.
        """
        local = {str(p.relative_to(self.root)): p for p in self._shards()}
        staging = self.root / ".stream"
        for path in sorted(set(local) | set(self._remote_shards())):
            table = read_table(local.get(path) or self.hub.download(path, staging))
            shutil.rmtree(staging, ignore_errors=True)
            rows = table["embedding"].combine_chunks().flatten().to_numpy()
            yield table["photo_id"].to_pylist(), rows.reshape(table.num_rows, -1)

    def done_ids(self) -> set[int]:
        """Return the ids of all photos that have an embedding."""
        return {
            i
            for shard in self._shards()
            for i in pq.read_table(shard, columns=["photo_id"])["photo_id"].to_pylist()
        }

    def add(self, photo_ids: Sequence[int], embeddings: np.ndarray) -> Path:
        """Write one shard and upload it. The shard name is a hash of its ids."""
        if len(photo_ids) != len(embeddings):
            raise ValueError("photo_ids and embeddings differ in length")
        digest = hashlib.sha256(np.asarray(photo_ids, np.int64).tobytes()).hexdigest()[:16]
        prefix = f"part-{self.tag}-" if self.tag else "part-"
        name = f"{self.remote_dir}/{prefix}{digest}.parquet"
        path = self.root / name
        flat = pa.array(np.asarray(embeddings, np.float16).ravel())
        column = pa.FixedSizeListArray.from_arrays(flat, embeddings.shape[1])
        write_table(
            pa.table({"photo_id": pa.array(photo_ids, pa.int64()), "embedding": column}), path
        )
        self.hub.upload_file(path, name, f"Add {len(photo_ids)} embeddings")
        return path

    def _shards(self) -> list[Path]:
        return sorted(self.local_dir.glob("part-*.parquet"))
