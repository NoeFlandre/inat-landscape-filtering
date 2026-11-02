"""Append-only embedding shards, kept locally and mirrored to the hub.

The shards are the only record of which photos have an embedding. Before a run, the
store pulls the remote shards, so no photo is embedded twice, even on a new machine.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.adapters.hub import Hub


class EmbeddingStore:
    """Embedding shards under `root / remote_dir`, mirrored at `remote_dir` on the hub."""

    def __init__(self, root: Path, remote_dir: str, hub: Hub) -> None:
        self.root = root
        self.remote_dir = remote_dir.strip("/")
        self.hub = hub

    @property
    def local_dir(self) -> Path:
        return self.root / self.remote_dir

    def pull(self) -> int:
        """Download remote shards that are not local yet. Return how many."""
        remote = set(self._remote_shards())
        missing = sorted(p for p in remote if not (self.root / p).exists())
        for path in missing:
            self.hub.download(path, self.root)
        return len(missing)

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
        name = f"{self.remote_dir}/part-{digest}.parquet"
        path = self.root / name
        flat = pa.array(np.asarray(embeddings, np.float16).ravel())
        column = pa.FixedSizeListArray.from_arrays(flat, embeddings.shape[1])
        write_table(
            pa.table({"photo_id": pa.array(photo_ids, pa.int64()), "embedding": column}), path
        )
        self.hub.upload_file(path, name, f"Add {len(photo_ids)} embeddings")
        return path

    def load(self) -> tuple[list[int], np.ndarray]:
        """Return all ids and the float16 embeddings (as stored). Each id appears once."""
        tables = [read_table(s) for s in self._shards()]
        if not tables:
            return [], np.zeros((0, 0), np.float16)
        table = pa.concat_tables(tables)
        ids = table["photo_id"].to_pylist()
        first = sorted({i: n for n, i in reversed(list(enumerate(ids)))}.values())
        flat = table["embedding"].combine_chunks().flatten().to_numpy()
        matrix = flat.reshape(table.num_rows, -1)
        return [ids[n] for n in first], matrix[first]

    def dim(self) -> int:
        """Return the embedding length, or 0 when the store is empty."""
        shards = self._shards()
        return pq.read_schema(shards[0]).field("embedding").type.list_size if shards else 0

    def _shards(self) -> list[Path]:
        return sorted(self.local_dir.glob("part-*.parquet"))
