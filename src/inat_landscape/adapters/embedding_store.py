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
        """Return all ids and the float16 embeddings (as stored). Each id appears once.

        The shards are copied one by one into one preallocated array, so the memory peak is
        the array plus one shard.
        """
        shards = self._shards()
        if not shards:
            return [], np.zeros((0, 0), np.float16)
        ids, matrix = self._read_into_one_array(shards)
        first = _first_positions(ids)
        if len(first) == len(ids):
            return ids, matrix
        return [ids[n] for n in first], matrix[first]

    def _read_into_one_array(self, shards: list[Path]) -> tuple[list[int], np.ndarray]:
        counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
        matrix = np.empty((sum(counts), self.dim()), np.float16)
        ids: list[int] = []
        start = 0
        for shard in shards:
            table = read_table(shard)
            rows = table["embedding"].combine_chunks().flatten().to_numpy()
            matrix[start : start + table.num_rows] = rows.reshape(table.num_rows, -1)
            ids += table["photo_id"].to_pylist()
            start += table.num_rows
        return ids, matrix

    def dim(self) -> int:
        """Return the embedding length, or 0 when the store is empty."""
        shards = self._shards()
        return pq.read_schema(shards[0]).field("embedding").type.list_size if shards else 0

    def _shards(self) -> list[Path]:
        return sorted(self.local_dir.glob("part-*.parquet"))


def _first_positions(ids: list[int]) -> list[int]:
    """Return, in order, the position of the first copy of each id."""
    return sorted({i: n for n, i in reversed(list(enumerate(ids)))}.values())
