"""Append-only embedding shards, kept locally and mirrored to the hub.

The shards are the only record of which photos have an embedding. Before a run, the
store reads the photo ids of the remote shards, so no photo is embedded twice, even on a new
machine. A worker slot tags its shards (`part-w1of4-…`). It reads the ids of its own shards,
of the untagged ones and of the shards of another worker count: the work that can overlap
with its own. The other slots of its count cannot overlap, so their shards are not read.
"""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.adapters.hub import Hub
from inat_landscape.domain.sampling import Slots, in_slots

_INDEX = "index"  # the done-ids snapshot; not a shard (the embeddings config reads *.parquet)


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
        """Upload the local shards that are not on the hub yet, in one commit. Return how many.

        The hub allows 128 commits per hour on a repository: tens of GPU jobs must not
        commit each shard.
        """
        remote = set(self._remote_shards())
        local = [str(p.relative_to(self.root)) for p in self._shards()]
        missing = [p for p in local if p not in remote]
        if missing:
            self.hub.upload_folder(self.root, f"Add {len(missing)} embedding shards", missing, [])
        return len(missing)

    def _remote_shards(self) -> list[str]:
        files = self.hub.list_files(self.remote_dir + "/")
        return [p for p in files if p.endswith(".parquet") and f"/{_INDEX}/" not in p]

    def remote_ids(self) -> set[int]:
        """Return the ids of the remote photos that can belong to this tag.

        The done-ids snapshot gives the ids of the shards it covers in one file; the other
        shards that can overlap this tag are read (`photo_id` column only). No shard is stored.
        """
        ids, covered = self._snapshot()
        own = _tag_slots(self.tag)
        if own is not None:
            ids = ids[in_slots(ids, own)]
        done = set(ids.tolist())
        for path in self._remote_shards():
            if path not in covered and self._mine(path):
                done.update(self.hub.read_ids(path))
        return done

    def snapshot(self) -> int:
        """Write and upload the done-ids snapshot of all the remote shards. Return its size.

        The shards that the old snapshot covers are not read again.
        """
        ids, covered = self._snapshot()
        shards = self._remote_shards()
        new = [np.asarray(self.hub.read_ids(p), np.int64) for p in shards if p not in covered]
        every = np.unique(np.concatenate([ids, *new]))
        table = pa.table({"photo_id": every}).replace_schema_metadata(
            {b"shards": json.dumps(sorted(set(shards) | covered)).encode()}
        )
        local = self.root / ".snapshot" / "done-ids.parquet"
        write_table(table, local)
        self.hub.upload_file(local, self._snapshot_path, f"Snapshot of {len(every)} done ids")
        shutil.rmtree(local.parent)
        return len(every)

    @property
    def _snapshot_path(self) -> str:
        return f"{self.remote_dir}/{_INDEX}/done-ids.parquet"

    def _snapshot(self) -> tuple[np.ndarray, set[str]]:
        """Return the ids and the covered shards of the remote snapshot (none if absent)."""
        if self._snapshot_path not in self.hub.list_files(f"{self.remote_dir}/{_INDEX}/"):
            return np.zeros(0, np.int64), set()
        staging = self.root / ".snapshot"
        table = pq.read_table(self.hub.download(self._snapshot_path, staging))
        shutil.rmtree(staging)
        covered = set(json.loads((table.schema.metadata or {}).get(b"shards", b"[]")))
        return table["photo_id"].to_numpy(), covered

    def _mine(self, path: str) -> bool:
        """Tell if a shard can hold photos of this tag: untagged, or a slot range (or another
        worker count) that overlaps this one."""
        name = path.rsplit("/", 1)[-1]
        own = _tag_slots(self.tag)
        other = _tag_slots(name.removeprefix("part-").split("-of")[0].rsplit("-", 1)[0])
        return own is None or other is None or own.overlaps(other)

    def stream(self) -> Iterator[tuple[list[int], np.ndarray]]:
        """Yield the ids and float16 embeddings of each shard, one shard at a time."""
        for path in self.shards():
            yield self.read(path)

    def shards(self) -> list[str]:
        """Return the paths of all the shards, local and remote, each once and in order."""
        local = {str(p.relative_to(self.root)) for p in self._shards()}
        return sorted(local | set(self._remote_shards()))

    def read(self, path: str) -> tuple[list[int], np.ndarray]:
        """Return the ids and float16 embeddings of one shard.

        A local shard is read in place. A remote shard with no local copy is downloaded into
        its own staging folder, read and deleted: shards can be read in parallel threads.
        """
        local = self.root / path
        staging = self.root / ".stream" / hashlib.sha256(path.encode()).hexdigest()[:16]
        table = read_table(local if local.exists() else self.hub.download(path, staging))
        shutil.rmtree(staging, ignore_errors=True)
        rows = table["embedding"].combine_chunks().flatten().to_numpy()
        return table["photo_id"].to_pylist(), rows.reshape(table.num_rows, -1)

    def prune(self, keep: np.ndarray, *, workers: int = 1, batch: int = 50) -> tuple[int, int]:
        """Remove from the remote shards the photos that are not in `keep`. Return the count of
        shards rewritten and of rows removed.

        A shard with rows to remove is replaced by a shard of its kept rows, with the same tag.
        Each commit replaces `batch` shards at most (the hub limits the commits per hour). The
        done-ids snapshot is deleted: it lists removed ids.
        """
        shards = self._remote_shards()
        batches = [shards[start : start + batch] for start in range(0, len(shards), batch)]
        counts = np.array([self._prune_batch(b, keep, workers) for b in batches]).reshape(-1, 2)
        return int(counts[:, 0].sum()), int(counts[:, 1].sum())

    def _prune_batch(self, chunk: list[str], keep: np.ndarray, workers: int) -> tuple[int, int]:
        """Replace the shards of `chunk` that have rows to remove, in one commit."""
        with ThreadPoolExecutor(workers) as pool:
            changed = [p for p in pool.map(self._pruned, chunk, [keep] * len(chunk)) if p.removed]
        if changed:
            self._replace(changed)
        return len(changed), sum(p.removed for p in changed)

    def _replace(self, changed: list["_Pruned"]) -> None:
        new = [p.new for p in changed if p.new]
        gone = [*(p.path for p in changed), self._snapshot_path]
        self.hub.upload_folder(self.root, f"Prune {len(changed)} embedding shards", new, gone)
        for name in new:
            (self.root / name).unlink()

    def _pruned(self, path: str, keep: np.ndarray) -> "_Pruned":
        """Write the kept rows of a remote shard as a new local shard, if it has rows to
        remove and rows to keep."""
        ids, embeddings = self.read(path)
        mask = np.isin(np.asarray(ids, np.int64), keep)
        removed = int((~mask).sum())
        if removed == 0 or removed == len(ids):
            return _Pruned(path, "", removed)
        tag = path.rsplit("/", 1)[-1].removeprefix("part-").rpartition("-")[0]
        kept = np.asarray(ids, np.int64)[mask].tolist()
        new = _write_shard(self.root, self.remote_dir, tag, kept, embeddings[mask])
        return _Pruned(path, str(new.relative_to(self.root)), removed)

    def done_ids(self) -> set[int]:
        """Return the ids of all photos that have an embedding."""
        return {
            i
            for shard in self._shards()
            for i in pq.read_table(shard, columns=["photo_id"])["photo_id"].to_pylist()
        }

    def add(self, photo_ids: Sequence[int], embeddings: np.ndarray, *, upload: bool = True) -> Path:
        """Write one shard, and upload it (else `push` uploads it later). The shard name is a
        hash of its ids."""
        if len(photo_ids) != len(embeddings):
            raise ValueError("photo_ids and embeddings differ in length")
        path = _write_shard(self.root, self.remote_dir, self.tag, photo_ids, embeddings)
        name = str(path.relative_to(self.root))
        if upload:
            self.hub.upload_file(path, name, f"Add {len(photo_ids)} embeddings")
        return path

    def _shards(self) -> list[Path]:
        return sorted(self.local_dir.glob("part-*.parquet"))


@dataclass(frozen=True)
class _Pruned:
    """One remote shard after the prune: the rows it lost, and its new shard ("" if none)."""

    path: str
    new: str
    removed: int


def _write_shard(
    root: Path, remote_dir: str, tag: str, photo_ids: Sequence[int], embeddings: np.ndarray
) -> Path:
    """Write one shard under `root / remote_dir`. Its name is a hash of its ids."""
    digest = hashlib.sha256(np.asarray(photo_ids, np.int64).tobytes()).hexdigest()[:16]
    prefix = f"part-{tag}-" if tag else "part-"
    path = root / remote_dir / f"{prefix}{digest}.parquet"
    flat = pa.array(np.asarray(embeddings, np.float16).ravel())
    column = pa.FixedSizeListArray.from_arrays(flat, embeddings.shape[1])
    write_table(pa.table({"photo_id": pa.array(photo_ids, pa.int64()), "embedding": column}), path)
    return path


def _tag_slots(tag: str) -> Slots | None:
    """Return the slots of a shard tag (`w3of9`, `w3-6of9`), or None for no tag."""
    match = re.fullmatch(r"w(\d+)(?:-(\d+))?of(\d+)", tag)
    if match is None:
        return None
    start, stop, count = match.groups()
    return Slots(int(start), int(stop) if stop else int(start) + 1, int(count))
